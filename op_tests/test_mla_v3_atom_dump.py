# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Replay ATOM's real DeepSeek-V3 dense MLA decode and validate the aiter op.

Companion to the V4 sparse variant (``test_mla_v4_*.py``) and the methodology
doc ``poc_kl/mi400/mla/toaiter/v3/mla_atom_dump_howto.md``.

ATOM (``ATOM_USE_TRITON_MLA=1``) runs the triton kernel
``aiter.ops.triton.attention.mla_decode.decode_attention_fwd`` on the V3 decode
path (``attention_mla.py::MLAAttention._forward_decode`` -> ``triton_block_table``
branch). ``atom/model_ops/mla_dump.py::dump_v3_dense_decode`` records the EXACT
kernel inputs + output ``o`` (the golden) right after that call. This script
replays the dump through three passes and compares against the golden:

  --pass triton : re-run aiter's decode_attention_fwd (the same kernel ATOM ran)
                  -> cos(triton, atom) ~= 0   (dump fidelity / determinism)
  --pass torch  : independent naive fp32 softmax MLA decode reference
                  -> cos(torch, atom) ~= 1e-7 (numeric correctness, cross-check)
  --pass aiter  : the op under test, aiter.mla.mla_decode_fwd (page_size=1)
                  -> cos(aiter, atom) (main) + cos(aiter, ref) (avoids the
                  self-consistency trap)
  --pass both   : aiter + the torch reference in one run.

Two input modes:
  --from-dump DIR        : read mla_v3_decode.rank*.*.pt (exact tensors + golden o)
  --from-params MANIFEST : read only mla_calls.rank*.jsonl, regenerate random
                           cases of the same (bs, H, D, ctx) shape, torch == golden

Unlike the V4 triton kernel (which lives in ATOM and had to be vendored), the V3
triton golden is itself an aiter kernel, so the triton pass imports it directly
from aiter -- no vendoring and no ``import atom`` needed.
"""

import argparse
import glob
import json
import math
import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def cos_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = (a.square() + b.square()).sum().clamp_min(1e-12)
    return float(1.0 - 2.0 * (a * b).sum() / denom)


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )


def _dequant_kv(kv_cache: torch.Tensor, k_scale) -> torch.Tensor:
    """Dequantize an fp8 token-major cache to bf16 (scale==1 for bf16 KV)."""
    if _is_fp8(kv_cache.dtype):
        scale = float(k_scale) if k_scale is not None else 1.0
        return (kv_cache.float() * scale).to(torch.bfloat16)
    return kv_cache.to(torch.bfloat16)


def _build_indptr(block_table, context_lens, total_q, kv_lora_rank):
    """Derive per-token (page_size=1) CSR metadata for aiter.mla.mla_decode_fwd
    from the dumped dense block_table [bs, max_blocks] + context_lens [bs]."""
    bs = block_table.shape[0]
    decode_qlen = max(total_q // bs, 1)
    ctx = context_lens.to(torch.int64)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=block_table.device)
    kv_indptr[1:] = torch.cumsum(ctx, 0).to(torch.int32)
    kv_indices = torch.empty(
        int(kv_indptr[-1].item()), dtype=torch.int32, device=block_table.device
    )
    for b in range(bs):
        L = int(ctx[b].item())
        kv_indices[int(kv_indptr[b]) : int(kv_indptr[b + 1])] = block_table[b, :L]
    qo_indptr = (
        torch.arange(bs + 1, dtype=torch.int32, device=block_table.device) * decode_qlen
    )
    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=block_table.device)
    return qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, decode_qlen


# --------------------------------------------------------------------------- #
# reference / passes
# --------------------------------------------------------------------------- #
def torch_ref_decode(q, kv_cache, block_table, context_lens, sm_scale,
                     kv_lora_rank, k_scale):
    """Naive fp32 MLA decode. q:[total_q,H,D] kv:[slots,1,D] (D=lora+pe).
    Single KV head broadcast across H query heads; v = k[:, :lora]. Causal for
    decode_qlen>1. Returns o:[total_q,H,kv_lora_rank]."""
    total_q, H, D = q.shape
    bs = block_table.shape[0]
    decode_qlen = max(total_q // bs, 1)
    kv = _dequant_kv(kv_cache, k_scale).float().squeeze(1)  # [slots, D]
    qf = q.float()
    o = torch.zeros(total_q, H, kv_lora_rank, dtype=torch.float32, device=q.device)
    ctx = context_lens.to(torch.int64)
    for i in range(total_q):
        b = i // decode_qlen
        qpos = i % decode_qlen
        eff = int(ctx[b].item()) - (decode_qlen - 1 - qpos)
        if eff <= 0:
            continue
        slots = block_table[b, :eff].to(torch.int64)
        k = kv.index_select(0, slots)            # [eff, D]
        v = k[:, :kv_lora_rank]                   # [eff, lora]
        scores = (qf[i] @ k.t()) * sm_scale       # [H, eff]
        probs = torch.softmax(scores, dim=-1)
        o[i] = probs @ v                          # [H, lora]
    return o


def run_triton_pass(q, kv_cache, block_table, context_lens, sm_scale, page_size,
                    num_kv_splits, kv_lora_rank, k_scale, v_scale):
    from aiter.ops.triton.attention.mla_decode import decode_attention_fwd

    total_q, H, D = q.shape
    k_buffer = kv_cache.unsqueeze(2)               # [slots, page_size, 1, D]
    v_buffer = k_buffer[..., :kv_lora_rank]
    o = torch.empty(total_q, H, kv_lora_rank, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty(total_q, H, dtype=torch.float32, device=q.device)
    attn_logits = torch.empty(
        total_q, H, num_kv_splits, kv_lora_rank + 1, dtype=torch.float32, device=q.device
    )
    ks = torch.tensor(float(k_scale) if k_scale is not None else 1.0,
                      dtype=torch.float32, device=q.device)
    vs = torch.tensor(float(v_scale) if v_scale is not None else 1.0,
                      dtype=torch.float32, device=q.device)
    decode_attention_fwd(
        q, k_buffer, v_buffer, o, lse, block_table, context_lens, attn_logits,
        num_kv_splits, sm_scale, page_size, k_scale=ks, v_scale=vs,
    )
    return o


def run_aiter_pass(q, kv_cache, block_table, context_lens, sm_scale,
                   kv_lora_rank, k_scale):
    from aiter.mla import mla_decode_fwd

    total_q, H, D = q.shape
    # Dequantize fp8 KV to bf16 so the page_size=1 decode runs a clean bf16
    # kernel (q is already bf16 on the triton dump; scales fold to 1).
    kv = _dequant_kv(kv_cache, k_scale).contiguous()
    qb = q.to(torch.bfloat16).contiguous()
    qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, decode_qlen = _build_indptr(
        block_table, context_lens, total_q, kv_lora_rank
    )
    kv_buffer = kv.view(kv.shape[0], 1, 1, D)      # page_size=1, nhead_kv=1
    o = torch.empty(total_q, H, kv_lora_rank, dtype=torch.bfloat16, device=q.device)
    one = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    mla_decode_fwd(
        qb, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
        max_seqlen_q=decode_qlen, page_size=1, sm_scale=sm_scale,
        q_scale=one, kv_scale=one,
    )
    return o


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def _to_dev(payload):
    out = {}
    for k, v in payload.items():
        out[k] = v.to(DEVICE) if torch.is_tensor(v) else v
    return out


def run_from_dump(args):
    files = sorted(glob.glob(os.path.join(args.from_dump, "mla_v3_decode.rank*.*.pt")))
    if not files:
        raise SystemExit(f"no mla_v3_decode.*.pt found under {args.from_dump}")
    npass = nfail = 0
    for f in files:
        d = _to_dev(torch.load(f, map_location=DEVICE))
        p = d["params"]
        q, kv_cache, bt = d["q"], d["kv_cache"], d["block_table"]
        ctx, golden = d["context_lens"], d["o"]
        sm_scale = p["softmax_scale"]
        page_size = p["page_size"]
        nks = p["num_kv_splits"]
        lora = p["kv_lora_rank"]
        k_scale = d.get("k_scale", None)
        v_scale = d.get("v_scale", None)

        ref = torch_ref_decode(q, kv_cache, bt, ctx, sm_scale, lora, k_scale)
        line = (f"[{os.path.basename(f)}] bs={p['bs']} total_q={p['total_q']} "
                f"H={p['H']} D={p['D']} max_ctx={p['max_ctx']} kv={p['kv_dtype']}")

        cos_torch = cos_diff(ref, golden)
        line += f" | cos(torch,atom)={cos_torch:.3e}"

        if args.pass_ in ("triton", "both"):
            o_t = run_triton_pass(q, kv_cache, bt, ctx, sm_scale, page_size, nks,
                                  lora, k_scale, v_scale)
            line += f" cos(triton,atom)={cos_diff(o_t, golden):.3e}"
        if args.pass_ in ("aiter", "both"):
            o_a = run_aiter_pass(q, kv_cache, bt, ctx, sm_scale, lora, k_scale)
            ca_atom = cos_diff(o_a, golden)
            ca_ref = cos_diff(o_a, ref)
            line += f" cos(aiter,atom)={ca_atom:.3e} cos(aiter,ref)={ca_ref:.3e}"
            main = max(ca_atom, ca_ref)
        else:
            main = cos_torch

        ok = math.isfinite(main) and main < args.cos_threshold
        line += "  PASS" if ok else "  FAIL"
        print(line)
        npass += int(ok)
        nfail += int(not ok)
    print(f"\n{npass} passed, {nfail} failed (threshold={args.cos_threshold})")
    raise SystemExit(1 if nfail else 0)


def run_from_params(args):
    lines = [json.loads(x) for x in open(args.from_params) if x.strip()]
    lines = [x for x in lines if x.get("op") == "decode_attention_fwd"]
    if not lines:
        raise SystemExit(f"no V3 decode_attention_fwd entries in {args.from_params}")
    torch.manual_seed(0)
    npass = nfail = 0
    for p in lines:
        bs, H = p["bs"], p["H"]
        lora, pe = p["kv_lora_rank"], p["qk_rope_head_dim"]
        D = lora + pe
        total_q = p["total_q"]
        ctx = torch.tensor(p["context_lens"], dtype=torch.int32, device=DEVICE)
        sm_scale = p["softmax_scale"]
        max_ctx = int(ctx.max().item())
        kv = torch.randn(max_ctx * bs, 1, D, dtype=torch.bfloat16, device=DEVICE) * 0.1
        q = torch.randn(total_q, H, D, dtype=torch.bfloat16, device=DEVICE) * 0.1
        bt = torch.zeros(bs, max_ctx, dtype=torch.int32, device=DEVICE)
        off = 0
        for b in range(bs):
            L = int(ctx[b].item())
            bt[b, :L] = torch.arange(off, off + L, dtype=torch.int32, device=DEVICE)
            off += L
        ref = torch_ref_decode(q, kv, bt, ctx, sm_scale, lora, None)
        o_a = run_aiter_pass(q, kv, bt, ctx, sm_scale, lora, None)
        c = cos_diff(o_a, ref)
        ok = math.isfinite(c) and c < args.cos_threshold
        print(f"[params bs={bs} H={H} total_q={total_q} max_ctx={max_ctx}] "
              f"cos(aiter,ref)={c:.3e}  {'PASS' if ok else 'FAIL'}")
        npass += int(ok)
        nfail += int(not ok)
    print(f"\n{npass} passed, {nfail} failed (threshold={args.cos_threshold})")
    raise SystemExit(1 if nfail else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-dump", metavar="DIR",
                   help="replay mla_v3_decode.*.pt tensor dumps (golden = ATOM o)")
    g.add_argument("--from-params", metavar="MANIFEST",
                   help="regenerate random cases from mla_calls.rank*.jsonl shapes")
    ap.add_argument("--pass", dest="pass_", default="both",
                    choices=["aiter", "triton", "torch", "both"])
    ap.add_argument("--cos-threshold", type=float, default=2e-2)
    args = ap.parse_args()
    if args.from_dump:
        run_from_dump(args)
    else:
        run_from_params(args)


if __name__ == "__main__":
    main()
