# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 / mi400 seg MLA decode: triton-as-golden numerical + perf compare.

Self-contained unit test (no tensor loading). For each (nhead, decode_qlen,
batch, ctx) combo it generates ONE logical fp8 Q/KV case and feeds the SAME
fp8 values to two kernels:

  - aiter  : aiter.mla.mla_decode_fwd, the gfx1250 seg asm decode. Consumes a
             page-level seg-packed KV cache ([num_pages, page_size*512 nope |
             page_size*64 pe]) and the 768-padded selected Q layout.
  - triton : op_tests/triton_tests/utils/mla_decode_ref.decode_attention_fwd,
             the known-good reference. Consumes a token-major interleaved KV
             cache ([num_tokens, 1, 576]) and token-level kv_indptr/indices.

triton runs in bf16 on the fp8-dequantized inputs, so it acts as the golden:
the only difference vs aiter is the kernel, not the quantization. The headline
pass/fail is cos_diff(aiter_o, triton_o) < threshold. Both kernels are timed
with run_perftest and reported side by side (us / TFLOPS / TB/s).

This mirrors the historical "feed the same input to triton (good) and the seg
path, compare the decode output" debug method, but as a repeatable UT.

Examples:
  # default sweep on gfx1250
  python3 test_mla_mi400_triton.py
  # single combo
  python3 test_mla_mi400_triton.py -n 128,1 -b 8 -c 1024
"""

import argparse
import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

import aiter
import aiter.mla
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.attention import mla_decode_stage1_asm_fwd
from aiter.test_common import benchmark, run_perftest

# triton golden
from triton_tests.utils.mla_decode_ref import decode_attention_fwd

# In lean containers aiter.__init__ may skip bulk op exports; register the op
# the seg decode needs explicitly (same as test_mla_mi400.py).
aiter.mla_decode_stage1_asm_fwd = mla_decode_stage1_asm_fwd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


@dataclass(frozen=True)
class MlaMi400KernelVariant:
    name: str
    nhead: int
    decode_qlen: int


_MI400_KERNEL_VARIANTS = [
    MlaMi400KernelVariant(name="qh16-q1-16mx1-32nx4-np-3p", nhead=16, decode_qlen=1),
    MlaMi400KernelVariant(name="qh16-q2-16mx2-32nx4-np-3p", nhead=16, decode_qlen=2),
    MlaMi400KernelVariant(name="qh32-q1-32mx1-32nx4-np-3p", nhead=32, decode_qlen=1),
    MlaMi400KernelVariant(name="qh16-q4-16mx4-64nx1-np", nhead=16, decode_qlen=4),
    MlaMi400KernelVariant(name="qh64-q1-16mx4-64nx1-np", nhead=64, decode_qlen=1),
    MlaMi400KernelVariant(name="qh128-q1-16mx4-64nx1-np", nhead=128, decode_qlen=1),
]
_MI400_VARIANT_BY_KEY = {(v.nhead, v.decode_qlen): v for v in _MI400_KERNEL_VARIANTS}
_MI400_VARIANT_BY_KEY_NAME = {v.name: v for v in _MI400_KERNEL_VARIANTS}

_DEFAULT_NHEAD = [(16, 1), (32, 1), (64, 1), (128, 1), (16, 2), (16, 4)]
_DEFAULT_CTX_LENS = [17, 65, 256, 1024, 4096]
_DEFAULT_BATCH_SIZES = [1, 4, 16, 64]


# --------------------------------------------------------------------------- #
# layout helpers (copied from test_mla_mi400.py to keep this file standalone;
# importing test_mla_mi400 would trigger its module-level argparse + sweep).
# --------------------------------------------------------------------------- #
def _pack_rope_split3_q_pages(tensor, nope_dim, rope_dim, padded_stride_bytes=768):
    shape = tensor.shape
    assert shape[-1] == nope_dim + rope_dim
    elem_size = tensor.element_size()
    if padded_stride_bytes % elem_size != 0:
        raise ValueError("rope_split3 padded stride must be element aligned")
    padded_dim = padded_stride_bytes // elem_size
    if padded_dim < shape[-1]:
        raise ValueError(
            f"rope_split3 padded dim {padded_dim} is smaller than Q dim {shape[-1]}"
        )
    rows = tensor.reshape(-1, shape[-1])
    padded = torch.zeros(
        (rows.shape[0], padded_dim), dtype=tensor.dtype, device=tensor.device
    )
    padded[:, : shape[-1]].copy_(rows)
    return torch.as_strided(
        padded,
        size=shape,
        stride=(shape[1] * shape[2] * padded_dim, shape[2] * padded_dim, padded_dim, 1),
    )


def _pack_rope_split2_kv_pages(tensor, nope_dim, rope_dim):
    pages, page_size, nhead_kv, head_dim = tensor.shape
    assert nhead_kv == 1
    assert head_dim == nope_dim + rope_dim
    packed = torch.cat(
        (
            tensor[..., :nope_dim].reshape(pages, page_size * nope_dim),
            tensor[..., nope_dim:].reshape(pages, page_size * rope_dim),
        ),
        dim=-1,
    )
    return packed.reshape(pages, page_size, nhead_kv, head_dim).contiguous()


def _make_page_permutation(num_pages, *, shuffle):
    if not shuffle or num_pages <= 1:
        return list(range(num_pages))
    for step in (7, 5, 3):
        if num_pages % step != 0:
            return [(i * step + 1) % num_pages for i in range(num_pages)]
    return list(reversed(range(num_pages)))


def _cosine_diff(actual, expected):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    numerator = 2 * (actual.double() * expected.double()).sum()
    denominator = (
        (actual.double().square() + expected.double().square()).sum().clamp_min(1e-12)
    )
    return (1 - (numerator / denominator)).item()


# --------------------------------------------------------------------------- #
# shared case construction
# --------------------------------------------------------------------------- #
def _build_case(
    *,
    batch,
    ctx_lens,
    nhead,
    decode_qlen,
    qk_head_dim,
    v_head_dim,
    page_size,
    mask,
    page_indices_oob,
    shuffle_pages,
):
    """Build one logical fp8 Q/KV case and the per-kernel views/metadata.

    Returns a dict with everything both kernels need. Both kernels read the
    EXACT same fp8 Q and KV values; only the memory layout differs.
    """
    device = torch.device("cuda")
    nhead_kv = 1
    rope_dim = qk_head_dim - v_head_dim
    npb = (ctx_lens + page_size - 1) // page_size
    total_pages = batch * npb
    total_page_indices = batch * (npb + page_indices_oob)
    last_page_len = ctx_lens % page_size or page_size

    torch.manual_seed(20260621 + batch * 1009 + ctx_lens + nhead * 7 + decode_qlen)

    # --- logical bf16 KV per page, then scatter pages to physical ids -------- #
    kv_logical_bf16 = torch.randn(
        (total_pages, page_size, nhead_kv, qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    perm = _make_page_permutation(total_pages, shuffle=shuffle_pages)
    kv_scattered_bf16 = torch.empty_like(kv_logical_bf16)
    # kv_indices: page-level block table, batch-major logical order, OOB pad at end.
    kv_indices = torch.zeros(total_page_indices, dtype=torch.int32, device=device)
    for logical_page, physical_page in enumerate(perm):
        kv_scattered_bf16[physical_page] = kv_logical_bf16[logical_page]
        kv_indices[logical_page] = physical_page
    kv_ref_fp8 = kv_scattered_bf16.to(dtypes.fp8)  # [total_pages, page, 1, 576] phys

    # aiter seg-packed KV: [num_pages, page_size*nope | page_size*pe]
    kv_seg = _pack_rope_split2_kv_pages(kv_ref_fp8, v_head_dim, rope_dim)

    # --- logical bf16 Q, fp8, plus the 768-padded selected layout ------------ #
    total_q = batch * decode_qlen
    q_bf16 = torch.randn(
        (total_q, nhead, qk_head_dim), dtype=torch.bfloat16, device=device
    )
    q_fp8 = q_bf16.to(dtypes.fp8)
    q_seg = _pack_rope_split3_q_pages(
        q_fp8.view(batch, decode_qlen, nhead, qk_head_dim), v_head_dim, rope_dim
    )
    q_seg = torch.as_strided(
        q_seg,
        size=(total_q, nhead, qk_head_dim),
        stride=(nhead * q_seg.stride(2), q_seg.stride(2), q_seg.stride(3)),
    )

    # --- page-level metadata (aiter) ----------------------------------------- #
    kv_indptr_page = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr_page[1:] = torch.cumsum(
        torch.full((batch,), npb, dtype=torch.int32, device=device), dim=0
    )
    kv_last_page_lens = torch.full(
        (batch,), last_page_len, dtype=torch.int32, device=device
    )
    qo_indptr = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * decode_qlen
    )
    num_kv_splits, num_kv_splits_indptr = aiter.mla.get_meta_param(
        None, batch, total_pages, nhead, decode_qlen, dtypes.fp8
    )
    num_kv_splits = int(num_kv_splits)

    # --- token-major views + per-query metadata (triton golden) -------------- #
    # Physical token index = physical_page * page_size + offset.
    k_buffer_tok = kv_ref_fp8.reshape(total_pages * page_size, nhead_kv, qk_head_dim).to(
        torch.bfloat16
    )
    v_buffer_tok = k_buffer_tok[..., :v_head_dim]  # value == nope part
    q_tok = q_fp8.view(total_q, nhead, qk_head_dim).to(torch.bfloat16)

    kv_indptr_tok = [0]
    kv_indices_tok = []
    for b in range(batch):
        # physical token ids for this batch's logical tokens (length ctx_lens)
        batch_tok_phys = []
        for j in range(npb):
            pp = int(kv_indices[b * npb + j])
            ntok = page_size if j < npb - 1 else last_page_len
            base = pp * page_size
            batch_tok_phys.extend(range(base, base + ntok))
        for qpos in range(decode_qlen):
            # causal: query token qpos sees ctx_lens - (decode_qlen-1-qpos) tokens
            seq_len_row = ctx_lens - (decode_qlen - 1 - qpos) if mask else ctx_lens
            kv_indices_tok.extend(batch_tok_phys[:seq_len_row])
            kv_indptr_tok.append(len(kv_indices_tok))
    kv_indptr_tok = torch.tensor(kv_indptr_tok, dtype=torch.int32, device=device)
    kv_indices_tok = torch.tensor(kv_indices_tok, dtype=torch.int32, device=device)

    return {
        "device": device,
        "nhead_kv": nhead_kv,
        "page_size": page_size,
        "total_q": total_q,
        "total_kv": batch * ctx_lens,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "sm_scale": 1.0 / (qk_head_dim**0.5),
        # aiter inputs
        "q_seg": q_seg,
        "kv_seg": kv_seg,
        "kv_indices": kv_indices,
        "kv_indptr_page": kv_indptr_page,
        "kv_last_page_lens": kv_last_page_lens,
        "qo_indptr": qo_indptr,
        "num_kv_splits": num_kv_splits,
        "num_kv_splits_indptr": num_kv_splits_indptr,
        # triton inputs
        "q_tok": q_tok,
        "k_buffer_tok": k_buffer_tok,
        "v_buffer_tok": v_buffer_tok,
        "kv_indptr_tok": kv_indptr_tok,
        "kv_indices_tok": kv_indices_tok,
    }


def _run_aiter(case, nhead, decode_qlen, *, perf):
    out = torch.zeros(
        (case["total_q"], nhead, case["v_head_dim"]), dtype=torch.bfloat16
    )
    q_scale = torch.ones([1], dtype=torch.float32, device=case["device"])
    kv_scale = torch.ones([1], dtype=torch.float32, device=case["device"])
    fn_args = (
        case["q_seg"],
        case["kv_seg"],
        out,
        case["qo_indptr"],
        case["kv_indptr_page"],
        case["kv_indices"],
        case["kv_last_page_lens"],
        decode_qlen,
        case["page_size"],
        case["nhead_kv"],
        case["sm_scale"],
    )
    fn_kwargs = dict(
        num_kv_splits=case["num_kv_splits"],
        num_kv_splits_indptr=case["num_kv_splits_indptr"],
        q_scale=q_scale,
        kv_scale=kv_scale,
        return_lse=True,
    )
    if perf:
        _, us = run_perftest(aiter.mla.mla_decode_fwd, *fn_args, **fn_kwargs)
        return out, us
    aiter.mla.mla_decode_fwd(*fn_args, **fn_kwargs)
    return out, None


def _run_triton(case, nhead, triton_kv_splits, *, perf):
    out = torch.empty(
        (case["total_q"], nhead, case["v_head_dim"]), dtype=torch.bfloat16
    )
    attn_logits = torch.empty(
        (case["total_q"], nhead, triton_kv_splits, case["v_head_dim"] + 1),
        dtype=torch.float32,
    )
    fn_args = (
        case["q_tok"],
        case["k_buffer_tok"],
        case["v_buffer_tok"],
        out,
        case["kv_indptr_tok"],
        case["kv_indices_tok"],
        attn_logits,
        triton_kv_splits,
        case["sm_scale"],
    )
    if perf:
        _, us = run_perftest(decode_attention_fwd, *fn_args)
        return out, us
    decode_attention_fwd(*fn_args)
    return out, None


@benchmark()
def test_mla_decode_triton(
    ctx_lens,
    batch_size,
    nhead,
    decode_qlen,
    kv_lora_rank,
    qk_rope_head_dim,
    page_size,
    mask,
    cos_threshold,
    triton_kv_splits,
    page_indices_oob,
    shuffle_pages,
):
    ret = {
        "nhead": nhead,
        "decode_qlen": decode_qlen,
        "batch": batch_size,
        "ctx": ctx_lens,
        "mask": mask,
        "skipped": True,
        "passed": None,
        "cos(aiter,triton)": None,
        "aiter_us": None,
        "triton_us": None,
        "aiter_TFLOPS": None,
        "triton_TFLOPS": None,
        "aiter_TB/s": None,
        "triton_TB/s": None,
        "speedup(triton/aiter)": None,
    }

    variant = _MI400_VARIANT_BY_KEY.get((nhead, decode_qlen))
    if variant is None:
        ret["reason"] = "unsupported (nhead,decode_qlen)"
        aiter.logger.info(
            "mla_decode-triton [nhead=%d q=%d]: skipped (unsupported combo)",
            nhead,
            decode_qlen,
        )
        return ret

    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    v_head_dim = kv_lora_rank
    ret["variant"] = variant.name
    ret["skipped"] = False

    case = _build_case(
        batch=batch_size,
        ctx_lens=ctx_lens,
        nhead=nhead,
        decode_qlen=decode_qlen,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        page_size=page_size,
        mask=mask,
        page_indices_oob=page_indices_oob,
        shuffle_pages=shuffle_pages,
    )
    triton_kv_splits = min(triton_kv_splits, max(1, ctx_lens))
    ret["num_kv_splits"] = case["num_kv_splits"]
    ret["triton_kv_splits"] = triton_kv_splits

    # ---- correctness: single clean launch of each kernel ------------------- #
    out_triton, _ = _run_triton(case, nhead, triton_kv_splits, perf=False)
    out_aiter, _ = _run_aiter(case, nhead, decode_qlen, perf=False)

    finite = (
        torch.isfinite(out_triton.detach().float().cpu()).all().item()
        and torch.isfinite(out_aiter.detach().float().cpu()).all().item()
    )
    if finite:
        cos_diff = _cosine_diff(out_aiter, out_triton)
    else:
        cos_diff = float("inf")
    passed = finite and cos_diff < cos_threshold
    ret["finite"] = finite
    ret["cos(aiter,triton)"] = cos_diff
    ret["passed"] = passed
    aiter.logger.info(
        "mla_decode-triton [%s | b=%d ctx=%d mask=%d]: cos(aiter,triton)=%.3e %s",
        variant.name,
        batch_size,
        ctx_lens,
        mask,
        cos_diff,
        "passed" if passed else "FAILED",
    )

    # ---- performance: time both kernels over the perftest loop ------------- #
    _, us_aiter = _run_aiter(case, nhead, decode_qlen, perf=True)
    _, us_triton = _run_triton(case, nhead, triton_kv_splits, perf=True)

    total_q, total_kv = case["total_q"], case["total_kv"]
    flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    fp8_b = torch.finfo(dtypes.fp8).bits // 8
    bf16_b = torch.finfo(torch.bfloat16).bits // 8
    bytes_aiter = (
        total_kv * case["nhead_kv"] * qk_head_dim * fp8_b
        + total_q * nhead * qk_head_dim * fp8_b
        + total_q * nhead * v_head_dim * bf16_b
    )
    bytes_triton = (
        total_kv * case["nhead_kv"] * qk_head_dim * bf16_b
        + total_q * nhead * qk_head_dim * bf16_b
        + total_q * nhead * v_head_dim * bf16_b
    )
    ret["aiter_us"] = us_aiter
    ret["triton_us"] = us_triton
    ret["aiter_TFLOPS"] = flops / us_aiter / 1e6
    ret["triton_TFLOPS"] = flops / us_triton / 1e6
    ret["aiter_TB/s"] = bytes_aiter / us_aiter / 1e6
    ret["triton_TB/s"] = bytes_triton / us_triton / 1e6
    ret["speedup(triton/aiter)"] = us_triton / us_aiter
    aiter.logger.info(
        "mla_decode-triton [%s | b=%d ctx=%d]: aiter %.2f us (%.2f TFLOPS) | "
        "triton %.2f us (%.2f TFLOPS) | speedup x%.2f",
        variant.name,
        batch_size,
        ctx_lens,
        us_aiter,
        ret["aiter_TFLOPS"],
        us_triton,
        ret["triton_TFLOPS"],
        ret["speedup(triton/aiter)"],
    )
    return ret


def _build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="triton-as-golden numerical + perf compare for gfx1250 seg MLA decode",
    )
    parser.add_argument("-k", "--kv_lora_rank", type=int, default=512)
    parser.add_argument("-qr", "--qk_rope_head_dim", type=int, default=64)
    parser.add_argument("-blk", "--block_size", type=int, default=64, help="page size")
    parser.add_argument(
        "-c",
        "--ctxLen",
        type=int,
        nargs="*",
        default=_DEFAULT_CTX_LENS,
        help="context length(s). e.g.: -c 1024",
    )
    parser.add_argument(
        "-b",
        "--batchSize",
        type=int,
        nargs="*",
        default=_DEFAULT_BATCH_SIZES,
        help="batch size(s). e.g.: -b 16",
    )
    parser.add_argument(
        "-n",
        "--nhead",
        type=dtypes.str2tuple,
        choices=list(_MI400_VARIANT_BY_KEY.keys()),
        nargs="*",
        const=None,
        default=_DEFAULT_NHEAD,
        help="(nhead, decode_qlen) pair(s). e.g.: -n 128,1",
    )
    parser.add_argument(
        "--mask",
        type=int,
        nargs="+",
        choices=[0, 1],
        default=[1],
        help="causal/tail mask selector (only affects decode_qlen>1).",
    )
    parser.add_argument(
        "--cos-threshold",
        type=float,
        default=5e-2,
        help="max cos_diff(aiter, triton) to count as passed.",
    )
    parser.add_argument(
        "--triton-kv-splits",
        type=int,
        default=16,
        help="num_kv_splits for the triton golden (numeric-invariant; affects triton perf).",
    )
    parser.add_argument(
        "--page-indices-oob",
        type=int,
        default=4,
        help="extra OOB padding entries appended to the page block table.",
    )
    parser.add_argument(
        "--no-shuffle-pages",
        action="store_true",
        help="disable physical page shuffle (default: shuffled).",
    )
    parser.add_argument(
        "--variant",
        choices=[v.name for v in _MI400_KERNEL_VARIANTS],
        default=None,
        help="restrict to a single kernel variant by name.",
    )
    return parser


def main():
    args = _build_parser().parse_args()

    gfx = None
    try:
        gfx = get_gfx()
    except Exception:
        pass
    if gfx != "gfx1250":
        aiter.logger.warning(
            "this test targets gfx1250 seg MLA decode; detected gfx=%s", gfx
        )

    # default ASM dir to the repo's hsa so the seg variants resolve.
    repo_hsa_dir = Path(__file__).resolve().parents[1] / "hsa"
    os.environ.setdefault("AITER_ASM_DIR", str(repo_hsa_dir))

    nhead_combos = args.nhead
    if args.variant is not None:
        v = _MI400_VARIANT_BY_KEY_NAME[args.variant]
        nhead_combos = [(v.nhead, v.decode_qlen)]

    failures = []
    for nhead, decode_qlen in nhead_combos:
        df = []
        for ctx_len, batch_size, mask in itertools.product(
            args.ctxLen, args.batchSize, args.mask
        ):
            ret = test_mla_decode_triton(
                ctx_len,
                batch_size,
                nhead,
                decode_qlen,
                args.kv_lora_rank,
                args.qk_rope_head_dim,
                args.block_size,
                mask,
                args.cos_threshold,
                args.triton_kv_splits,
                args.page_indices_oob,
                not args.no_shuffle_pages,
            )
            df.append(ret)
            if not ret.get("skipped", True) and not ret.get("passed", False):
                failures.append(
                    (
                        ret.get("variant"),
                        batch_size,
                        ctx_len,
                        mask,
                        ret.get("cos(aiter,triton)"),
                    )
                )
        if df:
            df_md = pd.DataFrame(df).to_markdown(index=False)
            aiter.logger.info("mla triton-vs-aiter summary:\n%s", df_md)

    if failures:
        raise AssertionError(f"triton-vs-aiter MLA numerics failed for: {failures}")


if __name__ == "__main__":
    main()
