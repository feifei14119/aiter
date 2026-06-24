# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM-dump driven V4 sparse MLA decode unit test: aiter vs golden.

ATOM's V4 (DeepSeek-V4) decode runs ``sparse_attn_v4_paged_decode`` -- on the
``ATOM_USE_TRITON_ATTN=1`` path this is the triton kernel
``_sparse_attn_v4_paged_decode_triton`` (atom/model_ops/v4_kernels/paged_decode.py).
The aiter counterpart (identical signature) is
``aiter.ops.triton.attention.pa_decode_sparse.pa_decode_sparse``.

Two modes:

  --from-dump DIR
      Feature 1 (tensor replay). Loads each ``mla_decode.rank*.*.pt`` written by
      ATOM (atom/model_ops/mla_dump.py): the real fp8/bf16 Q, the compact
      unified KV pool, the paged indices, attn_sink, softmax_scale, and ATOM's
      triton kernel output ``o`` (== the golden). For each dump we run the aiter
      op on the SAME inputs and compare aiter vs ATOM's output (and vs an
      independent pure-torch reference).

  --from-params MANIFEST
      Feature 2 (params replay, no tensors). Reads the JSONL manifest of V4
      decode call params (T, H, D, per-token kv spans, ...) ATOM recorded and,
      for each distinct shape, regenerates a fresh random case and runs the
      torch reference (golden) vs the aiter op.

Run inside the gfx1250 aiter container (see mla_atom_dump_howto.md):
  docker exec <ctr> bash -lc 'cd /home/carhuang/feifei/aiter/op_tests && \
      python3 test_mla_atom_dump.py --from-params /data/mla_dump/mla_calls.rank0.jsonl'
  docker exec <ctr> bash -lc 'cd /home/carhuang/feifei/aiter/op_tests && \
      python3 test_mla_atom_dump.py --from-dump /data/mla_dump'
"""

import argparse
import glob
import json
import os

import pandas as pd
import torch

import aiter
from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

_FP8_GROUP_SIZE = 64


# --------------------------------------------------------------------------- #
# Independent pure-torch golden (port of ATOM's
# sparse_attn_v4_paged_decode_reference / the aiter pa_decode_sparse test ref).
# --------------------------------------------------------------------------- #
def _sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [B,M,H,D], kv [B,N,D], attn_sink [H], topk_idxs [B,M,K] (-1 = skip)."""
    B, M, H, D = q.shape
    K = topk_idxs.shape[-1]
    device = q.device
    out_dtype = q.dtype

    valid = topk_idxs != -1
    safe_idxs = topk_idxs.clamp(min=0).long()
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, K)
    kv_gathered = kv[batch_idx, safe_idxs]  # [B, M, K, D]
    kv_f32 = kv_gathered.float()
    kv_f32 = torch.where(
        valid.unsqueeze(-1), kv_f32, torch.zeros((), dtype=kv_f32.dtype, device=device)
    )
    q_f32 = q.float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q_f32, kv_f32) * float(softmax_scale)
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    sink = attn_sink.float().view(1, 1, H, 1).expand(B, M, H, 1)
    combined = torch.cat([scores, sink], dim=-1)
    cmax = combined.amax(dim=-1, keepdim=True)
    cmax = torch.where(
        cmax == float("-inf"), torch.zeros((), dtype=cmax.dtype, device=device), cmax
    )
    weights = (combined - cmax).exp()
    denom = weights.sum(dim=-1, keepdim=True)
    weights = weights / denom.clamp(min=1e-30)
    out = torch.einsum("bmhk,bmkd->bmhd", weights[..., :K], kv_f32)
    return out.to(out_dtype)


def _reference(q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale):
    T = q.size(0)
    indptr = kv_indptr.to(torch.int64)
    spans = (indptr[1:] - indptr[:T]).clamp(min=0)
    k_dim = int(spans.max().item()) if T > 0 else 1
    k_dim = max(k_dim, 1)
    topk_idxs = torch.full((T, k_dim), -1, device=q.device, dtype=torch.int32)
    for t in range(T):
        s = int(indptr[t].item())
        n = int(spans[t].item())
        if n > 0:
            topk_idxs[t, :n] = kv_indices[s : s + n].to(torch.int32)
    return _sparse_attn_torch(
        q.unsqueeze(0),
        unified_kv.unsqueeze(0),
        attn_sink,
        topk_idxs.unsqueeze(0),
        softmax_scale,
    ).squeeze(0)


def _dequant_kv_fp8(kv_fp8, kv_scales, group_size=_FP8_GROUP_SIZE):
    total_pages, D = kv_fp8.shape
    num_groups = D // group_size
    kv_f32 = kv_fp8.float().view(total_pages, num_groups, group_size)
    scales_exp = kv_scales.unsqueeze(-1).expand(total_pages, num_groups, group_size)
    return (kv_f32 * scales_exp).view(total_pages, D)


def _cosine_diff(actual, expected):
    a = actual.detach().float().cpu()
    b = expected.detach().float().cpu()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("inf")
    num = 2 * (a.double() * b.double()).sum()
    den = (a.double().square() + b.double().square()).sum().clamp_min(1e-12)
    return (1 - (num / den)).item()


# --------------------------------------------------------------------------- #
# Feature 1 — tensor replay (use ATOM's dumped inputs + golden output)
# --------------------------------------------------------------------------- #
def run_from_dump(dump_dir, cos_threshold):
    files = sorted(glob.glob(os.path.join(dump_dir, "mla_decode.rank*.*.pt")))
    if not files:
        raise FileNotFoundError(
            f"no mla_decode.rank*.*.pt tensor dumps under {dump_dir}"
        )
    device = torch.device("cuda")
    rows = []
    failures = []
    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        p = payload["params"]
        q = payload["q"].to(device)
        unified_kv = payload["unified_kv"].to(device)
        kv_indices = payload["kv_indices"].to(device).to(torch.int32).contiguous()
        kv_indptr = payload["kv_indptr"].to(device).to(torch.int32).contiguous()
        attn_sink = payload["attn_sink"].to(device).to(torch.float32)
        o_atom = payload["o"].to(device)
        kv_scales = payload.get("kv_scales")
        kv_scales = kv_scales.to(device) if kv_scales is not None else None
        softmax_scale = float(p["softmax_scale"])
        has_invalid = bool(p.get("has_invalid", False))

        out_aiter = pa_decode_sparse(
            q,
            unified_kv,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            kv_scales=kv_scales,
            has_invalid=has_invalid,
        )

        # Independent torch golden on the SAME inputs (dequant fp8 KV first).
        ukv_ref = (
            _dequant_kv_fp8(unified_kv, kv_scales).to(q.dtype)
            if kv_scales is not None
            else unified_kv
        )
        ref = _reference(q, ukv_ref, kv_indices, kv_indptr, attn_sink, softmax_scale)

        cos_aiter_atom = _cosine_diff(out_aiter, o_atom.view_as(out_aiter))
        cos_aiter_ref = _cosine_diff(out_aiter, ref)
        cos_atom_ref = _cosine_diff(o_atom.view_as(ref), ref)
        passed = cos_aiter_atom < cos_threshold
        rows.append(
            {
                "src": os.path.basename(f),
                "layer": p.get("layer_id"),
                "ratio": p.get("ratio"),
                "T": p["T"],
                "H": p["H"],
                "D": p["D"],
                "max_kv": p.get("max_kv_len"),
                "fp8": kv_scales is not None,
                "cos(aiter,atom)": cos_aiter_atom,
                "cos(aiter,ref)": cos_aiter_ref,
                "cos(atom,ref)": cos_atom_ref,
                "passed": passed,
            }
        )
        aiter.logger.info(
            "dump [%s | L%s r%s T=%d H=%d ctx=%d]: cos(aiter,atom)=%.3e "
            "cos(aiter,ref)=%.3e cos(atom,ref)=%.3e %s",
            os.path.basename(f),
            p.get("layer_id"),
            p.get("ratio"),
            p["T"],
            p["H"],
            p.get("max_kv_len"),
            cos_aiter_atom,
            cos_aiter_ref,
            cos_atom_ref,
            "passed" if passed else "FAILED",
        )
        if not passed:
            failures.append((os.path.basename(f), cos_aiter_atom))
    return rows, failures


# --------------------------------------------------------------------------- #
# Feature 2 — params replay (regenerate random case, torch golden vs aiter)
# --------------------------------------------------------------------------- #
def _iter_manifest(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _dtype_from_str(s, default=torch.bfloat16):
    return {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
    }.get(s, default)


def _build_case_from_params(rec, seed):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    T = int(rec["T"])
    H = int(rec["H"])
    D = int(rec["D"])
    dtype = _dtype_from_str(rec.get("q_dtype"))
    softmax_scale = float(rec.get("softmax_scale") or D ** -0.5)
    kv_lens = rec.get("kv_lens")
    if kv_lens:
        kv_lens_t = torch.tensor(kv_lens[:T], dtype=torch.int64, device=device)
    else:
        kv_lens_t = torch.full(
            (T,), int(rec.get("max_kv_len") or 1), dtype=torch.int64, device=device
        )
    indptr = torch.zeros(T + 1, dtype=torch.int64, device=device)
    indptr[1:] = kv_lens_t.cumsum(0)
    total_indices = int(indptr[-1].item())
    total_pages = max(int(rec.get("total_pages") or total_indices), 1)

    q = torch.randn(T, H, D, dtype=dtype, device=device) * 0.5
    unified_kv = torch.randn(total_pages, D, dtype=dtype, device=device) * 0.5
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1
    indices = torch.randint(
        0, total_pages, (total_indices,), dtype=torch.int32, device=device
    )
    return q, unified_kv, indices, indptr.to(torch.int32), attn_sink, softmax_scale


def run_from_params(manifest_path, cos_threshold, dedup):
    seen = set()
    rows = []
    failures = []
    for i, rec in enumerate(_iter_manifest(manifest_path)):
        key = (int(rec["T"]), int(rec["H"]), int(rec["D"]), int(rec.get("max_kv_len") or 0))
        if dedup and key in seen:
            continue
        seen.add(key)
        if int(rec["T"]) <= 0 or int(rec.get("max_kv_len") or 0) <= 0:
            continue

        q, ukv, indices, indptr, sink, scale = _build_case_from_params(rec, seed=i)
        ref = _reference(q, ukv, indices, indptr, sink, scale)
        out = pa_decode_sparse(q, ukv, indices, indptr, sink, scale, has_invalid=False)
        cos = _cosine_diff(out, ref)
        passed = cos < cos_threshold
        rows.append(
            {
                "src": "params",
                "layer": rec.get("layer_id"),
                "ratio": rec.get("ratio"),
                "T": rec["T"],
                "H": rec["H"],
                "D": rec["D"],
                "max_kv": rec.get("max_kv_len"),
                "cos(aiter,ref)": cos,
                "passed": passed,
            }
        )
        aiter.logger.info(
            "params [L%s r%s T=%d H=%d ctx=%d]: cos(aiter,ref)=%.3e %s",
            rec.get("layer_id"),
            rec.get("ratio"),
            rec["T"],
            rec["H"],
            rec.get("max_kv_len"),
            cos,
            "passed" if passed else "FAILED",
        )
        if not passed:
            failures.append((key, cos))
    return rows, failures


def _build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="ATOM-dump driven V4 sparse MLA decode aiter test",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--from-dump",
        metavar="DIR",
        help="directory of ATOM mla_decode.rank*.*.pt tensor dumps (Feature 1).",
    )
    src.add_argument(
        "--from-params",
        metavar="MANIFEST",
        help="ATOM mla_calls.rank*.jsonl params manifest (Feature 2).",
    )
    parser.add_argument(
        "--cos-threshold",
        type=float,
        default=2e-2,
        help="max cos_diff to count as passed (bf16 sparse decode).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="(params mode) run every manifest line, not just distinct shapes.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    if args.from_params:
        rows, failures = run_from_params(
            args.from_params, args.cos_threshold, dedup=not args.no_dedup
        )
        label = "params"
    else:
        rows, failures = run_from_dump(args.from_dump, args.cos_threshold)
        label = "dump"

    if rows:
        aiter.logger.info(
            "mla atom-%s summary:\n%s",
            label,
            pd.DataFrame(rows).to_markdown(index=False),
        )
    if failures:
        raise AssertionError(f"atom-{label} V4 sparse MLA numerics failed: {failures}")


if __name__ == "__main__":
    main()
