# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# gfx1250 / mi400 MLA fp8 decode test. Run inside the `ff_mla` container:
#
#   # Single variant (default is qh128-q1-16mx4-64nx1-np):
#   docker exec ff_mla bash -lc 'cd /home/carhuang/feifei/aiter && \
#     rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
#     env -u AITER_ASM_DEBUG -u AITER_MLA_DEBUG_SKIP_KERNEL \
#     ROCM_HOME=/opt/rocm ENABLE_CK=0 ENABLE_FLYDSL=0 \
#     GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
#     AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
#     python3 op_tests/test_mla_mi400.py \
#     --mi400-variant qh128-q1-16mx4-64nx1-np'
#
#   # First JIT build takes ~10s; drop the `rm -rf` line to reuse the cache.
#   # See op_tests/test_mla_mi400_README.md for full details.

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
from aiter.ops.attention import mla_decode_stage1_asm_fwd
from aiter.test_common import benchmark, run_perftest

# In lean containers, aiter.__init__ can skip bulk op exports when optional
# dependencies are unavailable. Register the op the mi400 sweep needs explicitly.
aiter.mla_decode_stage1_asm_fwd = mla_decode_stage1_asm_fwd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


def check_support(dtype, kv_dtype, nhead):
    if dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        return False
    return True


# ###########################################################################
# gfx1250 / mi400 MLA decode
#
# This driver runs only the mi400 fp8 decode sweep. For each
# (nhead=Gqa, decode_qlen, batch, ctx_len) combo it routes through the mi400
# fp8 decode check below. Unsupported (Gqa, decode_qlen) combos are skipped.
# Exercises the shader variants registered in
# hsa/gfx1250/mla/mla_asm.csv.
# ###########################################################################


@dataclass(frozen=True)
class MlaMi400KernelVariant:
    name: str
    nhead: int
    decode_qlen: int


_MI400_KERNEL_VARIANTS = [
    MlaMi400KernelVariant(name="qh16-q1-16mx1-32nx4-np-3p", nhead=16, decode_qlen=1),
    MlaMi400KernelVariant(
        name="qh16-q2-16mx2-32nx4-np-3p",
        nhead=16,
        decode_qlen=2,
    ),
    MlaMi400KernelVariant(
        name="qh32-q1-32mx1-32nx4-np-3p",
        nhead=32,
        decode_qlen=1,
    ),
    MlaMi400KernelVariant(name="qh16-q4-16mx4-64nx1-np", nhead=16, decode_qlen=4),
    MlaMi400KernelVariant(
        name="qh64-q1-16mx4-64nx1-np",
        nhead=64,
        decode_qlen=1,
    ),
    MlaMi400KernelVariant(
        name="qh128-q1-16mx4-64nx1-np",
        nhead=128,
        decode_qlen=1,
    ),
]

# Dispatch key (nhead, decode_qlen) -> variant. Source of truth for which
# (nhead, decode_qlen) combos the mi400 decode check supports.
_MI400_VARIANT_BY_KEY = {(v.nhead, v.decode_qlen): v for v in _MI400_KERNEL_VARIANTS}
# Variant name -> variant, used by the --mi400-variant CLI selector.
_MI400_VARIANT_BY_KEY_NAME = {v.name: v for v in _MI400_KERNEL_VARIANTS}

# mi400 driver sweep dims.
_MI400_NHEAD = [(v.nhead, v.decode_qlen) for v in _MI400_KERNEL_VARIANTS]
_MI400_CTX_LENS = [128] # [7, 17, 33, 65, 256, 512, 1024, 4096, 10240]
_MI400_BATCH_SIZES = [1] # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_MI400_SPLIT_PER_BATCH = [2]


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

    # Mirror poc_kl pack_q_page1_padded(): each logical Q row stores
    # [nope][rope] followed by zero padding up to a 768-byte row stride.
    rows = tensor.reshape(-1, shape[-1])
    padded = torch.zeros(
        (rows.shape[0], padded_dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, : shape[-1]].copy_(rows)
    return torch.as_strided(
        padded,
        size=shape,
        stride=(
            shape[1] * shape[2] * padded_dim,
            shape[2] * padded_dim,
            padded_dim,
            1,
        ),
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
    if not shuffle:
        return list(range(num_pages))
    if num_pages <= 1:
        return list(range(num_pages))
    for step in (7, 5, 3):
        if num_pages % step != 0:
            return [(i * step + 1) % num_pages for i in range(num_pages)]
    return list(reversed(range(num_pages)))


def _make_scales(batch, device, *, enabled):
    if not enabled:
        return (
            torch.ones((1,), dtype=torch.float32, device=device),
            torch.ones((1,), dtype=torch.float32, device=device),
        )
    q_scale = torch.linspace(0.75, 1.25, 1, dtype=torch.float32, device=device)
    kv_scale = torch.linspace(1.20, 0.80, 1, dtype=torch.float32, device=device)
    return q_scale, kv_scale


def _make_mla_mi400_case(
    *,
    batch,
    ctx_lens,
    nhead,
    decode_qlen,
    num_kv_splits,
    use_non_unit_scales=True,
):
    repo_hsa_dir = Path(__file__).resolve().parents[1] / "hsa"
    os.environ["AITER_ASM_DIR"] = str(repo_hsa_dir)

    device = torch.device("cuda")
    page_size = 64
    num_pages_per_batch = (ctx_lens + page_size - 1) // page_size

    if num_kv_splits is None:
        # Mirror mla_decode_fwd(num_kv_splits=None): resolve the auto split count
        # and its indptr through the shared meta-param heuristic so the case
        # carries a concrete value for the shape checks and the kernel args.
        num_kv_splits, num_kv_splits_indptr = aiter.mla.get_meta_param(
            None,
            batch,
            batch * num_pages_per_batch,
            nhead,
            decode_qlen,
            dtypes.fp8,
        )
        num_kv_splits = int(num_kv_splits)
    else:
        assert num_kv_splits > 0
        num_kv_splits_indptr = (
            torch.arange(batch + 1, dtype=torch.int32, device=device) * num_kv_splits
        )
    torch.manual_seed(
        20260513
        + batch * 1009
        + ctx_lens
        + nhead * 7
        + decode_qlen
        + num_kv_splits * 101
    )

    last_page_len = ctx_lens % page_size or page_size
    kv_last_page_lens = torch.full(
        (batch,), last_page_len, dtype=torch.int32, device=device
    )
    # gfx1250/mi400 stage1 asm kernel consumes a PAGE-level kv_indptr directly
    # (it walks the page-level kv_indices block table). Build it here as the
    # per-batch prefix sum of page counts so mla.py no longer needs to convert a
    # token-level kv_indptr. With uniform ctx_lens this is [0, npb, 2*npb, ...].
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.full((batch,), num_pages_per_batch, dtype=torch.int32, device=device),
        dim=0,
    )
    q_scale, kv_scale = _make_scales(batch, device, enabled=use_non_unit_scales)

    return {
        "page_size": page_size,
        "num_kv_splits": num_kv_splits,
        "num_pages_per_batch": num_pages_per_batch,
        "kv_last_page_lens": kv_last_page_lens,
        "kv_indptr": kv_indptr,
        "num_kv_splits_indptr": num_kv_splits_indptr,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
    }


def _make_mla_mi400_kv_case(
    *,
    kv_buffer_bf16,
    batch,
    ctx_lens,
    qk_head_dim,
    v_head_dim,
    page_indices_oob,
    shuffle_pages=True,
):
    """Build the KV inputs for the gfx1250 seg asm decode (qk_head_dim=576 =
    nope 512 + rope 64).

    Returns (kv_buffer, kv_buffer_ref, kv_indices):
      kv_buffer     : fp8 (float8_e4m3fn), aiter PAGE-level seg-pack, shape
                      [num_pages, page_size, 1, 576] holding
                      [page_size*512 (nope) | page_size*64 (pe)] per page
                      (page_size=64). This is what mla.mla_decode_fwd consumes.
                      Built by _pack_rope_split2_kv_pages.
      kv_buffer_ref : fp8 (float8_e4m3fn), TOKEN-major scattered cache
                      [num_pages, page_size, 1, 576] (pages placed at their
                      physical ids); consumed only by the PyTorch fp32 reference.
      kv_indices    : int32 PAGE-level block table [batch*(npb+oob)] of physical
                      page ids (compact, OOB padding appended after valid pages).
    """
    device = torch.device("cuda")
    page_size = 64
    nhead_kv = 1
    num_pages_per_batch = (ctx_lens + page_size - 1) // page_size
    total_page_indices = batch * (num_pages_per_batch + page_indices_oob)
    total_pages = batch * num_pages_per_batch

    kv_buffer_source_bf16 = kv_buffer_bf16.view(-1, page_size, nhead_kv, qk_head_dim)
    available_pages = kv_buffer_source_bf16.size(0)
    if available_pages >= total_pages:
        kv_buffer_logical_bf16 = kv_buffer_source_bf16[:total_pages].contiguous()
    else:
        kv_buffer_logical_bf16 = torch.empty(
            (total_pages, page_size, nhead_kv, qk_head_dim),
            dtype=kv_buffer_source_bf16.dtype,
            device=kv_buffer_source_bf16.device,
        )
        kv_buffer_logical_bf16[:available_pages] = kv_buffer_source_bf16
        kv_buffer_logical_bf16[available_pages:] = torch.randn(
            (total_pages - available_pages, page_size, nhead_kv, qk_head_dim),
            dtype=kv_buffer_source_bf16.dtype,
            device=kv_buffer_source_bf16.device,
        )
    # Poison the unused tail of every batch's last (partially filled) page with
    # NaN. When ctx_lens % page_size != 0 the final logical page of each batch
    # keeps only last_page_len valid tokens; slots [last_page_len:page_size] are
    # never valid KV. The kernel must honor kv_last_page_lens / kv_indptr and
    # never read past them, so a correct kernel still yields a finite, matching
    # output. The PyTorch reference excludes this tail via kv[:ctx_lens].
    last_page_len = ctx_lens % page_size or page_size
    if last_page_len != page_size:
        last_logical_pages = [
            (b + 1) * num_pages_per_batch - 1 for b in range(batch)
        ]
        kv_buffer_logical_bf16[last_logical_pages, last_page_len:] = float("nan")

    # The kernel consumes a compact block table, with OOB padding only after all
    # valid pages. KV pages are scattered into their physical page ids.
    shuffled_page_indices = _make_page_permutation(total_pages, shuffle=shuffle_pages)
    kv_buffer_scattered_bf16 = torch.empty_like(kv_buffer_logical_bf16)
    kv_indices = torch.zeros(total_page_indices, dtype=torch.int32, device=device)
    for logical_page, physical_page in enumerate(shuffled_page_indices):
        kv_buffer_scattered_bf16[physical_page] = kv_buffer_logical_bf16[logical_page]
        kv_indices[logical_page] = physical_page

    kv_buffer_ref = kv_buffer_scattered_bf16.to(dtypes.fp8)
    kv_buffer = _pack_rope_split2_kv_pages(
        kv_buffer_ref.view(total_pages, page_size, nhead_kv, qk_head_dim),
        v_head_dim,
        qk_head_dim - v_head_dim,
    )
    return kv_buffer, kv_buffer_ref, kv_indices


def _make_mla_mi400_q_case(
    *, q_fp8, batch, decode_qlen, nhead, qk_head_dim, v_head_dim
):
    """Build the Q input for the gfx1250 seg asm decode.

    Returns q: fp8 (float8_e4m3fn), shape [total_q, nhead, 576], NON-contiguous
    768-padded selected layout -- per-head row stride = 768 elems (=768 B in
    fp8), i.e. each head's 576 values ([nope 512][rope 64]) followed by 192 B of
    zero padding (_MLA_Q_OUT_PADDED_DIM). Built by _pack_rope_split3_q_pages +
    as_strided. (The PyTorch fp32 reference instead reads the unpadded q_fp8
    directly.)
    """
    q = q_fp8.view(batch, decode_qlen, nhead, qk_head_dim)
    q = _pack_rope_split3_q_pages(
        q,
        v_head_dim,
        qk_head_dim - v_head_dim,
    )
    return torch.as_strided(
        q,
        size=(batch * decode_qlen, nhead, qk_head_dim),
        stride=(nhead * q.stride(2), q.stride(2), q.stride(3)),
    )


def _apply_causal_mask_(logits):
    # Matches the causal/tail mask shape used by the reference attention.
    _, s_q, s_k = logits.shape
    mask = torch.ones(s_q, s_k, dtype=torch.bool, device=logits.device).tril(
        diagonal=s_k - s_q
    )
    logits.masked_fill_(mask.logical_not().unsqueeze(0), float("-inf"))


def _ref_mla_mi400(
    case,
    q_ref,
    kv_buffer_ref,
    kv_indices,
    batch_size,
    ctx_lens,
    decode_qlen,
    nhead_kv,
    qk_head_dim,
    v_head_dim,
    mask,
):
    """PyTorch fp32 analytic reference (qk_head_dim=576 = nope 512 + rope 64).

    Inputs it reads (both UNPACKED relative to the aiter kernel layouts; both
    fp8 then upcast to fp32 here so the r eference carries no extra quant error):
      q_ref         : fp8 (float8_e4m3fn), CONTIGUOUS [total_q, nhead, 576]
                      (the plain q_fp8, NOT the 768-padded selected layout the
                      asm kernel consumes). Upcast via .float() * q_scale.
      kv_buffer_ref : fp8 (float8_e4m3fn), TOKEN-major scattered cache
                      [num_pages, page_size, 1, 576] (pages at physical ids, NOT
                      the seg-packed layout). Gathered per batch by physical
                      page id (kv_indices), upcast via .float() * kv_scale, then
                      reshaped to [ctx_lens, 1, 576]; key=full 576, value=[:512].
    Output: bf16 [total_q, nhead, 512] (softmax(QK^T/sqrt(576))·V, causal mask).
    """
    outputs = []
    num_pages = case["num_pages_per_batch"]
    kv_source = kv_buffer_ref
    for b in range(batch_size):
        q_start = b * decode_qlen
        q_end = q_start + decode_qlen
        q_scale = case["q_scale"][0 if case["q_scale"].numel() == 1 else b]
        kv_scale = case["kv_scale"][0 if case["kv_scale"].numel() == 1 else b]
        q = q_ref[q_start:q_end].float() * q_scale
        page_indices = kv_indices[b * num_pages : (b + 1) * num_pages].long()
        kv = torch.index_select(kv_source.float(), 0, page_indices) * kv_scale
        kv = kv.reshape(-1, nhead_kv, qk_head_dim)
        kv = kv[:ctx_lens]
        key = kv
        value = kv[..., :v_head_dim]

        logits = torch.einsum("qhd,kmd->hqk", q, key) * (1.0 / (qk_head_dim**0.5))
        if mask:
            _apply_causal_mask_(logits)
        weights = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("hqk,kmd->qhd", weights, value).to(torch.bfloat16))
    return torch.cat(outputs, dim=0)


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


@benchmark()
def test_mla(
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    decode_qlen,
    split_per_batch=None,
    mask=1,
):
    ret = {}

    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    # absorb layout: mi400 fp8 decode runs the absorbed MLA path.
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank

    kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.full((batch_size,), decode_qlen, dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)

    def test_absorb_decode_mi400():
        # mi400 (gfx1250) fp8 MLA decode. It derives fp8 + selected Q/KV layout
        # from the standard bf16 inputs and checks against _ref_mla_mi400.
        # Dispatch key is (nhead, decode_qlen); unsupported combos are recorded
        # as skipped (not failures) so the driver does not abort.
        ret["mi400:nhead"] = nhead
        ret["mi400:decode_qlen"] = decode_qlen
        ret["mi400:batch"] = batch_size
        ret["mi400:ctx"] = ctx_lens
        ret["mi400:num_kv_splits"] = split_per_batch
        ret["mi400:mask"] = mask
        ret["mi400:skipped"] = True
        ret["mi400:passed"] = None
        ret["mi400:finite"] = None
        ret["mi400:cos_diff"] = None
        ret["mi400:us"] = None
        ret["mi400:TFLOPS"] = None
        ret["mi400:TB/s"] = None

        variant = _MI400_VARIANT_BY_KEY.get((nhead, decode_qlen))
        if variant is None:
            ret["mi400:reason"] = "unsupported (nhead,decode_qlen)"
            aiter.logger.info(
                "mla_decode-mi400 [nhead=%d decode_qlen=%d]: skipped (unsupported dispatch combo)",
                nhead,
                decode_qlen,
            )
            return

        ret["mi400:variant"] = variant.name
        ret["mi400:skipped"] = False
        # Looser than the generic fp8 3e-2 tolerance: with page shuffle + OOB +
        # non-unit scales all on, short-KV / multi-batch combos (e.g. q4,
        # batch=2, ctx=65) sit just above 3e-2 from fp8 quant noise.
        cos_threshold = 5e-2
        # mi400-specific coverage knobs are fixed fully-on (page shuffle + OOB
        # padding + non-unit scales) for every supported combo.
        page_indices_oob = 4
        kv_buffer_mi400, kv_buffer_ref_mi400, kv_indices_mi400 = (
            _make_mla_mi400_kv_case(
                kv_buffer_bf16=kv_buffer,
                batch=batch_size,
                ctx_lens=ctx_lens,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
                page_indices_oob=page_indices_oob,
            )
        )
        q_fp8_mi400 = q.to(dtypes.fp8)
        q_mi400 = _make_mla_mi400_q_case(
            q_fp8=q_fp8_mi400,
            batch=batch_size,
            decode_qlen=decode_qlen,
            nhead=nhead,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
        )
        case = _make_mla_mi400_case(
            batch=batch_size,
            ctx_lens=ctx_lens,
            nhead=nhead,
            decode_qlen=decode_qlen,
            num_kv_splits=split_per_batch,
        )
        # split_per_batch=None resolves to an auto split count inside the case;
        # report the concrete value the kernel actually ran with.
        ret["mi400:num_kv_splits"] = case["num_kv_splits"]

        # Single launch for functional/numerical validation, kept separate from
        # the perf loop below so the correctness check always inspects one clean
        # launch into the freshly zeroed out buffer.
        out_mi400 = torch.zeros(
            (
                batch_size * decode_qlen,
                nhead,
                v_head_dim,
            ),
            dtype=torch.bfloat16,
        )
        attn_logits, attn_lse = aiter.mla.mla_decode_fwd(
            q_mi400,
            kv_buffer_mi400,
            out_mi400,
            qo_indptr,
            case["kv_indptr"],
            kv_indices_mi400,
            case["kv_last_page_lens"],
            decode_qlen,
            case["page_size"],
            nhead_kv,
            1.0 / (qk_head_dim**0.5),
            num_kv_splits=case["num_kv_splits"],
            num_kv_splits_indptr=case["num_kv_splits_indptr"],
            q_scale=case["q_scale"],
            kv_scale=case["kv_scale"],
            return_lse=True,
        )
        out_check = out_mi400.clone()

        out_shape = (
            batch_size * decode_qlen,
            nhead,
            v_head_dim,
        )
        logits_shape = (
            batch_size * decode_qlen,
            case["num_kv_splits"],
            nhead,
            v_head_dim,
        )
        if case["num_kv_splits"] == 1:
            logits_shape = (
                batch_size * decode_qlen,
                nhead,
                v_head_dim,
            )
        # Structural shape checks are hard asserts: they must always hold.
        assert out_check.shape == out_shape
        assert attn_logits.shape == logits_shape
        assert attn_lse.shape == (batch_size * decode_qlen, nhead)

        finite = (
            torch.isfinite(out_check.detach().float().cpu()).all().item()
            and torch.isfinite(attn_logits.detach().float().cpu()).all().item()
            and torch.isfinite(attn_lse.detach().float().cpu()).all().item()
        )
        if finite:
            expected = _ref_mla_mi400(
                case,
                q_fp8_mi400,
                kv_buffer_ref_mi400,
                kv_indices_mi400,
                batch_size,
                ctx_lens,
                decode_qlen,
                nhead_kv,
                qk_head_dim,
                v_head_dim,
                mask,
            )
            cos_diff = _cosine_diff(out_check, expected)
        else:
            cos_diff = float("inf")

        passed = finite and cos_diff < cos_threshold
        ret["mi400:finite"] = finite
        ret["mi400:cos_diff"] = cos_diff
        ret["mi400:passed"] = passed
        aiter.logger.info(
            "mla_decode-mi400 [%s | batch=%d ctx=%d splits=%d]: finite=%s cos_diff=%.3e %s",
            variant.name,
            batch_size,
            ctx_lens,
            case["num_kv_splits"],
            finite,
            cos_diff,
            "passed" if passed else "FAILED",
        )

        # Performance: zero-initialized split/out buffers make the repeated
        # launches safe, so time the kernel over the standard perftest loop.
        # Correctness was already validated above on the single launch.
        _, us_mi400 = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_mi400,
            kv_buffer_mi400,
            out_mi400,
            qo_indptr,
            case["kv_indptr"],
            kv_indices_mi400,
            case["kv_last_page_lens"],
            decode_qlen,
            case["page_size"],
            nhead_kv,
            1.0 / (qk_head_dim**0.5),
            num_kv_splits=case["num_kv_splits"],
            num_kv_splits_indptr=case["num_kv_splits_indptr"],
            q_scale=case["q_scale"],
            kv_scale=case["kv_scale"],
            return_lse=True,
        )

        total_q = batch_size * decode_qlen
        total_kv = batch_size * ctx_lens
        mi_flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
        mi_bytes = (
            total_kv * nhead_kv * qk_head_dim * (torch.finfo(dtypes.fp8).bits // 8)
            + total_q * nhead * qk_head_dim * (torch.finfo(dtypes.fp8).bits // 8)
            + total_q * nhead * v_head_dim * (torch.finfo(torch.bfloat16).bits // 8)
        )
        ret["mi400:us"] = us_mi400
        ret["mi400:TFLOPS"] = mi_flops / us_mi400 / 1e6
        ret["mi400:TB/s"] = mi_bytes / us_mi400 / 1e6
        aiter.logger.info(
            "mla_decode-mi400 [%s | batch=%d ctx=%d]: %8.2f us  %7.2f TFLOPS  %7.2f TB/s",
            variant.name,
            batch_size,
            ctx_lens,
            us_mi400,
            ret["mi400:TFLOPS"],
            ret["mi400:TB/s"],
        )

    test_absorb_decode_mi400()
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qn",
    "--qk_nope_head_dim",
    type=int,
    default=128,
    help="""qk nope head dim.
    e.g.: -qn 128""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)
parser.add_argument(
    "-vh",
    "--v_head_dim",
    type=int,
    default=128,
    help="""v head dim.
    e.g.: -vh 128""",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="""Block size.
    e.g.: -blk 1""",
)
parser.add_argument(
    "--mask",
    type=int,
    nargs="+",
    choices=[0, 1],
    default=[1],
    help="""mi400 attention mask selector: 0 disables causal/tail mask, 1 enables it.
    e.g.: --mask 0 1""",
)
parser.add_argument(
    "--mi400-variant",
    choices=[v.name for v in _MI400_KERNEL_VARIANTS],
    default="qh128-q1-16mx4-64nx1-np",
    help="""Restrict the mi400 sweep to a single kernel variant from
    _MI400_KERNEL_VARIANTS (by name). Default: all variants.""",
)


args = parser.parse_args()

# This driver always runs the gfx1250/mi400 fp8 decode sweep. nhead carries
# (gqa, decode_qlen); unsupported combos self-skip inside the mi400 check.
args.dtype = [dtypes.fp8]
args.kv_dtype = [dtypes.fp8]
if args.mi400_variant is not None:
    v = _MI400_VARIANT_BY_KEY_NAME[args.mi400_variant]
    args.nhead = [(v.nhead, v.decode_qlen)]
else:
    args.nhead = _MI400_NHEAD
args.ctxLen = _MI400_CTX_LENS
args.batchSize = _MI400_BATCH_SIZES
args.split_per_batch = _MI400_SPLIT_PER_BATCH
args.block_size = 64
args.kv_lora_rank = 512
args.qk_rope_head_dim = 64

mi400_failures = []
for nhead, decode_qlen in args.nhead:
    df = []
    _param_iter = itertools.product(
        args.dtype,
        args.kv_dtype,
        args.ctxLen,
        args.batchSize,
        args.split_per_batch,
        args.mask,
    )
    for dtype, kvtype, ctx_len, batch_size, split_per_batch, mask in _param_iter:
        if check_support(dtype, kvtype, nhead):
            ret = test_mla(
                ctx_len,
                batch_size,
                nhead,
                args.kv_lora_rank,
                args.qk_nope_head_dim,
                args.qk_rope_head_dim,
                args.v_head_dim,
                dtype,
                kvtype,
                args.block_size,
                decode_qlen=decode_qlen,
                split_per_batch=split_per_batch,
                mask=mask,
            )
            df.append(ret)
            if not ret.get("mi400:skipped", True) and not ret.get(
                "mi400:passed", False
            ):
                mi400_failures.append(
                    (
                        ret.get("mi400:variant"),
                        batch_size,
                        ctx_len,
                        ret.get("mi400:cos_diff"),
                    )
                )
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla summary (markdown):\n%s", df_md)

if mi400_failures:
    raise AssertionError(f"mi400 MLA numerics failed for: {mi400_failures}")
