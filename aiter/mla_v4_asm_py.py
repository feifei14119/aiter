# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-Python launcher for the gfx1250 v4-nm MLA decode `.co`.

Drop-in replacement for the stage1 launch that `mla_decode_fwd_v4_nm`
(aiter/mla.py) normally issues through `aiter.mla_decode_v4_asm` (the compiled
csrc/py_itfs_cu/asm_mla_v4.cu C-ABI). Instead of a `.cu` host stub, this module
reproduces the gfx1250 *preload* dispatch path in Python and launches the SAME
assembly `.co` directly via the HIP runtime (ctypes -> libamdhip64).

Only gfx1250 is handled here (the shipped preload ABI). Other arches keep using
the compiled `.cu` path — see the caller branch in mla.py.

Stream / CUDA-graph correctness
--------------------------------
The compiled C-ABI caller passes `torch.cuda.current_stream(device).cuda_stream`
as the kernel's stream (aiter/jit/core.py::_ctypes_call). We do EXACTLY the same
so that:
  * calls placed on a non-default stream launch on that stream, and
  * during `torch.cuda.graph` / `hipStreamCaptureMode` capture the launch is
    issued on the capture stream and is therefore recorded into the graph.

To keep capture state consistent we bind to the SAME libamdhip64 instance torch
already has mapped (discovered from /proc/self/maps), and we NEVER synchronize
or allocate on the launch path. `hipModuleLoad` happens once, lazily, on the
first call (a warm-up iteration outside capture — same lazy-load contract as the
C++ `AiterAsmKernel`), and is cached; it never runs during capture.
"""

import ctypes
import csv
import glob
import math
import os

import torch

from aiter.jit.core import get_asm_dir
from aiter.jit.utils.chip_info import get_gfx

# kV4DimNope + kV4DimRope = 448 + 64 = 512. The kernel hardcodes 1/sqrt(512) as
# its softmax pre-scale (the softmax_scale arg is ignored — kept for parity).
_KV4_DIM_NOPE = 448
_KV4_DIM_ROPE = 64

# HIP magic launch-param constants (hip_runtime_api.h). Identical extra-config
# protocol to AiterAsmKernel::launch_kernel in csrc/include/aiter_hip_common.h.
_HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
_HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
_HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)


# ----------------------------------------------------------------------------
# libamdhip64 binding (module load + launch only; torch owns all device memory)
# ----------------------------------------------------------------------------
_hip = None


def _load_hip():
    """Bind to the SAME libamdhip64 torch already has mapped.

    Using the exact instance torch loaded (rather than dlopen'ing another copy,
    e.g. /opt/rocm's, which has clashing ROCR symbol versions) is what keeps HIP
    graph-capture state consistent between torch and our ctypes launches.
    """
    global _hip
    if _hip is not None:
        return _hip

    candidates = []
    # 1) Whatever is already mapped in this process is guaranteed to be torch's.
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.rstrip().split(" ")[-1]
                if "libamdhip64.so" in path and os.path.exists(path):
                    candidates.append(path)
    except OSError:
        pass
    # 2) torch's bundled copy, 3) the rocm sdk wheel copy.
    candidates += glob.glob(
        os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so*")
    )
    try:
        import _rocm_sdk_core  # noqa: F401

        candidates += glob.glob(
            os.path.join(
                os.path.dirname(_rocm_sdk_core.__file__), "lib", "libamdhip64.so*"
            )
        )
    except ImportError:
        pass
    candidates.append("libamdhip64.so")

    last = None
    for cand in candidates:
        try:
            lib = ctypes.CDLL(cand)
        except OSError as exc:
            last = exc
            continue
        lib.hipGetErrorString.restype = ctypes.c_char_p
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipModuleLoad.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
        ]
        lib.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        _hip = lib
        return _hip
    raise RuntimeError(f"mla_v4_asm_py: could not load libamdhip64: {last}")


def _check(err, what):
    if err != 0:
        msg = _load_hip().hipGetErrorString(err).decode()
        raise RuntimeError(f"HIP error in {what}: ({err}) {msg}")


# co_path -> module handle ; (co_path, symbol) -> function handle.
_module_cache = {}
_func_cache = {}


def _get_function(co_path, symbol):
    hip = _load_hip()
    module = _module_cache.get(co_path)
    if module is None:
        module = ctypes.c_void_p()
        _check(hip.hipModuleLoad(ctypes.byref(module), co_path.encode()), "hipModuleLoad")
        _module_cache[co_path] = module
    key = (co_path, symbol)
    func = _func_cache.get(key)
    if func is None:
        func = ctypes.c_void_p()
        _check(
            hip.hipModuleGetFunction(ctypes.byref(func), module, symbol.encode()),
            "hipModuleGetFunction",
        )
        _func_cache[key] = func
    return func


# ----------------------------------------------------------------------------
# 120-byte packed-preload kernarg — ports MlaV4KernelArgsPreload (asm_mla_v4.cu).
# ----------------------------------------------------------------------------
class MlaV4KernelArgsPreload(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("ptr_R", ctypes.c_void_p),          # 0x00 splitData (logits) FP32 (rw)
        ("ptr_Q", ctypes.c_void_p),          # 0x08 Q packed FP8 + e8m0 scale
        ("ptr_KV", ctypes.c_void_p),         # 0x10 KV packed FP8
        ("ptr_LTP", ctypes.c_void_p),        # 0x18 kv_indptr
        ("ptr_LTL", ctypes.c_void_p),        # 0x20 kv_last_page_lens
        ("ptr_QTP", ctypes.c_void_p),        # 0x28 qo_indptr
        ("ptr_QROPE", ctypes.c_void_p),      # 0x30 Q rope BF16
        ("ptr_KVROPE", ctypes.c_void_p),     # 0x38 KV rope BF16
        ("scalar_f", ctypes.c_float),        # 0x40 1/sqrt(512)
        ("s_gqa_ratio", ctypes.c_uint32),    # 0x44 gqa_ratio * max_seqlen_q (MQA)
        ("s_kv_split", ctypes.c_uint32),     # 0x48 num_kv_splits == passes
        ("s_total_kv", ctypes.c_uint32),     # 0x4C kv_seq_lens * num_seqs
        ("out_16_nosplit", ctypes.c_uint32), # 0x50 0=fp32 split, 1=bf16 nosplit
        ("ptr_LSE", ctypes.c_void_p),        # 0x54 splitLse (attn_lse) FP32 (rw)
        ("ptr_LTD", ctypes.c_void_p),        # 0x5C kv_page_indices
        ("ptr_valid_split", ctypes.c_void_p),# 0x64 [num_seqs] i32 scratch (rw)
        ("s_use_valid_split", ctypes.c_uint32),  # 0x6C gates valid_split write
        ("ptr_sink", ctypes.c_void_p),       # 0x70 [num_heads] FP32 sink logit
    ]


assert ctypes.sizeof(MlaV4KernelArgsPreload) == 120


# ----------------------------------------------------------------------------
# CSV kernel table (ports get_heuristic_kernel_mla_v4 + cfg_mla_v4_asm), read
# from hsa/<gfx>/mla_v4/mla_v4_asm.csv — the same file codegen.py ingests.
# ----------------------------------------------------------------------------
_cfgs_cache = {}


def _load_cfg(asm_dir):
    cfgs = _cfgs_cache.get(asm_dir)
    if cfgs is not None:
        return cfgs
    csv_path = os.path.join(asm_dir, "mla_v4", "mla_v4_asm.csv")
    cfgs = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cfgs.append(
                dict(
                    qType=row["qType"].strip(),
                    kvType=row["kvType"].strip(),
                    Gqa=int(row["Gqa"]),
                    ps=int(row["ps"]),
                    qSeqLen=int(row["qSeqLen"]),
                    prefill=int(row["prefill"]),
                    causal=int(row["causal"]),
                    lse=int(row["lse"]),
                    knl_name=row["knl_name"].strip(),
                    co_name=row["co_name"].strip(),
                )
            )
    _cfgs_cache[asm_dir] = cfgs
    return cfgs


def _get_heuristic_kernel(
    asm_dir, q_type, kv_type, gqa, ps, prefill, causal, qseqlen, lse
):
    for cfg in _load_cfg(asm_dir):
        if cfg["qType"] != q_type or cfg["kvType"] != kv_type:
            continue
        if cfg["Gqa"] != gqa or cfg["ps"] != ps or cfg["prefill"] != prefill:
            continue
        if cfg["causal"] != causal or cfg["qSeqLen"] != qseqlen:
            continue
        if cfg["lse"] != lse:
            continue
        return cfg
    raise RuntimeError(
        f"mla_v4_asm_py: no shipped variant for q_type:{q_type} kv_type:{kv_type} "
        f"gqa:{gqa} ps:{ps} qSeqLen:{qseqlen} prefill:{prefill} causal:{causal} "
        f"lse:{lse}"
    )


def _dtype_str(t):
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return "fp8"
    if t.dtype == torch.bfloat16:
        return "bf16"
    raise RuntimeError(f"mla_v4_asm_py: unsupported dtype {t.dtype}")


def _warp_size():
    try:
        return int(torch.cuda.get_device_properties(0).warp_size)
    except AttributeError:
        return 32  # gfx1250 (RDNA family) is wave32


# ----------------------------------------------------------------------------
# Entry — mirrors the AITER_C_ITFS `mla_decode_v4_asm` gfx1250 preload path.
# Same positional arg order the mla.py wrapper uses (stream is picked up from
# the current torch stream on the tensors' device, exactly like the C ABI).
# ----------------------------------------------------------------------------
def mla_decode_v4_asm_py(
    Q,                 # [total_q, num_heads, head_size]  FP8 packed Q+e8m0
    qrope,             # [total_q, num_heads, kv_rotary]  BF16
    KV,                # [num_page, page_size, num_kv_heads, head_size] FP8
    kvrope,            # [num_page, page_size, num_kv_heads, kv_rotary] BF16
    qo_indptr,         # [num_seqs+1]
    kv_indptr,         # [num_seqs+1]
    kv_page_indices,   # [num_page_used]
    kv_last_page_lens, # [num_seqs]
    split_indptr,      # [num_seqs+1] (unused on preload; parity with the C ABI)
    sink,              # [num_heads] FP32
    max_seqlen_q,
    softmax_scale,     # ignored; kernel hardcodes 1/sqrt(512)
    out_16_nosplit,
    num_kv_splits,
    splitData,         # out: [total_q, num_kv_splits, num_heads, v_head_dim] FP32
    splitLse,          # out: [total_q, num_kv_splits, num_heads, 1] FP32
    output,            # out: [total_q, num_heads, v_head_dim] BF16 (out_16_nosplit==1)
    valid_split_count, # [num_seqs] int32 scratch, nullable
    use_valid_split_count_reduce,
):
    del softmax_scale, split_indptr  # not consumed by the gfx1250 preload ABI

    if sink is None or sink.data_ptr() == 0:
        raise RuntimeError(
            "mla_v4_asm_py: `sink` must be allocated (torch.full(-inf) for no-sink)"
        )
    assert Q.is_contiguous() and KV.is_contiguous()
    assert qrope.is_contiguous() and kvrope.is_contiguous()

    arch = get_gfx()
    if arch != "gfx1250":
        raise RuntimeError(
            f"mla_v4_asm_py implements the gfx1250 preload path only, got {arch}"
        )

    num_seqs = qo_indptr.shape[0] - 1
    num_heads = Q.size(1)
    num_kv_heads = KV.size(2)
    gqa_ratio = num_heads // num_kv_heads
    page_size = KV.size(1)
    dim_qk_packed = KV.size(3)  # per-token kernel stride in BYTES (FP8 = 1B/elem)

    assert num_kv_heads == 1, "mla_v4_asm_py: only support num_kv_heads==1 for now"
    assert Q.size(2) == dim_qk_packed, (
        "mla_v4_asm_py: Q head_size must equal KV head_size (= dim_qk_packed)"
    )

    scalar_f = 1.0 / math.sqrt(float(_KV4_DIM_NOPE + _KV4_DIM_ROPE))
    q_type = _dtype_str(Q)
    kv_type = _dtype_str(KV)

    # ---- V3-style per-shape heuristic (sub_Q + CSV lookup key) --------------
    sub_Q = 64
    config_max_seqlen_q = max_seqlen_q
    ps = 0
    prefill = 0
    causal = 0
    lse_flag = 0

    if gqa_ratio == 16 and q_type == "fp8" and kv_type == "fp8":
        if max_seqlen_q == 4:
            sub_Q, config_max_seqlen_q = 64, 4
        elif max_seqlen_q == 1:
            sub_Q, config_max_seqlen_q = 16, 1
        elif max_seqlen_q == 2:
            sub_Q, config_max_seqlen_q = 32, 2
        else:
            config_max_seqlen_q = 4
    elif gqa_ratio == 64 and q_type == "fp8" and kv_type == "fp8":
        if max_seqlen_q == 1:
            sub_Q, config_max_seqlen_q = 64, 1
        elif max_seqlen_q == 2:
            sub_Q, config_max_seqlen_q = 128, 2
        else:
            config_max_seqlen_q = 1
    elif gqa_ratio == 128 and q_type == "fp8" and kv_type == "fp8":
        if max_seqlen_q == 1:
            sub_Q, config_max_seqlen_q = 64, 1

    # ---- gfx1250 CSV lookup-key remap: shared qh64 .co serves (64,1)/(128,1)
    csv_gqa = gqa_ratio
    csv_qseqlen = config_max_seqlen_q
    if (
        q_type == "fp8"
        and kv_type == "fp8"
        and (gqa_ratio in (64, 128) and config_max_seqlen_q == 1)
    ):
        csv_gqa, csv_qseqlen = 64, 1

    asm_dir = get_asm_dir()  # hsa/<gfx>
    cfg = _get_heuristic_kernel(
        asm_dir, q_type, kv_type, csv_gqa, ps, prefill, causal, csv_qseqlen, lse_flag
    )
    co_path = os.path.join(asm_dir, "mla_v4", cfg["co_name"])
    func = _get_function(co_path, cfg["knl_name"])

    # ---- build the preload kernarg (gfx1250 overrides folded in) ------------
    args = MlaV4KernelArgsPreload()
    args.ptr_R = splitData.data_ptr()
    args.ptr_Q = Q.data_ptr()
    args.ptr_KV = KV.data_ptr()
    args.ptr_LTP = kv_indptr.data_ptr()
    args.ptr_LTL = kv_last_page_lens.data_ptr()
    args.ptr_QTP = qo_indptr.data_ptr()
    args.ptr_QROPE = qrope.data_ptr()
    args.ptr_KVROPE = kvrope.data_ptr()
    args.scalar_f = scalar_f
    args.s_gqa_ratio = gqa_ratio * max_seqlen_q  # gfx1250 flattened MQA
    args.s_kv_split = int(num_kv_splits)
    args.s_total_kv = KV.size(0) * page_size     # gfx1250 real total_kv
    args.out_16_nosplit = int(out_16_nosplit)
    args.ptr_LSE = splitLse.data_ptr()
    args.ptr_LTD = kv_page_indices.data_ptr()

    if use_valid_split_count_reduce != 0:
        if valid_split_count is None or valid_split_count.data_ptr() == 0:
            raise RuntimeError(
                "mla_v4_asm_py: gfx1250 requires valid_split_count scratch when "
                "use_valid_split_count_reduce!=0"
            )
    if valid_split_count is not None and valid_split_count.data_ptr() != 0:
        assert valid_split_count.dtype == torch.int32, "valid_split_count must be int32"
        assert valid_split_count.size(0) >= num_seqs
        args.ptr_valid_split = valid_split_count.data_ptr()
    else:
        args.ptr_valid_split = None
    args.s_use_valid_split = 1 if use_valid_split_count_reduce != 0 else 0
    args.ptr_sink = sink.data_ptr()

    # ---- launch geometry ----------------------------------------------------
    block_dim = 4 * _warp_size()
    q_seq_lens_internal = gqa_ratio * max_seqlen_q
    gdx = (q_seq_lens_internal + sub_Q - 1) // sub_Q
    gdy = num_seqs
    gdz = int(num_kv_splits)

    # ---- extra config: single packed kernarg buffer via BUFFER_POINTER/SIZE -
    arg_size = ctypes.c_size_t(ctypes.sizeof(MlaV4KernelArgsPreload))
    extra = (ctypes.c_void_p * 5)(
        _HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(ctypes.byref(args), ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.byref(arg_size), ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_END,
    )

    # Same stream the compiled C ABI would use: the current torch stream on the
    # tensors' device. During graph capture this is the capture stream, so the
    # launch is recorded into the graph.
    stream = torch.cuda.current_stream(Q.device).cuda_stream

    hip = _load_hip()
    _check(
        hip.hipModuleLaunchKernel(
            func,
            gdx, gdy, gdz,
            block_dim, 1, 1,
            0,
            ctypes.c_void_p(stream),
            None,
            extra,
        ),
        "hipModuleLaunchKernel",
    )
