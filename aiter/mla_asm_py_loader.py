"""Python loader backend for the gfx1250 MLA decode stage1 asm kernel.

This is an OPT-IN alternative to the C++ host glue in
    csrc/py_itfs_cu/asm_mla.cu :: mla_decode_mi400_dispatch
It loads + launches the assembly `.co` directly from Python via the HIP runtime
(ctypes -> libamdhip64), instead of going through the compiled `.cu` op.

It is wired into aiter/mla.py behind the env switch AITER_MLA_PY_LOADER=1 and is
currently only enabled/verified for the qh128-q1 fp8 gfx1250 kernel:
    hsa/gfx1250/mla/mla_a8w8_qh128_1tg_16mx4_64nx1_np.co   (288-byte v3 ABI)

The tensor interface matches what aiter/mla.py passes to
`aiter.mla_decode_stage1_asm_fwd`, so it is a drop-in for the stage1 call.
"""

import csv
import ctypes
import glob
import math
import os

import torch

HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)


def _load_hip():
    """Load the SAME libamdhip64 torch uses (avoids ROCR symbol-version clashes
    with the system /opt/rocm copy)."""
    candidates = list(
        glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so*"))
    )
    try:
        import _rocm_sdk_core

        candidates += glob.glob(
            os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib", "libamdhip64.so*")
        )
    except ImportError:
        pass
    candidates.append("libamdhip64.so")
    last = None
    for cand in candidates:
        try:
            return ctypes.CDLL(cand)
        except OSError as exc:
            last = exc
    raise RuntimeError(f"could not load libamdhip64: {last}")


_hip = None


def _get_hip():
    global _hip
    if _hip is not None:
        return _hip
    hip = _load_hip()
    hip.hipGetErrorString.restype = ctypes.c_char_p
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    hip.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    hip.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _hip = hip
    return _hip


def _check(err, what):
    hip = _get_hip()
    if err != 0:
        raise RuntimeError(f"HIP error in {what}: ({err}) {hip.hipGetErrorString(err).decode()}")


_U32 = ctypes.c_uint32


class MlaMi400KernelArgs(ctypes.Structure):
    """288-byte kernarg layout (18 x 16-byte slots), mirrors MlaMi400KernelArgs
    in asm_mla.cu and the .co's patched .args offsets."""

    _pack_ = 1
    _fields_ = [
        ("ptr_R", ctypes.c_void_p), ("_p0", _U32 * 2),
        ("ptr_LSE", ctypes.c_void_p), ("_p1", _U32 * 2),
        ("ptr_Q", ctypes.c_void_p), ("_p2", _U32 * 2),
        ("ptr_KV", ctypes.c_void_p), ("_p3", _U32 * 2),
        ("ptr_LTP", ctypes.c_void_p), ("_p4", _U32 * 2),
        ("ptr_LTD", ctypes.c_void_p), ("_p5", _U32 * 2),
        ("ptr_LTL", ctypes.c_void_p), ("_p6", _U32 * 2),
        ("scalar", ctypes.c_float), ("_p7", _U32 * 3),
        ("q_seq_lens", _U32), ("_p8", _U32 * 3),
        ("passes", _U32), ("_p9", _U32 * 3),
        ("stride_Q", _U32), ("_p10", _U32 * 3),  # .args names this slot total_kv
        ("stride_page", _U32), ("_p11", _U32 * 3),
        ("log2_page", _U32), ("_p12", _U32 * 3),
        ("ptr_QTP", ctypes.c_void_p), ("_p13", _U32 * 2),
        ("ptr_STP", ctypes.c_void_p), ("_p14", _U32 * 2),
        ("out_16_nosplit", _U32), ("_p15", _U32 * 3),
        ("ptr_QROPE", ctypes.c_void_p), ("_p16", _U32 * 2),
        ("ptr_KVROPE", ctypes.c_void_p), ("_p17", _U32 * 2),
    ]


assert ctypes.sizeof(MlaMi400KernelArgs) == 288, ctypes.sizeof(MlaMi400KernelArgs)


def _default_mla_dir():
    """Resolve the gfx1250 MLA .co directory: AITER_ASM_DIR/gfx1250/mla if set,
    else the in-package hsa/gfx1250/mla (next to this aiter package)."""
    asm_dir = os.environ.get("AITER_ASM_DIR", "")
    if asm_dir:
        return os.path.join(asm_dir, "gfx1250", "mla")
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../aiter
    return os.path.join(pkg_root, "hsa", "gfx1250", "mla")


def select_kernel(mla_dir, q_type, kv_type, gqa, ps, qseqlen, prefill, causal, lse):
    """Mirror get_heuristic_kernel_mla in asm_mla.cu via mla_asm.csv."""
    csv_path = os.path.join(mla_dir, "mla_asm.csv")
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["qType"] == q_type
                and row["kvType"] == kv_type
                and int(row["Gqa"]) == gqa
                and int(row["ps"]) == ps
                and int(row["qSeqLen"]) == qseqlen
                and int(row["prefill"]) == prefill
                and int(row["causal"]) == causal
                and int(row["lse"]) == lse
            ):
                return row["knl_name"], row["co_name"]
    raise RuntimeError(
        f"no kernel in {csv_path} for q={q_type} kv={kv_type} gqa={gqa} ps={ps} "
        f"qSeqLen={qseqlen} prefill={prefill} causal={causal} lse={lse}"
    )


# Process-level cache of loaded (module, function) handles keyed by symbol: the
# first call loads + registers the .co, later calls reuse it (JIT behaviour).
_KERNEL_CACHE = {}


def _get_function(co_path, symbol):
    if symbol in _KERNEL_CACHE:
        return _KERNEL_CACHE[symbol]
    hip = _get_hip()
    module = ctypes.c_void_p()
    _check(hip.hipModuleLoad(ctypes.byref(module), co_path.encode()), "hipModuleLoad")
    func = ctypes.c_void_p()
    _check(
        hip.hipModuleGetFunction(ctypes.byref(func), module, symbol.encode()),
        "hipModuleGetFunction",
    )
    _KERNEL_CACHE[symbol] = (module, func)
    return module, func


def mla_decode_stage1_asm_fwd_py(
    Q,
    KV,
    qo_indptr,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    num_kv_splits_indptr,
    max_seqlen_q,
    page_size,
    nhead_kv,
    softmax_scale,
    splitData,
    splitLse,
    output,
    q_scale,
    kv_scale,
    mla_dir=None,
    stream=None,
    verbose=False,
):
    """Python re-implementation of asm_mla.cu::mla_decode_mi400_dispatch (gfx1250).

    Builds the 288-byte kernarg pack, selects the kernel from mla_asm.csv, loads
    the `.co` from `mla_dir` (auto-resolved if None), and launches it via
    hipModuleLaunchKernel on the current torch HIP stream.
    """
    assert q_scale is not None and kv_scale is not None, "fp8 path needs q_scale/kv_scale"
    if mla_dir is None:
        mla_dir = _default_mla_dir()

    batch = qo_indptr.size(0) - 1
    num_heads = Q.size(1)
    gqa_ratio = num_heads // nhead_kv
    kv_split = splitData.size(1)
    qk_head_dim = Q.size(2)
    q_elem_size = Q.element_size()
    sub_Q = gqa_ratio * max_seqlen_q
    q_seq_lens_kernel = max_seqlen_q * gqa_ratio

    knl_name, co_name = select_kernel(
        mla_dir, "fp8", "fp8", gqa_ratio, 0, max_seqlen_q, 0, 0, 0
    )
    co_path = os.path.join(mla_dir, co_name)
    _, func = _get_function(co_path, knl_name)

    args = MlaMi400KernelArgs()
    args.ptr_R = splitData.data_ptr()
    args.ptr_LSE = splitLse.data_ptr()
    args.ptr_Q = Q.data_ptr()
    args.ptr_KV = KV.data_ptr()
    args.ptr_LTP = kv_indptr.data_ptr()
    args.ptr_LTD = kv_page_indices.data_ptr()
    args.ptr_LTL = kv_last_page_lens.data_ptr()
    args.scalar = float(softmax_scale)
    args.q_seq_lens = q_seq_lens_kernel
    args.passes = kv_split
    args.stride_Q = nhead_kv * q_seq_lens_kernel * qk_head_dim * q_elem_size
    args.stride_page = KV.stride(0) * KV.element_size()
    args.log2_page = int(math.log2(page_size))
    args.ptr_QTP = qo_indptr.data_ptr()
    args.ptr_STP = num_kv_splits_indptr.data_ptr()
    args.out_16_nosplit = 1 if kv_split == 1 else 0
    args.ptr_QROPE = q_scale.data_ptr()
    args.ptr_KVROPE = kv_scale.data_ptr()

    arg_size = ctypes.c_size_t(ctypes.sizeof(args))
    extra = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(ctypes.byref(args), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.byref(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )

    gdx = (max_seqlen_q * gqa_ratio + sub_Q - 1) // sub_Q
    gdy = batch
    gdz = kv_split * 2 if gqa_ratio == 128 else kv_split
    bdx, bdy, bdz = 128, 1, 1

    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    c_stream = ctypes.c_void_p(stream)

    if verbose:
        print(
            f"[aiter][mla-py-loader] kernel={knl_name} co={co_path}\n"
            f"[aiter][mla-py-loader] grid=({gdx},{gdy},{gdz}) block=({bdx},{bdy},{bdz}) "
            f"q_seq_lens={args.q_seq_lens} passes={args.passes} stride_Q={args.stride_Q} "
            f"stride_page={args.stride_page} log2_page={args.log2_page} "
            f"out_16_nosplit={args.out_16_nosplit} scalar={args.scalar:.6g}"
        )

    hip = _get_hip()
    _check(
        hip.hipModuleLaunchKernel(
            func, gdx, gdy, gdz, bdx, bdy, bdz, 0, c_stream, None, extra
        ),
        "hipModuleLaunchKernel",
    )
