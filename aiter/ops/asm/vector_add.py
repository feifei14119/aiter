# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""vector_add from LLVM asm source (csrc/asm/gfx1250/vector_add/vector_add.s).

module_vector_add_asm is assembled into a code object on first use (or by
PREBUILD_KERNELS) and launched through asm_utils.
"""

import ctypes

import torch
from torch import Tensor

from ...jit.core import compile_ops
from .asm_utils import (
    current_raw_stream,
    device_guard,
    get_function,
    get_gfx_from_device,
    launch_co,
    register_asm_custom_op,
)

# Must match the source's default BLOCK_SIZE and .max_flat_workgroup_size.
BLOCK_SIZE = 256
KERNEL_NAME = "vector_add"


class VectorAddArgs(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("c", ctypes.c_void_p),
        ("n", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


@compile_ops("module_vector_add_asm", ffi_type="asm")
def _vector_add_code_objects() -> dict[str, str]: ...


def vector_add_asm_eager(a: Tensor, b: Tensor, out: Tensor) -> None:
    """out = a + b for contiguous fp32 tensors of equal size."""
    for name, t in (("a", a), ("b", b), ("out", out)):
        if t.dtype != torch.float32 or not t.is_contiguous() or not t.is_cuda:
            raise ValueError(
                f"vector_add_asm: {name} must be a contiguous fp32 GPU tensor"
            )
    if not a.numel() == b.numel() == out.numel():
        raise ValueError(
            "vector_add_asm: a, b and out must have the same number of elements"
        )
    if not a.device == b.device == out.device:
        raise ValueError("vector_add_asm: a, b and out must be on the same device")
    n = a.numel()
    # The kernel addresses elements with a 32-bit byte offset.
    if n >= 2**30:
        raise ValueError("vector_add_asm: at most 2**30 - 1 elements")
    if n == 0:
        return
    dev = a.device.index
    arch = get_gfx_from_device(dev)
    co_path = _vector_add_code_objects().get(arch)
    if co_path is None:
        raise RuntimeError(f"vector_add_asm: no code object for {arch}")
    args = VectorAddArgs(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, 0)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    with device_guard(dev):
        func = get_function(co_path, KERNEL_NAME, dev)
        launch_co(func, grid, (BLOCK_SIZE, 1, 1), args, stream=current_raw_stream(dev))


vector_add_asm = register_asm_custom_op(
    "vector_add_asm", vector_add_asm_eager, mutates_args=["out"]
)
