# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter vector_add_asm (LLVM asm kernel) against torch on gfx1250."""

import pytest
import torch

from aiter.ops.asm.asm_utils import get_gfx_from_device
from aiter.ops.asm.vector_add import BLOCK_SIZE, vector_add_asm

SENTINEL = -7.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_from_device(0) != "gfx1250",
    reason="vector_add_asm targets gfx1250",
)


@pytest.mark.parametrize("n", [1, 1000, 4096 + 37, 1 << 20])
def test_vector_add_asm_matches_torch(n):
    gen = torch.Generator(device="cuda").manual_seed(n)
    a = torch.randn(n, device="cuda", generator=gen)
    b = torch.randn(n, device="cuda", generator=gen)
    # The tail past n catches writes from the last, partial workgroup.
    buf = torch.full((n + BLOCK_SIZE,), SENTINEL, device="cuda")

    vector_add_asm(a, b, buf[:n])
    torch.cuda.synchronize()

    torch.testing.assert_close(buf[:n], a + b, rtol=0, atol=0)
    assert torch.all(buf[n:] == SENTINEL), "kernel wrote past n"
