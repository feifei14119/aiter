// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// c[i] = a[i] + b[i] (fp32) for i < n, one element per thread, wave32.
// Kernarg: a @0, b @8, c @16 (global pointers), n @24 (int32), 32 bytes.
// Entry state: s[0:1] kernarg pointer, v0 workitem id x. gfx1250 has
// architected SGPRs: a non-cluster launch puts workgroup id x in ttmp9.
// BLOCK_SIZE (threads per workgroup, <= 256) may be set with -defsym.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

.ifndef BLOCK_SIZE
.set BLOCK_SIZE, 256
.endif
.if BLOCK_SIZE > 256
.error "BLOCK_SIZE exceeds .max_flat_workgroup_size (256)"
.endif

.text
.globl vector_add
.p2align 8
.type vector_add,@function
vector_add:
    s_load_b64 s[4:5], s[0:1], 0x0
    s_load_b64 s[6:7], s[0:1], 0x8
    s_load_b64 s[8:9], s[0:1], 0x10
    s_load_b32 s10, s[0:1], 0x18
    s_mul_i32 s11, ttmp9, BLOCK_SIZE
    v_and_b32 v0, 0x3ff, v0
    v_add_nc_u32 v1, s11, v0
    s_wait_kmcnt 0x0
    v_cmp_gt_i32 vcc_lo, s10, v1
    s_and_saveexec_b32 s12, vcc_lo
    s_cbranch_execz .Lvector_add_end
    v_lshlrev_b32 v1, 2, v1
    global_load_b32 v2, v1, s[4:5]
    global_load_b32 v3, v1, s[6:7]
    s_wait_loadcnt 0x0
    v_add_f32 v2, v2, v3
    global_store_b32 v1, v2, s[8:9]
.Lvector_add_end:
    s_endpgm
.size vector_add, . - vector_add

.rodata
.p2align 6
.amdhsa_kernel vector_add
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_count 2
  .amdhsa_kernarg_size 32
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 13
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.kernels:
  - .name: vector_add
    .symbol: vector_add.kd
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 13
    .vgpr_count: 4
    .max_flat_workgroup_size: 256
    .args:
      - {.name: a, .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: b, .offset: 8, .size: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: c, .offset: 16, .size: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: n, .offset: 24, .size: 4, .value_kind: by_value}
...
.end_amdgpu_metadata
