# MLA decode stage2 优化 — 调用点入口 & 性能 baseline

> 目标:asm(fp8)路径的 stage2(`_fwd_kernel_stage2_asm`)在实际推理里比 bf16/triton 路径的 stage2 慢很多。本文件固定"真正在跑的调用点"和一份 baseline，供后续逐步优化对比。
>
> 硬件:gfx1250 (AMD Radeon, gfx1250)。torch 2.11 + rocm7.14。

---

## 1. 三个 stage2 实现，谁真正在跑

MLA decode 是两阶段(stage1 split-K 计算部分注意力 → stage2 跨 split 合并)。代码里有 **三份** stage2 归约实现，容易混淆：

| # | kernel | 文件:行 | 何时使用 | 与本优化关系 |
|---|--------|---------|----------|--------------|
| A | `_fwd_kernel_stage2_asm` (Triton) | `aiter/mla.py:20` | **fp8 汇编路径的 stage2**（真在用） | **优化目标** |
| B | `_pa_decode_sparse_reduce` (Gluon) | `aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_decode_sparse.py:558` | **bf16/triton 路径的 stage2**（gfx1250 上真在用） | **对标参照** |
| C | `_fwd_kernel_stage2` (Triton, SGLang 风格) | `aiter/ops/triton/attention/mla_decode.py:547` | 仅经 `decode_attention_fwd` / `decode_attention_fwd_grouped_rope` 触发；**aiter 库内部无任何调用**，只有测试和外部 caller 用 | 不在生产路径，**不作为参照** |

> 早期分析一度把 C 当作"triton 对照"，实为误判：ff/ATOM 部署（`ff/serve_dsv4.sh`：`ATOM_FORCE_ATTN_TRITON=1` + gfx1250）走的是 B。

## 2. 真正的调用点入口

### A. fp8 汇编路径 stage2（优化目标）

- kernel 定义：`aiter/mla.py:20` `_fwd_kernel_stage2_asm`
- 启动点 1：`aiter/mla.py:380`，在 `mla_decode_fwd()` 内
- 启动点 2：`aiter/mla.py:1307`，在 `mla_decode_fwd_v4_nm()` 内（benchmark 走这条）
- grid = `(num_seqs, num_heads)`，`num_warps=4, num_stages=2`
- 输入布局（v4_nm）：
  - `logits`   `[total_q, num_kv_splits, num_heads, dv]` fp32（split-major）
  - `attn_lse` `[total_q, num_kv_splits, num_heads, 1]` fp32（独立 buffer）
  - `output`   `[total_q, num_heads, dv]` bf16

### B. bf16/triton 路径 stage2（对标参照，gfx1250 上真在用）

- 分发入口：`aiter/ops/triton/attention/pa_decode_sparse.py:38` `pa_decode_sparse()`
- 后端选择：`pa_decode_sparse.py:144` `use_gluon = DEVICE_ARCH == "gfx1250"`；`:211-216` 据此选 `reduce_impl`
- stage2 归约 kernel（gluon）：`_gluon_kernels/gfx1250/attention/pa_decode_sparse.py:558` `_pa_decode_sparse_reduce`
- stage2 归约 kernel（非 gfx1250 回退，Triton）：`_triton_kernels/attention/pa_decode_sparse.py` 同名 `_pa_decode_sparse_reduce`
- 启动点：`pa_decode_sparse.py:282`，`grid_reduce = (T, cdiv(H, 1))`（每 (token, head) 一个 reduce CTA）
- `kv_splits == 1` 时（`:267`）直接返回，不发 stage2
- partials 布局：
  - `m_partial` / `l_partial` `[T, kv_splits, H_padded]` fp32
  - `acc_partial` `[T, kv_splits, H_padded, D]` fp32

### C. 未使用的 SGLang 风格 stage2（仅记录，避免再次误认）

- kernel：`aiter/ops/triton/attention/mla_decode.py:547` `_fwd_kernel_stage2`
- 仅经 `decode_attention_fwd`（`mla_decode.py:689`）/ `decode_attention_fwd_grouped_rope`（`mla_decode_rope.py:144`）调用；库内部不调用。

## 3. 复现 benchmark

脚本 `op_tests/_stage1_bench_pa.py` 已把 asm 与 pa **各自的 stage1 / stage2 分开计时**：
- asm 经 `mla_decode_fwd_v4_nm` → stage2 = A（`_fwd_kernel_stage2_asm`），用 `_profile_stage_times` 分离 s1/s2。
- pa 经 `pa_decode_sparse`（ATOM 方式，自动推断 split）→ `pa_s2 = pa_full - pa_s1`（stage2 = B，gluon reduce）。

原 `run_stage1_bench_pa.sh` 从宿主机 `docker exec` 进 `ff_mla` 容器跑。**若已在容器/等价环境内且无 docker**，直接跑 python（注意本环境 triton 3.7.0 需预导入 `triton.language`，否则 `import aiter` 触发 `torch._dynamo` 报 `triton has no attribute 'language'`）：

```bash
cd /app/aiter
# 完整默认 sweep
ENABLE_CK=0 python3 -c "import triton.language; import runpy,sys; \
  sys.argv=['_stage1_bench_pa.py']; \
  runpy.run_path('op_tests/_stage1_bench_pa.py', run_name='__main__')"
# 单组合: batch ctx gqa asm_split
ENABLE_CK=0 python3 -c "import triton.language; import runpy,sys; \
  sys.argv=['_stage1_bench_pa.py','64','512','64','4']; \
  runpy.run_path('op_tests/_stage1_bench_pa.py', run_name='__main__')"
```

（不要执行脚本 `.sh` 里的 `sudo rm -rf jit/built` 清缓存那步，否则每次重编。）

## 4. Baseline（gfx1250，batch=64，Q=1，50 iters / 2 warmup）

单位 us。`s2 pa/asm < 1` 表示 asm stage2 更慢（越小越慢）。asm_split=1 走 final-out 快路，asm_s2=0，不参与 stage2 对比。

| gqa | batch | ctx | asm_split | pa_split | asm_s1 | pa_s1 | s1 pa/asm | asm_s2 | pa_s2 | s2 pa/asm | asm_tot | pa_tot | tot pa/asm |
|----:|------:|----:|----:|----:|-----:|-----:|--:|-----:|-----:|--:|-----:|-----:|--:|
| 64 | 64 | 256 | 1 | 4 | 38.19 | 49.87 | 1.31x | 0.00 | 6.79 | — | 38.19 | 56.66 | 1.48x |
| 64 | 64 | 256 | 2 | 4 | 41.70 | 49.72 | 1.19x | 42.18 | 13.58 | 0.32x | 83.88 | 63.30 | 0.75x |
| 64 | 64 | 256 | 4 | 4 | 48.55 | 39.68 | 0.82x | 48.86 | 36.49 | 0.75x | 97.41 | 76.17 | 0.78x |
| 64 | 64 | 512 | 1 | 4 | 40.57 | 49.78 | 1.23x | 0.00 | 6.85 | — | 40.57 | 56.63 | 1.40x |
| 64 | 64 | 512 | 2 | 4 | 43.72 | 49.80 | 1.14x | 42.02 | 6.87 | 0.16x | 85.74 | 56.67 | 0.66x |
| 64 | 64 | 512 | 4 | 4 | 42.16 | 49.80 | 1.18x | 43.18 | 6.88 | 0.16x | 85.34 | 56.68 | 0.66x |
| 64 | 64 | 1024 | 1 | 4 | 42.18 | 49.85 | 1.18x | 0.00 | 7.20 | — | 42.18 | 57.05 | 1.35x |
| 64 | 64 | 1024 | 2 | 4 | 44.65 | 49.81 | 1.12x | 41.55 | 7.07 | 0.17x | 86.20 | 56.88 | 0.66x |
| 64 | 64 | 1024 | 4 | 4 | 46.55 | 49.99 | 1.07x | 46.26 | 6.70 | 0.14x | 92.80 | 56.69 | 0.61x |
| 128 | 64 | 256 | 1 | 2 | 45.99 | 39.64 | 0.86x | 0.00 | 18.35 | — | 45.99 | 57.99 | 1.26x |
| 128 | 64 | 256 | 2 | 2 | 44.27 | 49.85 | 1.13x | 45.35 | 8.20 | 0.18x | 89.63 | 58.05 | 0.65x |
| 128 | 64 | 256 | 4 | 2 | 46.15 | 49.87 | 1.08x | 48.67 | 8.07 | 0.17x | 94.82 | 57.93 | 0.61x |
| 128 | 64 | 512 | 1 | 2 | 37.27 | 41.35 | 1.11x | 0.00 | 17.55 | — | 37.27 | 58.90 | 1.58x |
| 128 | 64 | 512 | 2 | 2 | 45.66 | 49.86 | 1.09x | 45.95 | 9.89 | 0.22x | 91.61 | 59.75 | 0.65x |
| 128 | 64 | 512 | 4 | 2 | 47.80 | 49.82 | 1.04x | 48.85 | 11.56 | 0.24x | 96.66 | 61.38 | 0.64x |
| 128 | 64 | 1024 | 1 | 2 | 50.43 | 72.50 | 1.44x | 0.00 | 33.10 | — | 50.43 | 105.60 | 2.09x |
| 128 | 64 | 1024 | 2 | 2 | 45.70 | 70.09 | 1.53x | 49.03 | 29.99 | 0.61x | 94.73 | 100.08 | 1.06x |
| 128 | 64 | 1024 | 4 | 2 | 71.14 | 70.91 | 1.00x | 57.29 | 27.08 | 0.47x | 128.43 | 97.99 | 0.76x |

### 关键观察

1. **asm stage2 有 ~42–57us 的"地板"，几乎不随 split 数 / ctx 变化**（split=2 与 split=4、ctx 256→1024 都卡在 ~45us）。纯归约却与工作量脱钩 → 典型的"运行期循环不展开 + 访存延迟全暴露 + 固定开销主导"。
2. **对照 pa_s2 只有 ~7–12us 且随规模变化**（真正按工作量走）。常见 ctx≥512 区间差距达 **6–7×**。
3. **asm_s2 ≈ asm_s1**：轻量归约与整个 attention stage1 一样贵，量级明显不合理。
4. stage1（asm）本身有竞争力（`s1 pa/asm` 多数 ≥1）；短板集中在 stage2。

## 5. 重新分析:对标 gluon reduce，为什么它快 / 怎么借鉴

对照的 gluon `_pa_decode_sparse_reduce` 快在结构完全不同：

| 维度 | asm `_fwd_kernel_stage2_asm` | gluon `_pa_decode_sparse_reduce` |
|------|------------------------------|----------------------------------|
| split 数 | 运行期 `num_valid_kv_splits`（load 得来） | `KV_SPLITS: constexpr`（编译期常量） |
| split 归约方式 | **串行** online-softmax 递推（`for split_kv_id`，逐迭代更新 e_max/e_sum/acc） | **并行** 归约：一次性载入所有 split 后 `gl.max(axis=0)` / `gl.sum(axis=0)` |
| 访存 | 每个 split 独立 global load，延迟逐次暴露 | **TDM `async_load` 批量 DMA** 整个 `[KV_SPLITS, BLOCK_H, BLOCK_D]` slab 进 LDS，一次等待 |
| 依赖链 | 跨迭代串行依赖（下一步依赖上一步的 e_max/e_sum） | 无跨 split 串行依赖；D 轴 spread 到各 warp |
| 空 split 处理 | 运行期 `num_valid` 截断 | 载入全量后用 `seg_active` mask 掉（`m=-inf`→贡献 0） |

**根因**：asm 版把跨 split 合并写成了"运行期边界的串行 online-softmax 递推"，每轮 load→stall→exp→update，延迟无法重叠；这解释了那条与工作量无关的 ~45us 地板。gluon 版是"constexpr split + 批量 DMA 载入 + 沿 split 轴向量化并行归约"。

**借鉴方向（更新后的推荐）**：不是简单把现有串行循环 `static_range` 展开，而是改成 gluon/flash-decoding 的**向量化并行归约**形态：
1. `NUM_KV_SPLITS` 作为 `tl.constexpr` 传入（常规 decode 下各 batch 均匀，host 侧已知）。
2. 用一次 2D `tl.load` 取回 `[NUM_KV_SPLITS, BLOCK_DV]` 的 `tv` 和 `[NUM_KV_SPLITS]` 的 `lse`（split 作外层、`offs_d` 作内层，静态偏移）。
3. 用 `seg_active = split_id < num_valid` 掩码把无效 split 的 lse 置 `-inf`。
4. 向量化归约：`m = tl.max(lse); p = tl.exp(lse - m); l = tl.sum(p); acc = tl.sum(p[:,None]*tv, axis=0)`；输出 `acc / l`。

这样同时消除"运行期循环边界"和"跨迭代串行依赖"，逼近 pa 的 ~7us 量级。风险：`NUM_KV_SPLITS` 需 host 已知且各 batch 一致（本次范围仅覆盖常规 decode）；`BLOCK_DV × NUM_KV_SPLITS` 的寄存器/LDS 压力需实测 occupancy（split ≤ 16 有界）。

> **澄清（纠正早期表述）**：buffer 格式（asm 的 2-buffer 归一化 vs gluon 的 3-buffer 未归一化）与"串行 vs 并行合并"是**正交**的。asm 采用串行 online-softmax 递推是独立的实现选择，**不是归一化 / 2-buffer 造成的**——2-buffer 归一化格式一样能并行合并（`m=max_i(lse_i); w_i=exp(lse_i−m); out=Σ w_i·o_i / Σ w_i`）。因此上述改写**无需改动 stage1 输出格式**。

## 6. 复现口径备注

- 每次改完 kernel，用第 3 节命令复测，直接和第 4 节表对比 `asm_s2` 列。
- run-to-run 抖动约 ±1us（如 64/512/4 两次 44.08 / 43.18）。
- 对照 pa 为 bf16、asm 为 fp8，且此 bench 用 D=512（MLA QK 实为 576，pa QK 约低估 ~11%）——仅比 stage2 归约耗时时不受影响（stage2 两边都是 fp32 partials 归约）。
