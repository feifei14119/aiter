# MLA asm stage2 调优记录（tuning log）

> 本文件是 `_fwd_kernel_stage2_asm`（fp8 汇编路径的 stage2，`aiter/mla.py`）逐次调优的**滚动记录**。
> 每做一次优化就在"调优记录"下**追加一条**（不要覆盖历史条目）。
>
> 配套文档：
> - `mla_stage2_baseline.md`：调用点入口 + baseline（第 7 节为干净 GPU3 baseline，权威）。
> - `mla_stage2_analysis.md`：两个 pass 的 stage1 输出 / workload / 影响因素分析。

---

## 固定测量口径（每条记录都必须遵守）

- **必须独占 GPU3**：命令前加 `HIP_VISIBLE_DEVICES=3`。共享/争抢的 GPU 会产生假象（见记录 #1）。
- **复现命令**（在 `/app/aiter` 下）：

```bash
# 全量 sweep
HIP_VISIBLE_DEVICES=3 ENABLE_CK=0 python3 -c "import triton.language; import runpy,sys; \
  sys.argv=['_stage1_bench_pa.py']; \
  runpy.run_path('op_tests/_stage1_bench_pa.py', run_name='__main__')"
# 单组合: batch ctx gqa asm_split
HIP_VISIBLE_DEVICES=3 ENABLE_CK=0 python3 -c "import triton.language; import runpy,sys; \
  sys.argv=['_stage1_bench_pa.py','64','512','64','4']; \
  runpy.run_path('op_tests/_stage1_bench_pa.py', run_name='__main__')"
```

- 关注指标：`asm_s2`（本 kernel 的 device 时间，profiler 口径）；参照 `pa_s2`（gluon reduce）。
- 每条记录至少跑 **2 次**确认抖动 < ~1us 才可信。
- 数值正确性：`num_warps` 类调度改动天然 bit-identical；改动 kernel 数学结构时，必须与 torch 参考对比（见记录 #1 的 `/tmp/s2_correct.py` 方法）。
- 环境注意：本机 triton 3.7.0 需先 `import triton.language`，否则 `import aiter` 触发 `torch._dynamo` 报错。

### 记录条目模板

```
## 记录 #N — <一句话标题>   (YYYY-MM-DD)
- 改动: <文件:行 + 具体改了什么>
- 原理: <为什么这样改会更快>
- 结果: <asm_s2 前后对比表 / 关键数字>
- 正确性: <bit-identical 或与参考的误差>
- 结论: <采纳 / 放弃 / 待验证；下一步>
```

---

# 调优记录

## 记录 #0 — 建立 baseline & 定位（背景）   (2026-07-14)

- **问题**：实际推理中 fp8 汇编路径的 stage2 比 bf16/triton 路径慢很多。
- **实际在跑的 stage2**：asm = `_fwd_kernel_stage2_asm`（`aiter/mla.py:20`，启动 `:380`/`:1307`）；对照 = gluon `_pa_decode_sparse_reduce`（gfx1250 实际路径），**不是** `mla_decode.py` 的 `_fwd_kernel_stage2`（后者仅外部/测试用）。
- **一次弯路（重要教训）**：最早在**被其它作业争抢的 GPU** 上测出 `asm_s2` ~42–57us 且与 split/ctx 无关的"地板"，据此误判"运行期串行 online-softmax 递推是主因"，并尝试改成向量化并行归约——**无效**。随后发现 profiler 与 cuda-event 口径差 2.3×、跨次抖动大，最终定位到 **GPU 争抢**。重启 + 独占 GPU3 后，"地板"消失，`asm_s2` 恢复随 work 缩放。
- **教训**：先保证独占 GPU 与测量可信，再谈优化；不要在争抢环境下做微秒级归因。

## 记录 #1 — stage2 启动 num_warps 4 → 1   (2026-07-14)

- **改动**：`aiter/mla.py` 两处 `_fwd_kernel_stage2_asm` 启动配置：
  - `mla_decode_fwd`（约 `:405`）：`num_warps=4 → 1`
  - `mla_decode_fwd_v4_nm`（约 `:1338`）：`num_warps=4 → 1`
  - kernel 主体（串行 online-softmax）**保持原样**，仅改调度参数。

- **原理（详细）**：
  - stage2 每个 workgroup 处理一个 `(batch/token, head)`，把该 head 的 `BLOCK_DV = next_pow2(v_head_dim) = 512` 维输出向量沿 `offs_d` 分配给 workgroup 内的所有 lane。
  - gfx1250 是 **wave = 32 lane**。`num_warps=4` ⇒ 128 lane ⇒ 每 lane 只负责 `512/128 = 4` 个输出元素；`num_warps=1` ⇒ 32 lane ⇒ 每 lane 负责 `512/32 = 16` 个元素。
  - 这个 kernel 的核心是"沿 split 轴的归约"（每 lane 对自己的若干 D 元素串行累加所有 split）。**它不是靠多 warp 并行获益的**：更多 warp 只是把本就不大的 per-(batch,head) 工作切得更碎，每 lane 的算术强度（ILP）更低、寄存器/调度更零散，并且 512-wide 的 reduce 用 128 lane 反而让每 lane 工作量太小、无法有效摊薄访存与循环开销。
  - `num_warps=1` 让每 lane 拿 16 个元素、算术更密、循环体更"值当"，占用率/延迟隐藏反而更好。这与实际更快的 gluon `_pa_decode_sparse_reduce` 一致——它的 reduce 就用 `num_warps=1`（`BLOCK_H=1`，split 留在 thread 内）。
  - 对拍数据（GPU3，cuda-event，stripped kernel）：`serial@nw=1` 全场 ~12.6us；`serial@nw=4` 在大 work 时退化到 ~28us（nhead128/split4）。`num_warps` 是此 kernel 的**主导因素**，而"串行 vs 向量化"在 `nw=1` 下几乎无差（12.6 vs 13.4us）。

- **结果**（GPU3 独占，profiler 口径，`asm_s2` us，2 次稳定）：

| gqa | ctx | split | 前(nw=4) | 后(nw=1) | 提速 | pa_s2 |
|----:|----:|----:|-----:|-----:|:--|----:|
| 64 | 256 | 4 | 28.7 | 23.7 | ~17% | 8.9 |
| 64 | 512 | 4 | 28.4 | 23.5 | ~17% | 5.7 |
| 64 | 1024 | 4 | 28.7 | 23.7 | ~17% | 6.0 |
| 64 | * | 2 | ~14.3 | ~13.1 | ~8% | ~6 |
| 128 | 256 | 4 | 41.0 | 24.6 | **~40%** | 6.6 |
| 128 | 512 | 4 | 40.9 | 24.7 | **~40%** | 6.0 |
| 128 | 1024 | 4 | 39.3 | 24.0 | **~39%** | 6.9 |
| 128 | * | 2 | ~30.6 | ~22.7 | ~26% | ~6.5 |

  - 净效果：nhead=64 ~17%、nhead=128 ~26–40% 的 stage2 提速；nhead=128/split4 是最大受益点（41→24.6us）。

- **正确性**：`num_warps` 仅改变 D 维在 lane 间的划分，不改变每个输出元素的计算 → 与 `nw=4` 输出 **bit-identical（max abs diff = 0.0）**；与 torch 参考仅差 `0.0078`（bf16 输出舍入，量级 ~3.8，约 0.2%）。零风险。（验证脚本 `/tmp/s2_correct.py`。）

- **结论**：**采纳**。最小改动（2 行）、零正确性风险、稳定可复现的收益。
  - 剩余差距：`asm_s2` ~13–25us vs pa ~6us（约 2–4×）。
  - 下一步候选（均需在 GPU3 同口径验证）：①`num_warps` 按 nhead/split 自适应（大 split 时 gluon 用 4，可探）；②向量化并行归约（对大 split 更稳、去掉运行期循环）；③削减 stage2 运行期开销（`num_valid` 的 mgc/cdiv 计算、FINAL_OUT 分支）；④合并 V+lse 访存。
