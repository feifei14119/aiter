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

---

## #2 启动参数扫描（`num_stages` × `waves_per_eu`）—— 无收益，不采纳

- **动机**：在 nw=1 基础上，`num_stages`（软件流水）与 `waves_per_eu`（占用率提示）是**零 kernel 改动、零正确性风险**的可调项，优先穷举。
- **方法**：真实生产 kernel，GPU3 独占，扫 `num_warps∈{1,2} × num_stages∈{1,2,3} × waves_per_eu∈{1,2,4,8}`（脚本 `/tmp/s2_launch_sweep.py`，cuda-event）。
- **结果**：所有组合在 cuda-event 口径下**全部收敛到 ~19.4us**，彼此差异 <1%。
  - 教训：**cuda-event 循环微基准被 kernel launch/dispatch 开销(~19us 地板)主导**，无法分辨 20us 以下的 kernel 差异；对这些"快 kernel"，只有 **profiler device-time（真实 bench）**才是可信口径。
- **结论**：`num_stages`/`waves_per_eu` 在可信口径下无法区分，保持现值（ns=2, we=4/默认）。**不改**。

---

## #3 向量化并行归约（一次性加载所有 split，沿 split 轴 reduce）—— 无收益，回退

- **改动**：把 `else` 分支的串行 online-softmax 循环改为一次性 `tl.load` 全部 `NKS_P2=next_pow2(split)` 个 partial（`[NKS_P2, BLOCK_DV]`），用 `tl.max/tl.exp/tl.sum` 沿 split 轴向量化归约（无效 split 用 `-inf` 掩码）。新增 `NKS_P2` constexpr。
- **动机**：去掉运行期循环、提高访存并行度，理论上更接近 pa 的批量加载。
- **结果**（GPU3 独占，**真实 bench**，`asm_s2` us）：与 nw=1 串行 baseline **逐格几乎相同**：
  - nhead64/ctx512/split4：23.20 vs 23.5(串行)；nhead128/ctx512/split2：22.48 vs 22.4；split4 全场仍 ~23–24us。
- **结论**：向量化**不是杠杆**，回退为串行版（减少复杂度与编译期常量）。仅保留 #1 的 `num_warps=1`。

---

## 本质结论与差距归因

- `asm_s2` 耗时只取决于**读取的 partial 向量总数** `≈ bs×nhead×split`，与 `ctx` 无关：
  - split2/nhead64（8192 向量）→ ~13us；split2/nhead128 与 split4/nhead64（均 16384）→ ~23us；split4/nhead128（32768）→ ~24us（CTA 翻倍、并行度更高，故未线性翻倍）。
  - 33.5MB / 23us ≈ **1.46 TB/s**，远低于 gfx1250 HBM/cache 峰值 → **访存延迟/效率受限**，非算力或带宽峰值受限。
- pa 恒定 ~6us 的来源是 gluon `_pa_decode_sparse_reduce` 的 **TDM 异步批量 DMA + LDS 暂存 + LDS 内归约**；普通 triton 的逐 split `tl.load`（无论串行/向量化、无论 launch 参数）都补不上这个**结构性**差距。
- **判断**：在"仅调优 asm triton stage2 kernel"的范围内，直接杠杆已穷尽（`num_warps=1` 已拿到 17–40% 且零风险）。要进一步逼近 pa 的 ~6us，只能做**结构性改写**（改用 gluon 风格 TDM/LDS staged reduce），本质上等于复刻 pa 已有实现，超出本 kernel 微调范畴，建议作为独立评估项。

---

## #4 验证：大 ctx / 大 split 是否该用 num_warps=2/4？—— 否，nw=1 全场最优

- **疑问**：nw=1 的结论只测到 split=4。真实场景大 ctx（4K/8K）→ 自动推断的 `num_kv_splits` 更大 → stage2 reduce 循环更长，会不会 nw=2/4 反而更好（更多 warp 隐藏访存延迟）？
- **方法**：GPU3 独占，真实 bench（profiler device-time）。ctx=2048、gqa=128、batch=64，扫 `split∈{4,8,16,32} × num_warps∈{1,2,4}`（临时 env `AITER_S2_NW` 覆盖 launch，测完已移除）。
- **结果**（`asm_s2` us）：

| split | nw=1 | nw=2 | nw=4 |
|--:|--:|--:|--:|
| 4  | **25.8** | 28.5 | 38.6 |
| 8  | **39.4** | 40.4 | 59.2 |
| 16 | **64.0** | 65.8 | 101.6 |
| 32 | **106.3** | 118.3 | 183.6 |

- **结论**：**nw=1 在所有 split 下都最快**，且 split 越大多 warp 越吃亏（nw=4 的相对惩罚从 +50% 扩大到 +73%）。
- **原理**：`num_warps` 切的是 **D 轴(512)**，不是 split 轴。online-softmax 的跨迭代依赖链长度 = split 数，与 num_warps 无关；增 warp 只把 512 维 D 工作切得更碎（nw=4 → 每 lane 仅 4 元素），让**每一次** split 迭代的访存/算术更低效，大 split 下迭代更多 → 惩罚累积放大。延迟隐藏靠的是 CTA 数量（bs×nhead，数千个），而非单 CTA 内的 warp 数——nw=1 时一个 wave 的 32 lane 已一次性并行读完 512 维。
- **附带观察**：`asm_s2` 随 split 近似线性增长（25.8→39.4→64.0→106.3），而 pa 的 reduce 基本不随 split 线性放大——进一步印证差距是结构性的（TDM 批量加载 + LDS 归约），而非 launch 参数可弥补。**`num_warps=1` 硬编码是正确选择，无需按 ctx/split 自适应。**
