# MLA decode 两个 pass 的 stage2 分析(asm vs pa/gluon)

> 配套 `mla_stage2_baseline.md`(调用点入口 + 性能 baseline)。本文件聚焦：
> 1) 两个 pass 的 **stage1 输出 dtype/layout**；2) 两个 pass 的 **stage2 workload 分配**；
> 3) **影响性能的因素**；4) **改善 asm 的建议(按优化程度排序)**。
>
> 硬件:gfx1250(**wave = 32 lane**，见 gluon reduce 内 `WARP_SIZE=32`)。
>
> "两个 pass"：
> - **asm pass**(fp8 汇编)：stage2 = `_fwd_kernel_stage2_asm`(`aiter/mla.py:20`)。
> - **pa/triton pass**(bf16，gfx1250 实际在跑)：stage2 = gluon `_pa_decode_sparse_reduce`
>   (`aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_decode_sparse.py:558`)。

---

## 1. stage1 输出的数据类型与 layout(= stage2 的输入)

### asm pass

stage1 把部分结果写成 **两个独立张量**(split-major)：

| 张量 | shape | dtype | 说明 |
|------|-------|-------|------|
| `logits`(Mid_O) | `[total_q, num_kv_splits, num_heads, dv]` | `fp32` | 每 split 的 V 部分结果 |
| `attn_lse`(Mid_lse) | `[total_q, num_kv_splits, num_heads, 1]` | `fp32` | 每 split 的 lse，**独立 buffer** |

- 布局 `[seq, split, head, dim]`：split 维在 head 之前；V 与 lse 分处两个张量。
- stride(v4_nm)：`stride_mid_os = num_heads`(split 步长)，`stride_mid_ob = num_kv_splits*num_heads`。
- 最终输出 `output` `[total_q, num_heads, dv]` `bf16`。

### pa/triton pass(gluon)

stage1(main kernel)把部分结果写成 **三个张量**：

| 张量 | shape | dtype | 说明 |
|------|-------|-------|------|
| `m_partial` | `[T, kv_splits, H_padded]` | `fp32` | 每 split 的 max |
| `l_partial` | `[T, kv_splits, H_padded]` | `fp32` | 每 split 的 sum(exp) |
| `acc_partial` | `[T, kv_splits, H_padded, D]` | `fp32` | 每 split 的加权 V 累加 |

- 同样是 `[token, split, head, dim]` split-major；但 m/l/acc 分开(经典 flash-decoding partials）。
- 最终输出 `out` `[T, H, D]`，dtype 同 q(bf16)。

> 两个 pass 的 stage1 输出布局本质相同(都是 split-major fp32 partials)。**差距不在数据布局，而在 stage2 怎么消费它。**

### 为什么 asm 是 2 个 buffer、gluon 是 3 个(编码差异,非信息差异)

两者存的是**同一份信息的两种编码**(`o_i = acc_i/l_i`,`lse_i = m_i + log(l_i)`):

- **asm(2 buff)**:stage1 就把每 split **归一化**成 `o_i = acc_i/l_i`(存入 `logits`),并把 `m_i,l_i` **折成一个** `lse_i = m_i+log(l_i)`(存入 `attn_lse`)。好处:`num_kv_splits==1` 时 `logits` 直接就是最终输出(`mla.py:282` 令 `logits = o.view(...)`,stage2 走 `FINAL_OUT` 快路免合并);stage2 每 split 少读 1 个 float。代价:stage1 多做 per-split 除法 + `log`。
- **gluon(3 buff)**:保留**未归一化分子** `acc_i` + 独立 `m_i` + `l_i`(经典 flash-decoding partials)。因为不归一化,`l_i` 折不进 `o_i`,必须单独留,故 3 个。好处:stage1 省除法/`log`,最后只做一次除法(数值更稳、空 split 易 mask),且 2D(m/l)+3D(acc) 正好配 TDM 的 LDS 载入。

> **重要澄清(纠正早期表述):buffer 格式与"串行 vs 并行合并"是正交的两件事。**
> "asm 做了归一化 / 用 2-buffer" **并不会**逼出串行递推——2-buffer 归一化格式一样能并行合并:
> `m=max_i(lse_i); w_i=exp(lse_i−m); out=Σ w_i·o_i / Σ w_i`(与 gluon 的 `Σ acc_i·alpha_i / Σ l_i·alpha_i` 可并行性完全一致)。
> asm 采用串行 online-softmax 递推是**独立的实现选择**(沿用了 flash-attention stage1 的流式写法、便于处理运行期可变 split),不是数据格式造成的。**因此下面建议 1 在不改 stage1 输出格式(仍 2-buffer 归一化)的前提下即可完成。**

---

## 2. stage2 的 workload 分配对比

### 2.1 asm `_fwd_kernel_stage2_asm`(`aiter/mla.py:20`，启动 `:380` / `:1307`)

- **workgroup 划分**：`grid = (num_seqs, num_heads)` → **一个 WG = 一个 (batch, head)**。
- **workgroup 大小**：`num_warps=4` → gfx1250 wave32 ⇒ **128 lane**。`BLOCK_DV = next_pow2(dv) = 512`。
- **每个 workgroup 做什么**：对该 (batch,head) 的所有 query token(外层 `for cur_qo`，decode=1)、跨全部有效 split，做跨 split 的 online-softmax 合并，写出该 head 的 512 维输出。
- **每个 thread 做什么**：512 长的输出向量按 `offs_d` 均分到 128 lane ⇒ **每 lane 负责 4 个 dv 元素**；各 lane 保有自己的 `acc(4)`，`e_max/e_sum/tlogic` 为标量(全 lane 相同)。
- **归约方式(关键)**：`for split_kv_id in range(0, num_valid_kv_splits)` —— **运行期边界、串行 online-softmax 递推**；split 维不并行，每 lane 顺序遍历所有 split，逐轮 load `tv`(4 元素)+ 标量 `tlogic`，更新 `e_max/e_sum/acc`(跨迭代串行依赖)。无 LDS、无跨 lane 归约。

### 2.2 gluon `_pa_decode_sparse_reduce`(`..._gluon_kernels/gfx1250/...:558`，启动 `pa_decode_sparse.py:282`)

- **workgroup 划分**：`grid_reduce = (T, cdiv(H, BLOCK_H))`，`BLOCK_H=1` → **一个 WG = 一个 (token, head)**(每 head 一个 reduce CTA 的 fan-out，掩盖启动延迟)。
- **workgroup 大小**：`reduce_num_warps = 1`(split≤8；split>8 时 4) → wave32 ⇒ **32 lane**(或 128)。`BLOCK_D = 512`，`KV_SPLITS = constexpr`。
- **每个 workgroup 做什么**：TDM `async_load` 把整块 `[KV_SPLITS, BLOCK_H, BLOCK_D]` 的 acc 及 `[KV_SPLITS, BLOCK_H]` 的 m/l **批量 DMA 进 LDS**(一次 `async_wait`)，然后沿 split 轴做一次并行 log-sum-exp 合并，融合 sink，写最终输出。
- **每个 thread 做什么**：布局 `BlockedLayout([KV_SPLITS,BLOCK_H,SIZE_D], threads=[1,1,32], warps=[1,1,num_warps])`，即 **D 轴 spread 到各 lane/warp**，`KV_SPLITS` 与 `BLOCK_H` **整段留在 thread 内**。num_warps=1 时 `SIZE_D = 512/32 = 16` ⇒ 每 lane 持有 `[KV_SPLITS, 1, 16]`。
- **归约方式(关键)**：`m_max = gl.max(m_p, axis=0)`、`gl.sum(alpha*l, axis=0)`、`gl.sum(a_p*alpha, axis=0)` —— **沿 split 轴向量化并行归约**，且 split 轴在 thread 内 ⇒ 纯寄存器 reduce，无跨 lane shuffle、无跨迭代串行依赖。空 split 用 `seg_active` mask(置 `-inf`→贡献 0)。

### 2.3 对比小结

| 维度 | asm stage2 | gluon stage2 |
|------|-----------|--------------|
| WG 划分 | (batch, head) | (token, head) |
| WG 大小 | num_warps=4 → 128 lane | num_warps=1(或4) → 32(或128) lane |
| 每 lane 负责 | 4 个 dv 元素 + 串行遍历所有 split | 16 个 dv 元素 + 所有 split 在 thread 内 |
| split 数 | 运行期 `num_valid_kv_splits` | `KV_SPLITS: constexpr` |
| 归约 | 串行 online-softmax 递推 | 向量化并行 reduce(axis=split) |
| 访存 | 每 split 独立 global load | TDM 批量 DMA 整块进 LDS |
| 依赖链 | 跨 split 串行 | 无跨 split 串行依赖 |
| 空 split | 运行期截断循环 | 全量载入后 mask |

---

## 3. 影响性能的因素(按对 asm 的影响从大到小)

> **【更正】** 下列排序基于早期在**被争抢的 GPU** 上的测量，已被推翻。干净 GPU3 复测后：那条"~45us 地板"是争抢假象；stage2 实际随 split/nhead 缩放。经对拍验证，**真正的主因是 `num_warps=4` 对 `BLOCK_DV=512` 的 reduce 过度切分**（每 lane 仅 4 元素、ILP 低）；改 `num_warps=1` 即得 nhead64 ~17% / nhead128 ~40% 提速，且串行 vs 向量化在 nw=1 下基本无差。详见 `mla_stage2_baseline.md` 第 7 节。以下原始因素列表仅作历史参考。

1. **~~运行期循环边界 + 串行 online-softmax 递推(主因)~~（已证伪，见上方更正）**
   `num_valid_kv_splits` 由 load 得来 → 循环不可展开；且 `e_max/e_sum/acc` 跨迭代串行依赖。曾以为对应 baseline 的 ~45us 地板，实为 GPU 争抢假象。
2. **无批量 DMA / 无 LDS 暂存**
   asm 每 split 各发一次 global load，延迟逐次暴露；gluon 用 TDM 一次把整块搬进 LDS，延迟只付一次。
3. **split 维完全不并行**
   asm 128 lane 全部沿 D 展开、各自串行遍历 split；gluon 把 split 留在 thread 内做寄存器并行 reduce，等效"一把算完"。
4. **两条独立访存 stream(V 与 lse 分离)**
   `tv` 来自 `Mid_O`、`tlogic` 来自独立 `Mid_lse`(步长 `num_heads`)，第二条稀疏标量 stream 额外制造依赖点。
5. **(次要)layout 的跨 split 大步长**
   `[seq,split,head,dim]` 下同 head 跨 split 步长 `num_heads*dv`；但每次单 split 的 V load 本身连续/合并，带宽影响有限——不是主因。
6. **(次要)每 lane 负载偏小**
   128 lane 每 lane 仅 4 个 dv 元素、ILP 低;gluon 每 lane 16 元素、算术更密。

---

## 4. 改善 asm pass 的建议(按优化程度 / 收益排序)

> 范围:常规 decode(`num_kv_splits` 各 batch 均匀、host 侧已知)。

### 建议 1(收益最大):改写为"向量化并行归约"形态,对齐 gluon/flash-decoding

**不改 stage1 输出格式(仍是 2-buffer 归一化 `o_i`+`lse_i`)**,只重写 stage2 的合并算法。同时消除主因 1、3、4。做法:
1. `NUM_KV_SPLITS` 作 `tl.constexpr` 传入。
2. 一次 2D `tl.load` 取回 `tv[NUM_KV_SPLITS, BLOCK_DV]` 与 `lse[NUM_KV_SPLITS]`(split 外层、`offs_d` 内层,偏移全静态)。
3. `seg_active = split_id < num_valid`,把无效 split 的 `lse` 置 `-inf`。
4. 向量归约:`m=tl.max(lse); p=tl.exp(lse-m); l=tl.sum(p); acc=tl.sum(p[:,None]*tv, axis=0)`,输出 `acc/l`。

预期把 ~45us 地板打到接近 pa 的 ~7us 量级。风险:`BLOCK_DV × NUM_KV_SPLITS` 寄存器压力(split≤16 有界),需实测 occupancy。

### 建议 2(中等收益):保留串行结构,但 `NUM_KV_SPLITS` 常量化 + `tl.static_range` 展开 + 掩码

若建议 1 改动过大,退而求其次:把循环边界改成 constexpr 并 `static_range` 展开,用 `if split_id < num_valid`(或 `other=-inf` 无分支掩码)处理有效性。展开后静态偏移的 load 可被提前发射、和 `num_stages` 一起流水,部分掩盖延迟。收益不如建议 1(仍是串行依赖链),但改动小、风险低,可作第一步验证。

### 建议 3(增量收益):合并 V 与 lse 的访存 / 预取 lse

在建议 1 或 2 基础上,把 `NUM_KV_SPLITS` 个 `tlogic` 一次性向量 load 到寄存器再进归约,减少标量散射 load;或让 stage1 把 lse 紧跟 V 存到同一 buffer(需动 stage1,超出"不改 layout"约束,列为可选)。

### 建议 4(架构级,收益视场景):stage2 也做 split 并行 + 二次归约

当 (batch×head) CTA 数不足以填满 CU 时,可把 split 维也拆给不同 WG,再做一次小规模二次归约。但 gluon 并未这么做(它靠建议 1 的形态就够快),属过度设计,仅在极端 low-occupancy 场景考虑。

### 建议 5(调参,收益小):launch 配置微调

`num_warps` / `waves_per_eu` / `num_stages` 随建议 1 的新形态重扫;每 lane 负载变化后最优配置可能不同。作为收尾微调,不单独实施。

---

## 5. 验证方式

每改一版,用 `mla_stage2_baseline.md` 第 3 节命令复测,对比 `asm_s2` 列与该文件第 4 节 baseline。
