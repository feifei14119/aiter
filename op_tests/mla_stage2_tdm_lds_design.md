# MLA decode stage2 —— TDM + LDS 规约方案设计

> 目标读者：要把 asm `_fwd_kernel_stage2_asm`（triton）替换/补充为一个能逼近
> gluon `_pa_decode_sparse_reduce` 性能的原生 HIP kernel 的人。
>
> 结论前提（见 `mla_stage2_tuning_log.md` #1–#4）：
> - asm stage2 已经用 `num_warps=1` 拿到 17–40% 且零风险，直接调参杠杆已穷尽。
> - 剩余 ~2–4× 差距（`asm_s2` 13–25us vs `pa_s2` ~6us）是**结构性**的：
>   asm 逐 split 用 `tl.load` 从 **global** 串行读 partial，跨迭代还有 online-softmax
>   依赖链；实测有效带宽仅 ~1.46 TB/s，是**访存延迟/效率受限**。
> - gluon 之所以恒定 ~6us，是因为它用 **TDM 异步批量 DMA** 把一个 token 的所有 split
>   partial 一次性搬进 **LDS**，再在 LDS/寄存器里做归约。本文把这套机制落到 HIP。

---

## 1. 核心概念

### 1.1 TDM（Tensor DMA，gfx1250）
- gfx1250 上的**张量搬运引擎**：给定一个「张量描述符」（基址 + 各维 size/stride + 目标
  LDS 的 swizzle 布局），由**独立的 DMA 硬件**把 global memory 的一个多维子块**整块、
  合并（coalesced）**地异步拷贝进 LDS。
- 关键特性：
  - **异步**：`async_load` 发起后立即返回，DMA 与 ALU/后续指令**重叠**；用 `async_wait`
    在真正要读数据前同步。
  - **批量 + 合并**：一次描述符搬运覆盖 `[KV_SPLITS, (BLOCK_H,) D]` 整块，替代成百上千个
    零散的 `global_load`，把访存效率从"逐 split 依赖式小读"提升到"一次大块流式传输"。
  - **不占用 VALU/寄存器发射**：搬运期间计算单元可做别的事（预取下一 token、算 sink 等）。
- 对应 gluon API（真实实现里就是这几个）：
  ```python
  desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base_ptr, shape, strides, tile, smem_layout)
  gl.amd.gfx1250.tdm.async_load(desc, offsets, smem_buffer)
  gl.amd.gfx1250.tdm.async_wait(0)
  ```

### 1.2 LDS（Local Data Share / shared memory）
- CU 内的片上便签内存（gfx1250 每 CU 64KB），延迟 ~几十 cycle，远低于 HBM 的数百 cycle。
- 作用：把"沿 split 轴的归约"从**反复读 global** 变成**反复读 LDS**。online-softmax 的
  跨迭代依赖链依然存在，但每一步的数据来源从 HBM（高延迟）变成 LDS（低延迟），且大块
  搬运已被 DMA 重叠掉——这正是 ~4× 差距的来源。

---

## 2. 现有 asm stage2 的瓶颈（为什么慢）

现状（`aiter/mla.py` 的 `_fwd_kernel_stage2_asm`，2-buffer 归一化格式）：

```
grid = (num_seqs, num_heads)            # 每个 CTA 处理一个 (token, head)
每 CTA（num_warps=1 → 32 lane）:
  for split in range(num_valid_kv_splits):     # ← 串行，且每次都打 global
      tv     = load(Mid_O[..., split, ...])    # [D=512] fp32，从 global 读
      tlogic = load(Mid_lse[..., split, ...])  # 标量，从 global 读
      # online-softmax 合并（跨迭代依赖 e_max/e_sum/acc）
  out = acc / e_sum                            # bf16 写回
```

两个结构性问题：
1. **逐 split 从 global 读**：每次迭代 512×4B 的 `tv` 都是一次高延迟 HBM 往返，split 越多
   往返越多（实测 `asm_s2` 随 split 近似线性：25.8→39.4→64.0→106.3us）。
2. **依赖链踩在 global 上**：`acc *= old_scale; acc += p*tv` 依赖上一轮结果，无法靠增 warp
   并行（`num_warps` 只切 D 轴不切 split 轴，实测多 warp 反而更差）。

---

## 3. TDM + LDS 方案设计

### 3.1 数据布局（沿用 asm 的 2-buffer，做 drop-in 替换）
保持 asm stage1 的输出格式不变，直接替换 stage2：
- `Mid_O`   = `logits`   : `[total_q, KV_SPLITS, H, D]` fp32 —— 每 split 的**归一化输出**
- `Mid_lse` = `attn_lse` : `[total_q, KV_SPLITS, H, 1]` fp32 —— 每 split 的 log-sum-exp
- `O` (out) : `[total_q, H, D]` bf16 —— 最终输出

> 注：gluon 用的是 3-buffer（m/l/acc 未归一化）。本方案不改 stage1，沿用 2-buffer，
> 归约公式即 asm 现有的 online-softmax（数值等价）。若将来愿意改 stage1 为 3-buffer，
> 可进一步省掉一次 exp、与 gluon 完全对齐（见 §7 可选项）。

### 3.2 Grid / CTA 映射
- `grid = (total_q, cdiv(H, BLOCK_H))`，每个 CTA 处理**一个 token 的 BLOCK_H 个 head**。
- `BLOCK_H`：一次处理多少 head。`BLOCK_H=1` 最简单（对齐 asm 现有 grid）；`BLOCK_H>1`
  可摊薄 DMA 描述符与 sink 计算开销，但吃更多 LDS（见 §3.5 预算）。
- 线程组织：`blockDim = num_warps * 32`，沿 **D 轴（512）** 铺开（每线程负责 `D/blockDim`
  个输出元素）。**split 轴留在线程内**（每个线程对自己那几个 D 元素，遍历所有 split 做归约）。

### 3.3 执行流程
```
1. 计算本 CTA 负责的 (t, h0..h0+BLOCK_H)。
2. 由 kv_indptr 推出 num_valid_kv_splits（与 asm 同逻辑，屏蔽越界/早退 split）。
3. 建描述符：把 global 的 Mid_O[t, :, h0:h0+BH, :] 视为 [KV_SPLITS, BLOCK_H, D]，
   Mid_lse[t, :, h0:h0+BH] 视为 [KV_SPLITS, BLOCK_H]。
4. tdm.async_load 三块（或两块：acc + lse）到 LDS；async_wait(0)。
5. 从 LDS 读回，屏蔽无效 split（-inf / 0）。
6. 沿 split 轴做 online-softmax 归约（数据全在 LDS/寄存器，无 global 往返）。
7. 归一化 acc / e_sum，bf16 写回 global O。
```

### 3.4 归约怎么并行
- **D 轴并行**：512 个输出元素分给 `blockDim` 个线程，天然并行、无线程间通信。
- **split 轴**：每个线程在自己的 D 元素上，遍历 `num_valid_kv_splits` 做 online-softmax。
  依赖链仍在，但每步读 LDS（~30cyc）而非 HBM（~300cyc），且大块搬运已被 TDM 重叠。
- lse 的 `e_max`/`e_sum` 是**每 head 一个标量**、与 D 无关：可让每个线程各自从 LDS 的
  `s_lse[split]` 独立算（重复但极廉价），无需跨线程 reduce。

### 3.5 LDS 预算（重要约束）
每 CTA 的 LDS 占用 ≈ `KV_SPLITS × BLOCK_H × D × 4B`（acc）+ 小量（lse）。
- `BLOCK_H=1, D=512`：`KV_SPLITS × 2KB`。KV_SPLITS=4 → 8KB（宽裕）；=16 → 32KB；
  =32 → 64KB（**占满 gfx1250 单 CU 的 64KB LDS，占用率归零、且无双缓冲空间**）。
- 因此**大 split 必须分块**（见 §5 的 chunked 变体）：一次只搬 `CHUNK_K` 个 split 进 LDS，
  在线程内做**部分归约**（维护 running e_max/e_sum/acc），再搬下一块，用双缓冲让下一块的
  DMA 与当前块的计算重叠。这样 LDS 占用被 `CHUNK_K` 固定住，占用率可控。

---

## 4. HIP 伪代码（BLOCK_H=1，split 一次装得下的基础版）

> TDM 在 HIP 源码层通过 gfx1250 的 tensor-DMA 内建暴露（下文用 `tdm_*` 包装函数表示，
> 实际名字需对照 ROCm/LLVM 的 `__builtin_amdgcn_*` tensor-load-to-lds 系列确认；
> 语义等价于 gluon 的 `make_tensor_descriptor` + `async_load` + `async_wait`）。

```cpp
// 编译期常量
template <int KV_SPLITS,   // 本次 launch 的 split 数（<= 能装下的上限）
          int D,           // v_head_dim = 512
          int NUM_WARPS>   // 建议 1（见 tuning_log #1/#4）
__global__ void mla_stage2_tdm_lds(
    const float* __restrict__ Mid_O,     // [total_q, KV_SPLITS, H, D]
    const float* __restrict__ Mid_lse,   // [total_q, KV_SPLITS, H, 1]
    __hip_bfloat16* __restrict__ O,      // [total_q, H, D]
    const int*   __restrict__ qo_indptr,
    const int*   __restrict__ kv_indptr,
    const int*   __restrict__ valid_split_count,
    int H, int mgc,
    long s_mid_t, long s_mid_k, long s_mid_h,   // Mid_O 的 stride（token/split/head）
    long s_lse_t, long s_lse_k, long s_lse_h,   // Mid_lse 的 stride
    long s_out_t, long s_out_h)                 // O 的 stride
{
    const int t = blockIdx.x;          // token
    const int h = blockIdx.y;          // head（BLOCK_H=1）
    const int tid = threadIdx.x;       // 0 .. NUM_WARPS*32-1
    const int nthreads = NUM_WARPS * 32;
    constexpr int WSIZE = 32;          // gfx1250 wave size

    // --- 1. 有效 split 数（与 asm 同逻辑）---
    int kv_len = kv_indptr[t + 1] - kv_indptr[t];
    int nvalid = min(KV_SPLITS, (kv_len + mgc - 1) / mgc);
    nvalid = min(nvalid, valid_split_count[t]);

    // --- 2. LDS 便签：acc[split][D] + lse[split] ---
    __shared__ float s_acc[KV_SPLITS][D];   // 主体：KV_SPLITS*D*4 B
    __shared__ float s_lse[KV_SPLITS];

    // --- 3. TDM 批量异步搬运 global -> LDS ---
    // 描述符：把 Mid_O[t, :, h, :] 视为 [KV_SPLITS, D] 的二维子块
    const float* acc_base = Mid_O   + t * s_mid_t + h * s_mid_h;   // 起点
    const float* lse_base = Mid_lse + t * s_lse_t + h * s_lse_h;

    auto acc_desc = tdm_make_descriptor_2d(
        /*base=*/acc_base,
        /*shape=*/{KV_SPLITS, D},
        /*strides=*/{s_mid_k, /*沿 D=*/1},
        /*tile=*/{KV_SPLITS, D});
    auto lse_desc = tdm_make_descriptor_1d(
        /*base=*/lse_base, /*shape=*/{KV_SPLITS}, /*strides=*/{s_lse_k}, /*tile=*/{KV_SPLITS});

    tdm_async_load(acc_desc, /*offsets=*/{0, 0}, &s_acc[0][0]);  // 一次搬完所有 split
    tdm_async_load(lse_desc, /*offsets=*/{0},    &s_lse[0]);

    // （此处可插入与 DMA 重叠的工作：读 sink、算 h_off 等）

    tdm_async_wait(/*outstanding=*/0);   // 等两笔 DMA 落地
    __syncthreads();                     // 确保 LDS 对全 block 可见

    // --- 4. 无效 split 置 -inf / 0（TDM 会把早退 split 的脏数据也搬进来）---
    if (tid < KV_SPLITS) {
        if (tid >= nvalid) s_lse[tid] = -INFINITY;   // 死 split 的 lse -> -inf
    }
    __syncthreads();

    // --- 5. 沿 split 轴 online-softmax 归约（数据全在 LDS）---
    // 每个线程负责 D 上的 [tid, tid+nthreads, ...] 一组元素。
    for (int d = tid; d < D; d += nthreads) {
        float e_max = -INFINITY;
        float e_sum = 0.f;
        float acc   = 0.f;
        for (int k = 0; k < nvalid; ++k) {
            float lse = s_lse[k];           // 低延迟 LDS 读
            float tv  = s_acc[k][d];        // 低延迟 LDS 读
            float new_max = fmaxf(lse, e_max);
            float scale   = __expf(e_max - new_max);
            float p       = __expf(lse - new_max);
            acc   = acc * scale + p * tv;
            e_sum = e_sum * scale + p;
            e_max = new_max;
        }
        // --- 6. 归一化 + bf16 写回 ---
        float out = (e_sum > 0.f) ? (acc / e_sum) : 0.f;
        O[t * s_out_t + h * s_out_h + d] = __float2bfloat16(out);
    }
}
```

要点回顾：
- 第 3 步一次 TDM 把 `KV_SPLITS × D` 的 partial 整块搬进 LDS（合并、异步、可与计算重叠）。
- 第 5 步的 split 循环全部命中 LDS，取代了 asm 的逐 split global 往返——这是提速的关键。
- `e_max/e_sum` 每线程各算一份（标量、重复但廉价），避免跨线程 reduce。

---

## 5. HIP 伪代码（chunked over split：大 split / 控 LDS 占用）

当 `KV_SPLITS` 大到 LDS 装不下（如 32 → 64KB），改为**分块流式**：每次搬 `CHUNK_K` 个
split，双缓冲让下一块 DMA 与本块计算重叠，LDS 占用固定为 `2 × CHUNK_K × D × 4B`。

```cpp
constexpr int CHUNK_K = 4;                 // 每块 split 数，控 LDS
__shared__ float s_acc[2][CHUNK_K][D];     // 双缓冲
__shared__ float s_lse[2][CHUNK_K];

float e_max_r[D_PER_THREAD], e_sum_r[D_PER_THREAD], acc_r[D_PER_THREAD];
// running 状态初始化为 -inf/0/0（每线程持有自己那几个 D 元素的 running 归约）

int buf = 0;
tdm_async_load(chunk_desc(0), ..., s_acc[0], s_lse[0]);   // 预取第 0 块
for (int c = 0; c < num_chunks; ++c) {
    if (c + 1 < num_chunks)
        tdm_async_load(chunk_desc(c + 1), ..., s_acc[buf ^ 1], s_lse[buf ^ 1]); // 预取下一块
    tdm_async_wait(/*等本块=*/(c + 1 < num_chunks) ? 1 : 0);
    __syncthreads();

    // 用本块 [CHUNK_K] 个 split 更新 running 的 (e_max, e_sum, acc)
    for (int j = 0; j < this_chunk_valid; ++j) {
        float lse = s_lse[buf][j];
        for (int i = 0; i < D_PER_THREAD; ++i) {
            int d = tid + i * nthreads;
            float tv = s_acc[buf][j][d];
            float nm = fmaxf(lse, e_max_r[i]);
            float sc = __expf(e_max_r[i] - nm), p = __expf(lse - nm);
            acc_r[i]  = acc_r[i] * sc + p * tv;
            e_sum_r[i]= e_sum_r[i]* sc + p;
            e_max_r[i]= nm;
        }
    }
    buf ^= 1;
    __syncthreads();
}
// 归一化 + bf16 写回（同基础版第 6 步）
```

要点：LDS 占用与 `KV_SPLITS` **解耦**（只跟 `CHUNK_K` 有关），且**下一块的 DMA 与本块的
计算重叠**——这正是把访存延迟藏起来的机制，对大 ctx/大 split 尤其重要。

---

## 6. 与 gluon `_pa_decode_sparse_reduce` 的对应关系

| 步骤 | 本 HIP 方案 | gluon 参考（`pa_decode_sparse.py:557+`）|
|---|---|---|
| grid | `(total_q, cdiv(H,BLOCK_H))` | 同（`(T, cdiv(H, BLOCK_H))`）|
| 描述符 | `tdm_make_descriptor_*` | `gl.amd.gfx1250.tdm.make_tensor_descriptor` |
| 批量搬入 LDS | `tdm_async_load` → `__shared__` | `tdm.async_load` → `allocate_shared_memory` |
| 等待 | `tdm_async_wait(0)` | `tdm.async_wait(0)` |
| 屏蔽死 split | `nvalid` + `-inf`/0 | `seg_active` + `gl.where(-inf/0)` |
| split 归约 | 线程内沿 split 的 online-softmax | `gl.max/exp/sum(axis=0)` 向量化 |
| sink 融合 | （可选，见 §7）| `m_final=max(m,sink)`、`alpha_sink` |
| 写回 | `buffer_store` bf16 | `gl.amd.cdna4.buffer_store` |

主要差异：gluon 用 **3-buffer(m,l,acc 未归一化)** 并把 split 归约写成**沿 axis=0 的向量化
reduce**；本方案沿用 asm 的 **2-buffer 归一化**、split 归约用线程内 online-softmax。两者
数值等价，前者少一次 exp、后者不用改 stage1。

---

## 7. 预期收益、风险与验证

### 预期收益
- 把逐 split 的 global 往返换成"一次 TDM 大块搬入 + LDS 归约"，目标是把 `asm_s2` 从
  13–25us（小 split）/ 100us+（split=32）拉向 gluon 的 ~6us 量级。收益随 split 增大而增大
  （现状 asm 随 split 近似线性，TDM+LDS 的搬运可被 DMA 重叠、不再线性吃 HBM 延迟）。

### 风险 / 待确认
1. **TDM 内建的确切 HIP 接口**：gfx1250 tensor-DMA 在 HIP 源码层的 `__builtin_amdgcn_*`
   名称与描述符构造需对照当前 ROCm/LLVM 头文件确认（gluon 走的是编译器内建，HIP 直写需
   核对）。若 HIP 暂不便直接发 TDM，退路是用 `__builtin_amdgcn_global_load_lds`
   （async global→LDS，粒度较小）或普通 `ds_*` + 向量化 global load 先拿到 LDS 收益的一部分。
2. **LDS 占用 vs 占用率**：大 split 必须走 §5 的 chunked 变体，`CHUNK_K` 需按占用率调（建议
   从 4 起扫）。
3. **死 split 的脏数据**：TDM 会把 stage1 早退 split 的未初始化内存也搬进来，务必按
   `nvalid` 置 `-inf/0`（gluon 里就是 `seg_active` 那段），否则 `garbage*0=NaN`。
4. **数值正确性**：与现有 asm stage2 输出对拍（应 bit 级接近；exp 顺序不同可能有 ULP 级差异），
   再与 torch 参考对比（现状差 ~0.0078，bf16 舍入量级）。

### 验证方法（沿用现有基建）
- 性能：`op_tests/run_stage1_bench_pa.sh`（已支持 `GPU=<id>` 独占，profiler device-time
  口径，`pa_s2` 已改为直接抓 reduce kernel，不再相减）。对比新 kernel 的 `asm_s2` vs `pa_s2`。
- 正确性：与 `_fwd_kernel_stage2_asm` 同输入对拍（max abs diff），并与 torch 参考核对。
- 每步结果记入 `mla_stage2_tuning_log.md`（#5 起）。

---

## 8. 落地步骤建议（增量、可回退）
1. 先写 §4 基础版（BLOCK_H=1、小 split 装得下），对拍正确性 → 小 split 上跑通并测速。
2. 加 §5 chunked 变体，覆盖大 split（16/32），扫 `CHUNK_K`。
3. 加 sink 融合、`BLOCK_H>1` 摊薄开销（可选）。
4. 若 TDM 内建暂不可用，先用 `global_load_lds`/向量化 global→LDS 拿部分收益，验证 LDS
   归约方向正确后再替换为 TDM。
5. 全程与 gluon `pa_s2` 对标，达到同量级即可作为 asm 路径 stage2 的替代实现。
