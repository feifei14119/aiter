# Dump ATOM DeepSeek-V4 MLA → 跑 aiter 单测

> 配套脚本：`aiter/op_tests/test_mla_atom_dump.py`
> 被测 aiter 算子：`aiter.ops.triton.attention.pa_decode_sparse.pa_decode_sparse`
> 对应 ATOM kernel：`atom/model_ops/v4_kernels/paged_decode.py::sparse_attn_v4_paged_decode`
> 方法论：仓库根 `mla_aiter_triton_golden_howto.md`

把 ATOM 端到端真实跑出来的 **DeepSeek-V4 sparse MLA decode** 调用落盘，再在 aiter 容器里用
`pa_decode_sparse` 复算并和 ATOM 输出（golden）对比，验证 aiter 算子的数值正确性。

---

## 0. 这是哪个 MLA op？（先看这一段，别挂错 kernel）

DeepSeek-V4（含 V4-Flash）走 `DeepseekV4Backend`，decode 真实调用的是
`sparse_attn_v4_paged_decode`（`ATOM_USE_TRITON_ATTN=1` → triton 实现，即“mla triton pass”）。
它的 **aiter 对应算子**（签名完全一致）就是 `pa_decode_sparse`。

> 这条路径**不是** `attention_mla.py` 的 dense `decode_attention_fwd`（那是非 V4 模型 +
> `ATOM_USE_TRITON_MLA=1` 才走，对应 `test_mla_mi400_triton.py`）。两套别混。

签名（两边一致）：

```python
out = sparse_attn_v4_paged_decode / pa_decode_sparse(
    q,            # [T, H, D]  bf16
    unified_kv,   # [total_pages, D]  bf16（fp8 时另带 kv_scales）
    kv_indices,   # [total_indices] int32，每 token 的 slot 列表，扁平
    kv_indptr,    # [T+1] int32，前缀和
    attn_sink,    # [H] fp32
    softmax_scale,# float
    kv_scales=None,  # fp8 时 [total_pages, D//64] fp32
)  # -> [T, H, D]
```

---

## 1. ATOM 侧改了什么（已就绪）

- `atom/utils/envs.py`：新增 `ATOM_DUMP_MLA_DIR` / `ATOM_DUMP_MLA_MAX` / `ATOM_DUMP_MLA_LAYERS`。
- `atom/model_ops/mla_dump.py`：`dump_v4_sparse_decode(...)` 采集上面 7 个输入 + 输出 `o`。
- `atom/models/deepseek_v4.py`：在 `sparse_attn_v4_paged_decode(...)` 之后、**inverse RoPE 原地改写 `o` 之前**挂钩子。
- `scripts/dsv4/serve_dsv4.sh`：已内置 dump 开关，默认开到 `/data/mla_dump`。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ATOM_DUMP_MLA_DIR` | `/data/mla_dump`（serve 脚本里） | dump 目录；设为空字符串则**禁用** dump |
| `ATOM_DUMP_MLA_MAX` | `8` | 最多落盘多少份**完整 tensor**（params manifest 不限量） |
| `ATOM_DUMP_MLA_LAYERS` | 空（=所有层） | 逗号分隔的 layer id，只 dump 这些层 |

产物（落在 `ATOM_DUMP_MLA_DIR`，按 rank 区分）：
- `mla_calls.rank<r>.jsonl`：每次 decode 调用一行参数（T/H/D、每 token 的 kv span、ratio、
  softmax_scale...），**每次都写**，很轻 → 供功能 2。
- `mla_decode.rank<r>.<idx>.pt`：完整 tensor（只保留被 `kv_indices` 引用的 KV 页并重映射索引，
  所以即使 KV pool 很大单份也很小）+ 输出 `o`，最多 `ATOM_DUMP_MLA_MAX` 份 → 供功能 1。

---

## 2. 在 docker 里起 ATOM V4 服务并 dump

### 2.1 起服务（dump 默认开）

```bash
# 默认 dump 到 /data/mla_dump
bash scripts/dsv4/serve_dsv4.sh /data/DeepSeek-V4-Flash

# 自定义：多 dump 几份 tensor，只看前两层
ATOM_DUMP_MLA_MAX=16 ATOM_DUMP_MLA_LAYERS=0,1 \
  bash scripts/dsv4/serve_dsv4.sh /data/DeepSeek-V4-Flash

# 关闭 dump（传空目录）
ATOM_DUMP_MLA_DIR= bash scripts/dsv4/serve_dsv4.sh /data/DeepSeek-V4-Flash
```

等待服务就绪（`/v1/models` 可访问、日志无启动错误）。

### 2.2 发请求触发 decode

```bash
bash scripts/dsv4/send.sh
```

> dump 只在 **decode** 阶段触发（prefill 用的是另一个 kernel）。`send.sh` 的 `max_tokens: 128`
> 会产生足够多的 decode step。

### 2.3 确认产物

```bash
ls -l /data/mla_dump
# 期望看到：mla_calls.rank0.jsonl 以及若干 mla_decode.rank0.*.pt
```

> TP>1 时每个 rank 各写一份（`rank0/1/...`）。任选一个 rank 的产物跑单测即可。

---

## 3. 在 aiter 容器里跑单测

进对应 aiter 容器（含 gfx1250 gluon/jit；`pa_decode_sparse` 在 gfx1250 走 gluon，其它 arch 走 triton）。
若 host 直接 `import aiter` 报 `PermissionError`，说明 editable 映射指向容器内路径——必须在容器里跑。

### 功能 1：直接读 dump 的 tensor（aiter vs ATOM 输出 golden）

```bash
docker exec <ctr> bash -lc 'cd /home/carhuang/feifei/aiter/op_tests && \
  python3 test_mla_atom_dump.py --from-dump /data/mla_dump'
```

对每份 `.pt`：用 aiter `pa_decode_sparse` 跑同一份输入，报三个 cos：
- `cos(aiter, atom)`：**主判据**，aiter vs ATOM 真实 triton 输出 `o`。
- `cos(aiter, ref)` / `cos(atom, ref)`：再用独立 pure-torch reference 交叉验证（排除“自洽陷阱”）。

### 功能 2：只读参数（torch reference 作 golden）

```bash
docker exec <ctr> bash -lc 'cd /home/carhuang/feifei/aiter/op_tests && \
  python3 test_mla_atom_dump.py --from-params /data/mla_dump/mla_calls.rank0.jsonl'
```

按 manifest 里记录的 `(T, H, D, 每 token kv span)` 现场造随机 case，跑 torch reference 作 golden 验
aiter，报 `cos(aiter, ref)`。不需要任何 tensor 文件。

### 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--from-dump DIR` | — | 功能 1：dump tensor 目录 |
| `--from-params MANIFEST` | — | 功能 2：参数 manifest 文件（二选一，互斥） |
| `--cos-threshold` | `2e-2` | 判定阈值（bf16 sparse decode；fp8 可适当放宽） |
| `--no-dedup` | 关 | 功能 2：跑 manifest 每一行，而非仅去重后的不同 shape |

退出码：全 pass 退出 0；有 fail 抛 `AssertionError` 并列出失败项（CI 可直接用）。

---

## 4. 判定与对齐要点

- **golden 来源**：功能 1 的 `o` 就是 ATOM 真实 triton kernel 的输出，天然是 golden；再叠加独立
  torch reference 双保险。功能 2 没有 tensor，用 torch reference（ATOM 文档称其与 triton kernel
  bit-close）作 golden。
- **同源输入**：功能 1 两路（aiter / golden）读**同一份** dump 的 Q/KV，cos 差异只反映 kernel 实现。
- **fp8**：若 dump 的 `unified_kv` 是 fp8（带 `kv_scales`），aiter 走 in-kernel 1x64 block-scale
  反量化，reference 先 dequant 再算。本次 V4-Flash 的 decode 调用未传 `kv_scales`（KV 为 bf16），
  dump 会如实记录 dtype；单测自动按 dtype 分支。
- **阈值**：bf16 + fp32 累加，cos 通常远小于 `2e-2`。若某 case 偏高，先看是不是 fp8 或超长 ctx。

---

## 5. 排错

| 现象 | 原因 / 处理 |
|---|---|
| `/data/mla_dump` 没有任何文件 | 服务没走 decode（只发了 prefill？），或 `ATOM_DUMP_MLA_DIR` 被置空。确认 send 请求有输出 token；确认 serve 日志有 `mla_dump: wrote tensors ...` |
| 有 `mla_calls.jsonl` 但没有 `.pt` | tensor 配额用完（`ATOM_DUMP_MLA_MAX`）。调大它重跑，或用功能 2（只需 manifest） |
| `no mla_decode.rank*.*.pt tensor dumps under ...` | 功能 1 找不到 tensor 文件；确认目录对、`ATOM_DUMP_MLA_MAX>0` |
| host 上 `import aiter` 报 `PermissionError` | 必须在 aiter 容器内跑（含 asm/jit 的算子） |
| `pa_decode_sparse expects fp16/bf16 q` | q dump 出来不是 bf16/fp16；V4 decode 的 `q_sa` 应为 bf16，检查 dump dtype |
| cos 偏大但 finite | 先看是否 fp8 / 超长 ctx；用 `--cos-threshold` 临时放宽定位，再排查 layout/scale |

---

## 6. 覆盖边界

- **能抓**：`pa_decode_sparse` decode kernel 本身的数值错误、稀疏 gather（kv_indices/kv_indptr）
  处理、attn_sink 折叠、split-K reduce、fp8 反量化、长/变长 context。
- **抓不到**：写入端（SWA write / compressor / qk_norm_rope）的 kernel 正确性——decode 用的是已
  写好的 `unified_kv` 与已 rope 过的 q。要覆盖那类需另写写入端对比单测。
