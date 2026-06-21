# `test_mla_mi400.py` 测试方法

本文档记录如何在 gfx1250（MI400）容器中运行 `op_tests/test_mla_mi400.py`
的 MLA fp8 decode 测试，并说明相关参数。

## 1. 运行环境

- 容器：`ff_mla`（镜像 `rocm/fw-bringup:gfx1250-atom-dev-*`，gfx1250）。
- 仓库路径（容器内外一致）：`/home/carhuang/feifei/aiter`。
- kernel 二进制：仓库内 `aiter/hsa/gfx1250/mla/*.co`，并在 `mla_asm.csv`
  注册 `knl_name` / `co_name`。

## 2. 关键环境变量

| 环境变量 | 作用 |
|---|---|
| `ENABLE_CK=0` | 禁用 CK 后端（参照 `poc_kl/mi400/mla/toaiter/README.md`）。 |
| `ENABLE_FLYDSL=0` | 禁用 FlyDSL，避免容器内 flydsl 版本与源码不匹配导致 `import aiter` 崩溃；MLA 测试不依赖 flydsl。 |
| `ROCM_HOME=/opt/rocm` | ROCm 安装路径。 |
| `GPU_ARCHS=gfx1250` / `AITER_GPU_ARCHS=gfx1250` | 目标架构。 |
| `AITER_ASM_DIR=.../aiter/hsa` | 指向仓库内 `hsa`，加载对应 `.co` kernel。 |
| `-u AITER_ASM_DEBUG` | 取消 ASM debug 编译宏，真实 launch（非调试）。 |
| `-u AITER_MLA_DEBUG_SKIP_KERNEL` | 取消「跳过 kernel launch」，执行真实数值校验。 |

> 首次 merge main 后若 `import aiter` 报 `cannot import name 'MxScaleRoundMode'`，
> 需全量重建 `module_aiter_core`（删 `aiter/jit/module_aiter_core.so` 与
> `aiter/jit/build/module_aiter_core` 后再跑）。

## 3. 命令行参数

| 参数 | 取值 | 说明 |
|---|---|---|
| `--mi400` | `auto`(默认 off) / `on` / `off` | 切换到 gfx1250/mi400 fp8 decode sweep。`auto` 在 `get_gfx()=="gfx1250"` 时自动开启；`on` 强制开启。 |
| `--mi400-variant` | `_MI400_KERNEL_VARIANTS` 中的某个 `name` | **只测试指定的单个 kernel variant**；不传则测试全部 variant。 |

可选 variant 名称（即 `_MI400_KERNEL_VARIANTS[*].name`）：

| variant 名称 | `(nhead, decode_qlen)` |
|---|---|
| `qh16-q1-16mx1-32nx4-np-3p` | (16, 1) |
| `qh16-q2-16mx2-32nx4-np-3p` | (16, 2) |
| `qh32-q1-32mx1-32nx4-np-3p` | (32, 1) |
| `qh16-q4-16mx4-64nx1-np` | (16, 4) |
| `qh64-q1-16mx4-64nx1-np` | (64, 1) |
| `qh128-q1-16mx4-64nx1-np` | (128, 1) |

`--mi400` 活动时，driver 会把 dtype/page/维度等覆盖成 mi400 fp8 decode 组合，
并按下列源码常量做笛卡尔 sweep：

- `_MI400_CTX_LENS`：KV 上下文长度列表。
- `_MI400_BATCH_SIZES`：batch 大小列表。
- `_MI400_SPLIT_PER_BATCH`：每 batch 的 KV split 数；其中 **`None` 表示交由
  `mla_decode_fwd` 的 `get_meta_param` 自动选择**（见第 5 节）。

## 4. 运行命令

### 测试全部 variant（完整 sweep）

```bash
docker exec ff_mla bash -lc 'cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  env -u AITER_ASM_DEBUG -u AITER_MLA_DEBUG_SKIP_KERNEL \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 ENABLE_FLYDSL=0 \
  GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 op_tests/test_mla_mi400.py --mi400 on'
```

### 只测试单个 variant（推荐，速度快）

例如只测 `qh128-q1-16mx4-64nx1-np`：

```bash
docker exec ff_mla bash -lc 'cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  env -u AITER_ASM_DEBUG -u AITER_MLA_DEBUG_SKIP_KERNEL \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 ENABLE_FLYDSL=0 \
  GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 op_tests/test_mla_mi400.py --mi400 on \
  --mi400-variant qh128-q1-16mx4-64nx1-np'
```

> 首次运行会 JIT 编译 `module_mla_asm`（删 build 目录后约十几秒）；后续运行可去掉
> `rm -rf ...` 一行复用缓存。

## 5. `num_kv_splits=None` 处理

`aiter/mla.py::mla_decode_fwd` 允许 `num_kv_splits=None`，此时通过
`get_meta_param(...)` 按 CU 占用率启发式自动选择 split 数及其 indptr。

测试侧 `_make_mla_mi400_case` 与之对齐：当传入的 `split_per_batch=None` 时，
同样调用 `aiter.mla.get_meta_param(None, batch, batch*num_pages_per_batch,
nhead, decode_qlen, dtypes.fp8)` 解析出**具体的 split 数**与 `num_kv_splits_indptr`，
供 shape 断言和 kernel 入参使用；解析后的值会记录在结果列 `mi400:num_kv_splits`。
传入具体整数（如 1 / 2）时则直接使用该值。

## 6. 输出与判定

- 每个组合先做一次干净启动的功能校验：`finite`（有限性）+ `cos_diff`（余弦差，
  阈值 `5e-2`，吸收 fp8 量化 + page shuffle + OOB 噪声），日志形如：

  ```
  mla_decode-mi400 [qh128-q1-16mx4-64nx1-np | batch=4 ctx=962 splits=2]: finite=True cos_diff=1.756e-04 passed
  ```

- 随后用 `run_perftest`（101 iters）计时，记录性能列 `mi400:us / mi400:TFLOPS /
  mi400:TB/s`。
- 每个 nhead 分组打印一张 `mla summary (markdown)` 表，含上述全部列。
- 任何非跳过组合 `cos_diff` 超阈值会在末尾抛 `AssertionError` 并以非零码退出。

> 注：当前 stage2 为临时 PyTorch 实现，且测试覆盖的 ctx/batch 组合多偏小，
> TFLOPS/TB·s 不代表纯 kernel 峰值；`us` 也含 host 开销。

## 7. 性能报告

运行单 variant 命令后，从日志末尾的 `mla summary (markdown)` 表读取
`mi400:us / mi400:TFLOPS / mi400:TB/s` 列即为性能报告。建议把日志重定向到文件再统计：

```bash
... python3 op_tests/test_mla_mi400.py --mi400 on \
  --mi400-variant qh128-q1-16mx4-64nx1-np > /tmp/mla_qh128.log 2>&1
grep -E "mla_decode-mi400|TFLOPS" /tmp/mla_qh128.log
```
