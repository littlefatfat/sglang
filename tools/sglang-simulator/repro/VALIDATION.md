# 验证记录

## 环境门槛

```text
SGLang: v0.5.16 基线或当前 v0.5.16 适配分支
transformers: 5.12.1
sgl-kernel: 0.4.5
simulation mode: OFFLINE
```

检查：

```bash
python3 scripts/check_environment.py
```

## 代码回归

```bash
bash scripts/test_bundle.sh
pytest -q ../test/test_simulation_sglang_runner.py
bash scripts/validate_cpu_gpu.sh
```

验收：

```text
bundle tests: 5 pass
simulator unit/hook/predictor tests: 9 pass
deterministic runner: pass
CPU-only runner + trace replay: pass
GPU-visible runner + trace replay: pass
```

确定性 runner 必须覆盖：

```text
cold -> L1 device hit -> L2 host hit -> L3 storage hit
TTFT/E2E >= 0
request count exact
```

CPU/GPU 验证定义：

```text
CPU-only: CUDA_VISIBLE_DEVICES=""，torch.cuda.is_available() == false
GPU-visible: 指定物理 GPU，torch.cuda.is_available() == true
CPU spawned runner 覆盖 cold -> L1 -> L2 -> L3
CPU/GPU 均执行 page_size=256 的同一份 replay trace
```

GPU-visible replay 使用 `device=cuda`，因此
`PagedTokenToKVPoolAllocator.alloc_extend/alloc_decode` 实际调用 v0.5.16
的 Triton kernel。`load_format=dummy` 不加载模型权重，HiSim hook 不执行
真实模型 forward；step 时间仍来自所选 predictor。

2026-07-28 实跑：

```text
container: hisim-v0516-dev
CPU-only: torch.cuda.is_available() == false
GPU-visible: NVIDIA A100-SXM4-80GB
CPU spawned runner: pass (cold -> L1 -> L2 -> L3)
GPU-visible spawned runner: pass
CPU page_size=256 replay: 3/3 requests, pass
GPU page_size=256 replay: 3/3 requests, pass
```

v0.5.16 没有删除 GPU page allocator 优化：

```text
python/sglang/kernels/ops/memory/allocator.py
PagedTokenToKVPoolAllocator.alloc_extend -> alloc_extend_kernel (Triton)
PagedTokenToKVPoolAllocator.alloc_decode -> alloc_decode_kernel (Triton)
```

当前 kernel 还包含 `free_page_ptr` 的 `do_not_specialize` 和动态 blocked loop，
用于减少第二种指针对齐产生的额外 JIT，以及避免按 extend size 展开。

注意：GPU 首次运行包含 CUDA/Triton 初始化和 JIT，且当前实现会把这段
wall-clock 计入仿真 `cpu_overhead`。因此 CPU/GPU smoke 只做功能验收；
准确度或性能对比必须预热，并使用固定/回放 CPU overhead。

## 历史实测基线

来源：<https://github.com/sgl-project/sglang/issues/21891>。

Qwen3-8B / H20：

| Case | TTFT MAPE | TPOT MAPE | ITL MAPE | Input throughput MAPE | Duration MAPE | Prefix hit MAPE |
|---|---:|---:|---:|---:|---:|---:|
| no-cache | 3.42% | 1.64% | 1.78% | 1.41% | 1.39% | 0.00% |
| L1 | 4.15% | 3.47% | 3.54% | 2.40% | 2.33% | 0.04% |
| L2 | 2.38% | 4.05% | 4.07% | 2.35% | 2.29% | 0.00% |

Qwen3-32B-FP8 / H20：

| Case | TTFT MAPE | TPOT MAPE | ITL MAPE | Input throughput MAPE | Duration MAPE | Prefix hit MAPE |
|---|---:|---:|---:|---:|---:|---:|
| no-cache | 2.77% | 0.58% | 0.52% | 0.52% | 0.51% | 0.00% |
| L1 | 2.40% | 1.03% | 1.02% | 1.04% | 1.03% | 0.04% |
| L2 | 3.05% | 1.11% | 1.02% | 1.13% | 1.12% | 0.00% |

## v0.5.16 Replay 回归

同一 trace、同一 `server_args`、同一 replay table：

```bash
python3 scripts/compare_results.py \
  --old <0714-result-dir> \
  --new <v0516-result-dir>
```

必须报告：

```text
prefix/device/host/storage hit delta
iteration count
batch-size histogram
exact batch composition sequence
first mismatch iteration
TTFT/E2E APE
```

框架准确度门槛：

```text
prefix hit delta = 0
batch composition sequence = exact match
request count = exact match
```

时间准确度门槛：

```text
Replay miss rate near 0 后再判断 TTFT/E2E
ML step MAPE <= 5% 才进入 E2E 验证
AIC 必须单独报告 SOL/SILICON 与实测误差
```

## GLM5 v0.5.16 实跑

日期：2026-07-28。Case：

```text
hisim-num-node-1-glm-5-blksz-256-bucket-85-128-cnt-1816-time-60min-pod-9p7wt_slowdown_factor_1
1816 requests / 60 minutes / GLM-5.1-FP8 / B300 / TP8 replay config
```

旧版与 v0.5.16 使用相同 trace、`server_args` 和纯 forward replay table：

| 指标 | 旧版 | v0.5.16 | 相对误差 |
|---|---:|---:|---:|
| Prefix reused ratio | 0.6699217 | 0.6700850 | 0.0244% |
| L1 device hit ratio | 0.6230461 | 0.6304010 | 1.1805% |
| L2 host hit ratio | 0.0468756 | 0.0396840 | 15.3419% |
| Mean TTFT/E2E | 2542.83 ms | 2657.92 ms | 4.5261% |
| P90 TTFT/E2E | 5702.74 ms | 5952.61 ms | 4.3817% |

Batch：

```text
old iterations: 4747
new iterations: 4582
old/new mean batch size: 1.1106 / 1.1910
first mismatch iteration: 5 (zero-based)
exact sequence match: false
```

前五步 replay forward latency 完全一致。分叉前，旧版每步记录的
preprocess+postprocess 约 39–44 ms；v0.5.16 将本机 wall-clock
`cpu_overhead` 约 57 ms 推进仿真时钟，五步累计差约 80 ms，使下一请求进入
第六个 batch。`0714_mry_dev` 包含这段 wall-clock 推进，
`refactor-simplify` 删除了它。

结论：

```text
前缀树总命中量稳定；L1/L2 分层和 batch 时序未保持不变。
偏差源是仿真机 wall-clock CPU overhead，不是 replay predictor 或前缀树。
OFFLINE 准确度回归不能继续依赖本机实测 CPU overhead。
后续应使用固定/回放 CPU overhead，或使用包含 pre/post 的 replay table 并禁用重复计时。
```
