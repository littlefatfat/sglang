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
```

验收：

```text
bundle tests: all pass
simulator unit/hook/predictor tests: 9 pass
deterministic runner: pass
```

确定性 runner 必须覆盖：

```text
cold -> L1 device hit -> L2 host hit -> L3 storage hit
TTFT/E2E >= 0
request count exact
```

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
