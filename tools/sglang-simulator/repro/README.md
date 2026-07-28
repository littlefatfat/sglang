# HiSim 最简复现

适用版本：SGLang `v0.5.16` 基线；容器 `hisim-v0516-dev`。

## 1. 进入环境

```bash
ssh 33.255.171.37
docker exec -it -w /host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation \
  hisim-v0516-dev bash
cd tools/sglang-simulator/repro
python3 scripts/check_environment.py
```

必须满足：

- `sglang` 从 `/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/python` 导入。
- `/nfs` 已挂载。
- `SGLANG_SIMULATOR_CONFIG_PATH` 指向本次配置。
- 每次运行使用独立的 `SGLANG_SIMULATOR_OUTPUT_DIR` 和
  `SGLANG_SIMULATOR_HICACHE_STORAGE_KEYS_PATH`。

## 2. 仿真模式

```bash
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE   # 不 sleep，按逻辑时间运行
# 或
export SGLANG_SIMULATOR_OUTPUT_MODE=BLOCKING  # 按预测 step 时间 sleep
```

准确度回归优先用 `OFFLINE`。验证服务交互行为时用 `BLOCKING`。

## 3. 启动方式一：服务化

终端 A：

```bash
python3 scripts/start_service.py \
  --server-args configs/qwen3-8b-h20/server_args.json \
  --hisim-config configs/qwen3-8b-h20/hisim.aic.json \
  --mode OFFLINE \
  --output-dir /tmp/hisim/qwen3-8b-service
```

服务地址：`http://127.0.0.1:30000`。

### 3.1 终端打 benchmark 流量

Random：

```bash
python3 -m sglang_simulator.simulation.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name random \
  --request-rate 4 \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 100 \
  --warmup-requests 0
```

ShareGPT：

```bash
python3 -m sglang_simulator.simulation.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name sharegpt \
  --dataset-path workloads/sharegpt.example.json \
  --request-rate 4 \
  --num-prompts 2 \
  --warmup-requests 0
```

HiSim trace：

```bash
python3 scripts/send_trace.py \
  --trace workloads/trace.example.jsonl \
  --base-url http://127.0.0.1:30000
```

结果：

```text
/tmp/hisim/qwen3-8b-service/metrics.json
/tmp/hisim/qwen3-8b-service/request.jsonl
/tmp/hisim/qwen3-8b-service/iteration.jsonl
```

## 4. 启动方式二：Python 同进程

Trace 回放：

```bash
python3 scripts/run_inprocess.py \
  --server-args configs/qwen3-8b-h20/server_args.json \
  --hisim-config configs/qwen3-8b-h20/hisim.replay.json \
  --workload trace \
  --dataset workloads/trace.example.jsonl \
  --output-dir /tmp/hisim/qwen3-8b-inprocess
```

Random request-rate：

```bash
python3 scripts/run_inprocess.py \
  --server-args configs/qwen3-8b-h20/server_args.json \
  --hisim-config configs/qwen3-8b-h20/hisim.aic.json \
  --workload random \
  --num-prompts 100 \
  --input-len 1024 \
  --output-len 128 \
  --request-rate 4 \
  --output-dir /tmp/hisim/qwen3-8b-random
```

ShareGPT request-rate：

```bash
python3 scripts/run_inprocess.py \
  --server-args configs/qwen3-8b-h20/server_args.json \
  --hisim-config configs/qwen3-8b-h20/hisim.aic.json \
  --workload sharegpt \
  --dataset workloads/sharegpt.example.json \
  --num-prompts 2 \
  --request-rate 4 \
  --output-dir /tmp/hisim/qwen3-8b-sharegpt
```

## 5. Workload

### 5.1 Benchmark 生成

- `random`：固定输入/输出长度。
- `sharegpt`：读取 ShareGPT JSON 数组；每项至少包含两轮
  `conversations[].value` 或 `conversations[].content`。
- `request-rate=inf`：全部请求逻辑时间为 0。
- 有限 `request-rate`：按 Poisson 到达间隔生成逻辑时间，不在客户端 sleep。

### 5.2 Trace 回放格式

一行一个 JSON：

```json
{"created_time": 1710000000.0, "input_ids": [1, 2, 3], "input_length": 3, "output_length": 1, "request_id": "r0"}
```

必填字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `created_time` 或 `timestamp` | number | 到达时间；整份 trace 使用同一单位 |
| `input_ids` | `list[int]` | 输入 token |
| `input_length` | int | 必须等于 `len(input_ids)` |
| `output_length` | int | 输出 token 数；P 节点仿真通常设为 `1` |

脚本按最早时间归零。若输入 `timestamp` 为毫秒，增加
`--timestamp-scale 1000`。

## 6. 时间预测器

只修改 `hisim_config.json` 的 `predictor`。

AIC：

```json
{
  "name": "aiconfigurator",
  "database_path": "/host/aiconfigurator/src/aiconfigurator/systems/",
  "database_mode": "SILICON"
}
```

`database_mode=SOL` 用公式估算；`SILICON` 使用 perf database。AIC 结果必须先做
step-level 校验，不能默认当真实值。

ML：

```json
{
  "name": "ml",
  "database_path": "/host/insight_benchmark/test/hisim/hicache/hisim_results/b300_tp4_prefill_hgbmono_v2/latency_model_b300_v2_hgbmono_p50.pkl",
  "latency_scale": 1.0
}
```

当前 ML ABI 固定为以下 18 个特征，顺序不可改：

```text
batch_size, sum_extend, max_extend, min_extend,
sum_past, max_past, min_past,
sum_extend_x_past, sum_extend_squared, sum_past_squared,
sum_attn_flops, sum_extend_x_max_past,
log1p_sum_past, log1p_sum_attn_flops,
batch_size_x_sum_extend, max_past_minus_min_past,
is_decode, is_prefill
```

Replay：

```json
{
  "name": "replay",
  "database_path": "workloads/replay_table.example.json",
  "miss_strategy": "zero",
  "miss_fallback_seconds": 0.0
}
```

`zero` miss 会低估耗时；`knn` 会引入插值误差。回归时必须报告 exact/miss
比例。

## 7. 模型、硬件、部署

| 配置目录 | 模型 | 目标硬件 | 部署形态 | 状态 |
|---|---|---|---|---|
| `qwen3-8b-h20` | Qwen3-8B | H20 | P/D 混合单实例 | 历史 PR 已验证 |
| `qwen3-32b-fp8-h20` | Qwen3-32B-FP8 | H20 | P/D 混合单实例 | 历史 PR 已验证 |
| `glm5-p-b300` | GLM-5.1-FP8 | B300, TP8 | PD 分离的 P 节点 | 当前新增 |
| `dsv4pro-p-gb300` | DeepSeek-V4-Pro | GB300, TP4 | PD 分离的 P 节点 | 当前新增 |

当前目录不包含 D 节点、KV 传输、gateway 或多实例仿真。

历史范围：

- Roadmap：<https://github.com/sgl-project/sglang/issues/21891>
- PR：<https://github.com/sgl-project/sglang/pull/22250>
- PR 中验证：Qwen3-8B、Qwen3-32B-FP8；no-cache、L1、L2。

运行 GLM5 P 节点：

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
python3 scripts/run_inprocess.py \
  --server-args configs/glm5-p-b300/server_args.json \
  --hisim-config configs/glm5-p-b300/hisim.aic.json \
  --workload trace \
  --dataset /host/bl_data_trace/multi_node_trace_combine_glm-5/<trace>.jsonl \
  --output-dir /tmp/hisim/glm5-p
```

运行 DSv4Pro P 节点：

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
python3 scripts/run_inprocess.py \
  --server-args configs/dsv4pro-p-gb300/server_args.json \
  --hisim-config configs/dsv4pro-p-gb300/hisim.ml.json \
  --workload trace \
  --dataset /host/bl_data_trace/multi_node_trace_combine_dpskv4pro/<trace>.jsonl \
  --output-dir /tmp/hisim/dsv4pro-p
```

## 8. 验证

```bash
python3 scripts/validate_result.py /tmp/hisim/qwen3-8b-inprocess
```

0714 与 v0.5.16 对比：

```bash
python3 scripts/compare_results.py \
  --old <0714-result-dir> \
  --new <v0516-result-dir>
```

完整静态测试：

```bash
bash scripts/test_bundle.sh
```

通过条件：

- `completed == num_requests`
- TTFT、E2E、duration 非负
- `prefix_cache_reused_ratio` 与 device/host/storage 分项一致
- replay 回归的 prefix hit ratio 不变
- batch composition 首个差异可定位
- `kv_cache_kb_per_token` 与对应部署基线一致

## 9. 输出

| 文件 | 内容 |
|---|---|
| `result.metrics.json` | 汇总吞吐、命中率、TTFT/TPOT/ITL/E2E |
| `result.request.jsonl` | 每请求到达、排队、命中和 token latency |
| `result.iteration.jsonl` | 每 step batch composition、forward/load/backup 耗时 |
| `server_args.json` | 本次实际服务配置 |
| `hisim_config.json` | 本次实际硬件和 predictor 配置 |

缓存命中分项互斥：

```text
prefix = device + host + storage
```

每次运行使用新输出目录；不要覆盖历史结果。
