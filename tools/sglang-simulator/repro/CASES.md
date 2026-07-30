# HiSim 可逐条复现用例

环境：容器 `hisim-v0516-official-0729`，镜像
`lmsysorg/sglang:v0.5.16`。

先在容器内执行一次：

```bash
export HISIM_REPRO=/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}
```

每次服务化测试使用两个终端。终端 A 启动服务，终端 B 打流并验证；完成后在
终端 A 按 `Ctrl-C`。不同用例使用不同 `OUT` 和 `PORT`。

## 1. CPU 服务 + Random

终端 A：

```bash
export HISIM_REPRO=/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=BLOCKING
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/hisim-case-cpu-random

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /nfs/Qwen/Qwen3-8B \
  --sim-config-path "${HISIM_REPRO}/configs/qwen3-8b-h20/hisim.aic.json" \
  --port 31501
```

终端 B：

```bash
export HISIM_REPRO=/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:31501 \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name random \
  --dataset-path "${HISIM_REPRO}/workloads/sharegpt.example.json" \
  --random-input-len 1024 \
  --random-output-len 128 \
  --random-range-ratio 1 \
  --request-rate 4 \
  --num-prompts 2 \
  --warmup-requests 0 \
  --profile \
  --disable-tqdm

python3 "${HISIM_REPRO}/scripts/validate_result.py" \
  /tmp/hisim-case-cpu-random --expected-requests 2
```

`--random-range-ratio 1` 保证每个请求恰好是 1024 输入、128 输出；不设置时，
官方 RandomDataset 会在 `1..指定长度` 内随机采样。

预期最后一行：

```text
PASS /tmp/hisim-case-cpu-random/metrics.json
```

## 2. CPU 服务 + ShareGPT

终端 A 使用第 1 节启动命令，只修改：

```bash
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/hisim-case-cpu-sharegpt
```

并把端口改为 `31502`。终端 B：

```bash
export HISIM_REPRO=/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:31502 \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name sharegpt \
  --dataset-path "${HISIM_REPRO}/workloads/sharegpt.example.json" \
  --request-rate 4 \
  --num-prompts 2 \
  --warmup-requests 0 \
  --profile \
  --disable-tqdm

python3 "${HISIM_REPRO}/scripts/validate_result.py" \
  /tmp/hisim-case-cpu-sharegpt --expected-requests 2
```

区别：Random 只从 ShareGPT 取 token 语料，再重复或截断到指定长度；ShareGPT
保留数据集本身的 prompt 和 output 长度。

## 3. CPU 服务 + timestamp Autobench

### 3.1 OFFLINE

终端 A 使用第 1 节启动命令，修改：

```bash
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/hisim-case-cpu-autobench-offline
```

并把端口改为 `31503`。终端 B：

```bash
export HISIM_REPRO=/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:31503 \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name autobench \
  --dataset-path "${HISIM_REPRO}/workloads/trace.autobench.example.jsonl" \
  --num-prompts 3 \
  --warmup-requests 0 \
  --profile \
  --disable-tqdm

python3 "${HISIM_REPRO}/scripts/validate_result.py" \
  /tmp/hisim-case-cpu-autobench-offline --expected-requests 3
```

OFFLINE 不增加 `--use-trace-timestamps`：客户端立即提交请求，服务端从
`extra_request_body...simulation.created_time_ms` 恢复逻辑到达时间。

### 3.2 BLOCKING

终端 A 使用第 1 节启动命令，修改输出目录为
`/tmp/hisim-case-cpu-autobench-blocking`，端口改为 `31504`。终端 B 使用
3.1 的命令，只修改服务地址：

```text
--base-url http://127.0.0.1:31504
```

并在 benchmark 命令中增加：

```text
--use-trace-timestamps
```

BLOCKING 中客户端按 `timestamp` 等待，服务端同时 sleep forward 和可见的
L2→L1 load；OFFLINE 只推进相同的逻辑时钟。

## 4. GPU 服务 + Triton page allocator

终端 A：

```bash
export HISIM_REPRO=/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}
unset SGLANG_USE_CPU_ENGINE
export CUDA_VISIBLE_DEVICES=7
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/hisim-case-gpu-autobench

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /nfs/Qwen/Qwen3-8B \
  --sim-config-path "${HISIM_REPRO}/configs/qwen3-8b-h20/hisim.replay.json" \
  --device cuda \
  --page-size 256 \
  --max-total-tokens 8192 \
  --disable-cuda-graph \
  --attention-backend torch_native \
  --sampling-backend pytorch \
  --port 31505
```

终端 B：

```bash
export HISIM_REPRO=/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:31505 \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name autobench \
  --dataset-path "${HISIM_REPRO}/workloads/trace.autobench.example.jsonl" \
  --num-prompts 3 \
  --warmup-requests 0 \
  --profile \
  --disable-tqdm

python3 "${HISIM_REPRO}/scripts/validate_result.py" \
  /tmp/hisim-case-gpu-autobench --expected-requests 3
```

该用例加载 dummy weights，不执行真实 forward；page 分配实际走 GPU Triton
kernel。实测结果为 `num_requests=3, completed=3`。

## 5. 一个 Python 脚本完成完整生命周期

以下命令启动 CPU 服务、用官方 `sglang.benchmark.serving` 发送精确长度 Random
流量、停止服务，并保留 metrics：

```bash
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

python3 /host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro/scripts/run_service_random.py \
  --output-dir /tmp/hisim-python-lifecycle \
  --mode BLOCKING \
  --port 31506 \
  --num-prompts 2 \
  --input-len 1024 \
  --output-len 128 \
  --request-rate 4
```

预期最后一行：

```text
PASS metrics=/tmp/hisim-python-lifecycle/metrics.json server_log=/tmp/hisim-python-lifecycle.server.log
```

脚本总是进入 `finally` 停止服务。输出目录已存在时会拒绝运行，避免把两次结果
混在一起。

## 6. DSv4Pro 同 trace 对比 AIC、ML、Replay

先从历史 trace 生成 5-request Autobench 副本；不会修改源文件：

```bash
export HISIM_REPRO=/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro

python3 "${HISIM_REPRO}/scripts/prepare_autobench_trace.py" \
  --input /host/bl_data_trace/multi_node_trace_combine_dpskv4pro/hisim-num-node-4-dpskv4pro-blksz-256-bucket-0-32-cnt-7819-time-60min.jsonl \
  --output /tmp/dsv4pro-5.autobench.jsonl \
  --indices 4,57,58,77,119 \
  --force
```

该副本有 5 个请求、13056 个输入 token、55.165 秒的历史到达时间跨度，并保留
1536-token 的共享前缀。

依次选择一个 predictor：

```bash
# AIC
export HISIM_CONFIG="${HISIM_REPRO}/configs/dsv4pro-p-gb300/hisim.aic.json"
export HISIM_OUT=/tmp/hisim-dsv4-aic
export HISIM_PORT=31511

# ML（另一次独立运行）
export HISIM_CONFIG="${HISIM_REPRO}/configs/dsv4pro-p-gb300/hisim.ml.json"
export HISIM_OUT=/tmp/hisim-dsv4-ml
export HISIM_PORT=31512

# Replay（另一次独立运行）
export HISIM_CONFIG="${HISIM_REPRO}/configs/dsv4pro-p-gb300/hisim.replay.json"
export HISIM_OUT=/tmp/hisim-dsv4-replay
export HISIM_PORT=31513
```

终端 A，在每组 `export` 后执行：

```bash
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR="${HISIM_OUT}"

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /nfs/deepseek-ai/DeepSeek-V4-Pro/ \
  --sim-config-path "${HISIM_CONFIG}" \
  --trust-remote-code \
  --page-size 256 \
  --max-total-tokens 65536 \
  --max-running-requests 64 \
  --max-prefill-tokens 32768 \
  --chunked-prefill-size 8192 \
  --disable-overlap-schedule \
  --port "${HISIM_PORT}"
```

终端 B，使用与终端 A 相同的 `HISIM_OUT`、`HISIM_PORT`：

```bash
export HISIM_REPRO=/host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url "http://127.0.0.1:${HISIM_PORT}" \
  --model /nfs/deepseek-ai/DeepSeek-V4-Pro/ \
  --dataset-name autobench \
  --dataset-path /tmp/dsv4pro-5.autobench.jsonl \
  --num-prompts 5 \
  --warmup-requests 0 \
  --profile \
  --disable-tqdm

python3 "${HISIM_REPRO}/scripts/validate_result.py" \
  "${HISIM_OUT}" --expected-requests 5
```

终端 B 不读取 simulator 的设备、时钟模式或输出目录变量；这些只由终端 A
的服务进程读取。客户端发送节奏由 `--request-rate` 或
`--use-trace-timestamps` 控制。

本次实测：

| predictor | completed | prefix reused | mean TTFT |
|---|---:|---:|---:|
| AIC SILICON | 5 | 35.294% | 225.025 ms |
| ML HGBMono 18-feature | 5 | 35.294% | 277.150 ms |
| Replay | 5 | 35.294% | 319.652 ms |

Replay 共 5 个 step：4 个精确命中，1 个 3-NN fallback。CPU overhead 会使绝对
TTFT 有小幅波动；请求数、batch/cache 语义和前缀复用率应保持一致。

## 7. 输出文件

每个 `SGLANG_SIMULATOR_OUTPUT_DIR` 下都有：

```text
metrics.json
request.jsonl
iteration.jsonl
```

快速查看：

```bash
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); print({k:x.get(k) for k in ("num_requests","completed","duration","mean_ttft_ms","mean_e2e_latency_ms","prefix_cache_reused_ratio")})' \
  /tmp/hisim-case-cpu-random/metrics.json
```
