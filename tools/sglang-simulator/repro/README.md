# HiSim 最简复现

适用版本：官方镜像 `lmsysorg/sglang:v0.5.16`；分支
`codex/hisim-v0.5.16-adaptation-0729`。

## 1. 从官方镜像启动

在 37 服务器执行：

```bash
docker run -dit \
  --name hisim-v0516-official-0729 \
  --privileged \
  --gpus all \
  --network host \
  --ipc host \
  -v /data2/maruiyan.mry:/host \
  lmsysorg/sglang:v0.5.16 bash
```

`--privileged` 用于在容器内挂载 NFS；纯 CPU 且模型已由宿主机映射时可以去掉。
不要在此容器内重新安装仓库根目录的 SGLang，镜像内
`/sgl-workspace/sglang` 已是准确的 `0.5.16`。

### 1.1 挂载模型

```bash
docker exec -it hisim-v0516-official-0729 bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  nfs-common curl git
mkdir -p /nfs /disk2_20
mount -t nfs4 33.254.37.150:/nfs/lvmpv/models /nfs
mount -t nfs4 33.254.38.20:/apsarapangu/disk2 /disk2_20
```

容器重建后重新执行挂载命令。

### 1.2 安装依赖和 simulator

```bash
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
pip3 install -e /host/aiconfigurator --no-deps
pip3 install -e \
  /host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator
```

AIC 当前元数据要求 NumPy 1.26，但已验证其本 case 在镜像自带的 NumPy 2.3.5
可正常预测。必须给 AIC 加 `--no-deps`，否则 pip 会降级 NumPy，与镜像内
CUDA 13 CuPy 冲突。

Simulator 直接依赖 `sglang==0.5.16`、NumPy、scikit-learn 和 joblib。普通
editable 安装会复用官方镜像中已满足的版本，并自动补齐缺失依赖；不要增加
`--upgrade`。AIC 是可选依赖，未使用的 `xgboost` 不再是必装项。

### 1.3 环境检查和单元测试

```bash
cd /host/hisim-sglang/worktrees/sglang-v0.5.16-adaptation/tools/sglang-simulator/repro
python3 scripts/check_environment.py
bash scripts/test_bundle.sh
```

官方镜像预期：

- `sglang==0.5.16`，从 `/sgl-workspace/sglang/python` 导入。
- `sglang_simulator` 从当前 worktree 导入。
- `/nfs` 已挂载。
- 四个模型目录、AIC database、ML 模型均显示 `OK`。
- 当前测试结果为 repro `10 passed`，simulator `18 passed`。

### 1.4 一键验收

```bash
HISIM_PORT=31029 \
HISIM_GPU_ID=7 \
HISIM_ACCEPTANCE_DIR=/data2/maruiyan.mry/hisim-sglang/validation/v0516/official-image-0729-final-v6 \
  bash scripts/acceptance.sh
```

`HISIM_PORT` 可避开另一个并行验收中的 30000 端口；默认值仍为 30000。
每次运行必须使用新的输出目录。

覆盖静态测试、两种启动方式、OFFLINE/BLOCKING、三种 workload、三种 predictor、
四个模型配置、服务终端打流、CPU allocator 和 GPU Triton allocator。全部通过时
最后一行是 `PASS all acceptance checks`。

验收还会用同一配置、trace 和 replay table 比较服务化与 Python 同进程模式：
request、batch composition、forward、缓存命中和 replay coverage 必须完全一致；
时间差必须不超过两次运行实际记录的逐步 CPU overhead 差值。

### 1.5 CPU/GPU 环境功能验证

```bash
bash scripts/validate_cpu_gpu.sh
```

指定物理 GPU：

```bash
HISIM_GPU_ID=1 bash scripts/validate_cpu_gpu.sh
```

该命令分别在 `CUDA_VISIBLE_DEVICES=""` 和 GPU 可见环境运行：

- CPU spawned runner：cold -> L1 -> L2 -> L3，校验 TTFT/E2E。
- CPU replay trace：page size 256，使用 Python paged allocator。
- GPU replay trace：page size 256，实际使用 Triton paged allocator。

两种环境均为 `load_format=dummy`，不加载权重；GPU 模式会分配精简的
dummy KV tensor 并执行 page 分配 Triton kernel，但不执行真实模型 forward。
日常纯 CPU 运行无需增加参数；需要 GPU allocator 时增加：

```text
--device cuda --page-size 256
```

## 2. 仿真模式

```bash
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE   # 不 sleep，按逻辑时间运行
# 或
export SGLANG_SIMULATOR_OUTPUT_MODE=BLOCKING  # 按预测 step 时间 sleep
```

准确度回归用 `OFFLINE`。服务交互验收用 `BLOCKING`；此时 benchmark 客户端
也必须设置 `SGLANG_SIMULATOR_OUTPUT_MODE=BLOCKING`，客户端才会按
`request-rate` 实际等待。

`BLOCKING` 会同时阻塞 forward 和可见的 L2→L1 load：关闭 overlap schedule
时 sleep 全部有效 load 时延；开启 overlap schedule 时仅 sleep
`max(load - 上一轮 inference, 0)`。这段实际 sleep 不再计入 CPU overhead，
避免重复计时。`OFFLINE` 不 sleep，但逻辑时钟仍包含相同的可见 L2 load。

`total_l2_blocking_wall_s` 是本次 BLOCKING 实际 sleep 的 L2 墙钟总和；
OFFLINE 中固定为 `0`。

## 3. 启动方式一：服务化

终端 A：

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/hisim/qwen3-8b-service

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /nfs/Qwen/Qwen3-8B \
  --sim-config-path configs/qwen3-8b-h20/hisim.aic.json \
  --port 30000
```

服务地址：`http://127.0.0.1:30000`。

模块入口默认使用 `load_format=dummy`，不会加载模型权重。设置
`SGLANG_USE_CPU_ENGINE=1` 后会自动补齐 v0.5.16 的 CPU 安全参数；显式传入的
SGLang CLI 参数优先。`scripts/start_service.py` 仅供一键验收脚本从
`server_args.json` 展开参数，不是对外服务入口。

### 3.1 终端打 benchmark 流量

客户端进程固定使用 CPU；这只避免导入无关 GPU kernel，不改变服务端 allocator：

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/hisim/qwen3-8b-service
```

Random：

Random/ShareGPT 的服务化 request-rate 示例应把终端 A 和客户端环境都改为
`BLOCKING`；官方客户端会按 request rate 实际发流。OFFLINE request-rate
使用第 4 节的同进程入口。

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name random \
  --dataset-path workloads/sharegpt.example.json \
  --request-rate 4 \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 2 \
  --warmup-requests 0 \
  --profile
```

SGLang v0.5.16 的 random sampler 需要 `--dataset-path` 作为本地 prompt 语料。

ShareGPT：

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name sharegpt \
  --dataset-path workloads/sharegpt.example.json \
  --request-rate 4 \
  --num-prompts 2 \
  --warmup-requests 0 \
  --profile
```

HiSim trace：

下面是 OFFLINE 服务化回放，保持客户端环境为 `OFFLINE`。

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name autobench \
  --dataset-path workloads/trace.example.jsonl \
  --num-prompts 3 \
  --warmup-requests 0 \
  --profile
```

HiSim trace 直接使用官方 Autobench 格式，`timestamp` 固定为毫秒。
OFFLINE 模式不增加 `--use-trace-timestamps`，客户端立即提交所有请求，
服务端从 `simulation.created_time_ms` 恢复逻辑到达时间。BLOCKING 模式增加
`--use-trace-timestamps`，由官方 `get_request()` 按 timestamp 实际等待。

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
{
  "prompt": [1, 2, 3],
  "prompt_len": 3,
  "output_len": 1,
  "timestamp": 200,
  "extra_request_body": {
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 1,
      "ignore_eos": true,
      "custom_params": {
        "simulation": {
          "created_time_ms": 200,
          "total_request": 100
        }
      }
    }
  }
}
```

必填字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamp` | number | 相对到达时间，固定为毫秒 |
| `prompt` | `list[int]` | 输入 token IDs |
| `prompt_len` | int | 必须等于 `len(prompt)` |
| `output_len` | int | 输出 token 数；P 节点仿真通常设为 `1` |
| `extra_request_body` | object | 将相同的毫秒时间和总请求数传给 HiSim scheduler |

同进程入口默认用 `timestamp / 1000` 恢复为仿真秒。服务化 OFFLINE 入口使用
`simulation.created_time_ms / 1000`，两条路径语义一致。

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

`ml` 是统一的机器学习 predictor 接口，不限定具体回归算法。HGBMono、GBR
或其他模型均应离线训练并导出包含 `model`、`features` 的 joblib bundle；
`model` 必须支持 sklearn 风格的 `predict([[18 features]])`，输出单位为秒。
`features` 必须与上面的18项名称和顺序完全一致，否则加载时直接报错。运行时
不接受缺少 `features` 元数据的裸模型，也不再读取训练 JSONL 或现场训练
模型。

Replay：

```json
{
  "name": "replay",
  "database_path": "workloads/replay_table.example.json",
  "miss_strategy": "zero",
  "miss_fallback_seconds": 0.0
}
```

基线采集时：

```bash
export SGL_HOOK_FETCH_BATCH_INFO=1
```

Replay table 只使用同步测量的 `iter_latency` 替换 forward；CPU
preprocess/postprocess、HiCache IO 和其余仿真建模保持不变。不要使用
`replay_tables_pre_post`，也不要将 `preprocess_latency`、
`postprocess_latency` 折叠进 replay 值。

`zero` miss 会低估耗时；`knn` 会引入插值误差。回归时必须报告 exact/miss
比例。`result.metrics.json` 会输出 `replay_exact_match_steps`、
`replay_miss_steps`、两种 fallback step 数和 `replay_fallback_rate`。

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
python3 scripts/validate_result.py \
  /tmp/hisim/qwen3-8b-inprocess --expected-requests 3
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

`result.metrics.json` 中的 `total_l2_load_s` 是逻辑 L2 load 总时延，
`total_l2_blocking_wall_s` 是 BLOCKING 模式实际阻塞的墙钟时间。

缓存命中分项互斥：

```text
prefix = device + host + storage
```

每次运行使用新输出目录；不要覆盖历史结果。
