# SGLang Simulator

SGLang Simulator reuses SGLang's scheduler and cache implementation while replacing
model forward execution with a latency predictor. It runs on CPU by default in the
reproducibility tools and supports trace replay, synthetic workloads, hierarchical
cache simulation, and serving-compatible metrics.

## Requirements

- SGLang `0.5.16`.
- A local model directory containing model configuration and tokenizer files.
  Simulation uses dummy weights.
- Predictor data for AIConfigurator, ML, or replay mode.

The official SGLang image is the recommended validation environment:

```bash
docker run --gpus all --ipc=host --rm -it \
  -v /absolute/path/to/sglang:/sgl-workspace/sglang \
  lmsysorg/sglang:v0.5.16
```

## Installation

From the SGLang repository:

```bash
cd tools/sglang-simulator
pip install -e .
```

AIConfigurator is optional. Install the revision required by the selected performance
database separately with `--no-deps`; this avoids replacing NumPy and CUDA-related
packages supplied by the SGLang image.

## Quick start

The shortest deterministic path is an in-process replay run:

```bash
export SIMULATOR_ROOT=/absolute/path/to/sglang/tools/sglang-simulator
export PYTHONPATH=/absolute/path/to/sglang/python:${PYTHONPATH:-}

python3 "${SIMULATOR_ROOT}/repro/scripts/run_inprocess.py" \
  --server-args "${SIMULATOR_ROOT}/repro/configs/qwen3-8b-h20/server_args.json" \
  --sim-config "${SIMULATOR_ROOT}/repro/configs/qwen3-8b-h20/simulator.replay.json" \
  --mode OFFLINE \
  --workload trace \
  --dataset "${SIMULATOR_ROOT}/repro/workloads/trace.autobench.replay.example.jsonl" \
  --num-prompts 3 \
  --output-dir /tmp/sglang-simulator-replay
```

Update `model_path` in the server-args file when the model is stored elsewhere.
The run produces:

- `result.metrics.json`
- `result.request.jsonl`
- `result.iteration.jsonl`
- copies of the resolved server and simulator configuration

## Serving mode

Set the mode and output directory before starting the server:

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /absolute/path/to/model \
  --sim-config-path /absolute/path/to/simulator.json \
  --port 30000
```

Send traffic with the simulator-aware benchmark adapter:

```bash
python3 -m sglang_simulator.simulation.bench_serving \
  --simulator-mode offline \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /absolute/path/to/model \
  --dataset-name autobench \
  --dataset-path /absolute/path/to/trace.jsonl \
  --use-trace-timestamps \
  --num-prompts 100 \
  --warmup-requests 0 \
  --profile
```

Server options are normal SGLang command-line arguments. Python lifecycle helpers
construct the same `ServerArgs` object from JSON for reproducible cases.

## Simulation modes

| Mode | Behavior |
|---|---|
| `OFFLINE` | Advances the simulator's logical clock without sleeping. |
| `BLOCKING` | Sleeps for predicted forward and visible L2-to-L1 load latency. |

Use server-side simulator metrics for comparisons. Client wall-clock duration is not
the simulated timeline in `OFFLINE` mode.

## Configuration

A simulator JSON file has three sections:

```json
{
  "platform": {
    "accelerator": {"name": "h20_sxm"},
    "disk_read_bandwidth_gb": 8,
    "disk_write_bandwidth_gb": 8,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64,
    "num_device_per_node": 1
  },
  "predictor": {
    "name": "replay",
    "database_path": "/absolute/path/to/replay_table.json"
  },
  "scheduler": {
    "tp_size": 1,
    "ep_size": 1,
    "dp_size": 1,
    "data_type": "BF16",
    "kv_cache_data_type": "BF16",
    "backend_name": "sglang",
    "backend_version": "0.5.16"
  }
}
```

- `platform` describes the simulated accelerator and storage bandwidth.
- `predictor` selects forward-latency prediction.
- `scheduler` describes target parallelism and backend metadata. It does not launch
  physical tensor-parallel workers.

Supported predictors:

| Predictor | Purpose |
|---|---|
| `aiconfigurator` | Operator and module performance-database estimation. |
| `ml` | A trained sklearn-compatible 18-feature latency model. |
| `replay` | Exact or nearest-neighbor batch-composition replay. |

Environment variables in configured paths use `${NAME}` syntax and are resolved at
load time.

## HiCache smoke test

The smoke test forces HBM eviction, writes a prefix to host memory, and reloads it:

```bash
python3 repro/scripts/run_hicache_smoke.py \
  --output-dir /tmp/sglang-simulator-hicache
```

It passes only when all three requests complete and the final request restores at
least 768 prefix tokens from the host cache.

## Validation

From `tools/sglang-simulator` in the official SGLang v0.5.16 image:

```bash
bash -n repro/scripts/*.sh
python3 -m compileall -q src test repro/scripts repro/tests
python3 -m pytest -q test/unit repro/tests/test_bundle.py repro/tests/test_cases.py
```

Run the repository's pinned formatting, import-order, spelling, and file checks:

```bash
git ls-files -z tools/sglang-simulator | \
  xargs -0 env SKIP=no-commit-to-branch pre-commit run --files
```

The complete CPU/GPU, predictor, workload, serving, and HiCache matrix is:

```bash
export SGLANG_SIMULATOR_ML_MODEL_PATH=/absolute/path/to/latency_model.pkl
export SGLANG_SIMULATOR_ACCEPTANCE_DIR=/tmp/sglang-simulator-acceptance
export SGLANG_SIMULATOR_PORT=31029
export SGLANG_SIMULATOR_GPU_ID=0
bash repro/scripts/acceptance.sh
```

Use a fresh acceptance directory. A successful run ends with
`PASS all acceptance checks`.

See [repro/README.md](repro/README.md) for the artifact bundle and
[repro/CASES.md](repro/CASES.md) for copyable validation cases.
