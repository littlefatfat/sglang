# sglang_simulator
## Background
As large language models (LLMs) are rapidly deployed at scale for inference services, inference performance directly impacts user experience, service cost, and resource efficiency. Key metrics such as Time to First Token (TTFT), Time Per Output Token (TPOT), and system throughput are highly dependent on the complex interplay among model architecture, hardware platforms (e.g., A100/H100), inference engines (e.g., SGLang, vLLM, TensorRT-LLM), and runtime configurations (e.g., quantization, batching, and parallelism strategies).

Traditional end-to-end stress testing on real GPU clusters is expensive and time-consuming, making it impractical to efficiently explore the vast space of configuration combinations. To address this, we propose **sglang_simulator**, a high-fidelity CPU-based simulation system. sglang_simulator enables fast, low-cost, and high-fidelity prediction of key performance metrics across different models, target hardware, inference engines, and configurations by replaying real-world inference workload traces collected from production or representative scenarios, thereby accelerating the design and optimization of inference systems.

---
## Introduction
sglang_simulator is a simulation tool that provides a command-line interface compatible with SGLang. It launches a mock inference service that accepts user requests—either via real-world trace replay or synthetic load generation—using standard benchmarking scripts. sglang_simulator outputs performance metrics identical to those produced by `sglang bench_serving`. See **Quick Start** for usage examples.

---
## Installation
```bash
cd tools/sglang_simulator
pip install .
```

---
## Quick Start
### Step 1: Mock Simulation
This example runs inference simulation using real-world trace data. You may also use synthetic random workloads (see below).

#### Launch the Simulation Server
Set absolute paths once. The commands then work from any current directory:
```bash
export HISIM_ROOT=/absolute/path/to/sglang/tools/sglang-simulator
export MODEL_PATH=/absolute/path/to/Qwen3-32B-FP8
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/sglang_simulator/qwen3-32b

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --sim-config-path "${HISIM_ROOT}/test/assets/config.json"
```

> **Notes**:
> - Use `--sim-config-path` to specify the simulation configuration file, which is equivalent to the system environment variable `SGLANG_SIMULATOR_CONFIG_PATH`.
> - The simulator entry point defaults to `load_format=dummy`; it does not load model weights. The model path must still contain model config and tokenizer files.
> - With `SGLANG_USE_CPU_ENGINE=1`, the entry point selects the v0.5.16 CPU-safe device, attention/sampling backends and disables CUDA graph. Explicit SGLang CLI options take precedence.
> - The provided [config file](test/assets/config.json) is for testing. Adjust hardware bandwidth and other parameters to match your actual deployment scenario for higher fidelity.

#### Run the Simulation Benchmark
- **Autobench trace replay**:
  ```bash
  python3 -m sglang.benchmark.serving \
      --backend sglang \
      --model "${MODEL_PATH}" \
      --dataset-name autobench \
      --dataset-path "${HISIM_ROOT}/repro/workloads/trace.autobench.example.jsonl" \
      --num-prompts 3 \
      --warmup-requests 0 \
      --profile
  ```

- **Synthetic random workload**:
  Start the simulator in `BLOCKING` mode for service-side request-rate traffic.
  ```bash
  export SGLANG_SIMULATOR_OUTPUT_MODE=BLOCKING
  python3 -m sglang.benchmark.serving \
      --warmup-requests 0 \
      --model "${MODEL_PATH}" \
      --dataset-name random \
      --request-rate 4 \
      --random-input-len 1024 \
      --random-output-len 1024 \
      --num-prompts 10 \
      --profile
  ```

You have now completed an inference simulation using framework interception.

#### Example Output
```bash
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 not set
Successful requests:                     10
Benchmark duration (s):                  1.15
Total input tokens:                      1997
Total input text tokens:                 -1
Total generated tokens:                  2798
Total generated tokens (retokenized):    -1
Request throughput (req/s):              1.58
Input token throughput (tok/s):          316.20
Output token throughput (tok/s):         443.04
Peak output token throughput (tok/s):    -1.00
Peak concurrent requests:                -1
Total token throughput (tok/s):          759.24
Concurrency:                             -1.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   3431.84
Median E2E Latency (ms):                 3809.48
P90 E2E Latency (ms):                    5309.93
P99 E2E Latency (ms):                    6214.97
---------------Time to First Token----------------
Mean TTFT (ms):                          128.59
Median TTFT (ms):                        128.59
P99 TTFT (ms):                           128.59
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          12.14
Median TPOT (ms):                        12.44
P99 TPOT (ms):                           12.89
---------------Inter-Token Latency----------------
Mean ITL (ms):                           11.85
Median ITL (ms):                         11.95
P95 ITL (ms):                            12.05
P99 ITL (ms):                            12.07
Max ITL (ms):                            12.08
==================================================
```

> Fields marked with `-1` indicate unsupported metrics in simulation mode.

---
## Usage
### Mock Simulation (Inference Simulation)
sglang_simulator uses **dynamic interception** to hijack the execution flow of the inference framework, bypassing actual LLM computation.

- For supported launch options, run:
  ```bash
  python -m sglang_simulator.simulation.sglang.launch_server --help
  ```

---
### Configuration File Format (`SGLANG_SIMULATOR_CONFIG_PATH`)
The config file is a JSON file with three main sections:

- **`platform`**: Hardware and bandwidth settings
  - `accelerator`: GPU model (e.g., `"h100_sxm"`)
  - `disk_*_bandwidth_gb`: L3 (disk) read/write bandwidth (GB/s)
  - `memory_*_bandwidth_gb`: L2 (memory) read/write bandwidth (GB/s)

- **`predictor`**: Time prediction module
  - `name`: predictor type (`"aiconfigurator"`)
  - See the **TimePredictor** section below for details

- **`scheduler`**: Parallelism and backend metadata
  > ⚠️ **Note**: Multi-GPU parallelism (TP/EP) should not be configured at the framework level during server launch. Instead, specify `tp_size` and `ep_size` here; the predictor will simulate parallel execution overhead accordingly.
  - When `predictor.name = "aiconfigurator"`, `backend_version` must be provided.

**Example Config**:
```json
{
    "platform": {
        "accelerator": {
            "name": "a100_sxm",
            "hbm_capacity_gb": 80
        },
        "disk_read_bandwidth_gb": 8,
        "disk_write_bandwidth_gb": 8,
        "memory_read_bandwidth_gb": 64,
        "memory_write_bandwidth_gb": 64,
        "num_device_per_node": 8
    },
    "predictor": {
        "name": "aiconfigurator"
    },
    "scheduler": {
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "backend_version": "0.5.9"
    }
}
```

---
## TimePredictor
### AIConfigurator
- Project: <https://github.com/ai-dynamo/aiconfigurator>
- Parameters:
  - `database_path`: (optional) path to custom operator profiling data
  - `platform.accelerator.name`: device/system name; it should be one of those defined internally in AIConfigurator
  - `prefill_scale_factor` / `decode_scale_factor`: optional calibration factors to adjust predicted latency

**Example**:
```json
{
  "name": "aiconfigurator",
  "database_path": "/absolute/path/to/aiconfigurator/data",
  "prefill_scale_factor": 1.02040816,
  "decode_scale_factor": 1.01010101
}
```
