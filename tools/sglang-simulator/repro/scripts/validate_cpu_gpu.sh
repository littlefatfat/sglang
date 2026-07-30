#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMULATOR_ROOT="$(cd "${ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${SIMULATOR_ROOT}/../.." && pwd)"
GPU_ID="${SGLANG_SIMULATOR_GPU_ID:-0}"
RESULT_ROOT="$(mktemp -d /tmp/sglang-simulator-cpu-gpu.XXXXXX)"
trap 'rm -rf -- "${RESULT_ROOT}"' EXIT

cd "${ROOT}"

validate_torch_cpu() {
  CUDA_VISIBLE_DEVICES="" python3 -c \
    'import torch; assert not torch.cuda.is_available(); print("CPU environment: CUDA unavailable")'
}

validate_torch_gpu() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -c \
    'import torch; assert torch.cuda.is_available(); print(f"GPU environment: {torch.cuda.get_device_name(0)}")'
}

run_runner_test() {
  local visible_devices="$1"
  (
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${visible_devices}" pytest -q \
      tools/sglang-simulator/test/test_simulation_sglang_runner.py
  )
}

run_trace_replay() {
  local mode="$1"
  local visible_devices="$2"
  local output="${RESULT_ROOT}/${mode}"

  CUDA_VISIBLE_DEVICES="${visible_devices}" python3 scripts/run_inprocess.py \
    --server-args configs/qwen3-8b-h20/server_args.json \
    --sim-config configs/qwen3-8b-h20/simulator.replay.json \
    --device "${mode%%-*}" \
    --page-size 256 \
    --workload trace \
    --dataset workloads/trace.autobench.replay.example.jsonl \
    --output-dir "${output}"
  python3 scripts/validate_result.py "${output}"
}

echo "[1/4] CPU-only runner"
validate_torch_cpu
run_runner_test ""

echo "[2/4] CPU-only trace replay"
run_trace_replay cpu-only ""

echo "[3/4] GPU-visible runner (physical GPU ${GPU_ID})"
validate_torch_gpu
run_runner_test "${GPU_ID}"

echo "[4/4] GPU-visible trace replay (physical GPU ${GPU_ID})"
run_trace_replay cuda "${GPU_ID}"

echo "PASS: CPU-only and GPU-visible SGLang Simulator validation"
