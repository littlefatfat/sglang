#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${HISIM_ACCEPTANCE_DIR:-/tmp/hisim-acceptance.$(date +%Y%m%d-%H%M%S)}"
GPU_ID="${HISIM_GPU_ID:-0}"
PORT="${HISIM_PORT:-30000}"
SERVICE_PID=""

mkdir -p "${RESULT_ROOT}"
cd "${ROOT}"

cleanup_service() {
  if [[ -n "${SERVICE_PID}" ]] && kill -0 "${SERVICE_PID}" 2>/dev/null; then
    kill -TERM "${SERVICE_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${SERVICE_PID}" 2>/dev/null || break
      sleep 0.5
    done
    wait "${SERVICE_PID}" 2>/dev/null || true
  fi
  SERVICE_PID=""
}
trap cleanup_service EXIT

run_logged() {
  local name="$1"
  shift
  echo "RUN  ${name}"
  if "$@" >"${RESULT_ROOT}/${name}.log" 2>&1; then
    echo "PASS ${name}"
  else
    echo "FAIL ${name}; tail follows"
    tail -80 "${RESULT_ROOT}/${name}.log" || true
    exit 1
  fi
}

validate() {
  local name="$1"
  local count="$2"
  run_logged "validate-${name}" \
    python3 scripts/validate_result.py "${RESULT_ROOT}/${name}" \
    --expected-requests "${count}"
}

start_service() {
  local name="$1"
  local mode="$2"
  local server_args="$3"
  local hisim_config="$4"

  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "FAIL port ${PORT} is already occupied"
    exit 1
  fi

  echo "RUN  service-${name}"
  python3 scripts/start_service.py \
    --server-args "${server_args}" \
    --hisim-config "${hisim_config}" \
    --mode "${mode}" \
    --output-dir "${RESULT_ROOT}/${name}" \
    --port "${PORT}" \
    >"${RESULT_ROOT}/service-${name}.log" 2>&1 &
  SERVICE_PID=$!

  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 \
      && grep -q "The server is fired up and ready to roll" \
        "${RESULT_ROOT}/service-${name}.log"; then
      echo "PASS service-${name}"
      return
    fi
    if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
      echo "FAIL service-${name}; tail follows"
      tail -100 "${RESULT_ROOT}/service-${name}.log" || true
      exit 1
    fi
    sleep 1
  done

  echo "FAIL service-${name}: readiness timeout"
  tail -100 "${RESULT_ROOT}/service-${name}.log" || true
  exit 1
}

stop_service() {
  cleanup_service
  for _ in $(seq 1 30); do
    if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      return
    fi
    sleep 0.5
  done
  echo "FAIL service did not release port ${PORT}"
  exit 1
}

echo "HiSim v0.5.16 acceptance"
echo "results=${RESULT_ROOT}"

run_logged static bash scripts/test_bundle.sh

run_logged inprocess-replay \
  python3 scripts/run_inprocess.py \
  --server-args configs/qwen3-8b-h20/server_args.json \
  --hisim-config configs/qwen3-8b-h20/hisim.replay.json \
  --mode OFFLINE \
  --workload trace \
  --dataset workloads/trace.autobench.example.jsonl \
  --output-dir "${RESULT_ROOT}/inprocess-replay"
validate inprocess-replay 3

run_logged inprocess-blocking-aic \
  python3 scripts/run_inprocess.py \
  --server-args configs/qwen3-8b-h20/server_args.json \
  --hisim-config configs/qwen3-8b-h20/hisim.aic.json \
  --mode BLOCKING \
  --workload random \
  --num-prompts 2 \
  --input-len 8 \
  --output-len 1 \
  --request-rate 4 \
  --output-dir "${RESULT_ROOT}/inprocess-blocking-aic"
validate inprocess-blocking-aic 2

run_logged inprocess-sharegpt \
  python3 scripts/run_inprocess.py \
  --server-args configs/qwen3-8b-h20/server_args.json \
  --hisim-config configs/qwen3-8b-h20/hisim.aic.json \
  --mode OFFLINE \
  --workload sharegpt \
  --dataset workloads/sharegpt.example.json \
  --num-prompts 2 \
  --request-rate 4 \
  --output-dir "${RESULT_ROOT}/inprocess-sharegpt"
validate inprocess-sharegpt 2

run_logged qwen3-32b-aic \
  python3 scripts/run_inprocess.py \
  --server-args configs/qwen3-32b-fp8-h20/server_args.json \
  --hisim-config configs/qwen3-32b-fp8-h20/hisim.aic.json \
  --mode OFFLINE \
  --workload trace \
  --dataset workloads/trace.autobench.example.jsonl \
  --output-dir "${RESULT_ROOT}/qwen3-32b-aic"
validate qwen3-32b-aic 3

run_logged glm5-p-aic \
  python3 scripts/run_inprocess.py \
  --server-args configs/glm5-p-b300/server_args.json \
  --hisim-config configs/glm5-p-b300/hisim.aic.json \
  --mode OFFLINE \
  --workload trace \
  --dataset workloads/trace.autobench.example.jsonl \
  --output-dir "${RESULT_ROOT}/glm5-p-aic"
validate glm5-p-aic 3

run_logged dsv4pro-p-ml \
  env SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  python3 scripts/run_inprocess.py \
  --server-args configs/dsv4pro-p-gb300/server_args.json \
  --hisim-config configs/dsv4pro-p-gb300/hisim.ml.json \
  --mode OFFLINE \
  --workload trace \
  --dataset workloads/trace.autobench.example.jsonl \
  --output-dir "${RESULT_ROOT}/dsv4pro-p-ml"
validate dsv4pro-p-ml 3

start_service \
  service-offline-replay OFFLINE \
  configs/qwen3-8b-h20/server_args.json \
  configs/qwen3-8b-h20/hisim.replay.json
run_logged service-offline-trace \
  env SGLANG_SIMULATOR_OUTPUT_DIR="${RESULT_ROOT}/service-offline-replay" \
  python3 -m sglang_simulator.simulation.bench_serving \
  --simulator-mode offline \
  --backend sglang \
  --base-url "http://127.0.0.1:${PORT}" \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name autobench \
  --dataset-path workloads/trace.autobench.example.jsonl \
  --use-trace-timestamps \
  --num-prompts 3 \
  --profile \
  --warmup-requests 0 \
  --disable-tqdm
validate service-offline-replay 3
run_logged service-inprocess-parity \
  python3 scripts/validate_mode_parity.py \
  --inprocess "${RESULT_ROOT}/inprocess-replay" \
  --service "${RESULT_ROOT}/service-offline-replay"
stop_service

start_service \
  service-blocking-aic BLOCKING \
  configs/qwen3-8b-h20/server_args.json \
  configs/qwen3-8b-h20/hisim.aic.json

run_logged terminal-random \
  env SGLANG_SIMULATOR_OUTPUT_DIR="${RESULT_ROOT}/service-blocking-aic" \
  python3 -m sglang_simulator.simulation.bench_serving \
  --simulator-mode blocking \
  --backend sglang \
  --base-url "http://127.0.0.1:${PORT}" \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name random \
  --dataset-path workloads/sharegpt.example.json \
  --request-rate 4 \
  --random-input-len 8 \
  --random-output-len 1 \
  --num-prompts 2 \
  --warmup-requests 0 \
  --profile \
  --disable-tqdm
validate service-blocking-aic 2

run_logged terminal-sharegpt \
  env SGLANG_SIMULATOR_OUTPUT_DIR="${RESULT_ROOT}/service-blocking-aic" \
  python3 -m sglang_simulator.simulation.bench_serving \
  --simulator-mode blocking \
  --backend sglang \
  --base-url "http://127.0.0.1:${PORT}" \
  --model /nfs/Qwen/Qwen3-8B \
  --dataset-name sharegpt \
  --dataset-path workloads/sharegpt.example.json \
  --request-rate 4 \
  --num-prompts 2 \
  --warmup-requests 0 \
  --profile \
  --disable-tqdm
validate service-blocking-aic 2
stop_service

run_logged cpu-gpu \
  env HISIM_GPU_ID="${GPU_ID}" bash scripts/validate_cpu_gpu.sh

touch "${RESULT_ROOT}/PASS"
echo "PASS all acceptance checks"
echo "results=${RESULT_ROOT}"

# HISIM_ACCEPTANCE_COMPLETE
