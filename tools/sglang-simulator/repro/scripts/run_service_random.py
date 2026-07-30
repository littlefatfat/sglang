#!/usr/bin/env python3
"""Start HiSim, run official random serving benchmark, stop, keep metrics."""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def benchmark_command(
    port: int,
    model_path: str,
    dataset_path: Path,
    num_prompts: int,
    input_len: int,
    output_len: int,
    request_rate: float,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        model_path,
        "--dataset-name",
        "random",
        "--dataset-path",
        str(dataset_path),
        "--request-rate",
        str(request_rate),
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        "1",
        "--num-prompts",
        str(num_prompts),
        "--warmup-requests",
        "0",
        "--profile",
        "--disable-tqdm",
    ]


def wait_until_ready(process: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"server did not become ready within {timeout}s")


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/nfs/Qwen/Qwen3-8B")
    parser.add_argument(
        "--sim-config-path",
        type=Path,
        default=ROOT / "configs/qwen3-8b-h20/hisim.aic.json",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=ROOT / "workloads/sharegpt.example.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["OFFLINE", "BLOCKING"], default="BLOCKING")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--request-rate", type=float, default=4)
    parser.add_argument("--ready-timeout", type=int, default=180)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"{output_dir} already exists; choose a new output directory")

    server_env = os.environ.copy()
    server_env.update(
        {
            "SGLANG_USE_CPU_ENGINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "SGLANG_SIMULATOR_OUTPUT_MODE": args.mode,
            "SGLANG_SIMULATOR_OUTPUT_DIR": str(output_dir),
        }
    )
    server = [
        sys.executable,
        "-m",
        "sglang_simulator.simulation.sglang.launch_server",
        "--model-path",
        args.model_path,
        "--sim-config-path",
        str(args.sim_config_path.resolve()),
        "--port",
        str(args.port),
    ]

    server_log_path = output_dir.parent / f"{output_dir.name}.server.log"
    with server_log_path.open("w", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            server,
            env=server_env,
            start_new_session=True,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_until_ready(process, args.port, args.ready_timeout)
            subprocess.run(
                benchmark_command(
                    args.port,
                    args.model_path,
                    args.dataset_path.resolve(),
                    args.num_prompts,
                    args.input_len,
                    args.output_len,
                    args.request_rate,
                ),
                check=True,
                env=os.environ.copy(),
            )
            metrics = output_dir / "metrics.json"
            if not metrics.is_file():
                raise RuntimeError(f"benchmark completed without {metrics}")
        finally:
            stop_process_group(process)
    print(f"PASS metrics={metrics} server_log={server_log_path}")


if __name__ == "__main__":
    main()
