#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_workload() -> list[dict]:
    shared = [1000 + index % 997 for index in range(768)]
    prompt_a = shared + [3000 + index for index in range(256)]
    prompt_b = [5000 + index for index in range(1024)]
    return [
        {
            "prompt": prompt,
            "prompt_len": len(prompt),
            "output_len": 1,
            "timestamp": timestamp,
        }
        for prompt, timestamp in (
            (prompt_a, 0),
            (prompt_b, 10_000),
            (prompt_a, 20_000),
        )
    ]


def build_server_args(base: dict) -> dict:
    config = dict(base)
    config.update(
        {
            "max_total_num_tokens": 1536,
            "max_running_requests": 1,
            "max_prefill_tokens": 1024,
            "chunked_prefill_size": 1024,
            "page_size": 256,
            "enable_hierarchical_cache": True,
            "hicache_ratio": 4.0,
            "hicache_write_policy": "write_through",
            "hicache_io_backend": "kernel",
            "hicache_mem_layout": "page_first",
        }
    )
    return config


def validate(output: Path) -> None:
    metrics = json.loads((output / "result.metrics.json").read_text())
    requests = [
        json.loads(line)
        for line in (output / "result.request.jsonl").read_text().splitlines()
    ]

    assert metrics["num_requests"] == 3
    assert metrics["completed"] == 3
    assert metrics["kv_cache_host_hit_ratio"] > 0
    assert len(requests) == 3
    assert requests[0]["final_host_hit_len"] == 0
    assert requests[1]["final_host_hit_len"] == 0
    assert requests[2]["final_host_hit_len"] >= 768


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small L2 eviction-and-reload simulation."
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    inputs = output.with_name(f"{output.name}.inputs")
    if output.exists() or inputs.exists():
        parser.error(f"output or input directory already exists: {output}, {inputs}")
    inputs.mkdir(parents=True)

    base_path = ROOT / "configs/qwen3-8b-h20/server_args.json"
    server_args_path = inputs / "server_args.json"
    workload_path = inputs / "trace.jsonl"
    server_args_path.write_text(
        json.dumps(build_server_args(json.loads(base_path.read_text())), indent=2)
        + "\n"
    )
    workload_path.write_text(
        "".join(json.dumps(row) + "\n" for row in build_workload())
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_inprocess.py"),
            "--server-args",
            str(server_args_path),
            "--sim-config",
            str(ROOT / "configs/qwen3-8b-h20/simulator.aic.json"),
            "--mode",
            "OFFLINE",
            "--workload",
            "trace",
            "--dataset",
            str(workload_path),
            "--num-prompts",
            "3",
            "--output-dir",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    validate(output)
    print(f"PASS HiCache host reload: {output}")


if __name__ == "__main__":
    main()
