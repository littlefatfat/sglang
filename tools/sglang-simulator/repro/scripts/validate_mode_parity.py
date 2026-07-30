#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

EXACT_METRICS = (
    "num_requests",
    "completed",
    "total_input",
    "total_output",
    "prefix_cache_reused_ratio",
    "kv_cache_device_hit_ratio",
    "kv_cache_host_hit_ratio",
    "kv_cache_storage_hit_ratio",
    "iterations",
    "replay_exact_match_steps",
    "replay_miss_steps",
    "replay_zero_fallback_steps",
    "replay_knn_fallback_steps",
    "replay_fallback_rate",
)

TIME_METRICS_MS = (
    "mean_ttft_ms",
    "median_ttft_ms",
    "p90_ttft_ms",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p90_e2e_latency_ms",
)


def read_json(path: Path) -> dict:
    return json.load(open(path, encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def request_signature(row: dict) -> tuple:
    return (
        row.get("input_length"),
        row.get("output_length"),
        row.get("final_device_hit_len"),
        row.get("final_host_hit_len"),
        row.get("final_storage_hit_len"),
    )


def batch_signature(row: dict) -> tuple:
    return tuple(sorted(tuple(req) for req in row.get("requests", [])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inprocess", type=Path, required=True)
    parser.add_argument("--service", type=Path, required=True)
    args = parser.parse_args()

    inprocess_metrics = read_json(args.inprocess / "metrics.json")
    service_metrics = read_json(args.service / "metrics.json")
    for key in EXACT_METRICS:
        assert inprocess_metrics.get(key) == service_metrics.get(key), (
            key,
            inprocess_metrics.get(key),
            service_metrics.get(key),
        )

    inprocess_iterations = read_jsonl(args.inprocess / "iteration.jsonl")
    service_iterations = read_jsonl(args.service / "iteration.jsonl")
    assert [batch_signature(row) for row in inprocess_iterations] == [
        batch_signature(row) for row in service_iterations
    ]
    assert [row.get("forward_latency") for row in inprocess_iterations] == [
        row.get("forward_latency") for row in service_iterations
    ]

    inprocess_requests = read_jsonl(args.inprocess / "request.jsonl")
    service_requests = read_jsonl(args.service / "request.jsonl")
    assert sorted(map(request_signature, inprocess_requests)) == sorted(
        map(request_signature, service_requests)
    )

    cpu_delta_ms = (
        sum(
            abs(
                float(left.get("cpu_overhead", 0)) - float(right.get("cpu_overhead", 0))
            )
            for left, right in zip(inprocess_iterations, service_iterations)
        )
        * 1000
    )
    for key in TIME_METRICS_MS:
        delta_ms = abs(inprocess_metrics[key] - service_metrics[key])
        assert delta_ms <= cpu_delta_ms + 1e-6, (
            key,
            delta_ms,
            cpu_delta_ms,
        )
    duration_delta_ms = (
        abs(inprocess_metrics["duration"] - service_metrics["duration"]) * 1000
    )
    assert duration_delta_ms <= cpu_delta_ms + 1e-6, (
        "duration",
        duration_delta_ms,
        cpu_delta_ms,
    )

    print(
        "PASS service/in-process parity: exact requests, batches, cache, "
        f"forward and replay coverage; timing delta <= {cpu_delta_ms:.6f} ms "
        "recorded CPU-overhead delta"
    )


if __name__ == "__main__":
    main()
