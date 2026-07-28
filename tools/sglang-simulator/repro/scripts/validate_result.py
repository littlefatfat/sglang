#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def require_nonnegative(metrics: dict, key: str) -> None:
    value = metrics.get(key)
    if value is None or not math.isfinite(value) or value < 0:
        raise AssertionError(f"{key} must be finite and >= 0, got {value}")


def validate(path: Path) -> None:
    metrics_path = path / "result.metrics.json"
    if not metrics_path.exists():
        metrics_path = path / "metrics.json"
    metrics = json.load(open(metrics_path, encoding="utf-8"))

    expected = metrics.get("num_requests", metrics.get("completed"))
    assert metrics["completed"] == expected, (
        f"completed={metrics['completed']} num_requests={expected}"
    )
    for key in ("duration", "mean_ttft_ms", "mean_e2e_latency_ms"):
        require_nonnegative(metrics, key)

    keys = (
        "prefix_cache_reused_ratio",
        "kv_cache_device_hit_ratio",
        "kv_cache_host_hit_ratio",
        "kv_cache_storage_hit_ratio",
    )
    for key in keys:
        value = metrics.get(key, 0.0)
        assert 0 <= value <= 1, f"{key} outside [0, 1]: {value}"
    parts = sum(metrics.get(key, 0.0) for key in keys[1:])
    assert math.isclose(metrics.get(keys[0], 0.0), parts, abs_tol=1e-9), (
        f"prefix={metrics.get(keys[0])} device+host+storage={parts}"
    )

    request_path = path / "result.request.jsonl"
    if request_path.exists():
        request_count = sum(1 for line in open(request_path) if line.strip())
        assert request_count == metrics["completed"], (
            f"request rows={request_count} completed={metrics['completed']}"
        )
    print(f"PASS {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    validate(parser.parse_args().result_dir)


if __name__ == "__main__":
    main()
