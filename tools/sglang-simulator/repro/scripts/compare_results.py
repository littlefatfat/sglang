#!/usr/bin/env python3
import argparse
import collections
import json
import statistics
from pathlib import Path

METRICS = (
    "prefix_cache_reused_ratio",
    "kv_cache_device_hit_ratio",
    "kv_cache_host_hit_ratio",
    "kv_cache_storage_hit_ratio",
    "duration",
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


def metric_path(root: Path) -> Path:
    for name in ("result.metrics.json", "metrics.json"):
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"metrics not found under {root}")


def batch_summary(rows: list[dict]) -> dict:
    compositions = [
        tuple(sorted(tuple(x) for x in row.get("requests", []))) for row in rows
    ]
    sizes = [len(x) for x in compositions]
    return {
        "iterations": len(rows),
        "unique_compositions": len(set(compositions)),
        "mean_batch_size": statistics.fmean(sizes) if sizes else 0,
        "max_batch_size": max(sizes, default=0),
        "batch_size_histogram": dict(sorted(collections.Counter(sizes).items())),
        "total_extend_tokens": sum(x[0] for comp in compositions for x in comp),
        "total_past_tokens": sum(x[1] for comp in compositions for x in comp),
        "_compositions": compositions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    args = parser.parse_args()

    old_metrics = read_json(metric_path(args.old))
    new_metrics = read_json(metric_path(args.new))
    metric_diff = {}
    for key in METRICS:
        old = old_metrics.get(key)
        new = new_metrics.get(key)
        metric_diff[key] = {
            "old": old,
            "new": new,
            "delta": None if old is None or new is None else new - old,
            "ape_pct": (
                None
                if old in (None, 0) or new is None
                else abs(new - old) / abs(old) * 100
            ),
        }

    old_batches = batch_summary(read_jsonl(args.old / "result.iteration.jsonl"))
    new_batches = batch_summary(read_jsonl(args.new / "result.iteration.jsonl"))
    old_comp = old_batches.pop("_compositions")
    new_comp = new_batches.pop("_compositions")
    first_mismatch = next(
        (i for i, pair in enumerate(zip(old_comp, new_comp)) if pair[0] != pair[1]),
        None,
    )
    result = {
        "metrics": metric_diff,
        "batch": {
            "old": old_batches,
            "new": new_batches,
            "exact_sequence_match": old_comp == new_comp,
            "first_mismatch_iteration": first_mismatch,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
