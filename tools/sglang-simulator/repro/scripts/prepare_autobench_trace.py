#!/usr/bin/env python3
"""Convert a HiSim timestamp trace into SGLang Autobench JSONL."""

import argparse
import json
from pathlib import Path


def parse_indices(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item) for item in value.split(",") if item.strip()}


def convert_rows(
    source: Path, indices: set[int] | None, limit: int | None
) -> list[dict]:
    selected = []
    with source.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if indices is not None and index not in indices:
                continue
            selected.append(json.loads(line))
            if indices is None and limit is not None and len(selected) >= limit:
                break

    if not selected:
        raise ValueError("the selection produced no requests")

    base_time = float(selected[0]["created_time"])
    converted = []
    for row in selected:
        created_time_ms = (float(row["created_time"]) - base_time) * 1000.0
        converted.append(
            {
                "prompt": row["input_ids"],
                "prompt_len": int(row["input_length"]),
                "output_len": int(row["output_length"]),
                "timestamp": created_time_ms,
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--indices",
        help="zero-based comma-separated rows; preserves the specified trace rows",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.indices and args.limit:
        parser.error("--indices and --limit are mutually exclusive")
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} already exists; pass --force to replace it")

    rows = convert_rows(args.input, parse_indices(args.indices), args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} requests to {args.output.resolve()}")


if __name__ == "__main__":
    main()
