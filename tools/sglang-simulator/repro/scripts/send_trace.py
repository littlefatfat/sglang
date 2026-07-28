#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path

import aiohttp


def load_trace(path: Path, timestamp_scale: float) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"empty trace: {path}")
    for i, row in enumerate(rows, 1):
        if "created_time" not in row and "timestamp" not in row:
            raise ValueError(f"row {i}: missing created_time/timestamp")
        if row["input_length"] != len(row["input_ids"]):
            raise ValueError(f"row {i}: input_length != len(input_ids)")
    rows.sort(key=lambda row: row.get("created_time", row.get("timestamp")))
    start = rows[0].get("created_time", rows[0].get("timestamp"))
    for row in rows:
        row["_relative_time"] = (
            row.get("created_time", row.get("timestamp")) - start
        ) / timestamp_scale
    return rows


async def post_json(session, url: str, body: dict):
    async with session.post(url, json=body) as response:
        response.raise_for_status()
        text = await response.text()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


async def run(args) -> None:
    rows = load_trace(Path(args.trace), args.timestamp_scale)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await post_json(session, f"{args.base_url}/start_profile", {})
        tasks = []
        for row in rows:
            body = {
                "input_ids": row["input_ids"],
                "sampling_params": {
                    "ignore_eos": True,
                    "max_new_tokens": row["output_length"],
                    "custom_params": {
                        "simulation": {
                            "created_time": row["_relative_time"],
                            "total_request": len(rows),
                        }
                    },
                },
            }
            tasks.append(
                asyncio.create_task(
                    post_json(session, f"{args.base_url}/generate", body)
                )
            )
        await asyncio.gather(*tasks)
        await post_json(session, f"{args.base_url}/stop_profile", {})
    print(f"completed={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timestamp-scale", type=float, default=1.0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
