#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path

from common import build_server_args, configure_environment


def load_trace(path: Path, timestamp_scale: float):
    from sglang_simulator.dataset import GenericRequest, SimpleDataset

    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            ts = row.get("created_time", row.get("timestamp"))
            if ts is None:
                raise ValueError(f"{path}:{line_no}: missing created_time/timestamp")
            ids = row["input_ids"]
            if row["input_length"] != len(ids):
                raise ValueError(f"{path}:{line_no}: input_length != len(input_ids)")
            rows.append((float(ts) / timestamp_scale, row))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    start = min(ts for ts, _ in rows)
    return SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=row["input_ids"],
                input_length=row["input_length"],
                output_length=row["output_length"],
                custom_params={"created_time": ts - start},
            )
            for ts, row in sorted(rows, key=lambda item: item[0])
        ]
    )


def random_dataset(count: int, input_len: int, output_len: int):
    from sglang_simulator.dataset import GenericRequest, SimpleDataset

    rng = random.Random(0)
    return SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=[rng.randrange(100, 10000) for _ in range(input_len)],
                input_length=input_len,
                output_length=output_len,
            )
            for _ in range(count)
        ]
    )


def sharegpt_dataset(path: Path, model_path: str, count: int):
    from sglang_simulator.dataset import GenericRequest, SimpleDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data = json.load(open(path, encoding="utf-8"))
    requests = []
    for item in data:
        turns = item.get("conversations", item.get("conversation", []))
        if len(turns) < 2:
            continue
        prompt = turns[0].get("value", turns[0].get("content"))
        answer = turns[1].get("value", turns[1].get("content"))
        input_ids = tokenizer.encode(prompt)
        output_len = max(1, len(tokenizer.encode(answer)))
        requests.append(
            GenericRequest(
                token_ids=input_ids,
                input_length=len(input_ids),
                output_length=output_len,
            )
        )
        if len(requests) == count:
            break
    if not requests:
        raise ValueError(f"no valid ShareGPT rows in {path}")
    return SimpleDataset(reqs=requests)


def persist(runner, metrics: dict, output: Path) -> None:
    with open(output / "result.metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    for name, getter in (
        ("result.request.jsonl", runner.get_request_stats),
        ("result.iteration.jsonl", runner.get_iteration_stats),
    ):
        with open(output / name, "w", encoding="utf-8") as f:
            for row in getter():
                f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-args", required=True)
    parser.add_argument("--hisim-config", required=True)
    parser.add_argument("--mode", choices=["OFFLINE", "BLOCKING"], default="OFFLINE")
    parser.add_argument("--workload", choices=["trace", "random", "sharegpt"], required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--timestamp-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = configure_environment(args.hisim_config, args.output_dir, args.mode)

    from sglang_simulator.simulation.benchmark import BenchmarkConfig
    from sglang_simulator.simulation.sglang.bench_runner import SGLangBenchmarkRunner

    server_args = build_server_args(args.server_args)
    raw_server_args = json.load(open(args.server_args, encoding="utf-8"))
    if args.workload == "trace":
        if not args.dataset:
            parser.error("--dataset is required for trace")
        dataset = load_trace(Path(args.dataset), args.timestamp_scale)
        benchmark = BenchmarkConfig(ignore_request_timestamp=False)
    elif args.workload == "sharegpt":
        if not args.dataset:
            parser.error("--dataset is required for sharegpt")
        dataset = sharegpt_dataset(
            Path(args.dataset), raw_server_args["model_path"], args.num_prompts
        )
        benchmark = BenchmarkConfig(
            request_rate=args.request_rate,
            ignore_request_timestamp=True,
        )
    else:
        dataset = random_dataset(args.num_prompts, args.input_len, args.output_len)
        benchmark = BenchmarkConfig(
            request_rate=args.request_rate,
            ignore_request_timestamp=True,
        )

    shutil.copy(args.server_args, output / "server_args.json")
    shutil.copy(args.hisim_config, output / "hisim_config.json")
    runner = SGLangBenchmarkRunner(server_args)
    try:
        metrics = runner.benchmark(benchmark, dataset)
        persist(runner, metrics, output)
        print(json.dumps(metrics, indent=2))
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
