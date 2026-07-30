#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from common import build_server_args, configure_environment


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
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--mode", choices=["OFFLINE", "BLOCKING"], default="OFFLINE")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--page-size", type=int)
    parser.add_argument(
        "--workload", choices=["trace", "random", "sharegpt"], required=True
    )
    parser.add_argument("--dataset")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--timestamp-scale", type=float, default=1000.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = configure_environment(
        args.sim_config, args.output_dir, args.mode, args.device
    )

    from sglang_simulator.simulation.benchmark import BenchmarkConfig
    from sglang_simulator.simulation.sglang.bench_runner import SGLangBenchmarkRunner
    from sglang_simulator.workload import load_inprocess_workload

    server_args = build_server_args(args.server_args, args.device, args.page_size)
    raw_server_args = json.load(open(args.server_args, encoding="utf-8"))
    try:
        dataset = load_inprocess_workload(
            name=args.workload,
            model_path=raw_server_args["model_path"],
            dataset_path=args.dataset,
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
            timestamp_scale=args.timestamp_scale,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.workload == "trace":
        benchmark = BenchmarkConfig(ignore_request_timestamp=False)
    else:
        benchmark = BenchmarkConfig(
            request_rate=args.request_rate,
            ignore_request_timestamp=True,
        )

    shutil.copy(args.server_args, output / "server_args.json")
    shutil.copy(args.sim_config, output / "sim_config.json")
    runner = SGLangBenchmarkRunner(server_args)
    try:
        metrics = runner.benchmark(benchmark, dataset)
        persist(runner, metrics, output)
        print(json.dumps(metrics, indent=2))
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
