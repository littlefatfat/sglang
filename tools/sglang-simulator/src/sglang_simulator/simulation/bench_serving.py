import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import fields
from typing import AsyncGenerator, List, Optional

import aiohttp
import numpy as np

from sglang.benchmark import serving as bench_serving
from sglang.benchmark.serving import DatasetRow

# Hijack aiohttp request sending to rewrite simulation metadata into a server-visible path.
#
# The benchmark attaches simulation data into the request payload, but the server-side
# parser does not read arbitrary top-level fields such as `simulation`.
# To make the metadata available to the backend, this hook intercepts
# `aiohttp.ClientSession._request` and moves:
#
#     payload["simulation"]
#
# into:
#
#     payload["sampling_params"]["custom_params"]["simulation"]
#
# so the simulation parameters are preserved during request serialization and can be
# consumed by the server through its existing sampling/custom-parameter pipeline.
_ORIG_AIOHTTP_REQUEST = None


def install_aiohttp_json_hijack(
    *,
    hijack_url_regex: Optional[str],
) -> None:
    global _ORIG_AIOHTTP_REQUEST
    if _ORIG_AIOHTTP_REQUEST is not None:
        return

    pattern = re.compile(hijack_url_regex) if hijack_url_regex else None
    _ORIG_AIOHTTP_REQUEST = aiohttp.ClientSession._request

    async def _patched_request(self, method, url, **kwargs):
        if pattern is not None and pattern.search(url):
            payload = kwargs.get("json", None)
            if isinstance(payload, dict) and "simulation" in payload:
                if "sampling_params" not in payload:
                    payload["sampling_params"] = {}
                if "custom_params" not in payload["sampling_params"]:
                    payload["sampling_params"]["custom_params"] = {}
                payload["sampling_params"]["custom_params"]["simulation"] = payload.pop(
                    "simulation"
                )
                kwargs["json"] = payload

        return await _ORIG_AIOHTTP_REQUEST(self, method, url, **kwargs)

    aiohttp.ClientSession._request = _patched_request


# Override request generation for simulation mode.
#
# In OFFLINE mode, `request_rate` is translated into `simulation.created_time`.
# The simulator replays logical arrivals without client-side sleeps.
#
# BLOCKING mode keeps the same metadata, paces the client in real time, and records
# `server_created_time`, so request, queue, and step timestamps use one origin.
#
# Trace timestamps are normalized relative to the first request. Without a trace,
# Poisson inter-arrival intervals are accumulated.
async def override_get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
    use_trace_timestamps: bool = False,
    slowdown_factor: float = 1.0,
    timestamp_scale_s: float = 1000,  # 1000: ms -> s
) -> AsyncGenerator[DatasetRow, None]:

    blocking = os.getenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE") == "BLOCKING"
    if use_trace_timestamps:
        print(
            f"Using trace timestamps for request generation with slowdown factor {slowdown_factor}."
        )
        # Sort requests by timestamp for correct replay
        input_requests.sort(key=lambda r: r.timestamp)

        start_time = time.perf_counter()
        trace_start_time = input_requests[0].timestamp if input_requests else 0

        for request in input_requests:
            created_time = (
                (request.timestamp - trace_start_time)
                / timestamp_scale_s
                * slowdown_factor
            )
            if blocking:
                target_time = start_time + created_time
                sleep_duration = target_time - time.perf_counter()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
            simulation = {
                "created_time": created_time,
                "total_request": len(input_requests),
            }
            if blocking:
                simulation["server_created_time"] = time.time()
            request.extra_request_body = {
                "simulation": simulation
            }

            yield request
    else:
        input_requests_iter = iter(input_requests)
        start_time = 0
        for request in input_requests_iter:
            simulation = {
                "created_time": start_time,
                "total_request": len(input_requests),
            }
            if blocking:
                simulation["server_created_time"] = time.time()
            request.extra_request_body = {
                "simulation": simulation
            }
            yield request

            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue

            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)
            if blocking:
                await asyncio.sleep(interval)
            # The next request will be sent after the interval.
            start_time += interval


# Replace benchmark-side metrics with backend-generated simulation metrics.
#
# In simulation mode, metrics computed by the benchmark client are not trustworthy,
# because they describe the local execution path of the benchmark tool rather than
# the actual simulated execution on the backend.
# Therefore, after the benchmark finishes, this wrapper loads the metrics file
# produced by the simulator backend and uses it as the final benchmark result.
#
# If the backend metrics file is missing, the wrapper falls back to the original
# benchmark-side metrics.

original_calculate_metrics = bench_serving.calculate_metrics


def wrapped_calculate_metrics(*args, **simulation_metrics):
    real_metrics, output_lens = original_calculate_metrics(*args, **simulation_metrics)

    out_dir = os.getenv("SGLANG_SIMULATOR_OUTPUT_DIR")
    if not out_dir:
        print("SGLANG_SIMULATOR_OUTPUT_DIR is unset; using client-side metrics")
        return real_metrics, output_lens
    metrics_path = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Fail to get simulation metrics from {out_dir}")
        return real_metrics, output_lens

    with open(metrics_path) as f:
        data = json.load(f)

    metrics_fields = {f.name for f in fields(bench_serving.BenchmarkMetrics)}
    # -1 means invalid value.
    simulation_metrics = {k: data.get(k, -1) for k in metrics_fields}

    return bench_serving.BenchmarkMetrics(**simulation_metrics), output_lens


original_run_benchmark = bench_serving.run_benchmark


def wrapped_run_benchmark(args_: argparse.Namespace):
    # The profile API has been hooked and is used as a trigger to start or stop the simulation.
    args_.profile = True
    return original_run_benchmark(args_)


bench_serving.get_request = override_get_request
bench_serving.calculate_metrics = wrapped_calculate_metrics
bench_serving.run_benchmark = wrapped_run_benchmark

install_aiohttp_json_hijack(hijack_url_regex=r"generate$")


if __name__ == "__main__":
    bench_serving.cli_main()
