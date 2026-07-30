"""SGLang serving benchmark adapter for simulator traffic.

This module deliberately reuses SGLang's benchmark implementation and dataset
loaders.  It only owns the simulator-specific parts of the protocol:

* convert request-rate or trace timestamps into logical arrival timestamps;
* inject the internal ``sampling_params.custom_params.simulation`` metadata;
* avoid client-side pacing in OFFLINE mode; and
* display backend-produced simulator metrics when they are locally available.

User datasets must not contain simulator metadata.
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import aiohttp
import numpy as np

from sglang.benchmark import serving
from sglang.benchmark.datasets.common import DatasetRow

_ORIGINAL_AIOHTTP_REQUEST = None
_ORIGINAL_CALCULATE_METRICS = serving.calculate_metrics
_ORIGINAL_GET_REQUEST = serving.get_request
_ORIGINAL_RUN_BENCHMARK = serving.run_benchmark
_SIMULATOR_MODE = "offline"
_USE_TRACE_TIMESTAMPS = False


def _metrics_path() -> Path:
    output_dir = Path(
        os.getenv("SGLANG_SIMULATOR_OUTPUT_DIR", "/tmp/sglang_simulator/output")
    )
    return output_dir / "metrics.json"


def _load_backend_metrics() -> dict:
    metrics_path = _metrics_path()
    if not metrics_path.is_file():
        raise RuntimeError(
            f"Simulator backend metrics are not available at {metrics_path}. "
            "Set SGLANG_SIMULATOR_OUTPUT_DIR in the benchmark process to the "
            "same directory used by the simulator server."
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


class _ServingResultFilter(io.TextIOBase):
    """Hide SGLang's client-side summary while streaming all other output."""

    def __init__(self, target):
        self.target = target
        self.suppress = False

    def write(self, text):
        if "Serving Benchmark Result" in text:
            self.suppress = True
        if self.suppress:
            if text.strip() == "=" * 50:
                self.suppress = False
            return len(text)
        return self.target.write(text)

    def flush(self):
        return self.target.flush()


def _set_simulation_metadata(
    request: DatasetRow, *, created_time_ms: float, total_request: int
) -> None:
    """Attach transient metadata without replacing dataset-specific parameters."""
    extra_request_body = dict(request.extra_request_body or {})
    extra_request_body["simulation"] = {
        "created_time_ms": created_time_ms,
        "total_request": total_request,
    }
    request.extra_request_body = extra_request_body


async def simulator_get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
    use_trace_timestamps: bool = False,
    slowdown_factor: float = 1.0,
) -> AsyncGenerator[DatasetRow, None]:
    """Generate simulator traffic while retaining official BLOCKING pacing."""
    # SGLang v0.5.16 parses --use-trace-timestamps but its benchmark() does not
    # forward the value to get_request(). Preserve it in our entry point.
    use_trace_timestamps = use_trace_timestamps or _USE_TRACE_TIMESTAMPS
    if _SIMULATOR_MODE == "blocking":
        async for request in _ORIGINAL_GET_REQUEST(
            input_requests,
            request_rate,
            use_trace_timestamps=use_trace_timestamps,
            slowdown_factor=slowdown_factor,
        ):
            yield request
        return

    total_request = len(input_requests)
    if use_trace_timestamps:
        if any(request.timestamp is None for request in input_requests):
            raise ValueError(
                "--use-trace-timestamps requires every request to have timestamp"
            )
        input_requests.sort(key=lambda request: request.timestamp)
        trace_start_time_ms = input_requests[0].timestamp if input_requests else 0.0
        for request in input_requests:
            created_time_ms = (
                float(request.timestamp) - float(trace_start_time_ms)
            ) * slowdown_factor
            _set_simulation_metadata(
                request,
                created_time_ms=created_time_ms,
                total_request=total_request,
            )
            yield request
        return

    created_time_ms = 0.0
    for request in input_requests:
        _set_simulation_metadata(
            request,
            created_time_ms=created_time_ms,
            total_request=total_request,
        )
        yield request
        if request_rate != float("inf"):
            created_time_ms += np.random.exponential(1.0 / request_rate) * 1000.0


def install_aiohttp_json_hijack(
    *, hijack_url_regex: Optional[str] = r"/generate(?:\?.*)?$"
) -> None:
    """Move transient metadata into the already-built sampling parameters."""
    global _ORIGINAL_AIOHTTP_REQUEST
    if _ORIGINAL_AIOHTTP_REQUEST is not None:
        return

    pattern = re.compile(hijack_url_regex) if hijack_url_regex else None
    _ORIGINAL_AIOHTTP_REQUEST = aiohttp.ClientSession._request

    async def patched_request(self, method, url, **kwargs):
        if pattern is None or pattern.search(str(url)):
            payload = kwargs.get("json")
            if isinstance(payload, dict) and "simulation" in payload:
                simulation = payload.pop("simulation")
                sampling_params = payload.setdefault("sampling_params", {})
                custom_params = sampling_params.setdefault("custom_params", {})
                custom_params["simulation"] = simulation
                kwargs["json"] = payload
        return await _ORIGINAL_AIOHTTP_REQUEST(self, method, url, **kwargs)

    aiohttp.ClientSession._request = patched_request


def simulator_calculate_metrics(*args, **kwargs):
    """Supply backend metrics to code paths inside SGLang's benchmark."""
    client_metrics, output_lens = _ORIGINAL_CALCULATE_METRICS(*args, **kwargs)
    backend_metrics = _load_backend_metrics()
    metric_names = {field.name for field in fields(serving.BenchmarkMetrics)}
    values = {
        name: backend_metrics.get(name, getattr(client_metrics, name))
        for name in metric_names
    }
    return serving.BenchmarkMetrics(**values), output_lens


def _replace_explicit_output_file(args: argparse.Namespace, metrics: dict) -> None:
    output_file = getattr(args, "output_file", None)
    if not output_file:
        return
    path = Path(output_file)
    lines = path.read_text(encoding="utf-8").splitlines()
    authoritative = json.dumps(metrics)
    if lines:
        lines[-1] = authoritative
    else:
        lines.append(authoritative)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def simulator_run_benchmark(args: argparse.Namespace):
    global _USE_TRACE_TIMESTAMPS
    if args.backend != "sglang":
        raise ValueError(
            "sglang_simulator.simulation.bench_serving requires --backend sglang"
        )
    if args.dataset_name == "mooncake":
        raise ValueError(
            "Mooncake's multi-round scheduler is not supported by the simulator "
            "benchmark adapter"
    )
    _USE_TRACE_TIMESTAMPS = getattr(args, "use_trace_timestamps", False)
    args.profile = True
    with contextlib.redirect_stdout(_ServingResultFilter(sys.stdout)):
        _ORIGINAL_RUN_BENCHMARK(args)

    backend_metrics = _load_backend_metrics()
    _replace_explicit_output_file(args, backend_metrics)
    print("\n============ Simulator Backend Metrics ============")
    print(f"Source: {_metrics_path()}")
    print(json.dumps(backend_metrics, indent=4))
    print("=" * 51)
    return backend_metrics


def _extract_simulator_args(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--simulator-mode",
        choices=("offline", "blocking"),
        default="offline",
        help=argparse.SUPPRESS,
    )
    args, remaining = parser.parse_known_args(argv)
    return args.simulator_mode, remaining


def cli_main() -> None:
    global _SIMULATOR_MODE
    if any(argument in ("-h", "--help") for argument in sys.argv[1:]):
        print(
            "SGLang Simulator option: "
            "--simulator-mode {offline,blocking} (default: offline)\n"
        )
    _SIMULATOR_MODE, remaining = _extract_simulator_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    serving.get_request = simulator_get_request
    serving.calculate_metrics = simulator_calculate_metrics
    serving.run_benchmark = simulator_run_benchmark
    install_aiohttp_json_hijack()

    print(f"SGLang Simulator benchmark mode: {_SIMULATOR_MODE.upper()}")
    serving.cli_main()


if __name__ == "__main__":
    cli_main()
