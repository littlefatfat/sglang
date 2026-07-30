import asyncio

import aiohttp

from sglang.benchmark.datasets.common import DatasetRow
from sglang_simulator.simulation import bench_serving


async def _collect(generator):
    return [request async for request in generator]


def _row(timestamp=None, extra_request_body=None):
    return DatasetRow(
        prompt=[1, 2, 3],
        prompt_len=3,
        output_len=7,
        timestamp=timestamp,
        extra_request_body=extra_request_body,
    )


def test_offline_trace_injects_normalized_metadata_without_mutating_contract():
    rows = [_row(1200.0, {"rid": "a"}), _row(1000.0, {"rid": "b"})]
    bench_serving._SIMULATOR_MODE = "offline"
    bench_serving._USE_TRACE_TIMESTAMPS = False

    result = asyncio.run(
        _collect(
            bench_serving.simulator_get_request(
                rows,
                request_rate=4,
                use_trace_timestamps=True,
                slowdown_factor=2,
            )
        )
    )

    assert [row.timestamp for row in result] == [1000.0, 1200.0]
    assert result[0].extra_request_body == {
        "rid": "b",
        "simulation": {"created_time_ms": 0.0, "total_request": 2},
    }
    assert result[1].extra_request_body["simulation"] == {
        "created_time_ms": 400.0,
        "total_request": 2,
    }


def test_offline_request_rate_generates_logical_time_without_sleep(monkeypatch):
    rows = [_row(), _row(), _row()]
    bench_serving._SIMULATOR_MODE = "offline"
    bench_serving._USE_TRACE_TIMESTAMPS = False
    monkeypatch.setattr(bench_serving.np.random, "exponential", lambda _: 0.25)

    result = asyncio.run(
        _collect(bench_serving.simulator_get_request(rows, request_rate=4))
    )

    assert [
        row.extra_request_body["simulation"]["created_time_ms"] for row in result
    ] == [0.0, 250.0, 500.0]
    assert all(
        row.extra_request_body["simulation"]["total_request"] == 3 for row in result
    )


def test_v0516_cli_trace_flag_survives_missing_benchmark_forwarding():
    rows = [_row(1000.0), _row(1250.0)]
    bench_serving._SIMULATOR_MODE = "offline"
    bench_serving._USE_TRACE_TIMESTAMPS = True

    result = asyncio.run(
        _collect(
            bench_serving.simulator_get_request(
                rows,
                request_rate=float("inf"),
                use_trace_timestamps=False,
            )
        )
    )

    assert [
        row.extra_request_body["simulation"]["created_time_ms"] for row in result
    ] == [0.0, 250.0]


def test_json_hijack_merges_metadata_into_existing_sampling_params(monkeypatch):
    captured = {}

    async def original_request(self, method, url, **kwargs):
        captured.update(kwargs["json"])
        return "response"

    monkeypatch.setattr(aiohttp.ClientSession, "_request", original_request)
    monkeypatch.setattr(bench_serving, "_ORIGINAL_AIOHTTP_REQUEST", None)
    bench_serving.install_aiohttp_json_hijack()
    payload = {
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 7, "temperature": 0},
        "simulation": {"created_time_ms": 10, "total_request": 1},
    }

    result = asyncio.run(
        aiohttp.ClientSession._request(
            object(), "POST", "http://127.0.0.1:30000/generate", json=payload
        )
    )

    assert result == "response"
    assert "simulation" not in captured
    assert captured["sampling_params"]["max_new_tokens"] == 7
    assert captured["sampling_params"]["custom_params"]["simulation"] == {
        "created_time_ms": 10,
        "total_request": 1,
    }


def test_simulator_cli_argument_is_removed_before_official_parser():
    mode, remaining = bench_serving._extract_simulator_args(
        ["--backend", "sglang", "--simulator-mode", "blocking", "--num-prompts", "2"]
    )
    assert mode == "blocking"
    assert remaining == ["--backend", "sglang", "--num-prompts", "2"]
