import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_json_files_parse():
    for path in ROOT.rglob("*.json"):
        json.load(open(path, encoding="utf-8"))


def test_package_declares_runtime_dependencies():
    metadata = tomllib.loads((ROOT.parent / "pyproject.toml").read_text())
    dependencies = set(metadata["project"]["dependencies"])
    assert dependencies == {
        "sglang==0.5.16",
        "numpy",
        "scikit-learn",
        "joblib",
    }


def test_trace_contract():
    from sglang_simulator.workload import load_hisim_trace_rows

    rows = load_hisim_trace_rows(
        ROOT / "workloads/trace.example.jsonl",
        timestamp_scale=1.0,
    )
    assert len(rows) == 3
    assert rows[0].timestamp == 0
    assert rows[2].timestamp == 0.2
    assert rows[0].prompt_len == len(rows[0].prompt)


def test_inprocess_trace_uses_shared_trace_loader():
    from sglang_simulator.workload import load_inprocess_workload

    dataset = load_inprocess_workload(
        name="trace",
        model_path="unused",
        dataset_path=str(ROOT / "workloads/trace.example.jsonl"),
        num_prompts=3,
        input_len=1,
        output_len=1,
        timestamp_scale=1.0,
    )
    assert [row.custom_params["created_time"] for row in dataset] == [
        0,
        0.1,
        0.2,
    ]


def test_compare_batch_summary():
    module = load_script("compare_results.py")
    summary = module.batch_summary(
        [
            {"requests": [[10, 0], [1, 10]]},
            {"requests": [[1, 11]]},
        ]
    )
    assert summary["iterations"] == 2
    assert summary["total_extend_tokens"] == 12
    assert summary["total_past_tokens"] == 21


def test_validate_result(tmp_path):
    module = load_script("validate_result.py")
    metrics = {
        "num_requests": 1,
        "completed": 1,
        "duration": 1.0,
        "mean_ttft_ms": 2.0,
        "mean_e2e_latency_ms": 3.0,
        "prefix_cache_reused_ratio": 0.6,
        "kv_cache_device_hit_ratio": 0.2,
        "kv_cache_host_hit_ratio": 0.3,
        "kv_cache_storage_hit_ratio": 0.1,
    }
    (tmp_path / "result.metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "result.request.jsonl").write_text("{}\n")
    module.validate(tmp_path)

    (tmp_path / "result.metrics.json").unlink()
    (tmp_path / "result.request.jsonl").unlink()
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "request.jsonl").write_text("{}\n")
    module.validate(tmp_path, expected_requests=1)

    try:
        module.validate(tmp_path, expected_requests=2)
    except AssertionError as error:
        assert "expected=2" in str(error)
    else:
        raise AssertionError("expected request-count validation to fail")


def test_send_trace_accepts_empty_text_and_json_responses():
    module = load_script("send_trace.py")

    class Response:
        def __init__(self, text):
            self.value = text

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        async def text(self):
            return self.value

    class Session:
        def __init__(self, text):
            self.value = text

        def post(self, *args, **kwargs):
            return Response(self.value)

    async def request(text):
        return await module.post_json(Session(text), "http://test", {})

    assert asyncio.run(request("")) is None
    assert asyncio.run(request("OK")) == "OK"


def test_cpu_capability_patch_is_scoped_to_cpu_simulation(monkeypatch):
    module = load_script("common.py")
    import torch

    original = torch.cuda.get_device_capability
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("SGLANG_USE_CPU_ENGINE", raising=False)
    module.patch_cpu_device_capability()
    assert torch.cuda.get_device_capability is original

    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")
    module.patch_cpu_device_capability()
    assert torch.cuda.get_device_capability() == (10, 0)


def test_spawned_python_installs_cpu_capability_shim():
    env = os.environ.copy()
    env["SGLANG_USE_CPU_ENGINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.get_device_capability())",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "(10, 0)"


def test_target_topology_is_kept_out_of_dummy_engine():
    module = load_script("common.py")
    engine = {"tp_size": 8, "ep_size": 4, "dp_size": 2, "pp_size": 3}
    module.normalize_dummy_engine_topology(engine)

    assert engine == {
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "pp_size": 1,
    }
    target = json.load(open(ROOT / "configs/glm5-p-b300/hisim.aic.json"))
    assert target["scheduler"]["tp_size"] == 8


def test_benchmark_client_supports_offline_and_blocking_metadata():
    env = os.environ.copy()
    env["SGLANG_USE_CPU_ENGINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    script = r"""
import asyncio
import os
from dataclasses import dataclass, field

from sglang_simulator.simulation.bench_serving import override_get_request

@dataclass
class Row:
    extra_request_body: dict = field(default_factory=dict)

async def collect(mode):
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = mode
    rows = [Row(), Row()]
    result = []
    async for row in override_get_request(rows, float("inf")):
        result.append(row.extra_request_body["simulation"])
    return result

offline = asyncio.run(collect("OFFLINE"))
assert [item["created_time"] for item in offline] == [0, 0]
assert all("server_created_time" not in item for item in offline)

blocking = asyncio.run(collect("BLOCKING"))
assert [item["created_time"] for item in blocking] == [0, 0]
assert all(item["server_created_time"] > 0 for item in blocking)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0


def test_benchmark_client_accepts_hisim_trace_cli():
    from sglang_simulator.simulation import bench_serving

    argv = bench_serving.prepare_hisim_cli_args(
        [
            "bench",
            "--dataset-name",
            "hisim-trace",
            "--dataset-path",
            "trace.jsonl",
            "--hisim-timestamp-scale",
            "1000",
            "--trace-slowdown-factor",
            "2",
        ]
    )
    assert argv == [
        "bench",
        "--dataset-name",
        "custom",
        "--dataset-path",
        "trace.jsonl",
    ]


def test_acceptance_script_has_completion_sentinel():
    script = ROOT / "scripts/acceptance.sh"
    assert script.read_text().rstrip().endswith(
        "# HISIM_ACCEPTANCE_COMPLETE"
    )
