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
    from sglang.benchmark.datasets.autobench import sample_autobench_requests

    rows = sample_autobench_requests(
        dataset_path=str(ROOT / "workloads/trace.autobench.example.jsonl"),
        num_requests=100,
        tokenizer=None,
    )
    assert len(rows) == 100
    assert rows[0].timestamp == 0
    assert rows[-1].timestamp == 149599.82
    assert all(row.prompt_len == len(row.prompt) for row in rows)
    assert min(row.prompt_len for row in rows) == 128
    assert max(row.prompt_len for row in rows) == 512
    assert rows[1].prompt[:128] == rows[0].prompt
    assert all(row.extra_request_body == {} for row in rows)


def test_inprocess_trace_uses_shared_trace_loader():
    from sglang_simulator.workload import load_inprocess_workload

    dataset = load_inprocess_workload(
        name="trace",
        model_path="unused",
        dataset_path=str(ROOT / "workloads/trace.autobench.example.jsonl"),
        num_prompts=100,
        input_len=1,
        output_len=1,
        timestamp_scale=1000.0,
    )
    assert len(dataset) == 100
    assert dataset[0].custom_params["created_time"] == 0
    assert round(dataset[-1].custom_params["created_time"], 6) == 149.59982


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
    target = json.load(open(ROOT / "configs/glm5-p-b300/simulator.aic.json"))
    assert target["scheduler"]["tp_size"] == 8


def test_acceptance_script_has_completion_sentinel():
    script = ROOT / "scripts/acceptance.sh"
    assert script.read_text().rstrip().endswith(
        "# SGLANG_SIMULATOR_ACCEPTANCE_COMPLETE"
    )
