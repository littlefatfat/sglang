import asyncio
import importlib.util
import json
import sys
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


def test_trace_contract():
    module = load_script("run_inprocess.py")
    dataset = module.load_trace(ROOT / "workloads/trace.example.jsonl", 1.0)
    assert len(dataset) == 3
    assert dataset[0].custom_params["created_time"] == 0
    assert dataset[2].custom_params["created_time"] == 0.2


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
