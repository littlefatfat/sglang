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


def test_prepare_autobench_trace(tmp_path):
    module = load_script("prepare_autobench_trace.py")
    source = tmp_path / "trace.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "a",
                        "input_ids": [1, 2],
                        "input_length": 2,
                        "output_length": 3,
                        "created_time": 10.0,
                    }
                ),
                json.dumps(
                    {
                        "request_id": "b",
                        "input_ids": [1, 2, 3],
                        "input_length": 3,
                        "output_length": 1,
                        "created_time": 10.25,
                    }
                ),
            ]
        )
        + "\n"
    )
    rows = module.convert_rows(source, {0, 1}, None)
    assert [row["timestamp"] for row in rows] == [0, 250]
    assert rows[1]["prompt_len"] == 3
    assert set(rows[1]) == {"prompt", "prompt_len", "output_len", "timestamp"}


def test_service_random_uses_simulator_benchmark():
    module = load_script("run_service_random.py")
    command = module.benchmark_command(
        30000, "/model", Path("/dataset.json"), 2, 1024, 128, 4
    )
    assert command[1:3] == [
        "-m",
        "sglang_simulator.simulation.bench_serving",
    ]
    assert command[command.index("--simulator-mode") + 1] == "blocking"
    assert command[command.index("--dataset-name") + 1] == "random"
    assert command[command.index("--request-rate") + 1] == "4"
    assert command[command.index("--random-input-len") + 1] == "1024"
    assert command[command.index("--random-output-len") + 1] == "128"
    assert command[command.index("--random-range-ratio") + 1] == "1"
