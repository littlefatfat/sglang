#!/usr/bin/env python3
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import sglang

    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "sglang",
            "sglang_simulator",
            "transformers",
            "sglang-kernel",
        )
    }
    print("sglang_version", versions["sglang"])
    print("sglang_source", sglang.__file__)
    print("simulator_version", versions["sglang_simulator"])
    print("transformers_version", versions["transformers"])
    print("sgl_kernel_version", versions["sglang-kernel"])
    assert versions["sglang"].startswith(("0.5.16", "0.5.17.dev")), versions["sglang"]
    assert versions["transformers"] == "5.12.1", versions["transformers"]
    assert versions["sglang-kernel"] == "0.4.5", versions["sglang-kernel"]

    required = (
        Path("/nfs/Qwen/Qwen3-8B"),
        Path("/nfs/Qwen/Qwen3-32B-FP8"),
        Path("/nfs/ZhipuAI/GLM-5.1-FP8"),
        Path("/nfs/deepseek-ai/DeepSeek-V4-Pro"),
    )
    for path in required:
        print("model", "OK" if path.exists() else "MISSING", path)
        assert path.exists(), path

    ml_model_path = os.environ.get("SGLANG_SIMULATOR_ML_MODEL_PATH")
    assert ml_model_path, "SGLANG_SIMULATOR_ML_MODEL_PATH is required"
    predictor_inputs = (
        Path("/host/aiconfigurator/src/aiconfigurator/systems"),
        Path(os.path.expandvars(os.path.expanduser(ml_model_path))),
    )
    for path in predictor_inputs:
        print("predictor", "OK" if path.exists() else "MISSING", path)
        assert path.exists(), path

    for path in sorted((ROOT / "configs").rglob("*.json")):
        json.load(open(path, encoding="utf-8"))
        print("json", "OK", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
