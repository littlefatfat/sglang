import inspect
import json
import os
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def configure_environment(
    hisim_config: str | Path,
    output_dir: str | Path,
    mode: str,
) -> Path:
    mode = mode.upper()
    if mode not in {"OFFLINE", "BLOCKING"}:
        raise ValueError("mode must be OFFLINE or BLOCKING")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(Path(hisim_config).resolve())
    os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(output)
    os.environ["SGLANG_SIMULATOR_HICACHE_STORAGE_KEYS_PATH"] = str(
        output / "hicache_storage_keys.txt"
    )
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = mode
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return output


def build_server_args(path: str | Path):
    from sglang.srt.server_args import ServerArgs

    raw = load_json(path)
    raw.pop("_comment", None)
    raw.pop("version", None)
    kv_bytes = raw.pop("kv_bytes_per_token_per_gpu", None)
    if "max_total_num_tokens" in raw:
        raw["max_total_tokens"] = raw.pop("max_total_num_tokens")
    raw.update(
        {
            "load_format": "dummy",
            "device": "cpu",
            "disable_cuda_graph": True,
        }
    )
    allowed = set(inspect.signature(ServerArgs).parameters)
    kwargs = {k: v for k, v in raw.items() if k in allowed and v is not None}
    args = ServerArgs(**kwargs)
    if kv_bytes is not None:
        setattr(args, "kv_bytes_per_token_per_gpu", kv_bytes)
    return args
