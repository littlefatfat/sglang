import inspect
import json
import os
from pathlib import Path


def patch_cpu_device_capability() -> None:
    """Keep model-specific ServerArgs checks usable in CPU simulation."""
    if os.environ.get("SGLANG_USE_CPU_ENGINE") != "1":
        return

    import torch

    torch.cuda.get_device_capability = lambda *_args, **_kwargs: (10, 0)


# Spawn imports this module again before reconstructing ServerArgs. Install the
# shim at import time when the parent has already selected CPU simulation.
patch_cpu_device_capability()


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def configure_environment(
    sim_config: str | Path,
    output_dir: str | Path,
    mode: str,
    device: str = "cpu",
) -> Path:
    mode = mode.upper()
    if mode not in {"OFFLINE", "BLOCKING"}:
        raise ValueError("mode must be OFFLINE or BLOCKING")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(Path(sim_config).resolve())
    os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(output)
    os.environ["SGLANG_SIMULATOR_HICACHE_STORAGE_KEYS_PATH"] = str(
        output / "hicache_storage_keys.txt"
    )
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = mode
    if device == "cpu":
        os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
        patch_cpu_device_capability()
    else:
        os.environ.pop("SGLANG_USE_CPU_ENGINE", None)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return output


def normalize_dummy_engine_topology(raw: dict) -> None:
    """Keep the physical dummy Engine single-process."""
    # SGLang Simulator models the target topology through sim_config.scheduler.
    # Launching TP4/TP8 workers here only adds NUMA/NCCL requirements.
    for key in ("tp_size", "ep_size", "dp_size", "pp_size"):
        raw[key] = 1


def build_server_args(
    path: str | Path,
    device: str = "cpu",
    page_size: int | None = None,
):
    from sglang.srt.server_args import ServerArgs

    raw = load_json(path)
    raw.pop("_comment", None)
    raw.pop("version", None)
    if page_size is not None:
        raw["page_size"] = page_size
    kv_bytes = raw.pop("kv_bytes_per_token_per_gpu", None)
    if "max_total_num_tokens" in raw:
        raw["max_total_tokens"] = raw.pop("max_total_num_tokens")
    normalize_dummy_engine_topology(raw)
    raw.update(
        {
            "load_format": "dummy",
            "device": device,
            "disable_cuda_graph": True,
            "attention_backend": "torch_native",
            "sampling_backend": "pytorch",
        }
    )
    allowed = set(inspect.signature(ServerArgs).parameters)
    kwargs = {k: v for k, v in raw.items() if k in allowed and v is not None}
    args = ServerArgs(**kwargs)
    if kv_bytes is not None:
        setattr(args, "kv_bytes_per_token_per_gpu", kv_bytes)
    return args
