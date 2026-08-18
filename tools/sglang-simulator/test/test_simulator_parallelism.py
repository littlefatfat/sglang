import argparse
import json

import pytest
from sglang_simulator.compat import (
    SIMULATOR_SERVER_ARG_OVERRIDES,
    apply_simulator_server_args,
    validate_simulator_server_args,
)
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.types import SchedulerConfig

PARALLEL_FIELDS = (
    "tp_size",
    "ep_size",
    "dp_size",
    "pp_size",
    "attn_cp_size",
    "dcp_size",
)


def test_runtime_parallelism_is_forced_to_one():
    args = argparse.Namespace(**{name: 8 for name in PARALLEL_FIELDS})

    apply_simulator_server_args(args)

    assert all(getattr(args, name) == 1 for name in PARALLEL_FIELDS)
    assert all(SIMULATOR_SERVER_ARG_OVERRIDES[name] == 1 for name in PARALLEL_FIELDS)
    validate_simulator_server_args(args)


def test_runtime_parallelism_validation_rejects_bypassed_entry_point():
    args = argparse.Namespace(**SIMULATOR_SERVER_ARG_OVERRIDES)
    args.tp_size = 4

    with pytest.raises(RuntimeError, match="tp_size=4"):
        validate_simulator_server_args(args)


def test_modeled_parallelism_overrides_runtime(monkeypatch, tmp_path):
    config_path = tmp_path / "simulator.json"
    config_path.write_text(
        json.dumps(
            {
                "scheduler": {
                    "tp_size": 8,
                    "ep_size": 4,
                    "dp_size": 1,
                    "pp_size": 2,
                    "cp_size": 2,
                    "cp_style": "ring",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(ConfigManager, "_raw_config", None)
    monkeypatch.setattr(ConfigManager, "_scheduler_config", None)

    runtime_config = SchedulerConfig(
        tp_size=1, ep_size=1, dp_size=1, pp_size=1, cp_size=1
    )
    ConfigManager.set_scheduler_config(runtime_config)
    modeled = ConfigManager.get_scheduler_config()

    assert modeled.tp_size == 8
    assert modeled.ep_size == 4
    assert modeled.dp_size == 1
    assert modeled.pp_size == 2
    assert modeled.cp_size == 2
    assert modeled.cp_style == "ring"
    assert modeled.attn_tp_size == 4
    assert isinstance(modeled.attn_tp_size, int)
    assert modeled.moe_tp_size == 2
    assert isinstance(modeled.moe_tp_size, int)


def test_modeled_parallelism_requires_integral_groups():
    invalid_attention = SchedulerConfig(tp_size=8, dp_size=1, cp_size=3)
    with pytest.raises(ValueError, match="dp_size \\* cp_size"):
        _ = invalid_attention.attn_tp_size

    invalid_moe = SchedulerConfig(tp_size=8, ep_size=3)
    with pytest.raises(ValueError, match="divisible by ep_size"):
        _ = invalid_moe.moe_tp_size
