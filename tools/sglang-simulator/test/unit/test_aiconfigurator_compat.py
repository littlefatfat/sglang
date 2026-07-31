import pytest

pytest.importorskip("aiconfigurator")

from aiconfigurator.sdk.common import (
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)
from sglang_simulator.time_predictor.aiconfigurator import (
    _install_legacy_float16_quant_mode_aliases,
    _relax_legacy_nearest_1d_interpolation,
)


class LegacyDatabase:
    def __init__(self):
        self.calls = []

    def _nearest_1d_point_helper(self, x, values, inner_only=True):
        self.calls.append((x, values, inner_only))
        return values[0], values[-1]


class CurrentDatabase:
    pass


def test_relax_legacy_nearest_interpolation_defaults_to_extrapolation():
    database = LegacyDatabase()

    _relax_legacy_nearest_1d_interpolation(database)

    assert database._nearest_1d_point_helper(3, [1, 5]) == (1, 5)
    assert database.calls == [(3, [1, 5], False)]


def test_relax_legacy_nearest_interpolation_preserves_explicit_inner_only():
    database = LegacyDatabase()

    _relax_legacy_nearest_1d_interpolation(database)

    database._nearest_1d_point_helper(3, [1, 5], inner_only=True)
    assert database.calls == [(3, [1, 5], True)]


def test_relax_legacy_nearest_interpolation_accepts_aic_010_database():
    _relax_legacy_nearest_1d_interpolation(CurrentDatabase())


def test_install_legacy_float16_quant_mode_aliases():
    _install_legacy_float16_quant_mode_aliases()

    for enum_class in (
        GEMMQuantMode,
        MoEQuantMode,
        FMHAQuantMode,
        KVCacheQuantMode,
    ):
        assert enum_class["float16"] is enum_class.bfloat16
