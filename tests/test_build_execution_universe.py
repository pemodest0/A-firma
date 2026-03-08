from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "build_execution_universe.py"
_SPEC = importlib.util.spec_from_file_location("build_execution_universe", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_default_caps_cover_key_equity_groups() -> None:
    caps = _MODULE.DEFAULT_GROUP_CAPS
    assert caps["industrials"] == 40
    assert caps["technology"] == 40
    assert caps["equities_br_bluechips"] == 24


def test_default_excluded_groups_remove_non_execution_buckets() -> None:
    excluded = set(_MODULE.DEFAULT_EXCLUDED_GROUPS)
    for group in ["bonds_credit", "bonds_rates", "commodities", "crypto", "fx", "miscellaneous", "vol_regime"]:
        assert group in excluded


def test_default_force_include_assets_cover_global_sleeve() -> None:
    forced = set(_MODULE.DEFAULT_FORCE_INCLUDE_ASSETS)
    for asset in ["SPY", "QQQ", "VTI", "VT", "EFA", "EEM", "XLF", "XLK", "XLV"]:
        assert asset in forced
