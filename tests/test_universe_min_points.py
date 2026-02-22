from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "validation" / "04_universe_mini.py"
_SPEC = importlib.util.spec_from_file_location("universe_mini_04", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

MIN_POINTS_DEFAULT = _MODULE.MIN_POINTS_DEFAULT
_min_points_for_timeframe = _MODULE._min_points_for_timeframe


def test_min_points_uses_timeframe_override() -> None:
    cfg = {
        "min_points_default": 150,
        "min_points_by_timeframe": {"daily": 200, "weekly": 120, "monthly": 60},
    }
    assert _min_points_for_timeframe(cfg, "daily") == 200
    assert _min_points_for_timeframe(cfg, "weekly") == 120
    assert _min_points_for_timeframe(cfg, "monthly") == 60


def test_min_points_falls_back_to_default() -> None:
    cfg = {"min_points_default": 99, "min_points_by_timeframe": {"daily": 180}}
    assert _min_points_for_timeframe(cfg, "hourly") == 99
    assert _min_points_for_timeframe({}, "daily") == MIN_POINTS_DEFAULT
