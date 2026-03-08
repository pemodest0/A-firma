from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from execution.net_assumptions import load_net_assumption_profiles


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "validation" / "run_profit_group_oos_validation.py"
_SPEC = importlib.util.spec_from_file_location("run_profit_group_oos_validation", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_eval_slice_returns_net_metrics() -> None:
    candidate = pd.DataFrame(
        {
            "ym": ["2025-01", "2025-02", "2025-03"],
            "gross_ret": [0.10, -0.02, 0.05],
            "turnover": [0.2, 0.0, 0.2],
            "foreign_share": [1.0, 1.0, 1.0],
        }
    )
    spy = pd.Series([0.03, -0.01, 0.02], index=["2025-01", "2025-02", "2025-03"], dtype=float)
    profiles = load_net_assumption_profiles(Path("config/profit_net_assumptions.json"))

    out = _MODULE._eval_slice(candidate, spy, profiles)

    assert out["months"] == 3
    assert out["net_ann_return"] > 0.0
    assert out["edge_vs_spy_net_total_return"] > 0.0
