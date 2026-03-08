from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "validation" / "run_profit_group_methodology_suite.py"
_SPEC = importlib.util.spec_from_file_location("run_profit_group_methodology_suite", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_infer_asset_jurisdiction_handles_brazil_suffix() -> None:
    assert _MODULE._infer_asset_jurisdiction("PETR4.SA", "equities_br_bluechips") == "br_local"
    assert _MODULE._infer_asset_jurisdiction("SPY", "equities_us_broad") == "foreign"


def test_build_static_combo_equal_weights_monthly_groups() -> None:
    group_monthly = pd.DataFrame(
        {
            "technology": [0.10, 0.00],
            "financials": [0.00, 0.20],
        },
        index=["2026-01", "2026-02"],
    )
    group_meta = pd.DataFrame(
        [
            {"asset_group": "technology", "foreign_share_assets": 1.0},
            {"asset_group": "financials", "foreign_share_assets": 0.5},
        ]
    )

    out = _MODULE._build_static_combo(group_monthly, group_meta, ("technology", "financials"))

    assert out["gross_monthly"].round(6).tolist() == [0.05, 0.10]
    assert round(float(out["foreign_share_monthly"].iloc[0]), 6) == 0.75


def test_candidate_status_prefers_positive_net_edge() -> None:
    assert _MODULE._candidate_status(0.10, 0.6, 48) == "keep"
    assert _MODULE._candidate_status(-0.05, 0.4, 48) == "watch"
    assert _MODULE._candidate_status(-0.20, 0.2, 48) == "kill"
