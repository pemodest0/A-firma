from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "run_canonical_systematic_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_canonical_systematic_eval", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_load_allowed_assets_reads_asset_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "allowed_assets.csv"
    pd.DataFrame({"asset": ["AAA", " BBB ", "", None]}).to_csv(csv_path, index=False)

    allowed = _MODULE._load_allowed_assets(str(csv_path))

    assert allowed == {"AAA", "BBB"}


def test_build_snapshots_filters_to_execution_universe(tmp_path: Path) -> None:
    impact_csv = tmp_path / "impact.csv"
    pd.DataFrame(
        [
            {"date": "2026-01-15", "asset_id": "AAA", "impact_global": 0.9, "global_score": 0.7, "regime": "stable", "sector": "technology"},
            {"date": "2026-01-15", "asset_id": "BBB", "impact_global": 0.8, "global_score": 0.7, "regime": "stable", "sector": "financials"},
            {"date": "2026-01-15", "asset_id": "CCC", "impact_global": 0.7, "global_score": 0.7, "regime": "stable", "sector": "industrials"},
        ]
    ).to_csv(impact_csv, index=False)

    by_month, state = _MODULE._build_snapshots(
        impact_csv,
        max_assets_per_month=10,
        allowed_assets={"AAA", "CCC"},
    )

    assert list(by_month.keys()) == ["2026-01"]
    assert by_month["2026-01"]["asset_id"].tolist() == ["AAA", "CCC"]
    assert state.index.tolist() == ["2026-01"]
    assert state.loc["2026-01", "regime"] == "stable"
