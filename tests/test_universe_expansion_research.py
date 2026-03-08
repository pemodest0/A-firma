from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "validation" / "run_universe_expansion_research.py"
_SPEC = importlib.util.spec_from_file_location("universe_expansion_research", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_is_research_equity_name_filters_bad_instruments() -> None:
    assert _MODULE.is_research_equity_name("Acme Corp Common Stock")
    assert not _MODULE.is_research_equity_name("Acme Corp Warrant")
    assert not _MODULE.is_research_equity_name("Acme ETF Trust")
    assert not _MODULE.is_research_equity_name("Acme DepositArY Shares")


def test_normalize_group_maps_known_sectors() -> None:
    assert _MODULE.normalize_group("Technology") == "technology"
    assert _MODULE.normalize_group("Health Care") == "health_care"
    assert _MODULE.normalize_group("Telecommunications") == "telecommunications"
    assert _MODULE.normalize_group("Basic Materials") == "materials"


def test_round_robin_by_sector_spreads_selection() -> None:
    df = pd.DataFrame(
        [
            {"symbol": "A1", "group": "technology", "market_cap_num": 100.0},
            {"symbol": "A2", "group": "technology", "market_cap_num": 90.0},
            {"symbol": "B1", "group": "energy", "market_cap_num": 95.0},
            {"symbol": "B2", "group": "energy", "market_cap_num": 85.0},
            {"symbol": "C1", "group": "financials", "market_cap_num": 80.0},
        ]
    )
    out = _MODULE._round_robin_by_sector(df, limit=4)
    assert out["symbol"].tolist() == ["B1", "C1", "A1", "B2"]
