from __future__ import annotations

import pandas as pd

from execution.shadow_gods_historical import _max_drawdown, _proxy_operation_fields, _year_slice


def test_proxy_operation_fields_maps_stress_to_protection() -> None:
    payload = _proxy_operation_fields(regime="stress", gross_exposure=0.72)
    assert payload["recommended_mode"] == "proteção"
    assert payload["vigilance_status"] == "warn"
    assert payload["posture"] == "protecao"


def test_max_drawdown_uses_running_peak() -> None:
    assert _max_drawdown(pd.Series([100.0, 120.0, 90.0, 95.0], dtype=float)) == -0.25


def test_year_slice_filters_rows_by_prefix() -> None:
    rows = [
        {"as_of_date": "2023-01-02", "value": 1},
        {"as_of_date": "2024-01-03", "value": 2},
        {"date": "2023-12-31", "value": 3},
    ]
    filtered = _year_slice(rows, 2023)
    assert [row["value"] for row in filtered] == [1, 3]
