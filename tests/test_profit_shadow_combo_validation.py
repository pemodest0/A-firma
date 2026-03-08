from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_shadow_combo_validation import _combine_rows, choose_meta_source


def test_choose_meta_source_prefers_challenger_in_bull() -> None:
    main = pd.Series([0.01, 0.01, 0.01], dtype=float)
    challenger = pd.Series([0.02, 0.02, 0.02], dtype=float)

    out = choose_meta_source(
        ym="2026-04",
        pos=3,
        market_state_by_month={"2026-03": "bull"},
        main_ret=main,
        challenger_ret=challenger,
        lookback_months=3,
    )

    assert out == "challenger"


def test_choose_meta_source_prefers_trailing_winner_in_sideways() -> None:
    main = pd.Series([0.01, 0.01, 0.01, 0.0], dtype=float)
    challenger = pd.Series([0.02, 0.02, 0.02, 0.0], dtype=float)

    out = choose_meta_source(
        ym="2026-04",
        pos=3,
        market_state_by_month={"2026-03": "sideways"},
        main_ret=main,
        challenger_ret=challenger,
        lookback_months=3,
    )

    assert out == "challenger"


def test_combine_rows_averages_weights_and_cash() -> None:
    row_main = pd.Series({"ym": "2026-02", "executed_weights_json": "{\"AAA\": 0.6}", "cash_weight": 0.4, "hedge_weight": 0.0})
    row_challenger = pd.Series({"ym": "2026-02", "executed_weights_json": "{\"BBB\": 1.0}", "cash_weight": 0.0, "hedge_weight": 0.0})

    out = _combine_rows(row_main, row_challenger, alpha_main=0.5)

    assert out["ym"] == "2026-02"
    assert out["cash_weight"] == 0.2
    assert "\"AAA\": 0.3" in out["executed_weights_json"]
    assert "\"BBB\": 0.5" in out["executed_weights_json"]
