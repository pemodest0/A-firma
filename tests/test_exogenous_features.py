from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine.portfolio.exogenous_features import (
    apply_free_energy_penalty,
    build_attractor_persistence_score,
    build_criticality_score,
    build_direction_gradient_score,
    build_exogenous_feature_panel,
    build_market_mode_structure_panel,
    build_state_curvature_score,
    build_structural_stress_signal,
)


def test_build_exogenous_feature_panel_outputs_expected_columns(tmp_path: Path) -> None:
    idx = pd.date_range("2022-01-01", periods=260, freq="D")
    def _make_price(path: Path, level: float, drift: float) -> None:
        r = np.full(len(idx), drift, dtype=float)
        price = level * np.exp(np.cumsum(r))
        df = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "price": price, "r": r})
        df.to_csv(path, index=False)

    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    for ticker, level, drift in [
        ("^VIX", 20.0, 0.0005),
        ("UUP", 25.0, 0.0002),
        ("HYG", 80.0, 0.0003),
        ("LQD", 110.0, 0.0001),
        ("TLT", 120.0, -0.0001),
        ("SHY", 85.0, 0.00005),
        ("TIP", 105.0, 0.00008),
    ]:
        _make_price(prices_dir / f"{ticker}.csv", level, drift)

    cols = ["BTC-USD", "ETH-USD", "ADA-USD"]
    crypto_r = pd.DataFrame(
        {
            "BTC-USD": np.full(len(idx), 0.001, dtype=float),
            "ETH-USD": np.full(len(idx), 0.0008, dtype=float),
            "ADA-USD": np.full(len(idx), 0.0006, dtype=float),
        },
        index=idx,
    )
    crypto_p = 100.0 * np.exp(crypto_r.cumsum())

    out = build_exogenous_feature_panel(
        prices_dir=prices_dir,
        crypto_returns=crypto_r,
        crypto_prices=crypto_p,
        benchmark_crypto="BTC-USD",
    )
    expected = {
        "funding",
        "open_interest",
        "liquidation",
        "btc_dominance",
        "breadth",
        "critical_slowing_down",
        "crowding",
        "crypto_dependency_risk",
        "VIX",
        "credit_spreads",
        "rates",
        "dollar",
        "liquidity",
        "macro_stress",
        "exogenous_risk",
    }
    assert expected.issubset(set(out.panel.columns))
    assert ((out.panel[list(expected)] >= 0.0) & (out.panel[list(expected)] <= 1.0)).all().all()


def test_build_structural_stress_signal_daily_projection() -> None:
    idx = pd.date_range("2023-01-01", periods=90, freq="D")
    monthly_idx = pd.date_range("2023-02-01", periods=3, freq="MS")
    spectral = pd.DataFrame(
        {
            "lambda1": [3.0, 4.5, 5.5],
            "n_assets": [10, 10, 10],
            "deff": [4.5, 3.2, 2.4],
            "avg_abs_corr": [0.12, 0.18, 0.26],
        },
        index=monthly_idx,
    )
    stress = build_structural_stress_signal(spectral_panel=spectral, index=idx)
    assert stress.index.equals(idx)
    assert stress.notna().sum() > 0
    assert ((stress.dropna() >= 0.0) & (stress.dropna() <= 1.0)).all()
    assert float(stress.loc["2023-03-31"]) >= float(stress.loc["2023-02-05"])


def test_build_market_mode_structure_panel_has_expected_shape() -> None:
    idx = pd.date_range("2023-01-01", periods=180, freq="D")
    rng = np.random.default_rng(23)
    market = rng.normal(0.0004, 0.01, size=len(idx))
    tech = market + rng.normal(0.0, 0.006, size=len(idx))
    fin = 0.7 * market + rng.normal(0.0, 0.007, size=len(idx))
    returns = pd.DataFrame(
        {
            "BTC-USD": market + rng.normal(0.0, 0.012, size=len(idx)),
            "ETH-USD": market + rng.normal(0.0, 0.011, size=len(idx)),
            "AAPL": tech,
            "MSFT": tech + rng.normal(0.0, 0.004, size=len(idx)),
            "JPM": fin,
            "BAC": fin + rng.normal(0.0, 0.004, size=len(idx)),
        },
        index=idx,
    )
    panel = build_market_mode_structure_panel(
        returns=returns,
        sector_map={
            "BTC-USD": "crypto",
            "ETH-USD": "crypto",
            "AAPL": "technology",
            "MSFT": "technology",
            "JPM": "financials",
            "BAC": "financials",
        },
        window=90,
        step=10,
    )
    assert not panel.empty
    assert {
        "market_mode_share",
        "sector_structure_strength",
        "between_structure_strength",
        "sector_rotation_score",
        "residual_dispersion",
        "avg_abs_corr",
        "deff_ratio",
    }.issubset(panel.columns)


def test_build_criticality_score_and_free_energy_penalty_behave() -> None:
    idx = pd.date_range("2023-01-01", periods=120, freq="D")
    structure = pd.DataFrame(
        {
            "market_mode_share": np.linspace(0.18, 0.44, len(idx)),
            "avg_abs_corr": np.linspace(0.10, 0.28, len(idx)),
            "deff_ratio": np.linspace(0.62, 0.28, len(idx)),
            "n_assets": np.full(len(idx), 12, dtype=float),
        },
        index=idx,
    )
    csd = pd.Series(np.linspace(0.2, 0.9, len(idx)), index=idx, dtype=float)
    stress = pd.Series(np.linspace(0.1, 0.8, len(idx)), index=idx, dtype=float)
    criticality = build_criticality_score(
        structure_panel=structure,
        critical_slowing_down=csd,
        structural_stress=stress,
        index=idx,
    )
    assert criticality.index.equals(idx)
    assert float(criticality.iloc[-1]) >= float(criticality.iloc[10])
    base = pd.Series(np.full(len(idx), 0.8, dtype=float), index=idx)
    turnover = pd.Series(np.linspace(0.02, 0.20, len(idx)), index=idx)
    penalized = apply_free_energy_penalty(
        base_score=base,
        turnover=turnover,
        instability=criticality,
        gamma=0.15,
        eta=0.20,
    )
    assert penalized.index.equals(idx)
    assert float(penalized.iloc[-1]) < float(base.iloc[-1])
    assert ((penalized >= 0.0) & (penalized <= 1.0)).all()


def test_direction_persistence_and_curvature_scores_behave() -> None:
    idx = pd.date_range("2023-01-01", periods=160, freq="D")
    structure_daily = pd.DataFrame(
        {
            "market_mode_share_pct": np.linspace(0.72, 0.28, len(idx)),
            "sector_rotation_score": np.linspace(0.35, 0.78, len(idx)),
            "residual_dispersion": np.linspace(0.32, 0.70, len(idx)),
            "structural_stress": np.linspace(0.74, 0.31, len(idx)),
        },
        index=idx,
    )
    criticality = pd.Series(np.linspace(0.78, 0.26, len(idx)), index=idx, dtype=float)
    direction = build_direction_gradient_score(
        structure_panel=structure_daily,
        criticality=criticality,
        structural_stress=structure_daily["structural_stress"],
        index=idx,
    )
    persistence = build_attractor_persistence_score(
        direction_score=direction,
        criticality=criticality,
        index=idx,
        window=21,
    )
    curvature = build_state_curvature_score(
        direction_score=direction,
        criticality=criticality,
        index=idx,
    )
    assert ((direction.dropna() >= 0.0) & (direction.dropna() <= 1.0)).all()
    assert ((persistence.dropna() >= 0.0) & (persistence.dropna() <= 1.0)).all()
    assert ((curvature.dropna() >= 0.0) & (curvature.dropna() <= 1.0)).all()
    assert float(direction.iloc[-1]) > float(direction.iloc[20])
    assert float(persistence.iloc[-1]) >= float(persistence.iloc[40])
    assert float(curvature.iloc[-1]) >= float(curvature.iloc[30])
