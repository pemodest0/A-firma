import pandas as pd

from engine.portfolio.concentration_controls import (
    CryptoConcentrationConfig,
    apply_conditional_crypto_cap,
    compute_domain_concentration,
    crypto_concentration_risk,
)


def test_compute_domain_concentration_with_top_level_weights():
    weights = pd.DataFrame(
        [
            {"crypto": 0.8, "equity": 0.1, "cash": 0.1},
            {"crypto": 0.6, "equity": 0.3, "cash": 0.1},
        ]
    )
    result = compute_domain_concentration(weights)
    assert round(result["crypto"], 4) == 0.7
    assert round(result["equity"], 4) == 0.2
    assert round(result["cash"], 4) == 0.1


def test_crypto_concentration_risk_increases_with_fragility():
    weights = pd.DataFrame([{"crypto": 0.85, "equity": 0.1, "cash": 0.05}])
    low = crypto_concentration_risk(
        {"liquidation": 0.1, "breadth": 0.8, "structural_stress": 0.1, "crypto_volatility": 0.1},
        weights,
    )
    high = crypto_concentration_risk(
        {"liquidation": 0.8, "breadth": 0.2, "structural_stress": 0.8, "crypto_volatility": 0.8},
        weights,
    )
    assert high > low


def test_apply_conditional_crypto_cap_moves_overflow_to_equity_and_cash():
    cfg = CryptoConcentrationConfig(
        max_crypto_weight_normal=0.9,
        max_crypto_weight_stressed=0.5,
        crypto_risk_threshold=0.6,
    )
    weights = pd.DataFrame([{"crypto": 0.8, "equity": 0.15, "cash": 0.05}])
    capped = apply_conditional_crypto_cap(weights, risk_score=0.8, config=cfg)
    assert float(capped.iloc[0]["crypto"]) <= 0.5 + 1e-9
    assert abs(float(capped.sum(axis=1).iloc[0]) - 1.0) < 1e-9
    assert float(capped.iloc[0]["equity"]) > 0.15
    assert float(capped.iloc[0]["cash"]) > 0.05
