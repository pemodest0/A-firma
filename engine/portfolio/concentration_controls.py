from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CryptoConcentrationConfig:
    max_crypto_weight_normal: float = 0.90
    max_crypto_weight_stressed: float = 0.55
    crypto_risk_threshold: float = 0.62

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_domain_concentration(weights: pd.DataFrame, metadata: pd.DataFrame | None = None) -> dict[str, float]:
    if weights.empty:
        return {}
    cols = [c for c in weights.columns if c in {"crypto", "equity", "cash"}]
    if cols:
        totals = pd.to_numeric(weights[cols].mean(), errors="coerce").fillna(0.0).astype(float)
        return {str(k): float(v) for k, v in totals.items()}
    if metadata is None or metadata.empty:
        return {}
    meta = metadata.copy()
    if "ticker" not in meta.columns or "asset_group" not in meta.columns:
        return {}
    out: dict[str, float] = {}
    for domain, sub in meta.groupby("asset_group", sort=True):
        tickers = [ticker for ticker in sub["ticker"].astype(str).tolist() if ticker in weights.columns]
        if not tickers:
            continue
        out[str(domain)] = float(pd.to_numeric(weights[tickers], errors="coerce").fillna(0.0).sum(axis=1).mean())
    return out


def crypto_concentration_risk(signal_bundle: dict[str, Any], weights: pd.DataFrame, metadata: pd.DataFrame | None = None) -> float:
    concentration = compute_domain_concentration(weights, metadata)
    crypto_weight = float(concentration.get("crypto", 0.0))
    liquidation = float(signal_bundle.get("liquidation", 0.0) or 0.0)
    breadth = float(signal_bundle.get("breadth", 0.5) or 0.5)
    structural = float(signal_bundle.get("structural_stress", 0.0) or 0.0)
    volatility = float(signal_bundle.get("crypto_volatility", 0.0) or 0.0)
    risk = (
        0.42 * np.clip(crypto_weight, 0.0, 1.0)
        + 0.22 * np.clip(liquidation, 0.0, 1.0)
        + 0.16 * np.clip(structural, 0.0, 1.0)
        + 0.10 * np.clip(volatility, 0.0, 1.0)
        + 0.10 * np.clip(1.0 - breadth, 0.0, 1.0)
    )
    return float(np.clip(risk, 0.0, 1.0))


def apply_conditional_crypto_cap(weights: pd.DataFrame, risk_score: float, config: CryptoConcentrationConfig) -> pd.DataFrame:
    if weights.empty or "crypto" not in weights.columns:
        return weights.copy()
    out = weights.copy().astype(float)
    max_crypto = (
        float(config.max_crypto_weight_stressed)
        if float(risk_score) >= float(config.crypto_risk_threshold)
        else float(config.max_crypto_weight_normal)
    )
    capped_crypto = pd.to_numeric(out["crypto"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=max_crypto)
    overflow = pd.to_numeric(out["crypto"], errors="coerce").fillna(0.0) - capped_crypto
    out["crypto"] = capped_crypto
    if "equity" in out.columns:
        out["equity"] = pd.to_numeric(out["equity"], errors="coerce").fillna(0.0) + overflow * 0.35
    if "cash" in out.columns:
        out["cash"] = pd.to_numeric(out["cash"], errors="coerce").fillna(0.0) + overflow * 0.65
    total = out.sum(axis=1).replace(0.0, np.nan)
    out = out.div(total, axis=0).fillna(0.0)
    return out
