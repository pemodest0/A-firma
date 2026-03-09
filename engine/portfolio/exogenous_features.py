from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_series(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(np.nan, index=index, dtype=float)
    out = pd.to_numeric(values.reindex(index), errors="coerce").astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)


def _tail_return(series: pd.Series | pd.DataFrame, lookback: int) -> pd.Series | pd.DataFrame:
    min_periods = max(10, int(lookback) // 3)
    if isinstance(series, pd.DataFrame):
        values = series.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        return (1.0 + values).rolling(int(lookback), min_periods=min_periods).apply(np.prod, raw=True) - 1.0
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return (1.0 + values).rolling(int(lookback), min_periods=min_periods).apply(np.prod, raw=True) - 1.0


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.empty:
        return values
    min_periods = max(10, int(window) // 3)

    def _pct(arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return float("nan")
        return float(np.mean(arr <= float(arr[-1])))

    out = values.rolling(int(window), min_periods=min_periods).apply(_pct, raw=True)
    return out.replace([np.inf, -np.inf], np.nan)


def _realized_vol(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return values.rolling(int(window), min_periods=max(10, int(window) // 3)).std(ddof=0)


def _load_market_csv(prices_dir: Path, ticker: str) -> pd.DataFrame | None:
    path = Path(prices_dir) / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    if "r" not in df.columns:
        if "price" in df.columns:
            price = pd.to_numeric(df["price"], errors="coerce").astype(float)
            df["r"] = np.log(price / price.shift(1))
        else:
            return None
    if "price" not in df.columns:
        ret = pd.to_numeric(df["r"], errors="coerce").fillna(0.0).astype(float)
        df["price"] = float(100.0) * np.exp(ret.cumsum())
    df["r"] = pd.to_numeric(df["r"], errors="coerce").astype(float)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    return df[["price", "r"]].copy()


def load_market_series(prices_dir: Path, ticker: str, index: pd.Index) -> tuple[pd.Series, pd.Series]:
    df = _load_market_csv(prices_dir, ticker)
    if df is None:
        empty = pd.Series(np.nan, index=index, dtype=float)
        return empty, empty
    price = pd.to_numeric(df["price"].reindex(index), errors="coerce").astype(float)
    ret = pd.to_numeric(df["r"].reindex(index), errors="coerce").astype(float)
    return price, ret


def build_crypto_breadth(prices: pd.DataFrame, returns: pd.DataFrame, *, lookback: int = 21, ma_days: int = 100) -> pd.Series:
    idx = prices.index.intersection(returns.index)
    if idx.empty:
        return pd.Series(dtype=float)
    p = prices.reindex(idx).astype(float)
    r = returns.reindex(idx).astype(float)
    momentum = _tail_return(r.mean(axis=1), lookback)
    ma_ok = p.gt(p.rolling(ma_days, min_periods=max(20, ma_days // 2)).mean()).astype(float)
    pos_ok = (_tail_return(r, lookback) > 0.0).astype(float)
    breadth = 0.5 * ma_ok.mean(axis=1) + 0.5 * pos_ok.mean(axis=1)
    breadth = breadth.fillna(0.0)
    # Slightly reward broad positive participation when basket momentum is also positive.
    breadth = (0.85 * breadth + 0.15 * ((momentum + 0.10) / 0.20).clip(0.0, 1.0)).clip(0.0, 1.0)
    return breadth.astype(float)


@dataclass(frozen=True)
class ExogenousFeaturePanel:
    panel: pd.DataFrame
    crypto_columns: list[str]
    macro_columns: list[str]
    risk_columns: list[str]


def build_exogenous_feature_panel(
    *,
    prices_dir: Path,
    crypto_returns: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    benchmark_crypto: str = "BTC-USD",
    macro_index: pd.Index | None = None,
) -> ExogenousFeaturePanel:
    idx = crypto_returns.index
    if macro_index is not None:
        idx = idx.intersection(macro_index)
    if idx.empty:
        return ExogenousFeaturePanel(pd.DataFrame(index=idx), [], [], [])

    c_returns = crypto_returns.reindex(idx).astype(float)
    c_prices = crypto_prices.reindex(idx).astype(float)
    breadth = build_crypto_breadth(c_prices, c_returns)

    btc_ret = pd.to_numeric(c_returns.get(benchmark_crypto), errors="coerce").reindex(idx).fillna(0.0).astype(float)
    alt_cols = [c for c in c_returns.columns if c != str(benchmark_crypto)]
    alt_ret = c_returns[alt_cols].mean(axis=1) if alt_cols else pd.Series(0.0, index=idx)
    basket_ret = c_returns.mean(axis=1)

    btc_fast = _tail_return(btc_ret, 21)
    alt_fast = _tail_return(alt_ret, 21)
    basket_fast = _tail_return(basket_ret, 21)
    basket_vol = _realized_vol(basket_ret, 21)

    btc_dominance = _rolling_percentile((btc_fast - alt_fast).clip(lower=-0.25, upper=0.25), 63).fillna(0.5)
    funding = _rolling_percentile((0.50 * basket_fast.clip(lower=0.0) + 0.30 * basket_vol + 0.20 * btc_dominance), 63).fillna(0.5)
    open_interest = _rolling_percentile((0.45 * basket_fast.abs() + 0.35 * basket_vol + 0.20 * breadth), 63).fillna(0.5)
    liquidation = _rolling_percentile(
        ((-basket_ret).clip(lower=0.0) + 0.5 * basket_vol + 0.5 * (1.0 - breadth)).clip(lower=0.0),
        63,
    ).fillna(0.5)
    dependency_risk = (
        0.28 * funding
        + 0.22 * open_interest
        + 0.22 * liquidation
        + 0.18 * btc_dominance
        + 0.10 * (1.0 - breadth)
    ).clip(0.0, 1.0)

    vix_price, vix_ret = load_market_series(prices_dir, "^VIX", idx)
    uup_price, uup_ret = load_market_series(prices_dir, "UUP", idx)
    hyg_price, hyg_ret = load_market_series(prices_dir, "HYG", idx)
    lqd_price, lqd_ret = load_market_series(prices_dir, "LQD", idx)
    tlt_price, tlt_ret = load_market_series(prices_dir, "TLT", idx)
    shy_price, shy_ret = load_market_series(prices_dir, "SHY", idx)
    tip_price, tip_ret = load_market_series(prices_dir, "TIP", idx)

    vix = _rolling_percentile(vix_price.ffill(), 126).fillna(0.5)
    credit_spread = _rolling_percentile((_tail_return(lqd_ret, 21) - _tail_return(hyg_ret, 21)), 126).fillna(0.5)
    rates = _rolling_percentile((_tail_return(shy_ret, 21) - _tail_return(tlt_ret, 21)), 126).fillna(0.5)
    dollar = _rolling_percentile(_tail_return(uup_ret, 21), 126).fillna(0.5)
    liquidity_stress = _clip01(0.40 * vix + 0.25 * credit_spread + 0.20 * dollar + 0.15 * rates)
    liquidity = (1.0 - liquidity_stress).clip(0.0, 1.0)

    panel = pd.DataFrame(
        {
            "funding": _clip01(funding),
            "open_interest": _clip01(open_interest),
            "liquidation": _clip01(liquidation),
            "btc_dominance": _clip01(btc_dominance),
            "breadth": _clip01(breadth),
            "crypto_dependency_risk": _clip01(dependency_risk),
            "VIX": _clip01(vix),
            "credit_spreads": _clip01(credit_spread),
            "rates": _clip01(rates),
            "dollar": _clip01(dollar),
            "liquidity": _clip01(liquidity),
            "macro_stress": _clip01(liquidity_stress),
            "inflation_liquidity": _clip01(_rolling_percentile(_tail_return(tip_ret - shy_ret, 21), 126).fillna(0.5)),
        },
        index=idx,
    ).astype(float)
    panel["macro_stress"] = _clip01(0.35 * panel["VIX"] + 0.25 * panel["credit_spreads"] + 0.20 * panel["dollar"] + 0.20 * panel["rates"])
    panel["exogenous_risk"] = _clip01(0.55 * panel["crypto_dependency_risk"] + 0.45 * panel["macro_stress"])

    crypto_cols = ["funding", "open_interest", "liquidation", "btc_dominance", "breadth", "crypto_dependency_risk"]
    macro_cols = ["VIX", "credit_spreads", "rates", "dollar", "liquidity", "macro_stress", "inflation_liquidity"]
    risk_cols = ["crypto_dependency_risk", "macro_stress", "exogenous_risk"]
    return ExogenousFeaturePanel(panel=panel, crypto_columns=crypto_cols, macro_columns=macro_cols, risk_columns=risk_cols)


def adjust_confidence_with_feature(
    *,
    base_score: pd.Series,
    feature: pd.Series,
    mode: str,
    weight: float,
) -> pd.Series:
    base = pd.to_numeric(base_score, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    feat = pd.to_numeric(feature.reindex(base.index), errors="coerce").fillna(0.5).clip(0.0, 1.0).astype(float)
    w = float(max(0.0, weight))
    key = str(mode).strip().lower()
    if key in {"penalty", "risk", "stress"}:
        adjusted = base - w * feat
    elif key in {"boost", "support", "positive"}:
        adjusted = base + w * (feat - 0.5)
    elif key in {"breadth_boost"}:
        adjusted = base + w * (feat - 0.45)
    else:
        adjusted = base
    return adjusted.clip(0.0, 1.0).astype(float)


def feature_spectral_extremes(
    *,
    feature_panel: pd.DataFrame,
    spectral_panel: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if feature_panel.empty or spectral_panel.empty:
        return pd.DataFrame(rows)
    joined = feature_panel.join(spectral_panel, how="inner")
    if joined.empty:
        return pd.DataFrame(rows)
    for col in feature_cols:
        s = pd.to_numeric(joined[col], errors="coerce")
        if s.dropna().size < 20:
            continue
        q_hi = float(s.quantile(0.80))
        q_lo = float(s.quantile(0.20))
        hi = joined[s >= q_hi]
        lo = joined[s <= q_lo]
        if hi.empty or lo.empty:
            continue
        rows.append(
            {
                "feature": str(col),
                "p1_high": float(pd.to_numeric(hi["p1"], errors="coerce").mean()),
                "p1_low": float(pd.to_numeric(lo["p1"], errors="coerce").mean()),
                "p1_diff": float(pd.to_numeric(hi["p1"], errors="coerce").mean() - pd.to_numeric(lo["p1"], errors="coerce").mean()),
                "deff_high": float(pd.to_numeric(hi["deff"], errors="coerce").mean()),
                "deff_low": float(pd.to_numeric(lo["deff"], errors="coerce").mean()),
                "deff_diff": float(pd.to_numeric(hi["deff"], errors="coerce").mean() - pd.to_numeric(lo["deff"], errors="coerce").mean()),
                "avg_abs_corr_high": float(pd.to_numeric(hi["avg_abs_corr"], errors="coerce").mean()),
                "avg_abs_corr_low": float(pd.to_numeric(lo["avg_abs_corr"], errors="coerce").mean()),
                "avg_abs_corr_diff": float(pd.to_numeric(hi["avg_abs_corr"], errors="coerce").mean() - pd.to_numeric(lo["avg_abs_corr"], errors="coerce").mean()),
                "lambda1_high": float(pd.to_numeric(hi["lambda1"], errors="coerce").mean()),
                "lambda1_low": float(pd.to_numeric(lo["lambda1"], errors="coerce").mean()),
                "lambda1_diff": float(pd.to_numeric(hi["lambda1"], errors="coerce").mean() - pd.to_numeric(lo["lambda1"], errors="coerce").mean()),
                "n_high": float(len(hi)),
                "n_low": float(len(lo)),
            }
        )
    return pd.DataFrame(rows)
