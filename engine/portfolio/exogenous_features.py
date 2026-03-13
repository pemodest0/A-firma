from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from engine.structural.covariance_estimators import estimate_corr


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


def _rolling_lag1_autocorr(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    min_periods = max(12, int(window) // 2)

    def _acf1(arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            return float("nan")
        x = arr[:-1]
        y = arr[1:]
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    return values.rolling(int(window), min_periods=min_periods).apply(_acf1, raw=True)


def _sign_persistence(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    signs = np.sign(values)
    same = pd.Series((signs == np.roll(signs, 1)).astype(float), index=values.index)
    same.iloc[0] = np.nan
    return same.rolling(int(window), min_periods=max(10, int(window) // 2)).mean()


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


def build_critical_slowing_down_signal(
    *,
    returns: pd.DataFrame,
    benchmark_col: str = "BTC-USD",
    window: int = 63,
) -> pd.Series:
    idx = returns.index
    if idx.empty:
        return pd.Series(dtype=float)
    basket = pd.to_numeric(returns.mean(axis=1), errors="coerce").fillna(0.0).astype(float)
    bench = pd.to_numeric(returns.get(benchmark_col, basket), errors="coerce").reindex(idx).fillna(0.0).astype(float)
    composite = 0.55 * bench + 0.45 * basket
    acf1 = _rolling_percentile(_rolling_lag1_autocorr(composite, window), window).fillna(0.5)
    variance = _rolling_percentile(_realized_vol(composite, window), window).fillna(0.5)
    persistence = _rolling_percentile(_sign_persistence(composite, window), window).fillna(0.5)
    return _clip01(0.45 * acf1 + 0.35 * variance + 0.20 * persistence)


def build_crowding_signal(
    *,
    returns: pd.DataFrame,
    lookback: int = 21,
    vol_window: int = 63,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    tail = _tail_return(returns, lookback)
    vol = returns.apply(lambda s: _realized_vol(s, vol_window))
    standardized = tail.divide(vol.replace(0.0, np.nan))
    standardized = standardized.replace([np.inf, -np.inf], np.nan)

    rows: list[float] = []
    idx = standardized.index
    for ts, row in standardized.iterrows():
        vals = pd.to_numeric(row, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.size < 3:
            rows.append(float("nan"))
            continue
        extreme_share = float((vals >= 1.25).mean())
        pos = vals.clip(lower=0.0)
        if float(pos.sum()) <= 1e-12:
            concentration = 0.0
        else:
            weights = pos / float(pos.sum())
            concentration = float(np.square(weights).sum())
        rows.append(0.60 * extreme_share + 0.40 * concentration)
    raw = pd.Series(rows, index=idx, dtype=float)
    return _clip01(_rolling_percentile(raw, 63).fillna(0.5))


def build_structural_stress_signal(
    *,
    spectral_panel: pd.DataFrame,
    index: pd.Index,
) -> pd.Series:
    if spectral_panel.empty or index.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    spec = spectral_panel.copy()
    if "lambda1_over_n" not in spec.columns and {"lambda1", "n_assets"}.issubset(spec.columns):
        spec["lambda1_over_n"] = pd.to_numeric(spec["lambda1"], errors="coerce") / pd.to_numeric(spec["n_assets"], errors="coerce").replace(0.0, np.nan)
    if "spectral_entropy" not in spec.columns and "deff" in spec.columns:
        deff = pd.to_numeric(spec["deff"], errors="coerce").clip(lower=1.0)
        n_assets = pd.to_numeric(spec.get("n_assets"), errors="coerce").clip(lower=1.0)
        approx = np.log(deff) / np.log(n_assets.where(n_assets > 1.0, 2.0))
        spec["spectral_entropy"] = approx.clip(0.0, 1.0)

    lambda_term = _rolling_percentile(pd.to_numeric(spec.get("lambda1_over_n"), errors="coerce"), 12).fillna(0.5)
    entropy_term = 1.0 - _clip01(pd.to_numeric(spec.get("spectral_entropy"), errors="coerce").fillna(0.5))
    corr_term = _rolling_percentile(pd.to_numeric(spec.get("avg_abs_corr"), errors="coerce"), 12).fillna(0.5)
    stress = _clip01(0.40 * lambda_term + 0.30 * entropy_term + 0.30 * corr_term)
    daily = pd.to_numeric(stress.reindex(index.union(spec.index)).sort_index().ffill().reindex(index), errors="coerce")
    return daily.astype(float)


def build_market_mode_structure_panel(
    *,
    returns: pd.DataFrame,
    sector_map: dict[str, str],
    window: int = 120,
    step: int = 5,
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    clean = returns.apply(pd.to_numeric, errors="coerce").astype(float)
    idx = clean.index
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    upper = None
    for end in range(int(window), len(idx) + 1, max(1, int(step))):
        sub = clean.iloc[end - int(window) : end].dropna(axis=1, thresh=max(60, int(window) // 2)).fillna(0.0)
        cols = [c for c in sub.columns if c in sector_map]
        sub = sub[cols]
        if sub.shape[1] < 6:
            continue
        corr = estimate_corr(sub.to_numpy(dtype=float), method="ledoit_wolf")
        eigvals, eigvecs = np.linalg.eigh(corr)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.real(eigvals[order])
        eigvecs = np.real(eigvecs[:, order])
        total = float(np.sum(eigvals))
        if total <= 1e-12:
            continue
        p = eigvals / total
        deff = float(1.0 / np.sum(np.square(p)))
        market_share = float(p[0])
        v1 = eigvecs[:, 0]
        market_component = float(eigvals[0]) * np.outer(v1, v1)
        residual = corr - market_component
        if upper is None or upper.shape != corr.shape:
            upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sectors = [str(sector_map.get(c, "unknown")) for c in sub.columns]
        same = np.zeros_like(corr, dtype=bool)
        for i, sec_i in enumerate(sectors):
            for j in range(i + 1, len(sectors)):
                if sec_i == sectors[j]:
                    same[i, j] = True
        diff = upper & (~same)
        same_vals = np.abs(residual[same])
        diff_vals = np.abs(residual[diff])
        within_abs = float(np.nanmean(same_vals)) if same_vals.size else 0.0
        between_abs = float(np.nanmean(diff_vals)) if diff_vals.size else 0.0
        residual_abs = float(np.nanmean(np.abs(residual[upper]))) if int(upper.sum()) > 0 else 0.0
        rows.append(
            {
                "date": idx[end - 1],
                "market_mode_share": market_share,
                "sector_structure_strength": within_abs,
                "between_structure_strength": between_abs,
                "sector_rotation_score_raw": within_abs - between_abs,
                "residual_dispersion_raw": max(0.0, 1.0 - residual_abs),
                "avg_abs_corr": float(np.nanmean(np.abs(corr[upper]))) if int(upper.sum()) > 0 else 0.0,
                "deff_ratio": float(deff / max(1.0, sub.shape[1])),
                "n_assets": int(sub.shape[1]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.set_index("date").sort_index()
    out["market_mode_share_pct"] = _rolling_percentile(out["market_mode_share"], 63).fillna(0.5)
    out["sector_rotation_score"] = _rolling_percentile(out["sector_rotation_score_raw"], 63).fillna(0.5)
    out["residual_dispersion"] = _rolling_percentile(out["residual_dispersion_raw"], 63).fillna(0.5)
    return out


def build_criticality_score(
    *,
    structure_panel: pd.DataFrame,
    critical_slowing_down: pd.Series | None = None,
    structural_stress: pd.Series | None = None,
    index: pd.Index | None = None,
) -> pd.Series:
    if structure_panel.empty:
        idx = index if index is not None else pd.Index([], dtype="datetime64[ns]")
        return pd.Series(np.nan, index=idx, dtype=float)
    panel = structure_panel.copy()
    mm = pd.to_numeric(panel["market_mode_share"], errors="coerce").astype(float)
    avg_corr = pd.to_numeric(panel["avg_abs_corr"], errors="coerce").astype(float)
    deff_ratio = pd.to_numeric(panel["deff_ratio"], errors="coerce").astype(float)
    d_market = mm.diff().clip(lower=0.0)
    d_corr = avg_corr.diff().clip(lower=0.0)
    d_deff = (-deff_ratio.diff()).clip(lower=0.0)
    persistence = _clip01(_rolling_percentile(mm.rolling(8, min_periods=3).mean(), 63).fillna(0.5))
    raw = (
        0.30 * _rolling_percentile(d_market, 63).fillna(0.5)
        + 0.20 * _rolling_percentile(d_corr, 63).fillna(0.5)
        + 0.20 * _rolling_percentile(d_deff, 63).fillna(0.5)
        + 0.15 * persistence
    )
    if structural_stress is not None:
        stress = _clip01(pd.to_numeric(structural_stress.reindex(panel.index), errors="coerce").fillna(0.5))
        raw = raw + 0.10 * stress
    if critical_slowing_down is not None:
        csd = _clip01(pd.to_numeric(critical_slowing_down.reindex(panel.index), errors="coerce").fillna(0.5))
        raw = raw + 0.05 * csd
    score = _clip01(raw)
    if index is None:
        return score.astype(float)
    return pd.to_numeric(score.reindex(index.union(score.index)).sort_index().ffill().reindex(index), errors="coerce").astype(float)


def build_direction_gradient_score(
    *,
    structure_panel: pd.DataFrame,
    criticality: pd.Series | None = None,
    structural_stress: pd.Series | None = None,
    index: pd.Index | None = None,
) -> pd.Series:
    if structure_panel.empty:
        idx = index if index is not None else pd.Index([], dtype="datetime64[ns]")
        return pd.Series(np.nan, index=idx, dtype=float)
    panel = structure_panel.copy()
    mm = pd.to_numeric(
        panel.get("market_mode_share_pct", panel.get("market_mode_share")),
        errors="coerce",
    ).astype(float)
    rotation = pd.to_numeric(panel.get("sector_rotation_score"), errors="coerce").astype(float)
    residual = pd.to_numeric(panel.get("residual_dispersion"), errors="coerce").astype(float)
    crit = _safe_series(criticality, panel.index) if criticality is not None else _safe_series(panel.get("criticality"), panel.index)
    stress = (
        _safe_series(structural_stress, panel.index)
        if structural_stress is not None
        else _safe_series(panel.get("structural_stress"), panel.index)
    )

    d_rotation = _rolling_percentile(rotation.diff(), 63).fillna(0.5)
    d_residual = _rolling_percentile(residual.diff(), 63).fillna(0.5)
    d_market_relief = _rolling_percentile((-mm.diff()), 63).fillna(0.5)
    d_critical_relief = _rolling_percentile((-crit.diff()), 63).fillna(0.5)

    level = _clip01(
        0.35 * rotation.fillna(0.5)
        + 0.25 * residual.fillna(0.5)
        + 0.20 * (1.0 - mm.fillna(0.5))
        + 0.20 * (1.0 - crit.fillna(0.5))
    )
    raw = (
        0.30 * level
        + 0.20 * d_rotation
        + 0.15 * d_residual
        + 0.20 * d_market_relief
        + 0.10 * d_critical_relief
        + 0.05 * (1.0 - stress.fillna(0.5))
    )
    score = _clip01(raw)
    if index is None:
        return score.astype(float)
    return pd.to_numeric(score.reindex(index.union(score.index)).sort_index().ffill().reindex(index), errors="coerce").astype(float)


def build_attractor_persistence_score(
    *,
    direction_score: pd.Series,
    criticality: pd.Series | None = None,
    index: pd.Index | None = None,
    window: int = 21,
) -> pd.Series:
    idx = index if index is not None else direction_score.index
    if idx.empty:
        return pd.Series(dtype=float)
    direction = pd.to_numeric(direction_score.reindex(idx), errors="coerce").astype(float)
    crit = _safe_series(criticality, idx).fillna(0.5) if criticality is not None else pd.Series(0.5, index=idx, dtype=float)
    state_bias = (direction - crit).astype(float)
    stable_state = pd.Series(np.where(state_bias >= 0.03, 1.0, np.where(state_bias <= -0.03, -1.0, 0.0)), index=idx, dtype=float)
    same = pd.Series(np.nan, index=idx, dtype=float)
    same.iloc[1:] = (stable_state.iloc[1:].to_numpy() == stable_state.iloc[:-1].to_numpy()).astype(float)
    same_share = same.rolling(int(window), min_periods=max(8, int(window) // 2)).mean()
    variance_relief = 1.0 - _rolling_percentile(direction.rolling(int(window), min_periods=max(8, int(window) // 2)).std(ddof=0), 63).fillna(0.5)
    distance = _rolling_percentile(state_bias.abs(), 63).fillna(0.5)
    score = _clip01(0.50 * same_share.fillna(0.5) + 0.25 * variance_relief + 0.25 * distance)
    return score.astype(float)


def build_state_curvature_score(
    *,
    direction_score: pd.Series,
    criticality: pd.Series | None = None,
    index: pd.Index | None = None,
) -> pd.Series:
    idx = index if index is not None else direction_score.index
    if idx.empty:
        return pd.Series(dtype=float)
    direction = pd.to_numeric(direction_score.reindex(idx), errors="coerce").astype(float)
    crit = _safe_series(criticality, idx).fillna(0.5) if criticality is not None else pd.Series(0.5, index=idx, dtype=float)
    base = (direction - 0.25 * crit).astype(float)
    slope = base.diff()
    accel = slope.diff()
    slope_score = _rolling_percentile(slope, 63).fillna(0.5)
    accel_score = _rolling_percentile(accel, 63).fillna(0.5)
    score = _clip01(0.40 * slope_score + 0.60 * accel_score)
    return score.astype(float)


def apply_free_energy_penalty(
    *,
    base_score: pd.Series,
    turnover: pd.Series,
    instability: pd.Series,
    gamma: float,
    eta: float,
) -> pd.Series:
    base = pd.to_numeric(base_score, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    turn = _clip01(_rolling_percentile(pd.to_numeric(turnover.reindex(base.index), errors="coerce").fillna(0.0), 63).fillna(0.5))
    inst = _clip01(pd.to_numeric(instability.reindex(base.index), errors="coerce").fillna(0.5))
    penalty = float(max(0.0, gamma)) * turn + float(max(0.0, eta)) * inst
    return (base - penalty).clip(0.0, 1.0).astype(float)


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
    critical_slowing_down = build_critical_slowing_down_signal(returns=c_returns, benchmark_col=str(benchmark_crypto), window=63)
    crowding = build_crowding_signal(returns=c_returns, lookback=21, vol_window=63)

    panel = pd.DataFrame(
        {
            "funding": _clip01(funding),
            "open_interest": _clip01(open_interest),
            "liquidation": _clip01(liquidation),
            "btc_dominance": _clip01(btc_dominance),
            "breadth": _clip01(breadth),
            "critical_slowing_down": _clip01(critical_slowing_down),
            "crowding": _clip01(crowding),
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
    panel["exogenous_risk"] = _clip01(
        0.45 * panel["crypto_dependency_risk"]
        + 0.25 * panel["macro_stress"]
        + 0.15 * panel["critical_slowing_down"]
        + 0.15 * panel["crowding"]
    )

    crypto_cols = [
        "funding",
        "open_interest",
        "liquidation",
        "btc_dominance",
        "breadth",
        "critical_slowing_down",
        "crowding",
        "crypto_dependency_risk",
    ]
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
