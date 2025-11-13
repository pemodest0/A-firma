from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "modelos" / "core" / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.append(path_str)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_walk import simulate_classical_walk
from dados.brutos.financial_loader import (
    PriceSeries,
    YFINANCE_AVAILABLE,
    download_price_series,
    generate_return_windows,
    load_value_csv,
    prepare_returns,
    discretize_returns,
)
from financial_walk_model import MODE_CLASSICAL, MODE_GROVER, MODE_HADAMARD, classify_phase, _build_metrics
from graph_utils import line_graph
from quantum_walk import QISKIT_AVAILABLE, simulate_quantum_walk


DEFAULT_SYMBOLS = ["SPY", "^BVSP", "BTC-USD"]
PHASE_COLORS = {"difusiva": "#2ca02c", "transicao": "#ff7f0e", "coerente": "#d62728"}


def _resolve_symbol(raw: str) -> str:
    aliases = {
        "IBOV": "^BVSP",
        "IBOV.SA": "^BVSP",
        "^BVSP": "^BVSP",
        "BVSP": "^BVSP",
    }
    return aliases.get(raw.upper(), raw)


def _safe_label(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _load_series(args: argparse.Namespace, symbol: Optional[str], csv_path: Optional[str]) -> PriceSeries:
    if csv_path is not None:
        frame = load_value_csv(Path(csv_path), date_column=args.date_col, value_column=args.value_col)
    else:
        if symbol is None:
            raise ValueError("symbol must be provided when csv_path is None")
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance not available; supply --csv")
        df = download_price_series(
            symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
            price_column=args.price_column,
        )
        frame = df.rename(columns={"date": "date", "price": "price"})

    if args.month_zscore:
        frame["month"] = frame["date"].dt.month

        def _zscore(series: pd.Series) -> pd.Series:
            std = series.std(ddof=0)
            if std == 0 or np.isnan(std):
                return series - series.mean()
            return (series - series.mean()) / std

        frame["price"] = frame.groupby("month")["price"].transform(_zscore)
        frame.drop(columns=["month"], inplace=True)

    return prepare_returns(frame, method=args.return_method)


def _expected_return(distribution: np.ndarray, bin_centers: np.ndarray) -> float:
    return float(np.dot(distribution, bin_centers))


def _phase_thresholds(alpha_series: pd.Series, low_q: float, high_q: float, default_low: float, default_high: float) -> Tuple[float, float]:
    series = alpha_series.dropna()
    if series.empty:
        return default_low, default_high
    low = float(series.quantile(low_q))
    high = float(series.quantile(high_q))
    if low >= high:
        high = low + 1e-3
    return low, high


def _augment_lag_features(df: pd.DataFrame, columns: Iterable[str], group_col: str = "mode") -> pd.DataFrame:
    """Append 1- and 2-day lags for selected columns, preserving chronological order per mode."""
    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()
        for col in columns:
            if col in group.columns:
                group[f"{col}_lag1"] = group[col].shift(1)
                group[f"{col}_lag2"] = group[col].shift(2)
        return group

    return df.groupby(group_col, group_keys=False).apply(_apply)


def _validate_classical_block(df: pd.DataFrame, label: str) -> None:
    """Ensure classical forecasts are finite/coherent before downstream training."""
    classical = df[df["mode"] == MODE_CLASSICAL].copy()
    if classical.empty:
        raise RuntimeError(f"Nao foi possivel gerar previsoes classicas para {label}.")
    essential_cols = ("price_pred", "expected_return", "alpha", "entropy")
    issues = {
        col: int((~np.isfinite(classical[col])).sum())
        for col in essential_cols
        if col in classical.columns and (~np.isfinite(classical[col])).any()
    }
    if issues:
        raise ValueError(f"Valores nao finitos detectados na serie classica de {label}: {issues}")
    duplicate_mask = classical.duplicated(subset=["mode", "date"])
    if duplicate_mask.any():
        raise ValueError(f"Datas duplicadas encontradas nas previsoes classicas de {label}.")


def _normalize_coin_state(raw_alpha: complex, raw_beta: complex) -> Tuple[complex, complex]:
    norm = math.sqrt(abs(raw_alpha) ** 2 + abs(raw_beta) ** 2)
    if norm == 0:
        raise ValueError("quantum coin state nao pode ter norma zero.")
    return raw_alpha / norm, raw_beta / norm


def _fixed_coin_state(args: argparse.Namespace) -> Optional[Tuple[complex, complex]]:
    values = getattr(args, "quantum_coin_state", None)
    if not values:
        return None
    alpha_raw = complex(values[0])
    beta_mag = float(values[1])
    phase = getattr(args, "quantum_coin_phase", 0.0)
    beta_raw = beta_mag * complex(math.cos(phase), math.sin(phase))
    return _normalize_coin_state(alpha_raw, beta_raw)


def _compute_quantum_coin_state(feature_snapshot: Dict[str, float], args: argparse.Namespace) -> Tuple[Optional[Tuple[complex, complex]], bool]:
    mode = getattr(args, "quantum_coin_mode", "default")
    if mode == "default":
        return _fixed_coin_state(args), False
    if mode == "fixed":
        return (_fixed_coin_state(args) or _normalize_coin_state(1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))), False

    # Adaptive mode: combina sinais de momentum, volatilidade e drawdown.
    def _signal(value: float, scale: float) -> float:
        if not np.isfinite(value):
            return 0.0
        scale = max(scale, 1e-6)
        clamp = max(getattr(args, "quantum_coin_clamp", 3.0), 0.5)
        normalized = value / scale
        return float(max(-clamp, min(clamp, normalized)))

    momentum = feature_snapshot.get("momentum_10", float("nan"))
    vol_ratio = feature_snapshot.get("vol_ratio", float("nan"))
    drawdown = feature_snapshot.get("drawdown_long", float("nan"))

    momentum_signal = _signal(momentum, getattr(args, "quantum_coin_scale", 0.03))
    vol_signal = _signal((vol_ratio - 1.0) if np.isfinite(vol_ratio) else 0.0, getattr(args, "quantum_coin_vol_scale", 0.12))
    drawdown_signal = _signal(drawdown, getattr(args, "quantum_coin_drawdown_scale", 0.08))

    score = (
        getattr(args, "quantum_coin_momentum_weight", 1.0) * momentum_signal
        + getattr(args, "quantum_coin_vol_weight", 0.5) * vol_signal
        + getattr(args, "quantum_coin_drawdown_weight", 0.4) * drawdown_signal
    )
    clamp_total = max(getattr(args, "quantum_coin_clamp", 3.0), 0.5)
    score = max(-clamp_total, min(clamp_total, score))

    trend = math.tanh(score)
    vol_threshold = getattr(args, "quantum_coin_vol_threshold", 1.6)
    drawdown_threshold = getattr(args, "quantum_coin_drawdown_threshold", -0.12)
    risk_scale = max(0.0, min(1.0, getattr(args, "quantum_coin_risk_scale", 0.35)))
    risk_flag = (np.isfinite(vol_ratio) and vol_ratio >= vol_threshold) or (np.isfinite(drawdown) and drawdown <= drawdown_threshold)
    if risk_flag:
        trend *= (1.0 - risk_scale)
    floor = min(max(getattr(args, "quantum_coin_floor", 0.05), 0.0), 0.49)
    bias_right = 0.5 + 0.5 * trend
    bias_right = min(1.0 - floor, max(floor, bias_right))
    bias_left = 1.0 - bias_right
    alpha_raw = math.sqrt(bias_left)
    beta_mag = math.sqrt(bias_right)
    phase = getattr(args, "quantum_coin_phase", 0.0)
    beta_raw = beta_mag * complex(math.cos(phase), math.sin(phase))
    return _normalize_coin_state(alpha_raw, beta_raw), risk_flag


@dataclass
class ModeDailyMetrics:
    mode: str
    mode_label: str
    expected_return: float
    predicted_price: float
    alpha: float
    entropy: float
    metrics_raw: object


def _compute_feature_snapshot(
    price_series: PriceSeries,
    idx: int,
    short_window: int = 5,
    long_window: int = 21,
) -> Dict[str, float]:
    """Derive contextual features (volatility, regime clues) for a given day."""
    data = price_series.data
    if idx < 0 or idx >= len(data):
        return {
            "vol_realized_short": float("nan"),
            "vol_realized_long": float("nan"),
            "vol_ratio": float("nan"),
            "vol_ewm_30": float("nan"),
            "abs_return_short": float("nan"),
            "skew_long": float("nan"),
            "kurt_long": float("nan"),
            "drawdown_long": float("nan"),
            "vol_of_vol": float("nan"),
            "macd": float("nan"),
            "macd_signal": float("nan"),
            "macd_hist": float("nan"),
            "ppo": float("nan"),
            "cci": float("nan"),
            "trix": float("nan"),
            "kama": float("nan"),
            "williams_r": float("nan"),
            "tsi": float("nan"),
            "rsi_sma_diff": float("nan"),
            "price_sma_ratio": float("nan"),
            "volat_ratio_change": float("nan"),
            "sma_10": float("nan"),
            "sma_20": float("nan"),
            "ema_10": float("nan"),
            "rsi_14": float("nan"),
            "bollinger_bandwidth": float("nan"),
            "momentum_10": float("nan"),
            "rolling_max_20": float("nan"),
            "rolling_min_20": float("nan"),
            "price_zscore_20": float("nan"),
        }

    returns = data[price_series.return_column].iloc[: idx + 1]
    prices = data[price_series.price_column].iloc[: idx + 1]

    def _tail(series: pd.Series, window: int) -> pd.Series:
        return series.iloc[max(0, len(series) - window):]

    def _realized_vol(series: pd.Series) -> float:
        if series.shape[0] < 2:
            return float("nan")
        return float(series.std(ddof=0) * np.sqrt(252.0))

    short_returns = _tail(returns, short_window)
    long_returns = _tail(returns, long_window)
    vol_short = _realized_vol(short_returns)
    vol_long = _realized_vol(long_returns)
    if np.isnan(vol_short) or np.isnan(vol_long) or vol_long == 0.0:
        vol_ratio = float("nan")
    else:
        vol_ratio = float(vol_short / vol_long)

    if returns.shape[0] < 2:
        vol_ewm = float("nan")
    else:
        vol_ewm = float(returns.ewm(span=30, adjust=False).std().iloc[-1] * np.sqrt(252.0))

    if short_returns.empty:
        abs_return_short = float("nan")
    else:
        abs_return_short = float(short_returns.abs().mean())

    skew_long = float(long_returns.skew()) if long_returns.shape[0] >= 3 else float("nan")
    kurt_long = float(long_returns.kurt()) if long_returns.shape[0] >= 4 else float("nan")

    price_long = _tail(prices, long_window)
    if price_long.empty:
        drawdown_long = float("nan")
    else:
        rolling_peak = float(price_long.max())
        current_price = float(price_long.iloc[-1])
        drawdown_long = float((current_price / rolling_peak) - 1.0) if rolling_peak != 0 else float("nan")

    if long_returns.shape[0] < 2:
        vol_of_vol = float("nan")
    else:
        vol_of_vol = float(long_returns.abs().std(ddof=0) * np.sqrt(252.0))

    # Technical indicators and momentum features
    def _rolling_mean(window: int) -> float:
        segment = _tail(prices, window)
        return float(segment.mean()) if segment.shape[0] == window else float("nan")

    sma_10 = _rolling_mean(10)
    sma_20 = _rolling_mean(20)

    ema_10 = (
        float(prices.ewm(span=10, adjust=False).mean().iloc[-1])
        if prices.shape[0] >= 2
        else float("nan")
    )

    def _compute_rsi(window: int = 14) -> float:
        price_diff = prices.diff().dropna()
        if price_diff.shape[0] < window:
            return float("nan")
        gain = price_diff.clip(lower=0.0)
        loss = -price_diff.clip(upper=0.0)
        avg_gain = gain.rolling(window, min_periods=window).mean().iloc[-1]
        avg_loss = loss.rolling(window, min_periods=window).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    rsi_14 = _compute_rsi(14)

    def _bollinger_bandwidth(window: int = 20) -> float:
        segment = _tail(prices, window)
        if segment.shape[0] < window:
            return float("nan")
        middle = float(segment.mean())
        std = float(segment.std(ddof=0))
        if middle == 0:
            return float("nan")
        upper = middle + 2 * std
        lower = middle - 2 * std
        return float((upper - lower) / middle)

    bollinger_bandwidth = _bollinger_bandwidth(20)

    def _momentum(window: int = 10) -> float:
        if prices.shape[0] <= window:
            return float("nan")
        past_price = float(prices.iloc[-window - 1])
        if past_price == 0.0:
            return float("nan")
        return float(prices.iloc[-1] / past_price - 1.0)

    momentum_10 = _momentum(10)

    rolling_max_20 = float(_tail(prices, 20).max()) if prices.shape[0] >= 1 else float("nan")
    rolling_min_20 = float(_tail(prices, 20).min()) if prices.shape[0] >= 1 else float("nan")

    def _price_zscore(window: int = 20) -> float:
        segment = _tail(prices, window)
        if segment.shape[0] < window:
            return float("nan")
        mean = float(segment.mean())
        std = float(segment.std(ddof=0))
        if std == 0:
            return float("nan")
        return float((segment.iloc[-1] - mean) / std)

    price_zscore_20 = _price_zscore(20)

    # Additional indicators
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    macd_series = ema_12 - ema_26
    macd_signal_series = macd_series.ewm(span=9, adjust=False).mean()
    macd_hist_series = macd_series - macd_signal_series
    macd = float(macd_series.iloc[-1]) if not macd_series.empty else float("nan")
    macd_signal = float(macd_signal_series.iloc[-1]) if not macd_signal_series.empty else float("nan")
    macd_hist = float(macd_hist_series.iloc[-1]) if not macd_hist_series.empty else float("nan")

    ppo = (
        float((macd_series.iloc[-1] / ema_26.iloc[-1]) * 100.0)
        if not macd_series.empty and ema_26.iloc[-1] != 0
        else float("nan")
    )

    def _cci(window: int = 20) -> float:
        segment = _tail(prices, window)
        if segment.shape[0] < window:
            return float("nan")
        typical_price = segment
        sma = typical_price.mean()
        mean_dev = np.mean(np.abs(typical_price - sma))
        if mean_dev == 0:
            return float("nan")
        return float((typical_price.iloc[-1] - sma) / (0.015 * mean_dev))

    cci = _cci(20)

    def _trix(window: int = 15) -> float:
        if prices.shape[0] < window * 3:
            return float("nan")
        ema1 = prices.ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        trix_series = ema3.pct_change() * 100
        return float(trix_series.iloc[-1])

    trix = _trix(15)

    def _kama(window: int = 10, fast: int = 2, slow: int = 30) -> float:
        if prices.shape[0] < window + 2:
            return float("nan")
        er_num = abs(prices.iloc[-1] - prices.iloc[-window])
        er_den = prices.diff().abs().iloc[-window + 1 :].sum()
        if er_den == 0:
            er = 0
        else:
            er = er_num / er_den
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        smoothing = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_prev = prices.ewm(alpha=smoothing, adjust=False).mean().iloc[-2]
        return float(kama_prev + smoothing * (prices.iloc[-1] - kama_prev))

    kama = _kama()

    def _williams_r(window: int = 14) -> float:
        segment = _tail(prices, window)
        if segment.shape[0] < window:
            return float("nan")
        highest_high = segment.max()
        lowest_low = segment.min()
        if highest_high == lowest_low:
            return float("nan")
        return float(-100 * (highest_high - segment.iloc[-1]) / (highest_high - lowest_low))

    williams_r = _williams_r(14)

    def _tsi(window_slow: int = 25, window_fast: int = 13) -> float:
        if prices.shape[0] < window_slow + 2:
            return float("nan")
        momentum = prices.diff()
        double_smooth = momentum.ewm(span=window_fast, adjust=False).mean().ewm(
            span=window_slow, adjust=False
        ).mean()
        abs_double = momentum.abs().ewm(span=window_fast, adjust=False).mean().ewm(
            span=window_slow, adjust=False
        ).mean()
        if abs_double.iloc[-1] == 0:
            return float("nan")
        return float((double_smooth.iloc[-1] / abs_double.iloc[-1]) * 100)

    tsi = _tsi()

    rsi_sma_diff = float(rsi_14 - sma_10) if np.isfinite(rsi_14) and np.isfinite(sma_10) else float("nan")
    price_sma_ratio = (
        float(prices.iloc[-1] / sma_20) if sma_20 not in (0.0, float("nan")) and np.isfinite(sma_20) else float("nan")
    )
    volat_ratio_change = (
        float(vol_ratio / (vol_ewm / np.sqrt(252.0))) if np.isfinite(vol_ratio) and np.isfinite(vol_ewm) else float("nan")
    )

    return {
        "vol_realized_short": vol_short,
        "vol_realized_long": vol_long,
        "vol_ratio": vol_ratio,
        "vol_ewm_30": vol_ewm,
        "abs_return_short": abs_return_short,
        "skew_long": skew_long,
        "kurt_long": kurt_long,
        "drawdown_long": drawdown_long,
        "vol_of_vol": vol_of_vol,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "ppo": ppo,
        "cci": cci,
        "trix": trix,
        "kama": kama,
        "williams_r": williams_r,
        "tsi": tsi,
        "rsi_sma_diff": rsi_sma_diff,
        "price_sma_ratio": price_sma_ratio,
        "volat_ratio_change": volat_ratio_change,
        "sma_10": sma_10,
        "sma_20": sma_20,
        "ema_10": ema_10,
        "rsi_14": rsi_14,
        "bollinger_bandwidth": bollinger_bandwidth,
        "momentum_10": momentum_10,
        "rolling_max_20": rolling_max_20,
        "rolling_min_20": rolling_min_20,
        "price_zscore_20": price_zscore_20,
    }


def _simulate_modes(
    graph,
    distribution: np.ndarray,
    bin_centers: np.ndarray,
    walk_steps: int,
    fit_min: int,
    fit_max: Optional[int],
    noise: float,
    forecast_step: int,
    include_quantum: bool,
    coin_state: Optional[Tuple[complex, complex]],
) -> Dict[str, ModeDailyMetrics]:
    outputs: Dict[str, ModeDailyMetrics] = {}

    classical_result = simulate_classical_walk(graph, walk_steps, initial_distribution=distribution)
    idx_step = min(forecast_step, classical_result.distributions.shape[0] - 1)
    classical_metrics = _build_metrics(
        MODE_CLASSICAL,
        classical_result.entropies,
        classical_result.distributions,
        bin_centers,
        fit_min,
        fit_max,
        noise,
    )
    outputs[MODE_CLASSICAL] = ModeDailyMetrics(
        mode=MODE_CLASSICAL,
        mode_label="Classico",
        expected_return=_expected_return(classical_result.distributions[idx_step], bin_centers),
        predicted_price=float("nan"),
        alpha=classical_metrics.dispersion_alpha,
        entropy=classical_metrics.entropy_final,
        metrics_raw=classical_metrics,
    )

    if include_quantum and QISKIT_AVAILABLE:
        for coin, label in (("hadamard", MODE_HADAMARD), ("grover", MODE_GROVER)):
            quantum = simulate_quantum_walk(
                graph,
                walk_steps,
                coin=coin,
                initial_distribution=distribution,
                coin_state=coin_state,
            )
            idx_step_q = min(forecast_step, quantum.distributions.shape[0] - 1)
            quantum_metrics = _build_metrics(
                label,
                quantum.entropies,
                quantum.distributions,
                bin_centers,
                fit_min,
                fit_max,
                noise,
            )
            outputs[label] = ModeDailyMetrics(
                mode=label,
                mode_label="Quantico (Hadamard)" if label == MODE_HADAMARD else "Quantico (Grover)",
                expected_return=_expected_return(quantum.distributions[idx_step_q], bin_centers),
                predicted_price=float("nan"),
                alpha=quantum_metrics.dispersion_alpha,
                entropy=quantum_metrics.entropy_final,
                metrics_raw=quantum_metrics,
            )
    elif include_quantum:
        for label, pretty in ((MODE_HADAMARD, "Quantico (Hadamard)"), (MODE_GROVER, "Quantico (Grover)")):
            outputs[label] = ModeDailyMetrics(
                mode=label,
                mode_label=pretty,
                expected_return=float("nan"),
                predicted_price=float("nan"),
                alpha=float("nan"),
                entropy=float("nan"),
                metrics_raw=None,
            )
    return outputs


def _compute_daily_forecast(
    series: PriceSeries,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    args: argparse.Namespace,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    windows = generate_return_windows(series, bin_edges, window=args.window, step=1)
    price_df = series.data[["date", "price"]].copy()
    date_to_index = {pd.Timestamp(row.date): idx for idx, row in price_df.iterrows()}

    scale_series = price_df["price"].rolling(window=args.window, min_periods=max(5, args.window // 2)).std()
    rows: List[Dict[str, object]] = []
    for window in windows:
        end_date = pd.Timestamp(window.end)
        idx = date_to_index.get(end_date)
        ahead = args.forecast_days
        if idx is None or idx + ahead >= len(price_df):
            continue
        next_price = float(price_df.loc[idx + ahead, "price"])
        today_price = float(price_df.loc[idx, "price"])
        forecast_date = pd.Timestamp(price_df.loc[idx + ahead, "date"])
        if args.return_method == "log":
            actual_return = float(np.log(next_price / today_price)) if today_price > 0 else 0.0
        elif args.return_method == "simple":
            actual_return = float(next_price / today_price - 1.0) if today_price != 0 else float("nan")
        else:  # diff
            actual_return = float(next_price - today_price)

        feature_snapshot = _compute_feature_snapshot(series, idx)
        noise_value = args.noise
        if args.adaptive_noise:
            ratio = feature_snapshot.get("vol_ratio", float("nan"))
            if np.isfinite(ratio):
                noise_delta = args.noise_ratio_scale * (ratio - 1.0)
                noise_value = args.noise * (1.0 + noise_delta)
            noise_value = max(args.noise_min, min(args.noise_max, noise_value))

        graph = line_graph(len(bin_centers))
        coin_state = None
        coin_risk = False
        if not args.skip_quantum:
            try:
                coin_state, coin_risk = _compute_quantum_coin_state(feature_snapshot, args)
            except ValueError as exc:
                print(f"[WARN] Estado da moeda invalido ({exc}); utilizando padrao.")
                coin_state = None
                coin_risk = False
        feature_snapshot["coin_risk_flag"] = coin_risk
        mode_outputs = _simulate_modes(
            graph,
            window.distribution,
            bin_centers,
            args.walk_steps,
            args.fit_min,
            args.fit_max,
            noise_value,
            ahead,
            include_quantum=not args.skip_quantum,
            coin_state=coin_state,
        )

        mode_cache: Dict[str, Dict[str, float]] = {}
        for mode, result in mode_outputs.items():
            expected_return = result.expected_return
            predicted_price = float("nan")
            if np.isfinite(expected_return):
                predicted_price = float(today_price * np.exp(expected_return))
            mode_cache[mode] = {
                "expected_return": expected_return,
                "predicted_price": predicted_price,
                "alpha": result.alpha,
                "entropy": result.entropy,
            }

        for mode, result in mode_outputs.items():
            cached = mode_cache[mode]
            expected_return = cached["expected_return"]
            predicted_price = cached["predicted_price"]
            if args.return_method == "diff":
                scale = float(scale_series.iloc[idx]) if idx < len(scale_series) else float("nan")
                if not np.isfinite(scale) or scale == 0.0:
                    scale = float(max(abs(next_price), 1.0))
                if np.isfinite(predicted_price):
                    error_pct = float(100.0 * (predicted_price - next_price) / scale)
                else:
                    error_pct = float("nan")
            else:
                error_pct = (
                    float(100.0 * (predicted_price - next_price) / next_price)
                    if np.isfinite(predicted_price) and next_price != 0
                    else float("nan")
                )
                scale = abs(next_price) if next_price != 0 else 1.0
            direction_pred = np.sign(expected_return)
            direction_real = np.sign(actual_return)
            direction_match = direction_pred == direction_real or direction_real == 0.0
            row = {
                "symbol": label,
                "mode": mode,
                "mode_label": result.mode_label,
                "date": forecast_date,
                "price_today": today_price,
                "price_real": next_price,
                "price_pred": predicted_price,
                "expected_return": expected_return,
                "error_pct": error_pct,
                "error_scale": scale,
                "direction_match": direction_match,
                "alpha": result.alpha,
                "entropy": result.entropy,
                "actual_return": actual_return,
                "return_mode": args.return_method,
                "noise_used": noise_value,
                **feature_snapshot,
            }
            if mode == MODE_CLASSICAL:
                for q_mode, q_cache in mode_cache.items():
                    if q_mode == MODE_CLASSICAL:
                        continue
                    q_prefix = q_mode
                    row[f"{q_prefix}_expected_return"] = q_cache["expected_return"]
                    row[f"{q_prefix}_price_pred"] = q_cache["predicted_price"]
                    row[f"{q_prefix}_alpha"] = q_cache["alpha"]
                    row[f"{q_prefix}_entropy"] = q_cache["entropy"]
                    if np.isfinite(predicted_price) and np.isfinite(q_cache["predicted_price"]):
                        row[f"{q_prefix}_delta_price"] = q_cache["predicted_price"] - predicted_price
                    else:
                        row[f"{q_prefix}_delta_price"] = float("nan")
            rows.append(row)

    if not rows:
        raise RuntimeError(f"Nenhum dado util para {label}.")

    df = pd.DataFrame(rows)
    df.sort_values(["mode", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    lag_columns = ("expected_return", "vol_ratio", "alpha", "entropy", "noise_used")
    df = _augment_lag_features(df, lag_columns)
    df.sort_values(["mode", "date"], inplace=True)

    _validate_classical_block(df, label)

    summaries: List[Dict[str, object]] = []
    for mode, group in df.groupby("mode"):
        mode_df = group.copy()
        low_thr, high_thr = _phase_thresholds(
            mode_df["alpha"],
            args.phase_lower_quantile,
            args.phase_upper_quantile,
            args.phase_low,
            args.phase_high,
        )
        mode_df["phase"] = mode_df["alpha"].apply(lambda x: classify_phase(x, low_thr, high_thr))
        mode_df["dalpha"] = mode_df["alpha"].diff()
        mode_df["dentropy"] = mode_df["entropy"].diff()
        corr = float(mode_df["dalpha"].corr(mode_df["actual_return"]))
        df.loc[df["mode"] == mode, ["phase", "dalpha", "dentropy"]] = mode_df[["phase", "dalpha", "dentropy"]].values

        mae_pct = float(mode_df["error_pct"].abs().mean())
        direction_acc = float(mode_df["direction_match"].mean())
        summaries.append(
            {
                "symbol": label,
                "mode": mode,
                "mode_label": mode_df["mode_label"].iloc[0],
                "records": mode_df.shape[0],
                "mae_pct": mae_pct,
                "direction_accuracy": direction_acc,
                "alpha_mean": float(mode_df["alpha"].mean()),
                "entropy_mean": float(mode_df["entropy"].mean()),
                "corr_dalpha_return": corr,
            }
        )

    summary_df = pd.DataFrame(summaries)
    return df, summary_df


def _plot_daily_overview(df: pd.DataFrame, summary: pd.DataFrame, output_path: Path, label: str) -> None:
    df_hadamard = df[df["mode"] == MODE_HADAMARD].copy()
    if df_hadamard.empty:
        df_hadamard = df[df["mode"] == MODE_CLASSICAL].copy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df_hadamard["date"], df_hadamard["price_real"], color="black", linewidth=1.8, label="Preco real")
    for mode, mode_df in df.groupby("mode"):
        axes[0].plot(
            mode_df["date"],
            mode_df["price_pred"],
            linestyle="--",
            linewidth=1.4,
            label=f"{mode_df['mode_label'].iloc[0]} previsao",
        )

    phase_series = df_hadamard[["date", "phase"]].dropna()
    if not phase_series.empty:
        axes_phase = phase_series.set_index("date")["phase"]
        last_date = axes_phase.index.max()
        for phase_value, group in axes_phase.groupby(axes_phase):
            color = PHASE_COLORS.get(phase_value, "#dddddd")
            start = group.index.min()
            end = group.index.max()
            axes[0].axvspan(start, end, color=color, alpha=0.1)

    axes[0].set_title(f"{label} - Preco real vs previsoes")
    axes[0].set_ylabel("Preco")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    for mode, mode_df in df.groupby("mode"):
        axes[1].plot(mode_df["date"], mode_df["alpha"], linewidth=1.5, label=mode_df["mode_label"].iloc[0])
    axes[1].set_ylabel("alpha")
    axes[1].set_title("Expoente de dispersao")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    for mode, mode_df in df.groupby("mode"):
        axes[2].plot(mode_df["date"], mode_df["entropy"], linewidth=1.5, label=mode_df["mode_label"].iloc[0])
    axes[2].set_ylabel("Entropia (bits)")
    axes[2].set_xlabel("Data")
    axes[2].set_title("Entropia de Shannon")
    axes[2].grid(True, linestyle="--", alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_results(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "daily_forecast_metrics.csv", index=False)
    summary.to_csv(out_dir / "daily_forecast_summary.csv", index=False)
    _plot_daily_overview(df, summary, out_dir / "daily_forecast_overview.png", label)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Daily rolling forecasts with classical/quantum walks.")
    parser.add_argument("--symbol", action="append", default=None, help="Ticker symbol (repeatable). Defaults to SPY,^BVSP,BTC-USD.")
    parser.add_argument("--csv", action="append", default=None, help="CSV path(s) instead of downloading.")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--price-column", type=str, default="Adj Close")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--value-col", type=str, default="price")
    parser.add_argument("--return-method", type=str, choices=("log", "simple", "diff"), default="log")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--walk-steps", type=int, default=30)
    parser.add_argument("--fit-min", type=int, default=5)
    parser.add_argument("--fit-max", type=int, default=None)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--month-zscore", action="store_true", help="Normalize series by monthly z-score before computing returns.")
    parser.add_argument("--forecast-days", type=int, default=1, help="Forecast horizon (business days, default=1).")
    parser.add_argument("--phase-low", type=float, default=0.6)
    parser.add_argument("--phase-high", type=float, default=0.85)
    parser.add_argument("--phase-lower-quantile", type=float, default=0.35)
    parser.add_argument("--phase-upper-quantile", type=float, default=0.65)
    parser.add_argument("--output", type=str, default="results_finance/daily_walk_test")
    parser.add_argument("--adaptive-noise", action="store_true", help="Scale walk noise using recent volatility regime indicators.")
    parser.add_argument("--noise-ratio-scale", type=float, default=0.5, help="Sensitivity of noise to vol ratio deviations (only with --adaptive-noise).")
    parser.add_argument("--noise-min", type=float, default=0.005, help="Lower bound for adaptive noise.")
    parser.add_argument("--noise-max", type=float, default=0.25, help="Upper bound for adaptive noise.")
    parser.add_argument("--skip-quantum", action="store_true", help="Ignora modos quanticos (uso para testes rapidos ou ambientes sem Qiskit).")
    parser.add_argument(
        "--quantum-coin-mode",
        type=str,
        choices=("adaptive", "fixed", "default"),
        default="adaptive",
        help="Controla o estado inicial da moeda quantica.",
    )
    parser.add_argument(
        "--quantum-coin-state",
        type=float,
        nargs=2,
        metavar=("ALPHA", "BETA"),
        help="Amplitudes iniciais da moeda (serao normalizadas). Usado em modos default/fixed.",
    )
    parser.add_argument("--quantum-coin-scale", type=float, default=0.03, help="Escala do sinal de momentum.")
    parser.add_argument("--quantum-coin-vol-scale", type=float, default=0.12, help="Escala do sinal de volatilidade.")
    parser.add_argument("--quantum-coin-drawdown-scale", type=float, default=0.08, help="Escala do sinal de drawdown.")
    parser.add_argument(
        "--quantum-coin-floor",
        type=float,
        default=0.05,
        help="Limite minimo de probabilidade por lado na moeda adaptativa.",
    )
    parser.add_argument(
        "--quantum-coin-phase",
        type=float,
        default=0.0,
        help="Fase (rad) aplicada ao componente |1> da moeda.",
    )
    parser.add_argument("--quantum-coin-momentum-weight", type=float, default=1.0, help="Peso do sinal de momentum na moeda adaptativa.")
    parser.add_argument("--quantum-coin-vol-weight", type=float, default=0.5, help="Peso do sinal de volatilidade.")
    parser.add_argument("--quantum-coin-drawdown-weight", type=float, default=0.4, help="Peso do sinal de drawdown.")
    parser.add_argument("--quantum-coin-clamp", type=float, default=3.0, help="Limite absoluto para o escore agregado antes do tanh.")
    parser.add_argument("--quantum-coin-vol-threshold", type=float, default=1.6, help="Se vol_ratio exceder esse valor, reduz peso quântico.")
    parser.add_argument("--quantum-coin-drawdown-threshold", type=float, default=-0.12, help="Se drawdown cair abaixo desse valor, reduz peso quântico.")
    parser.add_argument("--quantum-coin-risk-scale", type=float, default=0.35, help="Escala multiplicativa aplicada ao escore em regimes de risco.")

    args = parser.parse_args(argv)

    horizon = max(1, args.forecast_days)

    tasks: List[Tuple[Optional[str], Optional[str]]] = []
    if args.csv:
        for csv_path in args.csv:
            tasks.append((None, csv_path))
    if args.symbol is not None:
        symbols = args.symbol
    elif args.csv:
        symbols = []
    else:
        symbols = DEFAULT_SYMBOLS
    tasks.extend((symbol, None) for symbol in symbols)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    for symbol, csv_path in tasks:
        resolved_symbol = None if csv_path else _resolve_symbol(symbol or "")
        label = Path(csv_path).stem if csv_path else (symbol or "dataset")
        print(f"\n=== {label} ===")
        try:
            price_series = _load_series(args, resolved_symbol, csv_path)
        except Exception as exc:  # pragma: no cover - data issues
            print(f"[WARN] {label} ignorado: {exc}")
            continue
        price_series.data["date"] = pd.to_datetime(price_series.data["date"])
        price_series.data.sort_values("date", inplace=True)
        price_series.data.reset_index(drop=True, inplace=True)

        bin_edges, bin_centers = discretize_returns(
            price_series.data[price_series.return_column],
            num_bins=args.bins,
            method="quantile",
        )

        df_metrics, df_summary = _compute_daily_forecast(
            price_series,
            bin_edges,
            bin_centers,
            args,
            label,
        )
        _save_results(df_metrics, df_summary, out_root / _safe_label(label), label)


if __name__ == "__main__":
    main()
