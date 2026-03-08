#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execution.returns import load_return_series_csv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICES_DIR = ROOT / "data" / "raw" / "finance" / "yfinance_daily"
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
MOMENTUM_ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "IEF", "SHY", *SECTOR_ETFS]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _perf_from_simple_returns(simple_returns: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(simple_returns, errors="coerce").dropna().astype(float)
    if x.empty:
        return {
            "total_return": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "positive_days_share": float("nan"),
        }
    eq = (1.0 + x).clip(lower=1e-9, upper=10.0).cumprod()
    ann_return = float(np.power(float(eq.iloc[-1]), 252.0 / max(int(x.shape[0]), 1)) - 1.0)
    ann_vol = float(x.std(ddof=0) * np.sqrt(252.0))
    drawdown = float((eq / eq.cummax() - 1.0).min())
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(ann_return / ann_vol) if ann_vol > 1e-12 else float("nan"),
        "max_drawdown": drawdown,
        "positive_days_share": float((x > 0.0).mean()),
    }


def _resolve_prices_dir(raw: str | Path | None) -> Path:
    text = str(raw or "").strip()
    if not text:
        return DEFAULT_PRICES_DIR
    p = Path(text)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _load_profile_daily(profile_dir: Path) -> pd.DataFrame:
    path = profile_dir / "daily_replay.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["portfolio_return", "benchmark_return", "capital", "benchmark_capital", "drawdown"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_ticker_returns(prices_dir: Path, tickers: list[str]) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for ticker in tickers:
        path = prices_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        try:
            s = load_return_series_csv(path, source_kind="log", target_kind="simple", series_name=ticker)
        except ValueError:
            continue
        if s.empty:
            continue
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    return out


def _align_to_strategy(strategy_daily: pd.DataFrame, benchmark_returns: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(strategy_daily["date"]).sort_values()
    aligned = benchmark_returns.reindex(dates).sort_index()
    return aligned.fillna(0.0)


def _weighted_return(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    out = pd.Series(np.zeros(len(returns), dtype=float), index=returns.index, dtype=float)
    for ticker, weight in weights.items():
        if ticker in returns.columns:
            out = out.add(float(weight) * pd.to_numeric(returns[ticker], errors="coerce").fillna(0.0), fill_value=0.0)
    return out.astype(float)


def _equal_weight_return(returns: pd.DataFrame, tickers: list[str]) -> tuple[pd.Series, list[str]]:
    present = [ticker for ticker in tickers if ticker in returns.columns]
    if not present:
        return pd.Series(np.zeros(len(returns), dtype=float), index=returns.index, dtype=float), []
    block = returns[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return block.mean(axis=1).astype(float), present


def _momentum_topk_return(
    returns: pd.DataFrame,
    *,
    tickers: list[str],
    lookback_days: int = 126,
    top_k: int = 3,
    fallback_ticker: str = "SPY",
) -> tuple[pd.Series, pd.Series]:
    present = [ticker for ticker in tickers if ticker in returns.columns]
    if not present:
        empty = pd.Series(np.zeros(len(returns), dtype=float), index=returns.index, dtype=float)
        return empty, pd.Series([""] * len(returns), index=returns.index, dtype=object)
    prices = (1.0 + returns[present].fillna(0.0)).cumprod()
    month_keys = prices.index.to_period("M")
    out = pd.Series(np.zeros(len(prices), dtype=float), index=prices.index, dtype=float)
    picks = pd.Series([""] * len(prices), index=prices.index, dtype=object)
    current_weights: dict[str, float] = {}
    current_label = ""
    prev_month: pd.Period | None = None
    fallback = fallback_ticker if fallback_ticker in present else present[0]
    lb = int(max(21, lookback_days))
    take = int(max(1, top_k))

    for idx, dt in enumerate(prices.index):
        month = month_keys[idx]
        if prev_month != month:
            if idx >= lb:
                window = prices.iloc[idx - lb : idx]
                score = window.iloc[-1] / window.iloc[0] - 1.0
                score = pd.to_numeric(score, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not score.empty:
                    sel = score.sort_values(ascending=False).head(take).index.tolist()
                else:
                    sel = [fallback]
            else:
                sel = [fallback]
            w = 1.0 / float(len(sel))
            current_weights = {ticker: w for ticker in sel}
            current_label = ",".join(sel)
            prev_month = month
        out.iloc[idx] = float(sum(float(current_weights.get(t, 0.0)) * float(returns.iloc[idx][t]) for t in current_weights))
        picks.iloc[idx] = current_label
    return out.astype(float), picks


def _compound_monthly(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    df = pd.DataFrame({"ret": pd.to_numeric(series, errors="coerce").fillna(0.0)}, index=series.index)
    ym = df.index.to_period("M").astype(str)
    return df.groupby(ym)["ret"].apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0))


def _window_total(series: pd.Series, window: int) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if x.empty:
        return float("nan")
    tail = x.iloc[-int(max(1, window)) :]
    return float(np.prod(1.0 + tail.to_numpy(dtype=float)) - 1.0)


def _build_report(
    *,
    profile_dir: Path,
    prices_dir: Path,
    strategy_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    strategy_name: str,
) -> dict[str, Any]:
    strategy_ret = pd.to_numeric(strategy_daily["portfolio_return"], errors="coerce").fillna(0.0).astype(float)
    daily = pd.DataFrame(index=benchmark_daily.index)
    daily["strategy_profile"] = strategy_ret.to_numpy(dtype=float)

    spy_series = pd.to_numeric(benchmark_daily.get("SPY"), errors="coerce").fillna(0.0).astype(float)
    daily["spy_buy_hold"] = spy_series.to_numpy(dtype=float)

    bond_ticker = next((ticker for ticker in ["IEF", "SHY", "TLT", "LQD"] if ticker in benchmark_daily.columns), "")
    sixty_forty_weights = {"SPY": 0.6}
    if bond_ticker:
        sixty_forty_weights[bond_ticker] = 0.4
    daily["sixty_forty"] = _weighted_return(benchmark_daily, sixty_forty_weights).to_numpy(dtype=float)

    sector_eqw, sector_used = _equal_weight_return(benchmark_daily, SECTOR_ETFS)
    daily["sector_equal_weight"] = sector_eqw.to_numpy(dtype=float)

    momentum_series, momentum_picks = _momentum_topk_return(benchmark_daily, tickers=MOMENTUM_ETFS, lookback_days=126, top_k=3, fallback_ticker="SPY")
    daily["momentum_global_top3"] = momentum_series.to_numpy(dtype=float)
    daily["momentum_picks"] = momentum_picks.astype(str).to_numpy()

    daily = daily.reset_index().rename(columns={"index": "date"})
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    monthly_out = pd.DataFrame({"ym": daily["date"].dt.to_period("M").astype(str)})
    for col in ["strategy_profile", "spy_buy_hold", "sixty_forty", "sector_equal_weight", "momentum_global_top3"]:
        monthly_out[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0).astype(float)
    monthly = monthly_out.groupby("ym", as_index=False).agg(
        strategy_profile=("strategy_profile", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
        spy_buy_hold=("spy_buy_hold", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
        sixty_forty=("sixty_forty", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
        sector_equal_weight=("sector_equal_weight", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
        momentum_global_top3=("momentum_global_top3", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
    )

    metrics: dict[str, Any] = {
        "strategy_profile": _perf_from_simple_returns(pd.to_numeric(daily["strategy_profile"], errors="coerce"))
    }
    comparisons: dict[str, Any] = {}
    for name in ["spy_buy_hold", "sixty_forty", "sector_equal_weight", "momentum_global_top3"]:
        metrics[name] = _perf_from_simple_returns(pd.to_numeric(daily[name], errors="coerce"))
        comparisons[name] = {
            "edge_total_return": _safe_float(metrics["strategy_profile"]["total_return"]) - _safe_float(metrics[name]["total_return"]),
            "edge_ann_return": _safe_float(metrics["strategy_profile"]["ann_return"]) - _safe_float(metrics[name]["ann_return"]),
            "weekly_edge": _window_total(pd.to_numeric(daily["strategy_profile"], errors="coerce"), 5)
            - _window_total(pd.to_numeric(daily[name], errors="coerce"), 5),
            "monthly_edge": _window_total(pd.to_numeric(daily["strategy_profile"], errors="coerce"), 21)
            - _window_total(pd.to_numeric(daily[name], errors="coerce"), 21),
            "quarter_edge": _window_total(pd.to_numeric(daily["strategy_profile"], errors="coerce"), 63)
            - _window_total(pd.to_numeric(daily[name], errors="coerce"), 63),
        }
    best_benchmark = max(
        ["spy_buy_hold", "sixty_forty", "sector_equal_weight", "momentum_global_top3"],
        key=lambda name: _safe_float(metrics[name].get("ann_return"), float("-inf")),
    )

    return {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_dir": str(profile_dir),
        "strategy_name": strategy_name,
        "prices_dir": str(prices_dir),
        "benchmarks": {
            "spy_buy_hold": {"ticker": "SPY"},
            "sixty_forty": {"weights": sixty_forty_weights},
            "sector_equal_weight": {"tickers": sector_used},
            "momentum_global_top3": {"tickers": [ticker for ticker in MOMENTUM_ETFS if ticker in benchmark_daily.columns], "lookback_days": 126, "top_k": 3},
        },
        "metrics": metrics,
        "comparisons_vs_strategy": comparisons,
        "winner_by_ann_return": best_benchmark,
        "artifacts": {
            "daily_csv": str(profile_dir / "hard_benchmarks_daily.csv"),
            "monthly_csv": str(profile_dir / "hard_benchmarks_monthly.csv"),
            "report_json": str(profile_dir / "hard_benchmarks_report.json"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build harder benchmark comparisons for a profit shadow profile.")
    ap.add_argument("--profile-dir", required=True)
    ap.add_argument("--prices-dir", default=str(DEFAULT_PRICES_DIR))
    args = ap.parse_args()

    profile_dir = Path(args.profile_dir).resolve()
    prices_dir = _resolve_prices_dir(args.prices_dir)
    if not profile_dir.exists():
        raise SystemExit(f"missing profile dir: {profile_dir}")
    if not prices_dir.exists():
        raise SystemExit(f"missing prices dir: {prices_dir}")

    daily = _load_profile_daily(profile_dir)
    if daily.empty:
        raise SystemExit(f"empty daily replay: {profile_dir}")
    tickers = sorted(set(["SPY", "IEF", "SHY", "TLT", "LQD", *SECTOR_ETFS, *MOMENTUM_ETFS]))
    benchmark_returns = _align_to_strategy(daily, _load_ticker_returns(prices_dir, tickers))
    if benchmark_returns.empty:
        raise SystemExit(f"no benchmark returns loaded from {prices_dir}")

    strategy_name = profile_dir.name
    report = _build_report(
        profile_dir=profile_dir,
        prices_dir=prices_dir,
        strategy_daily=daily,
        benchmark_daily=benchmark_returns,
        strategy_name=strategy_name,
    )

    daily_out = benchmark_returns.copy()
    daily_out = daily_out.reindex(pd.DatetimeIndex(daily["date"])).fillna(0.0)
    daily_out.insert(0, "strategy_profile", pd.to_numeric(daily["portfolio_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    daily_out.insert(0, "date", pd.DatetimeIndex(daily["date"]).date.astype(str))
    daily_out["sixty_forty"] = _weighted_return(benchmark_returns, report["benchmarks"]["sixty_forty"]["weights"]).to_numpy(dtype=float)
    sector_eqw, _ = _equal_weight_return(benchmark_returns, SECTOR_ETFS)
    daily_out["sector_equal_weight"] = sector_eqw.to_numpy(dtype=float)
    momentum_series, momentum_picks = _momentum_topk_return(benchmark_returns, tickers=MOMENTUM_ETFS, lookback_days=126, top_k=3, fallback_ticker="SPY")
    daily_out["momentum_global_top3"] = momentum_series.to_numpy(dtype=float)
    daily_out["momentum_picks"] = momentum_picks.astype(str).to_numpy()
    daily_out.to_csv(profile_dir / "hard_benchmarks_daily.csv", index=False)

    monthly = pd.DataFrame({"ym": pd.to_datetime(daily_out["date"], errors="coerce").dt.to_period("M").astype(str)})
    for col in ["strategy_profile", "SPY", "sixty_forty", "sector_equal_weight", "momentum_global_top3"]:
        monthly[col] = pd.to_numeric(daily_out[col], errors="coerce").fillna(0.0).astype(float)
    monthly = monthly.groupby("ym", as_index=False).agg(
        strategy_profile=("strategy_profile", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
        spy_buy_hold=("SPY", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
        sixty_forty=("sixty_forty", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
        sector_equal_weight=("sector_equal_weight", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
        momentum_global_top3=("momentum_global_top3", lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)),
    )
    monthly.to_csv(profile_dir / "hard_benchmarks_monthly.csv", index=False)
    _write_json(profile_dir / "hard_benchmarks_report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
