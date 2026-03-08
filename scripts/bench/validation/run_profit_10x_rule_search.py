#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.cost_model import summarize_return_series  # noqa: E402
from execution.net_assumptions import apply_net_assumptions, blend_profiles, load_net_assumption_profiles  # noqa: E402
from execution.returns import load_return_series_csv  # noqa: E402


DEFAULT_GROUP_FAMILIES = [
    ("all_assets", None),
    ("technology", ["technology"]),
    ("materials", ["materials"]),
    ("consumer_discretionary", ["consumer_discretionary"]),
    ("speed_growth", ["technology", "consumer_discretionary", "telecommunications"]),
    ("speed_core", ["technology", "materials", "consumer_discretionary"]),
    ("growth_plus", ["technology", "consumer_discretionary", "telecommunications", "health_care"]),
    ("oos_static", ["equities_us_other", "materials", "technology"]),
    ("cyclical_growth", ["technology", "materials", "industrials"]),
]


@dataclass(frozen=True)
class UniverseFamily:
    family_id: str
    groups: tuple[str, ...]


@dataclass(frozen=True)
class RuleConfig:
    family_id: str
    groups: tuple[str, ...]
    lookback_days: int
    rebalance_days: int
    top_k: int
    score_mode: str
    asset_ma_days: int
    market_ma_days: int
    relative_to_shy: bool

    @property
    def candidate_id(self) -> str:
        rel = "relshy" if self.relative_to_shy else "riskon"
        return (
            f"{self.family_id}"
            f"__lb{int(self.lookback_days):03d}"
            f"__rb{int(self.rebalance_days):02d}"
            f"__k{int(self.top_k)}"
            f"__{self.score_mode}"
            f"__ama{int(self.asset_ma_days):03d}"
            f"__mma{int(self.market_ma_days):03d}"
            f"__{rel}"
        )


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _infer_asset_jurisdiction(ticker: str, group: str) -> str:
    tt = str(ticker).strip().upper()
    gg = str(group).strip().lower()
    if tt.endswith(".SA") or gg == "equities_br_bluechips":
        return "br_local"
    return "foreign"


def _load_asset_table(asset_groups_csv: Path, asset_metadata_csv: Path) -> pd.DataFrame:
    groups = pd.read_csv(asset_groups_csv).rename(columns={"asset": "asset_id", "group": "asset_group"})
    meta = pd.read_csv(asset_metadata_csv)
    groups["asset_id"] = groups["asset_id"].astype(str).str.strip()
    meta["asset_id"] = meta["asset_id"].astype(str).str.strip()
    merged = groups.merge(meta, on="asset_id", how="left")
    merged["ticker"] = merged.get("ticker", merged["asset_id"]).astype(str).str.strip()
    merged["asset_group"] = merged["asset_group"].astype(str).str.strip()
    merged["liquidity_proxy"] = pd.to_numeric(merged.get("liquidity_proxy"), errors="coerce").fillna(0.0)
    merged["jurisdiction"] = [
        _infer_asset_jurisdiction(ticker=ticker, group=group)
        for ticker, group in zip(merged["ticker"], merged["asset_group"])
    ]
    return merged[["asset_id", "ticker", "asset_group", "liquidity_proxy", "jurisdiction"]].copy()


def _load_group_viability_groups(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty or "groups" not in df.columns:
        return []
    if "status" in df.columns:
        df = df[df["status"].astype(str).isin(["keep", "watch"])].copy()
    sort_cols = [col for col in ["net_blended_ann_return", "gross_ann_return"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    tokens: list[str] = []
    for text in df["groups"].astype(str).tolist():
        if "," in text:
            continue
        tokens.append(text.strip())
    return _dedupe_keep_order(tokens)


def _build_universe_families(asset_table: pd.DataFrame, group_viability_csv: Path) -> list[UniverseFamily]:
    available = _dedupe_keep_order(sorted(asset_table["asset_group"].astype(str).unique().tolist()))
    keep_groups = _load_group_viability_groups(group_viability_csv)
    merged_seed = _dedupe_keep_order(
        keep_groups
        + ["technology", "materials", "consumer_discretionary", "telecommunications", "equities_us_other", "health_care", "industrials"]
    )
    families: list[UniverseFamily] = []
    seen: set[tuple[str, ...]] = set()

    def add(family_id: str, raw_groups: list[str] | None) -> None:
        groups = tuple(sorted(g for g in (raw_groups or available) if g in available))
        if not groups:
            return
        if groups in seen:
            return
        seen.add(groups)
        families.append(UniverseFamily(family_id=str(family_id), groups=groups))

    for family_id, raw_groups in DEFAULT_GROUP_FAMILIES:
        add(str(family_id), raw_groups)
    for group in merged_seed[:6]:
        add(group, [group])
    add("keep_top2", merged_seed[:2])
    add("keep_top3", merged_seed[:3])
    add("keep_top4", merged_seed[:4])
    return families


def _load_price_and_return_csv(path: Path, ticker: str) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    df = pd.read_csv(path)
    if "date" not in df.columns or "r" not in df.columns:
        return None, None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if df.empty:
        return None, None
    ret = load_return_series_csv(path, source_kind="log", target_kind="simple", series_name=ticker)
    ret = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if ret.empty:
        return None, None
    if "price" in df.columns:
        price = pd.to_numeric(df["price"], errors="coerce")
        price = pd.Series(price.to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=ticker)
        price = price[price > 0.0].astype(float)
        price = price.reindex(ret.index).astype(float)
    else:
        price = (1.0 + ret.fillna(0.0)).cumprod()
        if not price.empty:
            first_valid = ret.first_valid_index()
            if first_valid is not None:
                price.loc[price.index < first_valid] = np.nan
    return ret.astype(float), pd.to_numeric(price, errors="coerce").astype(float)


def _load_daily_universe(
    *,
    prices_dir: Path,
    asset_table: pd.DataFrame,
    min_history_days: int,
    max_abs_daily_return: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return_frames: list[pd.Series] = []
    price_frames: list[pd.Series] = []
    viability_rows: list[dict[str, Any]] = []
    for row in asset_table.drop_duplicates(subset=["ticker"], keep="first").itertuples(index=False):
        ticker = str(row.ticker).strip()
        path = prices_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        ret, price = _load_price_and_return_csv(path, ticker)
        if ret is None or price is None:
            continue
        ret = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
        price = pd.to_numeric(price, errors="coerce").dropna().astype(float)
        if ret.shape[0] < int(max(30, min_history_days)):
            continue
        if bool((ret.abs() > float(max_abs_daily_return)).any()):
            continue
        return_frames.append(ret.rename(ticker))
        price_frames.append(price.rename(ticker))
        viability_rows.append(
            {
                "asset_id": str(row.asset_id),
                "ticker": ticker,
                "asset_group": str(row.asset_group),
                "jurisdiction": str(row.jurisdiction),
                "liquidity_proxy": float(row.liquidity_proxy),
                "days_available": int(ret.shape[0]),
                "start_date": str(ret.index.min().date()),
                "end_date": str(ret.index.max().date()),
                "gross_total_return": float(np.prod(1.0 + ret.to_numpy(dtype=float)) - 1.0),
            }
        )
    if not return_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    returns = pd.concat(return_frames, axis=1, sort=True).sort_index()
    prices = pd.concat(price_frames, axis=1, sort=True).sort_index()
    returns = returns.loc[:, ~returns.columns.duplicated()].copy()
    prices = prices.loc[:, ~prices.columns.duplicated()].copy()
    viability = pd.DataFrame(viability_rows).sort_values(["asset_group", "liquidity_proxy", "ticker"], ascending=[True, False, True]).reset_index(drop=True)
    return returns, prices, viability


def _ensure_benchmark_columns(returns: pd.DataFrame, prices: pd.DataFrame, prices_dir: Path, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret = returns.copy()
    prc = prices.copy()
    for ticker in tickers:
        if ticker in ret.columns and ticker in prc.columns:
            continue
        path = prices_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        s_ret, s_price = _load_price_and_return_csv(path, ticker)
        if s_ret is None or s_price is None:
            continue
        ret = ret.join(s_ret.rename(ticker), how="outer")
        prc = prc.join(s_price.rename(ticker), how="outer")
    return ret.sort_index(), prc.sort_index()


def _precompute_scores(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    lookbacks: list[int],
    asset_ma_days_list: list[int],
    benchmark_ticker: str,
) -> tuple[dict[tuple[int, str], pd.DataFrame], dict[int, pd.DataFrame], dict[int, pd.Series]]:
    score_map: dict[tuple[int, str], pd.DataFrame] = {}
    for lookback in lookbacks:
        lb = int(max(2, lookback))
        end_price = prices.shift(1)
        start_price = prices.shift(1 + lb)
        total = end_price / start_price - 1.0
        score_map[(lb, "mom_total")] = total.replace([np.inf, -np.inf], np.nan)

        vol = returns.shift(1).rolling(lb, min_periods=max(20, lb // 2)).std(ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_adj = total / vol.replace(0.0, np.nan)
        score_map[(lb, "mom_vol_adj")] = vol_adj.replace([np.inf, -np.inf], np.nan)

    asset_ma_filters: dict[int, pd.DataFrame] = {0: pd.DataFrame(True, index=prices.index, columns=prices.columns)}
    for days in asset_ma_days_list:
        dd = int(days)
        if dd <= 0:
            continue
        close = prices.shift(1)
        ma = close.rolling(dd, min_periods=max(20, dd // 2)).mean()
        asset_ma_filters[dd] = (close > ma).fillna(False)

    benchmark_filters: dict[int, pd.Series] = {0: pd.Series(True, index=prices.index, dtype=bool)}
    if benchmark_ticker in prices.columns:
        bench = pd.to_numeric(prices[benchmark_ticker], errors="coerce").astype(float)
        close = bench.shift(1)
        for days in asset_ma_days_list:
            dd = int(days)
            if dd <= 0:
                continue
            ma = close.rolling(dd, min_periods=max(20, dd // 2)).mean()
            benchmark_filters[dd] = (close > ma).fillna(False).astype(bool)
    return score_map, asset_ma_filters, benchmark_filters


def _top_k_indices(score_row: np.ndarray, valid_mask: np.ndarray, top_k: int) -> list[int]:
    if score_row.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("score_row and valid_mask must be 1d")
    idx = np.flatnonzero(valid_mask)
    if idx.size == 0:
        return []
    scores = score_row[idx]
    take = int(max(1, min(int(top_k), idx.size)))
    if idx.size <= take:
        ordered = idx[np.argsort(scores)[::-1]]
        return ordered.astype(int).tolist()
    part = np.argpartition(scores, -take)[-take:]
    ordered_local = part[np.argsort(scores[part])[::-1]]
    return idx[ordered_local].astype(int).tolist()


def _rolling_ten_x_stats(net_returns: pd.Series, *, monthly_start: bool = True, horizon_days: int = 1260, target_multiple: float = 10.0) -> dict[str, float]:
    ret = pd.to_numeric(net_returns, errors="coerce").fillna(0.0).astype(float)
    if ret.empty:
        return {
            "starts_considered": 0.0,
            "hit_rate": float("nan"),
            "median_years": float("nan"),
            "best_years": float("nan"),
        }
    wealth = (1.0 + ret).cumprod().astype(float)
    if monthly_start:
        start_positions = ret.groupby(ret.index.to_period("M")).head(1).index
        starts = [int(ret.index.get_loc(dt)) for dt in start_positions]
    else:
        starts = list(range(ret.shape[0]))
    hits = 0
    years: list[float] = []
    for start_pos in starts:
        stop_pos = min(ret.shape[0] - 1, start_pos + int(max(1, horizon_days)))
        base = float(wealth.iloc[start_pos - 1]) if start_pos > 0 else 1.0
        rel = pd.to_numeric(wealth.iloc[start_pos : stop_pos + 1], errors="coerce").astype(float) / float(base)
        target = rel[rel >= float(target_multiple)]
        if target.empty:
            continue
        hits += 1
        years.append((target.index[0] - ret.index[start_pos]).days / 365.25)
    total_starts = len(starts)
    return {
        "starts_considered": float(total_starts),
        "hit_rate": float(hits / total_starts) if total_starts else float("nan"),
        "median_years": float(np.median(years)) if years else float("nan"),
        "best_years": float(np.min(years)) if years else float("nan"),
    }


def _l1_turnover(prev_weights: dict[str, float], next_weights: dict[str, float]) -> float:
    keys = sorted(set(prev_weights) | set(next_weights))
    return 0.5 * float(sum(abs(float(prev_weights.get(key, 0.0)) - float(next_weights.get(key, 0.0))) for key in keys))


def _status_from_rule(row: dict[str, Any]) -> str:
    hit5 = _safe_float(row.get("hit_rate_10x_5y"), 0.0)
    ann = _safe_float(row.get("net_ann_return"), float("-inf"))
    edge = _safe_float(row.get("edge_vs_spy_net_total_return"), float("-inf"))
    if np.isfinite(hit5) and hit5 >= 0.10 and np.isfinite(ann) and ann >= 0.20:
        return "keep"
    if np.isfinite(hit5) and hit5 > 0.0:
        return "watch"
    if np.isfinite(edge) and edge > 0.0:
        return "watch"
    return "kill"


def _simulate_candidate(
    *,
    cfg: RuleConfig,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    score_map: dict[tuple[int, str], pd.DataFrame],
    asset_ma_filters: dict[int, pd.DataFrame],
    benchmark_filters: dict[int, pd.Series],
    benchmark_ticker: str,
    fallback_ticker: str,
    net_profiles: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    all_tickers = list(returns.columns.astype(str))
    ticker_to_col = {ticker: idx for idx, ticker in enumerate(all_tickers)}
    allowed_assets = asset_table[asset_table["asset_group"].astype(str).isin(list(cfg.groups))]["ticker"].astype(str).tolist()
    allowed_idx = np.array([ticker_to_col[ticker] for ticker in allowed_assets if ticker in ticker_to_col], dtype=int)
    if allowed_idx.size == 0:
        return {}, pd.DataFrame()

    score_df = score_map[(int(cfg.lookback_days), str(cfg.score_mode))]
    score_arr = score_df.reindex(index=returns.index, columns=all_tickers).to_numpy(dtype=float)
    asset_ma_arr = asset_ma_filters[int(cfg.asset_ma_days)].reindex(index=returns.index, columns=all_tickers).fillna(False).to_numpy(dtype=bool)
    benchmark_ok = benchmark_filters[int(cfg.market_ma_days)].reindex(returns.index).fillna(False).to_numpy(dtype=bool)
    ret_arr = returns.reindex(columns=all_tickers).to_numpy(dtype=float)

    shy_score = None
    if bool(cfg.relative_to_shy) and fallback_ticker in score_df.columns:
        shy_score = pd.to_numeric(score_df[fallback_ticker], errors="coerce").to_numpy(dtype=float)

    asset_meta = asset_table.drop_duplicates(subset=["ticker"], keep="first").set_index("ticker")
    foreign_flag = np.array(
        [1.0 if str(asset_meta.get("jurisdiction", pd.Series()).get(ticker, "foreign")) == "foreign" else 0.0 for ticker in all_tickers],
        dtype=float,
    )
    group_by_ticker = {str(ticker): str(asset_meta.get("asset_group", pd.Series()).get(ticker, "")) for ticker in all_tickers}
    fallback_idx = int(ticker_to_col[fallback_ticker]) if fallback_ticker in ticker_to_col else -1

    warmup = max(int(cfg.lookback_days), int(cfg.asset_ma_days), int(cfg.market_ma_days)) + 2
    rebalance_positions = list(range(int(max(1, warmup)), ret_arr.shape[0], int(max(1, cfg.rebalance_days))))
    if not rebalance_positions:
        return {}, pd.DataFrame()

    daily_ret = np.zeros(ret_arr.shape[0], dtype=float)
    daily_turnover = np.zeros(ret_arr.shape[0], dtype=float)
    daily_foreign_share = np.zeros(ret_arr.shape[0], dtype=float)
    selection_rows: list[dict[str, Any]] = []
    prev_weights: dict[str, float] = {"CASH": 1.0}

    for pos_idx, pos in enumerate(rebalance_positions):
        next_pos = rebalance_positions[pos_idx + 1] if pos_idx + 1 < len(rebalance_positions) else ret_arr.shape[0]
        score_row = score_arr[pos]
        valid = np.zeros(score_row.shape[0], dtype=bool)
        valid[allowed_idx] = True
        valid &= np.isfinite(score_row)
        valid &= np.isfinite(ret_arr[pos - 1])
        valid &= asset_ma_arr[pos]
        valid &= score_row > 0.0
        if shy_score is not None and pos < shy_score.shape[0] and np.isfinite(shy_score[pos]):
            valid &= score_row > float(shy_score[pos])
        if int(cfg.market_ma_days) > 0 and not bool(benchmark_ok[pos]):
            valid[:] = False

        selected_idx = _top_k_indices(score_row, valid, int(cfg.top_k))
        if not selected_idx and fallback_idx >= 0:
            selected_idx = [fallback_idx]

        if not selected_idx:
            weights: dict[str, float] = {"CASH": 1.0}
            period_ret = np.zeros(next_pos - pos, dtype=float)
            foreign_share = 0.0
            groups = ""
            tickers = "CASH"
        else:
            w = 1.0 / float(len(selected_idx))
            weights = {all_tickers[idx]: w for idx in selected_idx}
            period_block = np.nan_to_num(ret_arr[pos:next_pos, selected_idx], nan=0.0)
            period_ret = period_block.mean(axis=1).astype(float)
            foreign_share = float(np.mean(foreign_flag[selected_idx]))
            groups = ",".join(sorted({group_by_ticker.get(all_tickers[idx], "") for idx in selected_idx if group_by_ticker.get(all_tickers[idx], "")}))
            tickers = ",".join(sorted(weights.keys()))

        daily_ret[pos:next_pos] = period_ret
        daily_foreign_share[pos:next_pos] = foreign_share
        daily_turnover[pos] = _l1_turnover(prev_weights, weights)
        prev_weights = dict(weights)
        selection_rows.append(
            {
                "date": str(returns.index[pos].date()),
                "candidate_id": cfg.candidate_id,
                "family_id": cfg.family_id,
                "tickers": tickers,
                "groups": groups,
                "weights_json": json.dumps(weights, sort_keys=True),
                "turnover": float(daily_turnover[pos]),
                "foreign_share": foreign_share,
            }
        )

    daily_ret_s = pd.Series(daily_ret, index=returns.index, dtype=float)
    turnover_s = pd.Series(daily_turnover, index=returns.index, dtype=float)
    foreign_share_s = pd.Series(daily_foreign_share, index=returns.index, dtype=float)

    avg_foreign_share = float(foreign_share_s.mean()) if not foreign_share_s.empty else 1.0
    foreign_profile = net_profiles["profiles"]["foreign_financial_brazil_resident"]
    br_profile = net_profiles["profiles"]["br_local_equity"]
    blended_profile = blend_profiles(avg_foreign_share, foreign_profile=foreign_profile, br_profile=br_profile)
    net_frame = apply_net_assumptions(daily_ret_s, turnover_s, profile=blended_profile, periods_index=returns.index)
    net_ret = pd.to_numeric(net_frame["net_ret"], errors="coerce").fillna(0.0).astype(float)

    benchmark_ret = pd.to_numeric(returns.get(benchmark_ticker), errors="coerce").reindex(returns.index).fillna(0.0).astype(float)
    benchmark_net = apply_net_assumptions(
        benchmark_ret,
        pd.Series(np.zeros(len(benchmark_ret), dtype=float), index=benchmark_ret.index, dtype=float),
        profile=foreign_profile,
        periods_index=benchmark_ret.index,
    )

    gross_summary = summarize_return_series(daily_ret_s, periods_per_year=252)
    net_summary = summarize_return_series(net_ret, periods_per_year=252)
    benchmark_net_summary = summarize_return_series(benchmark_net["net_ret"], periods_per_year=252)
    edge_vs_benchmark = _safe_float(net_summary.get("total_return")) - _safe_float(benchmark_net_summary.get("total_return"))

    wealth = (1.0 + net_ret).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x_full = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    hit_3y = _rolling_ten_x_stats(net_ret, horizon_days=756)
    hit_5y = _rolling_ten_x_stats(net_ret, horizon_days=1260)

    row = {
        "candidate_id": cfg.candidate_id,
        "family_id": cfg.family_id,
        "groups": ",".join(cfg.groups),
        "lookback_days": int(cfg.lookback_days),
        "rebalance_days": int(cfg.rebalance_days),
        "top_k": int(cfg.top_k),
        "score_mode": str(cfg.score_mode),
        "asset_ma_days": int(cfg.asset_ma_days),
        "market_ma_days": int(cfg.market_ma_days),
        "relative_to_shy": bool(cfg.relative_to_shy),
        "benchmark_ticker": str(benchmark_ticker),
        "gross_total_return": _safe_float(gross_summary.get("total_return")),
        "gross_ann_return": _safe_float(gross_summary.get("annualized_return")),
        "gross_sharpe": _safe_float(gross_summary.get("sharpe")),
        "gross_max_drawdown": _safe_float(gross_summary.get("max_drawdown")),
        "net_total_return": _safe_float(net_summary.get("total_return")),
        "net_ann_return": _safe_float(net_summary.get("annualized_return")),
        "net_sharpe": _safe_float(net_summary.get("sharpe")),
        "net_max_drawdown": _safe_float(net_summary.get("max_drawdown")),
        "edge_vs_benchmark_net_total_return": edge_vs_benchmark,
        "edge_vs_spy_net_total_return": edge_vs_benchmark,
        "avg_turnover_daily": float(turnover_s.mean()) if not turnover_s.empty else float("nan"),
        "avg_foreign_share": avg_foreign_share,
        "actual_end_value_100_net": float(100.0 * (1.0 + _safe_float(net_summary.get("total_return"), -1.0))),
        "actual_years_to_10x_full": years_to_10x_full,
        "hit_rate_10x_3y": _safe_float(hit_3y.get("hit_rate")),
        "median_years_to_10x_3y": _safe_float(hit_3y.get("median_years")),
        "hit_rate_10x_5y": _safe_float(hit_5y.get("hit_rate")),
        "median_years_to_10x_5y": _safe_float(hit_5y.get("median_years")),
        "selection_count": int(len(selection_rows)),
    }
    row["status"] = _status_from_rule(row)
    row["goal_score"] = (
        10.0 * max(0.0, _safe_float(row["hit_rate_10x_5y"], 0.0))
        + 6.0 * max(0.0, _safe_float(row["hit_rate_10x_3y"], 0.0))
        + (3.0 / max(_safe_float(row["actual_years_to_10x_full"], 99.0), 0.25) if np.isfinite(_safe_float(row["actual_years_to_10x_full"])) else 0.0)
        + 2.0 * max(-1.0, _safe_float(row["net_ann_return"], -1.0))
        + 0.15 * max(-1.0, _safe_float(row["edge_vs_benchmark_net_total_return"], -1.0))
    )
    return row, pd.DataFrame(selection_rows)


def _build_research_rows(results_df: pd.DataFrame, *, outdir: Path, summary_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in results_df.to_dict(orient="records"):
        rows.append(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_id": str(row["candidate_id"]),
                "label": str(row["candidate_id"]),
                "methodology": "asset_concentrated_momentum_10x",
                "status": str(row["status"]),
                "gross_ann_return": _safe_float(row.get("gross_ann_return")),
                "net_ann_return": _safe_float(row.get("net_ann_return")),
                "gross_total_return": _safe_float(row.get("gross_total_return")),
                "net_total_return": _safe_float(row.get("net_total_return")),
                "sharpe": _safe_float(row.get("net_sharpe")),
                "max_drawdown": _safe_float(row.get("net_max_drawdown")),
                "benchmark_ticker": str(row.get("benchmark_ticker", "")),
                "edge_vs_benchmark_net_total_return": _safe_float(row.get("edge_vs_benchmark_net_total_return", row.get("edge_vs_spy_net_total_return"))),
                "edge_vs_spy_net_total_return": _safe_float(row.get("edge_vs_spy_net_total_return")),
                "avg_foreign_share": _safe_float(row.get("avg_foreign_share")),
                "groups": str(row.get("groups", "")),
                "artifacts": {
                    "suite_dir": str(outdir),
                    "summary_json": str(summary_path),
                },
                "notes": f"Goal 10x search vs {row.get('benchmark_ticker', 'benchmark')}; full_years_to_10x={row.get('actual_years_to_10x_full')}, hit5y={row.get('hit_rate_10x_5y')}.",
            }
        )
    return rows


def _top_pick_frequency(selection_df: pd.DataFrame, asset_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selection_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    meta = asset_table.drop_duplicates(subset=["ticker"], keep="first").set_index("ticker")
    asset_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for text in selection_df["tickers"].astype(str):
        tickers = [token.strip() for token in text.split(",") if token.strip() and token.strip() != "CASH"]
        for ticker in tickers:
            asset_counts[ticker] = asset_counts.get(ticker, 0) + 1
            group = str(meta.get("asset_group", pd.Series()).get(ticker, ""))
            if group:
                group_counts[group] = group_counts.get(group, 0) + 1
    asset_freq = pd.DataFrame(
        [{"ticker": ticker, "rebalance_count": count, "asset_group": str(meta.get("asset_group", pd.Series()).get(ticker, ""))} for ticker, count in asset_counts.items()]
    )
    if not asset_freq.empty:
        asset_freq = asset_freq.sort_values(["rebalance_count", "ticker"], ascending=[False, True]).reset_index(drop=True)
    group_freq = pd.DataFrame([{"asset_group": group, "rebalance_count": count} for group, count in group_counts.items()])
    if not group_freq.empty:
        group_freq = group_freq.sort_values(["rebalance_count", "asset_group"], ascending=[False, True]).reset_index(drop=True)
    return asset_freq, group_freq


def _family_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id, sub in results_df.groupby("family_id", sort=True):
        top = sub.sort_values(["goal_score", "net_ann_return"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "family_id": str(family_id),
                "candidate_id": str(top["candidate_id"]),
                "groups": str(top["groups"]),
                "goal_score": _safe_float(top.get("goal_score")),
                "net_ann_return": _safe_float(top.get("net_ann_return")),
                "net_total_return": _safe_float(top.get("net_total_return")),
                "actual_years_to_10x_full": _safe_float(top.get("actual_years_to_10x_full")),
                "hit_rate_10x_5y": _safe_float(top.get("hit_rate_10x_5y")),
                "hit_rate_10x_3y": _safe_float(top.get("hit_rate_10x_3y")),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["goal_score", "net_ann_return"], ascending=[False, False]).reset_index(drop=True)
    return out


def _build_configs(
    families: list[UniverseFamily],
    *,
    lookbacks: list[int],
    rebalances: list[int],
    topks: list[int],
    score_modes: list[str],
    asset_ma_days: list[int],
    market_ma_days: list[int],
    relative_modes: list[bool],
) -> list[RuleConfig]:
    rows: list[RuleConfig] = []
    for family in families:
        for lookback in lookbacks:
            for rebalance in rebalances:
                for top_k in topks:
                    for score_mode in score_modes:
                        for ama in asset_ma_days:
                            for mma in market_ma_days:
                                for rel in relative_modes:
                                    rows.append(
                                        RuleConfig(
                                            family_id=family.family_id,
                                            groups=family.groups,
                                            lookback_days=int(lookback),
                                            rebalance_days=int(rebalance),
                                            top_k=int(top_k),
                                            score_mode=str(score_mode),
                                            asset_ma_days=int(ama),
                                            market_ma_days=int(mma),
                                            relative_to_shy=bool(rel),
                                        )
                                    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Busca causal de regras agressivas para 10x usando o universo local.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--asset-groups", default="data/asset_groups_target_800.csv")
    ap.add_argument("--asset-metadata", default="data/asset_metadata_target_800.csv")
    ap.add_argument("--group-viability", default="results/validation/profit_group_methodology_suite/20260306T221117Z/group_viability.csv")
    ap.add_argument("--net-assumptions-config", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-ticker", default="SPY")
    ap.add_argument("--fallback-ticker", default="SHY")
    ap.add_argument("--family-ids", default="")
    ap.add_argument("--lookbacks", default="63,126,252")
    ap.add_argument("--rebalances", default="5,21")
    ap.add_argument("--topks", default="1,2")
    ap.add_argument("--score-modes", default="mom_total,mom_vol_adj")
    ap.add_argument("--asset-ma-days", default="0,200")
    ap.add_argument("--market-ma-days", default="0,200")
    ap.add_argument("--relative-modes", default="0,1")
    ap.add_argument("--min-history-days", type=int, default=252)
    ap.add_argument("--max-abs-daily-return", type=float, default=2.0)
    ap.add_argument("--start-date", default="2016-02-18")
    ap.add_argument("--end-date", default="")
    ap.add_argument("--research-top-n", type=int, default=40)
    ap.add_argument("--outdir-root", default="results/validation/profit_10x_rule_search")
    args = ap.parse_args()

    prices_dir = (ROOT / args.prices_dir).resolve()
    asset_groups_csv = (ROOT / args.asset_groups).resolve()
    asset_metadata_csv = (ROOT / args.asset_metadata).resolve()
    group_viability_csv = (ROOT / args.group_viability).resolve()
    net_cfg_path = (ROOT / args.net_assumptions_config).resolve()
    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    lookbacks = [int(x.strip()) for x in str(args.lookbacks).split(",") if str(x).strip()]
    rebalances = [int(x.strip()) for x in str(args.rebalances).split(",") if str(x).strip()]
    topks = [int(x.strip()) for x in str(args.topks).split(",") if str(x).strip()]
    score_modes = [str(x).strip() for x in str(args.score_modes).split(",") if str(x).strip()]
    asset_ma_days = [int(x.strip()) for x in str(args.asset_ma_days).split(",") if str(x).strip()]
    market_ma_days = [int(x.strip()) for x in str(args.market_ma_days).split(",") if str(x).strip()]
    relative_modes = [bool(int(x.strip())) for x in str(args.relative_modes).split(",") if str(x).strip()]

    asset_table = _load_asset_table(asset_groups_csv, asset_metadata_csv)
    families = _build_universe_families(asset_table, group_viability_csv)
    if str(args.family_ids).strip():
        wanted = {token.strip() for token in str(args.family_ids).split(",") if token.strip()}
        families = [family for family in families if family.family_id in wanted]
    if not families:
        raise SystemExit("no universe families selected")
    raw_returns, raw_prices, asset_viability = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=asset_table,
        min_history_days=int(args.min_history_days),
        max_abs_daily_return=float(args.max_abs_daily_return),
    )
    if raw_returns.empty:
        raise SystemExit("no asset returns loaded")
    returns, prices = _ensure_benchmark_columns(raw_returns, raw_prices, prices_dir, [str(args.benchmark_ticker), str(args.fallback_ticker)])
    returns = returns.sort_index()
    prices = prices.reindex(returns.index).sort_index()
    if str(args.start_date).strip():
        start_dt = pd.Timestamp(str(args.start_date).strip())
        returns = returns[returns.index >= start_dt].copy()
        prices = prices[prices.index >= start_dt].copy()
    if str(args.end_date).strip():
        end_dt = pd.Timestamp(str(args.end_date).strip())
        returns = returns[returns.index <= end_dt].copy()
        prices = prices[prices.index <= end_dt].copy()
    if str(args.benchmark_ticker) not in returns.columns:
        raise SystemExit(f"benchmark ticker not found: {args.benchmark_ticker}")
    if returns.empty:
        raise SystemExit("returns frame empty after date filter")

    score_map, asset_ma_filters, benchmark_filters = _precompute_scores(
        returns,
        prices,
        lookbacks=lookbacks,
        asset_ma_days_list=sorted(set([0, *asset_ma_days, *market_ma_days])),
        benchmark_ticker=str(args.benchmark_ticker),
    )
    net_profiles = load_net_assumption_profiles(net_cfg_path)
    configs = _build_configs(
        families,
        lookbacks=lookbacks,
        rebalances=rebalances,
        topks=topks,
        score_modes=score_modes,
        asset_ma_days=asset_ma_days,
        market_ma_days=market_ma_days,
        relative_modes=relative_modes,
    )

    candidate_rows: list[dict[str, Any]] = []
    selection_tables: dict[str, pd.DataFrame] = {}
    for cfg in configs:
        row, selection_df = _simulate_candidate(
            cfg=cfg,
            returns=returns,
            prices=prices,
            asset_table=asset_table,
            score_map=score_map,
            asset_ma_filters=asset_ma_filters,
            benchmark_filters=benchmark_filters,
            benchmark_ticker=str(args.benchmark_ticker),
            fallback_ticker=str(args.fallback_ticker),
            net_profiles=net_profiles,
        )
        if not row:
            continue
        candidate_rows.append(row)
        selection_tables[cfg.candidate_id] = selection_df

    results_df = pd.DataFrame(candidate_rows).sort_values(
        ["goal_score", "hit_rate_10x_5y", "net_ann_return", "actual_years_to_10x_full"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    if results_df.empty:
        raise SystemExit("no candidate rows produced")

    top_candidate = results_df.iloc[0].to_dict()
    best_speed = results_df.sort_values(["actual_years_to_10x_full", "net_ann_return"], ascending=[True, False], na_position="last").iloc[0].to_dict()
    best_hit_5y = results_df.sort_values(["hit_rate_10x_5y", "net_ann_return"], ascending=[False, False], na_position="last").iloc[0].to_dict()
    best_net_ann = results_df.sort_values(["net_ann_return", "goal_score"], ascending=[False, False]).iloc[0].to_dict()
    research_seed = results_df.head(int(max(1, args.research_top_n))).copy()

    winner_selection = selection_tables.get(str(top_candidate["candidate_id"]), pd.DataFrame())
    winner_asset_freq, winner_group_freq = _top_pick_frequency(winner_selection, asset_table)
    family_df = _family_summary(results_df)
    family_df.to_csv(outdir / "family_best_rules.csv", index=False)
    results_df.to_csv(outdir / "candidate_compare.csv", index=False)
    asset_viability.to_csv(outdir / "asset_viability.csv", index=False)
    pd.DataFrame([{"family_id": fam.family_id, "groups": ",".join(fam.groups), "n_groups": int(len(fam.groups))} for fam in families]).to_csv(outdir / "universe_families.csv", index=False)
    if not winner_selection.empty:
        winner_selection.to_csv(outdir / "winner_rebalance_log.csv", index=False)
    if not winner_asset_freq.empty:
        winner_asset_freq.to_csv(outdir / "winner_asset_frequency.csv", index=False)
    if not winner_group_freq.empty:
        winner_group_freq.to_csv(outdir / "winner_group_frequency.csv", index=False)

    research_rows = _build_research_rows(research_seed, outdir=outdir, summary_path=outdir / "summary.json")
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal": {
            "start_brl": 100.0,
            "target_brl": 1000.0,
            "multiple": 10.0,
        },
        "outdir": str(outdir),
        "inputs": {
            "prices_dir": str(prices_dir),
            "asset_groups_csv": str(asset_groups_csv),
            "asset_metadata_csv": str(asset_metadata_csv),
            "group_viability_csv": str(group_viability_csv),
            "net_assumptions_config": str(net_cfg_path),
            "benchmark_ticker": str(args.benchmark_ticker),
            "fallback_ticker": str(args.fallback_ticker),
            "family_ids": [fam.family_id for fam in families],
            "start_date": str(args.start_date),
            "end_date": str(args.end_date),
            "lookbacks": lookbacks,
            "rebalances": rebalances,
            "topks": topks,
            "score_modes": score_modes,
            "asset_ma_days": asset_ma_days,
            "market_ma_days": market_ma_days,
            "relative_modes": [int(x) for x in relative_modes],
            "candidate_count": int(results_df.shape[0]),
        },
        "universe": {
            "assets_loaded": int(asset_viability.shape[0]),
            "families_tested": int(len(families)),
            "foreign_asset_share": float((asset_viability["jurisdiction"].astype(str) == "foreign").mean()) if not asset_viability.empty else float("nan"),
        },
        "top_candidates": {
            "best_goal_score": top_candidate,
            "best_hit_rate_5y": best_hit_5y,
            "best_full_sample_speed": best_speed,
            "best_net_ann_return": best_net_ann,
        },
        "insights": [
            f"Melhor regra pelo objetivo 10x: {top_candidate['candidate_id']} com hit_rate_10x_5y={_safe_float(top_candidate.get('hit_rate_10x_5y')):.4f}, net_ann_return={_safe_float(top_candidate.get('net_ann_return')):.4f} e anos para 10x no full sample={_safe_float(top_candidate.get('actual_years_to_10x_full')):.4f}.",
            f"Melhor regra por velocidade full sample: {best_speed['candidate_id']} em {best_speed.get('actual_years_to_10x_full')}.",
            f"Grupos mais frequentes no vencedor: {winner_group_freq.head(5).to_dict(orient='records') if not winner_group_freq.empty else []}.",
        ],
        "official_sources": net_profiles["official_sources"],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "asset_viability_csv": str(outdir / "asset_viability.csv"),
            "universe_families_csv": str(outdir / "universe_families.csv"),
            "family_best_rules_csv": str(outdir / "family_best_rules.csv"),
            "winner_rebalance_log_csv": str(outdir / "winner_rebalance_log.csv"),
            "winner_asset_frequency_csv": str(outdir / "winner_asset_frequency.csv"),
            "winner_group_frequency_csv": str(outdir / "winner_group_frequency.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir=outdir,
        script="scripts/bench/validation/run_profit_10x_rule_search.py",
        params={
            "prices_dir": str(prices_dir),
            "asset_groups": str(asset_groups_csv),
            "asset_metadata": str(asset_metadata_csv),
            "group_viability": str(group_viability_csv),
            "net_assumptions_config": str(net_cfg_path),
            "candidate_count": int(results_df.shape[0]),
            "family_ids": [fam.family_id for fam in families],
            "start_date": str(args.start_date),
            "end_date": str(args.end_date),
            "lookbacks": lookbacks,
            "rebalances": rebalances,
            "topks": topks,
            "score_modes": score_modes,
            "asset_ma_days": asset_ma_days,
            "market_ma_days": market_ma_days,
            "relative_modes": [int(x) for x in relative_modes],
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
        gates={"summary_created": True, "candidate_compare_created": True},
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "summary_json": str(outdir / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
