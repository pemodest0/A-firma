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

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRICES_DIR = ROOT / "data" / "raw" / "finance" / "yfinance_daily"


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


def _resolve_path(raw: str | Path | None) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _json_weight_map(raw: Any) -> dict[str, float]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        weight = _safe_float(value)
        if np.isfinite(weight) and abs(weight) > 1e-14:
            out[str(key)] = float(weight)
    return out


def _weight_json(weights: dict[str, float]) -> str:
    clean = {str(k): float(v) for k, v in sorted(weights.items()) if abs(float(v)) > 1e-14}
    return json.dumps(clean, sort_keys=True)


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


def _load_returns_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"returns_wide missing date column: {path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    cols = [c for c in df.columns if c != "date"]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.set_index("date")[cols].astype(float).sort_index()


def _load_price_returns(prices_dir: Path, ticker: str) -> pd.Series:
    path = prices_dir / f"{ticker}.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    try:
        out = load_return_series_csv(path, source_kind="log", target_kind="simple", series_name=ticker)
    except ValueError:
        return pd.Series(dtype=float)
    return out.astype(float)


def _load_asset_to_sector(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "asset" not in df.columns or "group" not in df.columns:
            continue
        for _, row in df.iterrows():
            asset = str(row.get("asset", "")).strip()
            group = str(row.get("group", "")).strip()
            if asset and group and asset not in out:
                out[asset] = group
    return out


def _build_candidate_context(candidate_node: dict[str, Any]) -> dict[str, Any]:
    profile_dir = _resolve_path(candidate_node.get("profile_dir"))
    if profile_dir is None or not profile_dir.exists():
        raise SystemExit(f"missing profile_dir in lock: {candidate_node}")
    monthly_path = profile_dir / "monthly_systematic_eval.csv"
    daily_path = profile_dir / "daily_replay.csv"
    if not monthly_path.exists() or not daily_path.exists():
        raise SystemExit(f"missing monthly/daily artifacts for {profile_dir}")

    manifest_path = profile_dir / "RUN_MANIFEST.json"
    manifests: list[dict[str, Any]] = []
    manifest_paths: list[Path] = []
    if manifest_path.exists():
        manifests.append(_read_json(manifest_path))
        manifest_paths.append(manifest_path)
    else:
        for path in sorted(profile_dir.parent.glob(f"{profile_dir.name}__*/RUN_MANIFEST.json")):
            payload = _read_json(path)
            if payload:
                manifests.append(payload)
                manifest_paths.append(path)
    if not manifests:
        raise SystemExit(f"missing manifest context for {profile_dir}")

    first_params = manifests[0].get("params", {}) if isinstance(manifests[0].get("params"), dict) else {}
    returns_csv = _resolve_path(first_params.get("returns_csv"))
    benchmark_symbol = str(first_params.get("benchmark_symbol", "SPY")).strip() or "SPY"
    execution_csvs = []
    for payload in manifests:
        params = payload.get("params", {}) if isinstance(payload.get("params"), dict) else {}
        candidate_csv = _resolve_path(params.get("execution_universe_csv"))
        if candidate_csv is not None:
            execution_csvs.append(candidate_csv)
    if returns_csv is None or not returns_csv.exists():
        raise SystemExit(f"missing returns_csv for {profile_dir}")

    monthly = pd.read_csv(monthly_path)
    monthly["ym"] = monthly["ym"].astype(str)
    daily = pd.read_csv(daily_path)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return {
        "profile_dir": profile_dir,
        "monthly": monthly,
        "daily": daily,
        "returns_wide": _load_returns_wide(returns_csv),
        "benchmark_symbol": benchmark_symbol,
        "asset_to_sector": _load_asset_to_sector(execution_csvs),
        "manifest_paths": [str(p) for p in manifest_paths],
    }


def build_daily_replay_with_delay(
    *,
    monthly_eval: pd.DataFrame,
    returns_wide: pd.DataFrame,
    benchmark_returns: pd.Series,
    initial_capital: float,
    execution_delay_days: int,
) -> pd.DataFrame:
    if monthly_eval.empty or returns_wide.empty:
        return pd.DataFrame()
    month_rows = monthly_eval.copy()
    month_rows["ym"] = month_rows["ym"].astype(str)
    month_rows = month_rows.drop_duplicates(subset=["ym"], keep="last").sort_values("ym").reset_index(drop=True)
    benchmark = pd.to_numeric(benchmark_returns, errors="coerce").reindex(returns_wide.index).fillna(0.0).astype(float)

    capital = float(initial_capital)
    benchmark_capital = float(initial_capital)
    rows: list[dict[str, Any]] = []
    prev_row: pd.Series | None = None
    delay_days = int(max(0, execution_delay_days))

    for _, month_row in month_rows.iterrows():
        ym = str(month_row["ym"])
        period = pd.Period(ym, freq="M")
        month_mask = returns_wide.index.to_period("M") == period
        if not bool(month_mask.any()):
            prev_row = month_row
            continue
        month_dates = returns_wide.index[month_mask]
        for pos, dt in enumerate(month_dates):
            active_row = prev_row if prev_row is not None and pos < delay_days else month_row
            weights = _json_weight_map(active_row.get("executed_weights_json", "{}"))
            cash_weight = _safe_float(active_row.get("cash_weight"), 0.0)
            hedge_weight = _safe_float(active_row.get("hedge_weight"), 0.0)
            selected_assets = str(active_row.get("executed_assets", "")).strip()
            risk_bucket = str(active_row.get("risk_bucket", "")).strip()
            ret_row = returns_wide.loc[dt]
            core_ret = 0.0
            for asset_id, weight in weights.items():
                if asset_id in ret_row.index:
                    core_ret += float(weight) * float(ret_row[asset_id])
            bench_ret = float(benchmark.loc[dt])
            day_ret = float(core_ret + hedge_weight * bench_ret + cash_weight * 0.0)
            capital *= 1.0 + day_ret
            benchmark_capital *= 1.0 + bench_ret
            rows.append(
                {
                    "date": dt.date().isoformat(),
                    "ym": ym,
                    "risk_bucket": risk_bucket,
                    "selected_assets": selected_assets,
                    "n_assets": int(len(weights)),
                    "cash_weight": float(cash_weight),
                    "hedge_weight": float(hedge_weight),
                    "gross_exposure": float(sum(abs(float(v)) for v in weights.values()) + abs(float(hedge_weight))),
                    "net_exposure": float(sum(float(v) for v in weights.values()) + float(hedge_weight)),
                    "portfolio_return": float(day_ret),
                    "benchmark_return": float(bench_ret),
                    "capital": float(capital),
                    "benchmark_capital": float(benchmark_capital),
                }
            )
        prev_row = month_row

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["capital_peak"] = pd.to_numeric(out["capital"], errors="coerce").cummax()
    out["drawdown"] = pd.to_numeric(out["capital"], errors="coerce") / out["capital_peak"] - 1.0
    return out


def _summarize_history(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {"status": "empty"}
    perf = _perf_from_simple_returns(pd.to_numeric(history["portfolio_return"], errors="coerce"))
    bench = _perf_from_simple_returns(pd.to_numeric(history["benchmark_return"], errors="coerce"))
    return {
        "status": "ok",
        "n_days": int(history.shape[0]),
        "start_date": str(history.iloc[0]["date"]),
        "end_date": str(history.iloc[-1]["date"]),
        "portfolio": perf,
        "benchmark": bench,
        "edge_total_return": _safe_float(perf.get("total_return")) - _safe_float(bench.get("total_return")),
        "edge_ann_return": _safe_float(perf.get("ann_return")) - _safe_float(bench.get("ann_return")),
    }


def _enforce_sector_caps(weights: dict[str, float], asset_to_sector: dict[str, str], sector_cap: float) -> dict[str, float]:
    if sector_cap <= 0.0:
        return dict(weights)
    out = {str(k): float(v) for k, v in weights.items()}
    for _ in range(6):
        sector_totals: dict[str, float] = {}
        for asset, weight in out.items():
            sector = asset_to_sector.get(asset, "unknown")
            sector_totals[sector] = sector_totals.get(sector, 0.0) + max(0.0, float(weight))
        changed = False
        for sector, total in sector_totals.items():
            if total <= sector_cap + 1e-12 or total <= 0.0:
                continue
            scale = float(sector_cap) / float(total)
            for asset, weight in list(out.items()):
                if asset_to_sector.get(asset, "unknown") == sector and weight > 0.0:
                    out[asset] = float(weight) * scale
                    changed = True
        if not changed:
            break
    return out


def apply_concentration_caps(
    monthly_eval: pd.DataFrame,
    *,
    asset_to_sector: dict[str, str],
    asset_cap: float,
    sector_cap: float,
    gross_cap: float,
) -> pd.DataFrame:
    out = monthly_eval.copy()
    for idx, row in out.iterrows():
        weights = _json_weight_map(row.get("executed_weights_json", "{}"))
        if not weights:
            continue
        capped = {asset: min(max(float(weight), 0.0), float(asset_cap)) for asset, weight in weights.items()}
        capped = {asset: weight for asset, weight in capped.items() if weight > 1e-14}
        capped = _enforce_sector_caps(capped, asset_to_sector, float(sector_cap))
        hedge_weight = _safe_float(row.get("hedge_weight"), 0.0)
        gross = float(sum(abs(float(v)) for v in capped.values()) + abs(float(hedge_weight)))
        if gross_cap > 0.0 and gross > gross_cap + 1e-12:
            scale = float(gross_cap) / float(gross)
            capped = {asset: float(weight) * scale for asset, weight in capped.items()}
            hedge_weight = float(hedge_weight) * scale
        positive_total = float(sum(max(0.0, float(v)) for v in capped.values()))
        out.at[idx, "executed_weights_json"] = _weight_json(capped)
        out.at[idx, "executed_assets"] = ",".join(sorted(capped.keys()))
        out.at[idx, "selected_assets"] = ",".join(sorted(capped.keys()))
        out.at[idx, "n_selected"] = int(len(capped))
        out.at[idx, "cash_weight"] = float(max(0.0, 1.0 - positive_total))
        out.at[idx, "hedge_weight"] = float(hedge_weight)
        out.at[idx, "core_gross_exposure"] = positive_total
        out.at[idx, "net_exposure"] = positive_total + float(hedge_weight)
    return out


def classify_market_slices(benchmark_returns: pd.Series) -> pd.Series:
    x = pd.to_numeric(benchmark_returns, errors="coerce").fillna(0.0).astype(float)
    if x.empty:
        return pd.Series(dtype=object)
    equity = (1.0 + x).cumprod()
    dd = equity / equity.cummax() - 1.0
    ret63 = equity / equity.shift(63) - 1.0
    ret21 = equity / equity.shift(21) - 1.0
    labels = pd.Series("sideways", index=x.index, dtype=object)
    labels.loc[(dd <= -0.18) & (ret21 > 0.05)] = "recovery"
    labels.loc[(dd <= -0.18) & ~(ret21 > 0.05)] = "bear"
    labels.loc[(ret63 <= -0.10) & ~(dd <= -0.18)] = "bear"
    labels.loc[(ret63 >= 0.08) & ~(dd <= -0.18)] = "bull"
    warmup = ret63.isna() | ret21.isna()
    labels.loc[warmup] = "warmup"
    return labels


def summarize_market_slices(history: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(history["date"], errors="coerce")
    bench = pd.to_numeric(benchmark_returns, errors="coerce").reindex(idx).fillna(0.0).astype(float)
    labels = classify_market_slices(bench)
    rows: list[dict[str, Any]] = []
    for label in ["bull", "bear", "recovery", "sideways"]:
        mask = labels == label
        if not bool(mask.any()):
            continue
        strat = pd.to_numeric(history.loc[mask.to_numpy(dtype=bool), "portfolio_return"], errors="coerce").fillna(0.0)
        bench_slice = bench.loc[mask]
        perf_s = _perf_from_simple_returns(strat)
        perf_b = _perf_from_simple_returns(bench_slice)
        rows.append(
            {
                "slice": label,
                "n_days": int(mask.sum()),
                "share_days": float(mask.mean()),
                "strategy_total_return": _safe_float(perf_s.get("total_return")),
                "strategy_ann_return": _safe_float(perf_s.get("ann_return")),
                "strategy_sharpe": _safe_float(perf_s.get("sharpe")),
                "strategy_max_drawdown": _safe_float(perf_s.get("max_drawdown")),
                "benchmark_total_return": _safe_float(perf_b.get("total_return")),
                "benchmark_ann_return": _safe_float(perf_b.get("ann_return")),
                "alpha_total_return": _safe_float(perf_s.get("total_return")) - _safe_float(perf_b.get("total_return")),
                "alpha_ann_return": _safe_float(perf_s.get("ann_return")) - _safe_float(perf_b.get("ann_return")),
            }
        )
    return pd.DataFrame(rows)


def _candidate_rows(label: str, candidate_node: dict[str, Any], prices_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ctx = _build_candidate_context(candidate_node)
    monthly = ctx["monthly"]
    returns_wide = ctx["returns_wide"]
    benchmark_symbol = ctx["benchmark_symbol"]
    benchmark_series = _load_price_returns(prices_dir, benchmark_symbol)
    if benchmark_series.empty:
        benchmark_series = pd.Series(np.zeros(len(returns_wide), dtype=float), index=returns_wide.index, dtype=float)

    delay_rows: list[dict[str, Any]] = []
    for delay_days in [0, 1, 2]:
        history = build_daily_replay_with_delay(
            monthly_eval=monthly,
            returns_wide=returns_wide,
            benchmark_returns=benchmark_series,
            initial_capital=10000.0,
            execution_delay_days=delay_days,
        )
        summary = _summarize_history(history)
        delay_rows.append(
            {
                "candidate": label,
                "scenario": f"delay_d{delay_days}",
                "delay_days": int(delay_days),
                "portfolio_total_return": _safe_float(((summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}).get("total_return")),
                "portfolio_ann_return": _safe_float(((summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}).get("ann_return")),
                "portfolio_sharpe": _safe_float(((summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}).get("sharpe")),
                "portfolio_max_drawdown": _safe_float(((summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}).get("max_drawdown")),
                "edge_total_return": _safe_float(summary.get("edge_total_return")),
                "edge_ann_return": _safe_float(summary.get("edge_ann_return")),
            }
        )

    caps_rows: list[dict[str, Any]] = []
    cap_scenarios = [
        {"name": "caps_moderate", "asset_cap": 0.08, "sector_cap": 0.30, "gross_cap": 1.10},
        {"name": "caps_strict", "asset_cap": 0.06, "sector_cap": 0.25, "gross_cap": 1.00},
    ]
    for scenario in cap_scenarios:
        capped_monthly = apply_concentration_caps(
            monthly,
            asset_to_sector=ctx["asset_to_sector"],
            asset_cap=float(scenario["asset_cap"]),
            sector_cap=float(scenario["sector_cap"]),
            gross_cap=float(scenario["gross_cap"]),
        )
        history = build_daily_replay_with_delay(
            monthly_eval=capped_monthly,
            returns_wide=returns_wide,
            benchmark_returns=benchmark_series,
            initial_capital=10000.0,
            execution_delay_days=0,
        )
        summary = _summarize_history(history)
        caps_rows.append(
            {
                "candidate": label,
                "scenario": str(scenario["name"]),
                "asset_cap": float(scenario["asset_cap"]),
                "sector_cap": float(scenario["sector_cap"]),
                "gross_cap": float(scenario["gross_cap"]),
                "portfolio_total_return": _safe_float(((summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}).get("total_return")),
                "portfolio_ann_return": _safe_float(((summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}).get("ann_return")),
                "portfolio_sharpe": _safe_float(((summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}).get("sharpe")),
                "portfolio_max_drawdown": _safe_float(((summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}).get("max_drawdown")),
                "edge_total_return": _safe_float(summary.get("edge_total_return")),
                "edge_ann_return": _safe_float(summary.get("edge_ann_return")),
            }
        )

    baseline_history = build_daily_replay_with_delay(
        monthly_eval=monthly,
        returns_wide=returns_wide,
        benchmark_returns=benchmark_series,
        initial_capital=10000.0,
        execution_delay_days=0,
    )
    slices = summarize_market_slices(baseline_history, benchmark_series)
    if not slices.empty:
        slices.insert(0, "candidate", label)
    return pd.DataFrame(delay_rows), pd.DataFrame(caps_rows), slices


def main() -> None:
    ap = argparse.ArgumentParser(description="Reality-oriented simulation battery for canonical profit shadow candidates.")
    ap.add_argument("--lock-path", required=True)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--prices-dir", default=str(DEFAULT_PRICES_DIR))
    args = ap.parse_args()

    lock_path = Path(args.lock_path).resolve()
    lock = _read_json(lock_path)
    if not lock:
        raise SystemExit(f"missing lock path: {lock_path}")
    outdir = _resolve_path(args.outdir) or (ROOT / "results" / "validation" / "profit_shadow_realism_battery" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = _resolve_path(args.prices_dir)
    if prices_dir is None or not prices_dir.exists():
        raise SystemExit(f"missing prices dir: {args.prices_dir}")

    frames_delay: list[pd.DataFrame] = []
    frames_caps: list[pd.DataFrame] = []
    frames_slices: list[pd.DataFrame] = []
    for label in ["main", "challenger"]:
        candidate_node = lock.get(label, {})
        if not isinstance(candidate_node, dict):
            continue
        delay_df, caps_df, slices_df = _candidate_rows(label, candidate_node, prices_dir)
        frames_delay.append(delay_df)
        frames_caps.append(caps_df)
        frames_slices.append(slices_df)

    delay_df = pd.concat(frames_delay, ignore_index=True) if frames_delay else pd.DataFrame()
    caps_df = pd.concat(frames_caps, ignore_index=True) if frames_caps else pd.DataFrame()
    slices_df = pd.concat(frames_slices, ignore_index=True) if frames_slices else pd.DataFrame()

    if not delay_df.empty:
        delay_df.to_csv(outdir / "execution_delay.csv", index=False)
    if not caps_df.empty:
        caps_df.to_csv(outdir / "concentration_caps.csv", index=False)
    if not slices_df.empty:
        slices_df.to_csv(outdir / "market_slices.csv", index=False)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": str(lock_path),
        "outdir": str(outdir),
        "delay_findings": delay_df.to_dict(orient="records"),
        "caps_findings": caps_df.to_dict(orient="records"),
        "slice_findings": slices_df.to_dict(orient="records"),
    }
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
