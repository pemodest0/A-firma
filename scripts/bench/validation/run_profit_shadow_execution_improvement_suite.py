#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench.validation.run_profit_attack_validation_suite import (  # noqa: E402
    _l1_turnover,
    _perf_from_simple_returns,
    build_daily_replay_with_rebalance,
    summarize_replay,
)
from scripts.bench.validation.run_profit_shadow_realism_battery import (  # noqa: E402
    _build_candidate_context,
    _json_weight_map,
    _load_price_returns,
    _resolve_path,
    _safe_float,
    _weight_json,
    build_daily_replay_with_delay,
    classify_market_slices,
)


CAP_PROFILES: dict[str, dict[str, float]] = {
    "none": {"warmup": 1.0, "bull": 1.0, "recovery": 1.0, "sideways": 1.0, "bear": 1.0},
    "mild": {"warmup": 0.70, "bull": 0.85, "recovery": 0.80, "sideways": 0.70, "bear": 0.55},
    "moderate": {"warmup": 0.60, "bull": 0.75, "recovery": 0.70, "sideways": 0.60, "bear": 0.45},
}

STATE_SLEEVE_MAP: dict[str, list[str]] = {
    "bull": ["SPY", "QQQ", "XLK", "XLY", "XLI"],
    "recovery": ["SPY", "RSP", "XLI", "XLF", "XLK"],
    "sideways": ["SPY", "XLV", "XLP", "XLU", "RSP"],
    "bear": ["XLV", "XLP", "XLU", "XLRE", "IEF"],
    "warmup": ["SPY", "RSP", "XLV", "XLK", "XLI"],
}


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


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _month_labels(series: pd.Series) -> dict[str, str]:
    labels = classify_market_slices(series)
    if labels.empty:
        return {}
    by_month = (
        pd.DataFrame({"ym": labels.index.to_period("M").astype(str), "label": labels.astype(str)})
        .groupby("ym", as_index=True)["label"]
        .last()
    )
    return {str(k): str(v) for k, v in by_month.items()}


def _prev_month(ym: str) -> str:
    return (pd.Period(str(ym), freq="M") - 1).strftime("%Y-%m")


def _load_multi_price_returns(prices_dir: Path, tickers: list[str]) -> pd.DataFrame:
    parts: list[pd.Series] = []
    for ticker in tickers:
        series = _load_price_returns(prices_dir, ticker)
        if not series.empty:
            parts.append(series.rename(str(ticker)))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1).sort_index()


def _augment_returns_wide(returns_wide: pd.DataFrame, prices_dir: Path, tickers: list[str]) -> pd.DataFrame:
    extra = _load_multi_price_returns(prices_dir, tickers)
    if extra.empty:
        return returns_wide.copy()
    aligned = extra.reindex(returns_wide.index).fillna(0.0)
    out = returns_wide.copy()
    for col in aligned.columns:
        if col not in out.columns:
            out[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0).astype(float)
    return out.sort_index()


def _sanitize_json_value(x: Any) -> Any:
    if isinstance(x, float):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _sanitize_json_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_json_value(v) for v in x]
    return x


def _top_sector_weight(weights: dict[str, float], asset_to_sector: dict[str, str]) -> float:
    if not weights:
        return 0.0
    totals: dict[str, float] = {}
    for asset, weight in weights.items():
        sector = str(asset_to_sector.get(asset, "unknown")).strip() or "unknown"
        totals[sector] = float(totals.get(sector, 0.0)) + float(max(0.0, weight))
    return float(max(totals.values())) if totals else 0.0


def _apply_corr_penalty(
    weights: dict[str, float],
    trailing_returns: pd.DataFrame,
    *,
    strength: float,
    target_total: float,
) -> dict[str, float]:
    if strength <= 1e-12 or len(weights) <= 1 or trailing_returns.empty:
        return {str(k): float(v) for k, v in weights.items()}
    assets = [asset for asset in weights if asset in trailing_returns.columns]
    if len(assets) <= 1:
        return {str(k): float(v) for k, v in weights.items()}
    corr = trailing_returns[assets].corr().fillna(0.0)
    scores: dict[str, float] = {}
    for asset in assets:
        row = pd.to_numeric(corr.loc[asset], errors="coerce").drop(labels=[asset], errors="ignore").fillna(0.0)
        avg_pos = float(row.clip(lower=0.0).mean()) if not row.empty else 0.0
        scores[asset] = float(max(0.05, 1.0 - float(strength) * avg_pos))
    adjusted: dict[str, float] = {}
    for asset, weight in weights.items():
        adjusted[str(asset)] = float(weight) * float(scores.get(asset, 1.0))
    total = float(sum(max(0.0, v) for v in adjusted.values()))
    if total <= 1e-12:
        return {str(k): float(v) for k, v in weights.items()}
    scale = float(target_total) / total
    return {asset: float(weight) * scale for asset, weight in adjusted.items() if float(weight) > 1e-14}


def _apply_sector_cap(weights: dict[str, float], asset_to_sector: dict[str, str], *, cap_fraction: float) -> tuple[dict[str, float], float]:
    if cap_fraction >= 0.999999:
        return {str(k): float(v) for k, v in weights.items()}, 0.0
    target_total = float(sum(max(0.0, float(v)) for v in weights.values()))
    if target_total <= 1e-12:
        return {}, 0.0
    cap_abs = float(cap_fraction) * target_total
    sector_totals: dict[str, float] = {}
    for asset, weight in weights.items():
        sector = str(asset_to_sector.get(asset, "unknown")).strip() or "unknown"
        sector_totals[sector] = float(sector_totals.get(sector, 0.0)) + float(max(0.0, weight))
    out: dict[str, float] = {}
    for asset, weight in weights.items():
        sector = str(asset_to_sector.get(asset, "unknown")).strip() or "unknown"
        sector_total = float(sector_totals.get(sector, 0.0))
        if sector_total > cap_abs + 1e-12 and sector_total > 1e-12:
            out[str(asset)] = float(weight) * float(cap_abs / sector_total)
        else:
            out[str(asset)] = float(weight)
    kept_total = float(sum(max(0.0, v) for v in out.values()))
    freed = max(0.0, target_total - kept_total)
    return {asset: float(weight) for asset, weight in out.items() if float(weight) > 1e-14}, float(freed)


def _equal_weight_map(assets: list[str], total_weight: float) -> dict[str, float]:
    clean = [str(asset) for asset in assets if str(asset).strip()]
    if not clean or total_weight <= 1e-12:
        return {}
    ew = float(total_weight) / float(len(clean))
    return {asset: ew for asset in clean}


def _combine_weight_maps(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for asset in sorted(set(a) | set(b)):
        weight = float(a.get(asset, 0.0)) + float(b.get(asset, 0.0))
        if weight > 1e-14:
            out[str(asset)] = float(weight)
    return out


def _policy_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"policy_id": "baseline", "cap_profile": "none", "corr_strength": 0.0, "sleeve_share": 0.0, "sleeve_trigger": 1.01}]
    for cap_profile in ["mild", "moderate"]:
        for corr_strength in [0.0, 0.25, 0.50]:
            for sleeve_share in [0.0, 0.10, 0.20]:
                policy_id = f"cap-{cap_profile}_corr-{int(round(corr_strength*100)):02d}_sleeve-{int(round(sleeve_share*100)):02d}"
                rows.append(
                    {
                        "policy_id": policy_id,
                        "cap_profile": cap_profile,
                        "corr_strength": float(corr_strength),
                        "sleeve_share": float(sleeve_share),
                        "sleeve_trigger": 0.65,
                    }
                )
    return rows


def _state_for_month(ym: str, market_state_by_month: dict[str, str]) -> str:
    return str(market_state_by_month.get(_prev_month(ym), "warmup")).strip().lower() or "warmup"


def _sleeve_assets_for_state(state: str, available_assets: set[str]) -> list[str]:
    candidates = STATE_SLEEVE_MAP.get(str(state).strip().lower(), STATE_SLEEVE_MAP["warmup"])
    return [ticker for ticker in candidates if ticker in available_assets]


def _transform_monthly_eval(
    monthly_eval: pd.DataFrame,
    *,
    policy: dict[str, Any],
    asset_to_sector: dict[str, str],
    market_state_by_month: dict[str, str],
    returns_wide: pd.DataFrame,
) -> pd.DataFrame:
    out = monthly_eval.copy()
    available_assets = set(str(col) for col in returns_wide.columns)
    out["ym"] = out["ym"].astype(str)
    for idx, row in out.iterrows():
        ym = str(row.get("ym", "")).strip()
        weights = _json_weight_map(row.get("executed_weights_json", "{}"))
        if not weights:
            continue
        target_total = float(sum(max(0.0, float(v)) for v in weights.values()))
        if target_total <= 1e-12:
            continue
        state = _state_for_month(ym, market_state_by_month)
        period = pd.Period(ym, freq="M")
        trailing = returns_wide.loc[returns_wide.index < period.start_time].tail(126)

        adjusted = _apply_corr_penalty(weights, trailing, strength=float(policy["corr_strength"]), target_total=target_total)

        sleeve_alloc = 0.0
        top_sector = _top_sector_weight(adjusted, asset_to_sector)
        if float(policy["sleeve_share"]) > 1e-12 and top_sector > float(policy["sleeve_trigger"]) * target_total:
            sleeve_alloc = float(policy["sleeve_share"]) * target_total
            scale = max(0.0, target_total - sleeve_alloc) / max(target_total, 1e-12)
            adjusted = {asset: float(weight) * scale for asset, weight in adjusted.items()}

        cap_fraction = float(CAP_PROFILES[str(policy["cap_profile"])].get(state, CAP_PROFILES[str(policy["cap_profile"])]["warmup"]))
        capped, freed = _apply_sector_cap(adjusted, asset_to_sector, cap_fraction=cap_fraction)
        sleeve_total = float(sleeve_alloc + freed)
        sleeve_assets = _sleeve_assets_for_state(state, available_assets)
        if sleeve_total > 1e-12 and sleeve_assets:
            capped = _combine_weight_maps(capped, _equal_weight_map(sleeve_assets, sleeve_total))
        elif sleeve_total > 1e-12:
            # No sleeve available: keep the risk budget in cash rather than inventing allocation.
            pass

        final_total = float(sum(max(0.0, float(v)) for v in capped.values()))
        original_cash = float(max(0.0, _safe_float(row.get("cash_weight"), 0.0)))
        new_cash = float(max(0.0, min(1.0, original_cash + max(0.0, target_total - final_total))))
        out.at[idx, "executed_weights_json"] = _weight_json(capped)
        out.at[idx, "executed_assets"] = ",".join(sorted(capped.keys()))
        out.at[idx, "selected_assets"] = ",".join(sorted(capped.keys()))
        out.at[idx, "n_selected"] = int(len(capped))
        out.at[idx, "cash_weight"] = new_cash
        out.at[idx, "policy_state"] = state
        out.at[idx, "policy_id"] = str(policy["policy_id"])
        out.at[idx, "policy_top_sector_pre"] = float(top_sector)
        out.at[idx, "policy_core_total_post"] = final_total
        out.at[idx, "policy_sleeve_total"] = float(sleeve_total)
    return out


def _structure_stats(monthly_eval: pd.DataFrame, asset_to_sector: dict[str, str]) -> dict[str, float]:
    if monthly_eval.empty:
        return {}
    top_sector_weights: list[float] = []
    max_asset_weights: list[float] = []
    effective_ns: list[float] = []
    turnovers: list[float] = []
    prev_weights: dict[str, float] = {}
    prev_cash = 1.0
    for _, row in monthly_eval.iterrows():
        weights = _json_weight_map(row.get("executed_weights_json", "{}"))
        cash_weight = float(max(0.0, _safe_float(row.get("cash_weight"), 0.0)))
        top_sector_weights.append(_top_sector_weight(weights, asset_to_sector))
        positives = np.array([max(0.0, float(v)) for v in weights.values()], dtype=float)
        positives = positives[positives > 1e-14]
        max_asset_weights.append(float(positives.max()) if positives.size else 0.0)
        effective_ns.append(float(1.0 / np.square(positives).sum()) if positives.size else 0.0)
        turnovers.append(float(_l1_turnover(prev_weights, prev_cash, weights, cash_weight)))
        prev_weights, prev_cash = dict(weights), float(cash_weight)
    return {
        "avg_top_sector_weight": float(np.mean(top_sector_weights)) if top_sector_weights else float("nan"),
        "avg_max_asset_weight": float(np.mean(max_asset_weights)) if max_asset_weights else float("nan"),
        "avg_effective_n": float(np.mean(effective_ns)) if effective_ns else float("nan"),
        "avg_turnover_l1": float(np.mean(turnovers)) if turnovers else float("nan"),
    }


def _evaluate_policy(
    *,
    policy_id: str,
    monthly_eval: pd.DataFrame,
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_series: pd.Series,
    asset_to_sector: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    structure = _structure_stats(monthly_eval, asset_to_sector)

    delay_history = build_daily_replay_with_delay(
        monthly_eval=monthly_eval,
        returns_wide=returns_wide,
        benchmark_returns=benchmark_series,
        initial_capital=10000.0,
        execution_delay_days=0,
    )
    delay_perf = _perf_from_simple_returns(pd.to_numeric(delay_history["portfolio_return"], errors="coerce"), periods_per_year=252.0)
    delay_bench = _perf_from_simple_returns(pd.to_numeric(delay_history["benchmark_return"], errors="coerce"), periods_per_year=252.0)
    rows.append(
        {
            "policy_id": policy_id,
            "scenario": "delay_d0",
            "ann_return": _safe_float(delay_perf.get("ann_return")),
            "sharpe": _safe_float(delay_perf.get("sharpe")),
            "max_drawdown": _safe_float(delay_perf.get("max_drawdown")),
            "total_return": _safe_float(delay_perf.get("total_return")),
            "edge_total_return": _safe_float(delay_perf.get("total_return")) - _safe_float(delay_bench.get("total_return")),
            **structure,
        }
    )

    for cost_bps in [10.0, 30.0]:
        history = build_daily_replay_with_rebalance(
            monthly_eval=monthly_eval,
            returns_wide=returns_wide,
            benchmark_symbol=benchmark_symbol,
            benchmark_returns=benchmark_series,
            initial_capital=10000.0,
            cost_bps=cost_bps,
            rebalance_frequency="monthly",
        )
        summary = summarize_replay(history, return_col="net_return")
        port = (summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}
        rows.append(
            {
                "policy_id": policy_id,
                "scenario": f"monthly_{int(cost_bps)}bps",
                "ann_return": _safe_float(port.get("ann_return")),
                "sharpe": _safe_float(port.get("sharpe")),
                "max_drawdown": _safe_float(port.get("max_drawdown")),
                "total_return": _safe_float(port.get("total_return")),
                "edge_total_return": _safe_float(summary.get("edge_vs_benchmark_total_return")),
                **structure,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Test execution-level improvements on the new profit-shadow combo.")
    ap.add_argument("--lock-path", required=True)
    ap.add_argument("--candidate-monthly", required=True)
    ap.add_argument("--prices-dir", default=str(ROOT / "data" / "raw" / "finance" / "yfinance_daily"))
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    lock = _read_json(Path(args.lock_path).resolve())
    if not lock:
        raise SystemExit(f"missing lock: {args.lock_path}")
    outdir = _resolve_path(args.outdir) or (ROOT / "results" / "validation" / "profit_shadow_execution_improvement" / _run_id())
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = _resolve_path(args.prices_dir)
    if prices_dir is None or not prices_dir.exists():
        raise SystemExit(f"missing prices_dir: {args.prices_dir}")

    main_ctx = _build_candidate_context(lock.get("main", {}))
    challenger_ctx = _build_candidate_context(lock.get("challenger", {}))
    candidate_monthly = pd.read_csv(Path(args.candidate_monthly).resolve())
    candidate_monthly["ym"] = candidate_monthly["ym"].astype(str)

    asset_to_sector = dict(main_ctx.get("asset_to_sector", {}))
    asset_to_sector.update(challenger_ctx.get("asset_to_sector", {}))

    benchmark_symbol = str(main_ctx["benchmark_symbol"])
    all_sleeve_tickers = sorted({ticker for tickers in STATE_SLEEVE_MAP.values() for ticker in tickers} | {benchmark_symbol})
    returns_wide = _augment_returns_wide(main_ctx["returns_wide"], prices_dir, all_sleeve_tickers)
    benchmark_series = _load_price_returns(prices_dir, benchmark_symbol)
    if benchmark_series.empty:
        benchmark_series = pd.Series(np.zeros(len(returns_wide), dtype=float), index=returns_wide.index, dtype=float)
    benchmark_series = pd.to_numeric(benchmark_series, errors="coerce").reindex(returns_wide.index).fillna(0.0).astype(float)
    market_state_by_month = _month_labels(benchmark_series)

    policies = _policy_grid()
    variant_dir = outdir / "monthly_eval_variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        transformed = _transform_monthly_eval(
            candidate_monthly,
            policy=policy,
            asset_to_sector=asset_to_sector,
            market_state_by_month=market_state_by_month,
            returns_wide=returns_wide,
        )
        transformed.to_csv(variant_dir / f"{policy['policy_id']}.csv", index=False)
        rows.extend(
            _evaluate_policy(
                policy_id=str(policy["policy_id"]),
                monthly_eval=transformed,
                returns_wide=returns_wide,
                benchmark_symbol=benchmark_symbol,
                benchmark_series=benchmark_series,
                asset_to_sector=asset_to_sector,
            )
        )

    results = pd.DataFrame(rows).sort_values(["scenario", "ann_return", "sharpe"], ascending=[True, False, False]).reset_index(drop=True)
    results.to_csv(outdir / "policy_results.csv", index=False)

    baseline = results[results["policy_id"] == "baseline"].set_index("scenario")
    dominating_rows: list[dict[str, Any]] = []
    if not baseline.empty:
        for scenario, group in results.groupby("scenario"):
            base = baseline.loc[scenario] if scenario in baseline.index else None
            if base is None:
                continue
            dom = group[
                (group["ann_return"] >= float(base["ann_return"]) - 1e-12)
                & (group["sharpe"] >= float(base["sharpe"]) - 1e-12)
                & (group["max_drawdown"] >= float(base["max_drawdown"]) - 1e-12)
            ].copy()
            dominating_rows.extend(dom.to_dict(orient="records"))
    dominating_df = pd.DataFrame(dominating_rows).sort_values(["scenario", "ann_return"], ascending=[True, False]).reset_index(drop=True) if dominating_rows else pd.DataFrame()
    dominating_df.to_csv(outdir / "dominating_vs_baseline.csv", index=False)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": str(Path(args.lock_path).resolve()),
        "candidate_monthly": str(Path(args.candidate_monthly).resolve()),
        "n_policies": int(len(policies)),
        "best_by_scenario": {
            scenario: (
                results[results["scenario"] == scenario]
                .sort_values(["ann_return", "sharpe"], ascending=False)
                .head(1)
                .iloc[0]
                .to_dict()
                if not results[results["scenario"] == scenario].empty
                else {}
            )
            for scenario in sorted(results["scenario"].unique())
        },
        "dominating_policy_counts": {
            scenario: int(dominating_df[dominating_df["scenario"] == scenario].shape[0]) for scenario in sorted(results["scenario"].unique())
        },
        "artifacts": {
            "policy_results_csv": str(outdir / "policy_results.csv"),
            "dominating_vs_baseline_csv": str(outdir / "dominating_vs_baseline.csv"),
            "monthly_eval_variants_dir": str(variant_dir),
        },
    }
    _write_json(outdir / "summary.json", _sanitize_json_value(summary))
    print(json.dumps(_sanitize_json_value(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
