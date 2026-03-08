#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench.validation.run_profit_attack_validation_suite import (  # noqa: E402
    _load_price_returns,
    _load_returns_wide,
    build_daily_replay_with_rebalance,
    summarize_replay,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_candidate(raw: str) -> tuple[str, Path]:
    text = str(raw).strip()
    if "=" not in text:
        raise ValueError(f"candidate must be label=path, got: {raw}")
    label, path_value = text.split("=", 1)
    label = label.strip()
    path = Path(path_value.strip())
    if not label or not str(path).strip():
        raise ValueError(f"invalid candidate spec: {raw}")
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return label, path


def _load_profile_context(profile_dir: Path, prices_dir_override: Path | None = None) -> dict[str, Any]:
    manifest = _read_json(profile_dir / "RUN_MANIFEST.json")
    params = manifest.get("params", {}) if isinstance(manifest.get("params"), dict) else {}
    if not params:
        sim_summary = _read_json(profile_dir / "simulation_summary.json")
        members = sim_summary.get("ensemble_members", []) if isinstance(sim_summary.get("ensemble_members"), list) else []
        if members:
            member_dir = profile_dir.parent / str(members[0]).strip()
            member_manifest = _read_json(member_dir / "RUN_MANIFEST.json")
            params = member_manifest.get("params", {}) if isinstance(member_manifest.get("params"), dict) else {}
    monthly_path = profile_dir / "monthly_systematic_eval.csv"
    if not monthly_path.exists():
        raise FileNotFoundError(f"missing monthly eval: {monthly_path}")
    returns_csv = Path(str(params.get("returns_csv", "")).strip())
    if not returns_csv.is_absolute():
        returns_csv = (ROOT / returns_csv).resolve()
    if not returns_csv.exists():
        raise FileNotFoundError(f"missing returns csv from manifest: {returns_csv}")
    returns_wide = _load_returns_wide(returns_csv)
    benchmark_symbol = str(params.get("benchmark_symbol", "SPY")).strip() or "SPY"
    prices_dir_raw = str(params.get("prices_dir", "")).strip()
    prices_dir = prices_dir_override or (
        Path(prices_dir_raw).resolve() if prices_dir_raw else (ROOT / "data" / "raw" / "finance" / "yfinance_daily")
    )
    benchmark_returns: pd.Series | None = None
    try:
        benchmark_returns = _load_price_returns(prices_dir, benchmark_symbol)
    except FileNotFoundError:
        if benchmark_symbol not in returns_wide.columns:
            raise
    return {
        "manifest": manifest,
        "params": params,
        "monthly_eval": pd.read_csv(monthly_path),
        "returns_wide": returns_wide,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_returns": benchmark_returns,
        "initial_capital": 10000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare cost-stress behavior across profit shadow candidate profiles.")
    ap.add_argument("--candidate", action="append", required=True, help="Candidate in label=/abs/or/relative/profile_dir format. Repeatable.")
    ap.add_argument("--outdir", default="", help="Output dir (default: results/validation/profit_shadow_cost_stress_compare/<run_id>)")
    ap.add_argument("--cost-bps-list", default="10,20,30,50")
    ap.add_argument("--frequencies", default="monthly,weekly")
    ap.add_argument("--prices-dir", default="", help="Optional benchmark price directory override.")
    args = ap.parse_args()

    run_id = _run_id()
    outdir = Path(str(args.outdir)).resolve() if str(args.outdir).strip() else (ROOT / "results" / "validation" / "profit_shadow_cost_stress_compare" / run_id)
    outdir.mkdir(parents=True, exist_ok=True)

    costs = [float(x.strip()) for x in str(args.cost_bps_list).split(",") if x.strip()]
    frequencies = [str(x).strip().lower() for x in str(args.frequencies).split(",") if str(x).strip()]
    candidates = [_parse_candidate(raw) for raw in args.candidate]
    prices_dir_override = Path(str(args.prices_dir)).resolve() if str(args.prices_dir).strip() else None

    rows: list[dict[str, Any]] = []
    for label, profile_dir in candidates:
        ctx = _load_profile_context(profile_dir, prices_dir_override)
        monthly_eval = ctx["monthly_eval"]
        returns_wide = ctx["returns_wide"]
        benchmark_symbol = str(ctx["benchmark_symbol"])
        benchmark_returns = ctx.get("benchmark_returns")
        initial_capital = float(ctx["initial_capital"])
        candidate_root = outdir / label
        candidate_root.mkdir(parents=True, exist_ok=True)
        for freq in frequencies:
            for cost_bps in costs:
                history = build_daily_replay_with_rebalance(
                    monthly_eval=monthly_eval,
                    returns_wide=returns_wide,
                    benchmark_symbol=benchmark_symbol,
                    benchmark_returns=benchmark_returns,
                    initial_capital=initial_capital,
                    cost_bps=float(cost_bps),
                    rebalance_frequency=freq,
                )
                history_path = candidate_root / f"{freq}_{int(cost_bps)}bps_daily.csv"
                history.to_csv(history_path, index=False)
                summary = summarize_replay(history, return_col="net_return")
                (history_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                rows.append(
                    {
                        "candidate": label,
                        "profile_dir": str(profile_dir),
                        "rebalance_frequency": freq,
                        "cost_bps": float(cost_bps),
                        "daily_total_return": _safe_float(summary.get("portfolio", {}).get("total_return")),
                        "daily_ann_return": _safe_float(summary.get("portfolio", {}).get("ann_return")),
                        "daily_sharpe": _safe_float(summary.get("portfolio", {}).get("sharpe")),
                        "daily_max_drawdown": _safe_float(summary.get("portfolio", {}).get("max_drawdown")),
                        "edge_vs_benchmark_total_return": _safe_float(summary.get("edge_vs_benchmark_total_return")),
                        "avg_turnover_daily": _safe_float(summary.get("avg_turnover_daily")),
                        "total_turnover": _safe_float(summary.get("total_turnover")),
                        "capital_end": _safe_float(summary.get("capital_end")),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["rebalance_frequency", "cost_bps", "candidate"]).reset_index(drop=True)
    df.to_csv(outdir / "cost_stress_compare.csv", index=False)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "candidates": [{"label": label, "profile_dir": str(profile_dir)} for label, profile_dir in candidates],
        "cost_bps_list": costs,
        "frequencies": frequencies,
        "rows": rows,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir), "rows": int(len(rows))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
