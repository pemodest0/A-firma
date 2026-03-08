#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


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


def _perf_from_simple_returns(simple_returns: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(simple_returns, errors="coerce").dropna().astype(float)
    if x.empty:
        return {"total_return": float("nan"), "ann_return": float("nan"), "ann_vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
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
    }


def _window_total(series: pd.Series, window: int) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if x.empty:
        return float("nan")
    tail = x.iloc[-int(max(1, window)) :]
    return float(np.prod(1.0 + tail.to_numpy(dtype=float)) - 1.0)


def _load_candidate_from_lock(lock: dict[str, Any], key: str) -> dict[str, Any]:
    node = lock.get(key, {})
    return node if isinstance(node, dict) else {}


def _load_latest_summary_from_config(config_path: Path) -> tuple[Path | None, dict[str, Any]]:
    cfg = _read_json(config_path)
    shadow_outdir = _resolve_path(cfg.get("shadow_outdir"))
    if shadow_outdir is None:
        return None, {}
    latest_run = _read_json(shadow_outdir / "latest_run.json")
    summary_path = _resolve_path(latest_run.get("summary_path"))
    if summary_path is None or not summary_path.exists():
        summary_path = shadow_outdir / "latest_summary.json"
    return summary_path if summary_path.exists() else None, _read_json(summary_path) if summary_path and summary_path.exists() else {}


def _candidate_payload(label: str, lock_node: dict[str, Any]) -> dict[str, Any]:
    config_path = _resolve_path(lock_node.get("config_path"))
    if config_path is None or not config_path.exists():
        return {"label": label, "status": "missing"}
    summary_path, summary = _load_latest_summary_from_config(config_path)
    if not summary:
        return {"label": label, "status": "missing", "config_path": str(config_path)}
    profiles = summary.get("profiles", [])
    first_profile = profiles[0] if isinstance(profiles, list) and profiles else {}
    profile_dir = _resolve_path(first_profile.get("run_dir"))
    if profile_dir is None:
        return {"label": label, "status": "missing", "summary_path": str(summary_path) if summary_path else ""}
    daily_path = profile_dir / "daily_replay.csv"
    daily = pd.read_csv(daily_path)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    daily["portfolio_return"] = pd.to_numeric(daily["portfolio_return"], errors="coerce").fillna(0.0)
    daily["benchmark_return"] = pd.to_numeric(daily["benchmark_return"], errors="coerce").fillna(0.0)
    bench_report = _read_json(profile_dir / "hard_benchmarks_report.json")
    latest_row = daily.iloc[-1] if not daily.empty else pd.Series(dtype=object)
    strategy = daily["portfolio_return"]
    benchmark = daily["benchmark_return"]
    bench_metrics = ((bench_report.get("metrics") or {}) if isinstance(bench_report.get("metrics"), dict) else {})
    return {
        "label": label,
        "status": "ok",
        "candidate_name": str(lock_node.get("run_id", label)),
        "config_path": str(config_path),
        "summary_path": str(summary_path) if summary_path else "",
        "profile_dir": str(profile_dir),
        "daily_days": int(daily.shape[0]),
        "week_return": _window_total(strategy, 5),
        "month_return": _window_total(strategy, 21),
        "quarter_return": _window_total(strategy, 63),
        "benchmark_week_return": _window_total(benchmark, 5),
        "benchmark_month_return": _window_total(benchmark, 21),
        "perf": _perf_from_simple_returns(strategy),
        "latest_risk_bucket": str(latest_row.get("risk_bucket", "")),
        "latest_gross_exposure": _safe_float(latest_row.get("gross_exposure")),
        "latest_net_exposure": _safe_float(latest_row.get("net_exposure")),
        "latest_date": str(pd.Timestamp(latest_row.get("date")).date()) if not daily.empty else "",
        "benchmarks": {
            "report_json": str(profile_dir / "hard_benchmarks_report.json"),
            "winner_by_ann_return": str(bench_report.get("winner_by_ann_return", "")),
            "spy_ann_return": _safe_float(((bench_metrics.get("spy_buy_hold") or {}) if isinstance(bench_metrics.get("spy_buy_hold"), dict) else {}).get("ann_return")),
            "sixty_forty_ann_return": _safe_float(((bench_metrics.get("sixty_forty") or {}) if isinstance(bench_metrics.get("sixty_forty"), dict) else {}).get("ann_return")),
            "sector_equal_weight_ann_return": _safe_float(
                ((bench_metrics.get("sector_equal_weight") or {}) if isinstance(bench_metrics.get("sector_equal_weight"), dict) else {}).get("ann_return")
            ),
            "momentum_global_top3_ann_return": _safe_float(
                ((bench_metrics.get("momentum_global_top3") or {}) if isinstance(bench_metrics.get("momentum_global_top3"), dict) else {}).get("ann_return")
            ),
        },
    }


def _winner(candidates: list[dict[str, Any]], field: str) -> str:
    ok = [row for row in candidates if row.get("status") == "ok"]
    if not ok:
        return ""
    best = max(ok, key=lambda row: _safe_float(row.get(field), float("-inf")))
    return str(best.get("label", ""))


def _markdown(scorecard: dict[str, Any]) -> str:
    lines = [
        "# Profit Shadow Weekly Scorecard",
        "",
        f"- Generated at: {scorecard.get('generated_at_utc', '')}",
        f"- Main winner this week: {scorecard.get('winner_by_week_return', '') or 'n/a'}",
        f"- Best since inception Sharpe: {scorecard.get('winner_by_sharpe', '') or 'n/a'}",
        "",
        "| Candidate | Week | Month | Quarter | Ann. | Sharpe | MDD | Regime | Gross |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in scorecard.get("candidates", []):
        perf = row.get("perf", {}) if isinstance(row.get("perf", {}), dict) else {}
        lines.append(
            "| {label} | {week:.2%} | {month:.2%} | {quarter:.2%} | {ann:.2%} | {sharpe:.2f} | {mdd:.2%} | {bucket} | {gross:.2f} |".format(
                label=str(row.get("label", "")),
                week=_safe_float(row.get("week_return")),
                month=_safe_float(row.get("month_return")),
                quarter=_safe_float(row.get("quarter_return")),
                ann=_safe_float(perf.get("ann_return")),
                sharpe=_safe_float(perf.get("sharpe")),
                mdd=_safe_float(perf.get("max_drawdown")),
                bucket=str(row.get("latest_risk_bucket", "")),
                gross=_safe_float(row.get("latest_gross_exposure")),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build weekly scorecard for the canonical profit shadow candidates.")
    ap.add_argument("--lock-path", required=True)
    args = ap.parse_args()

    lock_path = Path(args.lock_path).resolve()
    lock = _read_json(lock_path)
    if not lock:
        raise SystemExit(f"missing canonical lock: {lock_path}")

    lock_root = lock_path.parent
    main_row = _candidate_payload("main", _load_candidate_from_lock(lock, "main"))
    challenger_row = _candidate_payload("challenger", _load_candidate_from_lock(lock, "challenger"))
    candidates = [main_row, challenger_row]
    scorecard = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": str(lock_path),
        "winner_by_week_return": _winner(candidates, "week_return"),
        "winner_by_month_return": _winner(candidates, "month_return"),
        "winner_by_sharpe": _winner(candidates, "perf.sharpe"),
        "candidates": candidates,
    }

    # Flatten sharpe winner after the generic helper, which cannot dereference nested keys.
    ok = [row for row in candidates if row.get("status") == "ok"]
    if ok:
        scorecard["winner_by_sharpe"] = max(
            ok,
            key=lambda row: _safe_float(((row.get("perf") or {}) if isinstance(row.get("perf"), dict) else {}).get("sharpe"), float("-inf")),
        ).get("label", "")

    rows: list[dict[str, Any]] = []
    for row in candidates:
        perf = row.get("perf", {}) if isinstance(row.get("perf", {}), dict) else {}
        rows.append(
            {
                "candidate": row.get("label", ""),
                "week_return": _safe_float(row.get("week_return")),
                "month_return": _safe_float(row.get("month_return")),
                "quarter_return": _safe_float(row.get("quarter_return")),
                "ann_return": _safe_float(perf.get("ann_return")),
                "sharpe": _safe_float(perf.get("sharpe")),
                "max_drawdown": _safe_float(perf.get("max_drawdown")),
                "latest_risk_bucket": row.get("latest_risk_bucket", ""),
                "latest_gross_exposure": _safe_float(row.get("latest_gross_exposure")),
                "latest_net_exposure": _safe_float(row.get("latest_net_exposure")),
            }
        )
    df = pd.DataFrame(rows)
    scorecard_dir = lock_root / "scorecards" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(scorecard_dir / "weekly_scorecard.csv", index=False)
    _write_json(scorecard_dir / "weekly_scorecard.json", scorecard)
    (scorecard_dir / "weekly_scorecard.md").write_text(_markdown(scorecard), encoding="utf-8")
    _write_json(lock_root / "latest_scorecard.json", scorecard)
    print(json.dumps(scorecard, ensure_ascii=False))


if __name__ == "__main__":
    main()
