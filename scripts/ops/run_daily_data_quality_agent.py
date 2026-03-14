#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops.agent_guides import attach_agent_guide
from scripts.ops.data_review_policy import DEFERRED_REVIEW_TICKERS, is_deferred_review_ticker

CRITICAL_TICKERS = {
    "SPY",
    "QQQ",
    "IWM",
    "GLD",
    "SLV",
    "LQD",
    "SHY",
    "RSP",
    "USO",
    "VTI",
    "VT",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "MATIC-USD",
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
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


def _parse_date(value: Any) -> datetime.date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return pd.to_datetime(raw, errors="raise").date()
    except Exception:
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _load_universe_assets(paths: list[Path]) -> set[str]:
    assets: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = [col for col in df.columns if col.lower() in {"asset", "ticker", "symbol"}]
        if not cols:
            continue
        assets.update(df[cols[0]].dropna().astype(str))
    return assets


def _latest_csv_date(path: Path) -> datetime.date | None:
    try:
        df = pd.read_csv(path, usecols=["date"])
    except Exception:
        return None
    if df.empty:
        return None
    return _parse_date(df["date"].dropna().iloc[-1])


def _fresh_tolerance_days(ticker: str) -> int:
    if ticker.endswith(".SA"):
        return 2
    if ticker.endswith("-USD"):
        return 1
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Audita frescor e relevância dos preços usados pelo motor e pelo site.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--ingestion-summary", default="results/ops/agents/daily_ingestion/latest_summary.json")
    ap.add_argument("--operation-summary", default="results/ops/agents/daily_operation/latest_summary.json")
    ap.add_argument("--site-snapshot", default="results/ops/site_data/latest_site_snapshot.json")
    ap.add_argument("--outdir-root", default="results/ops/agents/daily_data_quality")
    ap.add_argument("--prune-stale-days", type=int, default=20)
    args = ap.parse_args()

    prices_dir = (ROOT / args.prices_dir).resolve()
    ingestion = _read_json((ROOT / args.ingestion_summary).resolve())
    operation = _read_json((ROOT / args.operation_summary).resolve())
    site_snapshot = _read_json((ROOT / args.site_snapshot).resolve())
    universe_assets = _load_universe_assets(
        [
            ROOT / "data" / "asset_groups_crypto_top_liquid_plus.csv",
            ROOT / "data" / "asset_groups_target_800_clean_plus.csv",
        ]
    )

    reference_date = _parse_date(ingestion.get("max_latest_date")) or datetime.now(timezone.utc).date()
    site_as_of_date = _parse_date(site_snapshot.get("as_of_date"))
    operation_generated = str(operation.get("generated_at_utc") or "").strip()

    stale_rows: list[dict[str, Any]] = []
    fresh_assets = 0
    for path in sorted(prices_dir.glob("*.csv")):
        latest_date = _latest_csv_date(path)
        if latest_date is None:
            continue
        stale_days = max((reference_date - latest_date).days, 0)
        ticker = path.stem
        role = "peripheral"
        if ticker in CRITICAL_TICKERS:
            role = "critical"
        elif ticker in universe_assets and not is_deferred_review_ticker(ticker):
            role = "core"
        if stale_days <= _fresh_tolerance_days(ticker):
            fresh_assets += 1
            continue
        stale_rows.append(
            {
                "ticker": ticker,
                "latest_date": latest_date.isoformat(),
                "stale_days": stale_days,
                "role": role,
                "deferred_review": ticker in DEFERRED_REVIEW_TICKERS,
            }
        )

    stale_rows.sort(key=lambda row: (-int(row["stale_days"]), str(row["ticker"])))
    critical_stale = [row for row in stale_rows if row["role"] == "critical"]
    core_stale = [row for row in stale_rows if row["role"] == "core"]
    peripheral_stale = [row for row in stale_rows if row["role"] == "peripheral"]
    prune_candidates = [row for row in peripheral_stale if int(row["stale_days"]) >= max(1, args.prune_stale_days)]

    site_lag_days = None
    if site_as_of_date is not None:
        site_lag_days = max((reference_date - site_as_of_date).days, 0)

    alerts: list[dict[str, str]] = []
    if str(ingestion.get("status") or "").strip().lower() == "fail":
        alerts.append(
            {
                "level": "fail",
                "code": "ingestion_failed",
                "message": "A ingestão diária falhou e deixou a base sem confirmação de frescor.",
            }
        )
    if critical_stale:
        alerts.append(
            {
                "level": "warn",
                "code": "critical_assets_stale",
                "message": f"{len(critical_stale)} ativos críticos estão atrasados frente à data-base mais nova.",
            }
        )
    if core_stale:
        alerts.append(
            {
                "level": "warn",
                "code": "core_assets_stale",
                "message": f"{len(core_stale)} ativos do núcleo do motor estão atrasados e precisam de cobertura melhor.",
            }
        )
    if site_lag_days is not None and site_lag_days > 0:
        alerts.append(
            {
                "level": "warn",
                "code": "site_snapshot_lagging",
                "message": f"O snapshot do site ficou {site_lag_days} dias atrás da melhor data local.",
            }
        )
    if prune_candidates:
        alerts.append(
            {
                "level": "info",
                "code": "peripheral_prune_candidates",
                "message": f"{len(prune_candidates)} ativos periféricos estão velhos o suficiente para revisão ou poda.",
            }
        )

    if any(alert["level"] == "fail" for alert in alerts):
        status = "fail"
    elif any(alert["level"] == "warn" for alert in alerts):
        status = "warn"
    else:
        status = "ok"

    summary = attach_agent_guide({
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_data_date": reference_date.isoformat(),
        "site_as_of_date": site_as_of_date.isoformat() if site_as_of_date else "",
        "site_lag_days": site_lag_days,
        "operation_generated_at_utc": operation_generated,
        "prices_dir": str(prices_dir),
        "total_assets": fresh_assets + len(stale_rows),
        "fresh_assets": fresh_assets,
        "stale_assets": len(stale_rows),
        "critical_stale_assets": len(critical_stale),
        "core_stale_assets": len(core_stale),
        "peripheral_stale_assets": len(peripheral_stale),
        "prune_candidate_count": len(prune_candidates),
        "deferred_review_tickers": sorted(DEFERRED_REVIEW_TICKERS),
        "sample_critical_stale": critical_stale[:20],
        "sample_core_stale": core_stale[:20],
        "sample_prune_candidates": prune_candidates[:20],
        "ingestion_status": ingestion.get("status"),
        "ingestion_warning_reasons": ingestion.get("warning_reasons") or [],
        "ingestion_provider_counts": ingestion.get("provider_counts") or {},
        "notes": [
            "Não apagar ativos atrasados às cegas. Primeiro separar o que é crítico, núcleo do motor e periferia.",
            "Ativos críticos atrasados devem ganhar fallback melhor; ativos periféricos muito velhos podem virar candidatos de poda.",
        ],
        "alerts": alerts,
    }, "daily-data-quality-agent")

    outroot = (ROOT / args.outdir_root).resolve()
    ts_dir = outroot / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _write_json(ts_dir / "summary.json", summary)
    _write_json(outroot / "latest_summary.json", summary)
    _write_json(outroot / "latest_data_quality.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
