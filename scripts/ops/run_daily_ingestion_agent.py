from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops.agent_guides import attach_agent_guide
from scripts.ops.cycle_context import attach_cycle_context, resolve_cycle_run_id, utc_now_iso, utc_run_id
from scripts.finance.yf_fetch_or_load import fetch_market_data, load_existing_base, unify_to_daily

DEFAULT_PRICES_DIR = ROOT / "data" / "raw" / "finance" / "yfinance_daily"
RESULTS_ROOT = ROOT / "results" / "ops" / "agents" / "daily_ingestion"
CRITICAL_FALLBACK_TICKERS = {
    "SPY",
    "QQQ",
    "LQD",
    "SHY",
    "RSP",
    "SLV",
    "USO",
    "VT",
    "VTI",
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
    "MATIC-USD",
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
}
CORE_FALLBACK_TICKERS = {
    "LREN3.SA",
    "MRVE3.SA",
    "MULT3.SA",
    "PCAR3.SA",
    "PRIO3.SA",
    "RADL3.SA",
    "RAIL3.SA",
    "STAG",
    "STX",
    "SU",
    "TAP",
    "TD",
    "TEL",
    "TIGO",
    "TLK",
    "TM",
    "TRP",
    "TSCO",
    "TSM",
    "TT",
    "TTE",
    "TU",
    "UBS",
    "ULTA",
    "WCN",
    "WPM",
    "XOP",
}
REMOTE_FALLBACK_TICKERS = CRITICAL_FALLBACK_TICKERS | CORE_FALLBACK_TICKERS
PRIORITY_DAILY_TICKERS = REMOTE_FALLBACK_TICKERS | {
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "IWM",
    "GLD",
}


@dataclass
class IngestionResult:
    ticker: str
    status: str
    previous_last_date: str | None
    latest_date: str | None
    rows_before: int
    rows_after: int
    changed: bool
    provider: str | None = None
    error: str | None = None


def read_existing_series(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "price", "log_price", "r"])
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"missing_date_column:{path.name}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def iso_date(value: Any) -> str | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def update_one_csv(path: Path, lookback_days: int, skip_remote: bool) -> IngestionResult:
    ticker = path.stem
    existing = read_existing_series(path)
    previous_last_date = iso_date(existing["date"].iloc[-1]) if not existing.empty else None
    rows_before = int(len(existing))
    today_iso = datetime.now(timezone.utc).date().isoformat()

    if skip_remote:
        return IngestionResult(
            ticker=ticker,
            status="skipped_remote",
            previous_last_date=previous_last_date,
            latest_date=previous_last_date,
            rows_before=rows_before,
            rows_after=rows_before,
            changed=False,
            provider=None,
        )

    if previous_last_date and previous_last_date >= today_iso:
        return IngestionResult(
            ticker=ticker,
            status="unchanged",
            previous_last_date=previous_last_date,
            latest_date=previous_last_date,
            rows_before=rows_before,
            rows_after=rows_before,
            changed=False,
            provider="local_fresh",
        )

    start = "2009-01-01"
    stale_days = None
    if previous_last_date:
        start_dt = datetime.fromisoformat(previous_last_date) - timedelta(days=lookback_days)
        start = start_dt.date().isoformat()
        stale_days = max((datetime.now(timezone.utc).date() - datetime.fromisoformat(previous_last_date).date()).days, 0)

    allow_yfinance = ticker in REMOTE_FALLBACK_TICKERS

    fetched, provider = fetch_market_data(ticker, start=start, end=None, allow_yfinance=allow_yfinance)
    if fetched is None or fetched.empty:
        return IngestionResult(
            ticker=ticker,
            status="no_remote_data",
            previous_last_date=previous_last_date,
            latest_date=previous_last_date,
            rows_before=rows_before,
            rows_after=rows_before,
            changed=False,
            provider=provider,
        )

    fetched_daily = unify_to_daily(fetched)
    merged = fetched_daily
    if not existing.empty:
        base = load_existing_base(path)
        if not base.empty and base["price"].notna().any():
            merged = pd.concat([base[["date", "price"]], fetched_daily[["date", "price"]]], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged.dropna(subset=["date", "price"]).sort_values("date").drop_duplicates("date", keep="last")
            merged = unify_to_daily(merged)

    latest_date = iso_date(merged["date"].iloc[-1]) if not merged.empty else previous_last_date
    rows_after = int(len(merged))
    changed = rows_after != rows_before or latest_date != previous_last_date

    if changed:
        merged.to_csv(path, index=False)

    return IngestionResult(
        ticker=ticker,
        status="updated" if changed else "unchanged",
        previous_last_date=previous_last_date,
        latest_date=latest_date,
        rows_before=rows_before,
        rows_after=rows_after,
        changed=changed,
        provider=provider,
    )


def build_summary(
    results: list[IngestionResult],
    prices_dir: Path,
    skip_remote: bool,
    run_id: str,
    cycle_run_id: str,
    max_assets: int,
    requested_tickers: set[str],
    priority_only: bool,
) -> dict[str, Any]:
    updated = [r for r in results if r.status == "updated"]
    unchanged = [r for r in results if r.status == "unchanged"]
    no_remote_data = [r for r in results if r.status == "no_remote_data"]
    skipped_remote = [r for r in results if r.status == "skipped_remote"]
    failed = [r for r in results if r.status == "failed" or r.error]
    provider_counts: dict[str, int] = {}
    for result in results:
        if result.provider:
            provider_counts[result.provider] = provider_counts.get(result.provider, 0) + 1
    latest_dates = [r.latest_date for r in results if r.latest_date]
    updated_assets = len(updated)
    unchanged_assets = len(unchanged)
    refreshed_assets = updated_assets + unchanged_assets
    stale_days: int | None = None
    max_latest_date = max(latest_dates) if latest_dates else None
    if max_latest_date:
        try:
            latest_dt = datetime.fromisoformat(max_latest_date).date()
            stale_days = max((datetime.now(timezone.utc).date() - latest_dt).days, 0)
        except Exception:
            stale_days = None

    warning_reasons: list[str] = []
    fatal_reason: str | None = None
    if (
        refreshed_assets == 0
        and results
        and not failed
        and not no_remote_data
        and not skipped_remote
        and max_latest_date
    ):
        # If every asset already matches the latest remote date, we still want the
        # summary to reflect that the universe is fresh instead of pretending that
        # nothing was refreshed.
        unchanged_assets = len(results)
        refreshed_assets = len(results)
        warning_reasons.append("asset_counts_reconstructed")
    if max_assets and max_assets > 0:
        warning_reasons.append("limited_scope")
    if priority_only:
        warning_reasons.append("priority_daily_scope")

    remote_unavailable_but_local_fresh = (
        not skip_remote
        and not failed
        and no_remote_data
        and len(no_remote_data) == len(results)
        and stale_days is not None
        and stale_days <= 1
    )

    if skip_remote:
        fatal_reason = "skip_remote_enabled"
    elif failed:
        fatal_reason = "fetch_failed"
    elif refreshed_assets == 0 and not remote_unavailable_but_local_fresh:
        fatal_reason = "no_assets_refreshed"
    elif not max_latest_date:
        fatal_reason = "missing_latest_date"
    elif stale_days is not None and stale_days > 4:
        fatal_reason = "stale_price_history"

    if no_remote_data:
        warning_reasons.append("assets_without_remote_data")
    if remote_unavailable_but_local_fresh:
        warning_reasons.append("remote_unavailable_local_fresh")
    if skipped_remote:
        warning_reasons.append("assets_skipped_remote")
    if stale_days is not None and stale_days > 2 and fatal_reason is None:
        warning_reasons.append("data_getting_stale")

    status = "fail" if fatal_reason else ("warn" if warning_reasons else "ok")
    return attach_agent_guide(attach_cycle_context({
        "status": status,
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "prices_dir": str(prices_dir),
        "skip_remote": skip_remote,
        "limited_scope": bool((max_assets and max_assets > 0) or requested_tickers),
        "priority_only": priority_only,
        "max_assets": int(max_assets or 0),
        "requested_tickers": sorted(requested_tickers),
        "attempted_assets": len(results),
        "updated_assets": updated_assets,
        "unchanged_assets": unchanged_assets,
        "refreshed_assets": refreshed_assets,
        "no_remote_data_assets": len(no_remote_data),
        "skipped_remote_assets": len(skipped_remote),
        "failed_assets": len(failed),
        "max_latest_date": max_latest_date,
        "stale_days": stale_days,
        "fatal_reason": fatal_reason,
        "warning_reasons": warning_reasons,
        "sample_updated": [r.ticker for r in updated[:15]],
        "sample_no_remote_data": [r.ticker for r in no_remote_data[:15]],
        "sample_skipped_remote": [r.ticker for r in skipped_remote[:15]],
        "sample_failed": [{"ticker": r.ticker, "error": r.error or r.status} for r in failed[:15]],
        "provider_counts": provider_counts,
    }, cycle_run_id=cycle_run_id, agent_run_id=run_id), "daily-ingestion-agent")


def main() -> None:
    parser = argparse.ArgumentParser(description="Atualiza preços diários crus antes do recálculo do motor.")
    parser.add_argument("--prices-dir", default=str(DEFAULT_PRICES_DIR))
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--max-assets", type=int, default=0)
    parser.add_argument("--tickers", default="", help="Lista separada por vírgula para testar tickers específicos.")
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument("--priority-only", action="store_true", help="Limita a coleta ao núcleo diário que afeta o motor e o site.")
    parser.add_argument("--cycle-run-id", default="")
    args = parser.parse_args()

    prices_dir = Path(args.prices_dir)
    csv_paths = sorted(prices_dir.glob("*.csv"))
    requested_tickers = {item.strip().upper() for item in str(args.tickers or "").split(",") if item.strip()}
    if requested_tickers:
        csv_paths = [path for path in csv_paths if path.stem.upper() in requested_tickers]
    elif args.priority_only:
        csv_paths = [path for path in csv_paths if path.stem.upper() in PRIORITY_DAILY_TICKERS]
    if args.max_assets and args.max_assets > 0:
        csv_paths = csv_paths[: args.max_assets]

    run_id = utc_run_id()
    cycle_run_id = resolve_cycle_run_id(args.cycle_run_id)
    results: list[IngestionResult] = []
    for path in csv_paths:
        try:
            results.append(update_one_csv(path, lookback_days=max(5, args.lookback_days), skip_remote=args.skip_remote))
        except Exception as exc:
            results.append(
                IngestionResult(
                    ticker=path.stem,
                    status="failed",
                    previous_last_date=None,
                    latest_date=None,
                    rows_before=0,
                    rows_after=0,
                    changed=False,
                    provider=None,
                    error=str(exc),
                )
            )

    summary = build_summary(
        results,
        prices_dir=prices_dir,
        skip_remote=args.skip_remote,
        run_id=run_id,
        cycle_run_id=cycle_run_id,
        max_assets=args.max_assets,
        requested_tickers=requested_tickers,
        priority_only=bool(args.priority_only),
    )
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not summary.get("limited_scope"):
        (RESULTS_ROOT / "latest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        (RESULTS_ROOT / "latest_partial_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if summary.get("status") == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
