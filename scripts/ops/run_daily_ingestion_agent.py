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

from scripts.finance.yf_fetch_or_load import fetch_yfinance, unify_to_daily

DEFAULT_PRICES_DIR = ROOT / "data" / "raw" / "finance" / "yfinance_daily"
RESULTS_ROOT = ROOT / "results" / "ops" / "agents" / "daily_ingestion"


@dataclass
class IngestionResult:
    ticker: str
    status: str
    previous_last_date: str | None
    latest_date: str | None
    rows_before: int
    rows_after: int
    changed: bool
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

    if skip_remote:
        return IngestionResult(
            ticker=ticker,
            status="skipped_remote",
            previous_last_date=previous_last_date,
            latest_date=previous_last_date,
            rows_before=rows_before,
            rows_after=rows_before,
            changed=False,
        )

    start = "2009-01-01"
    if previous_last_date:
        start_dt = datetime.fromisoformat(previous_last_date) - timedelta(days=lookback_days)
        start = start_dt.date().isoformat()

    fetched = fetch_yfinance(ticker, start=start, end=None)
    if fetched is None or fetched.empty:
        return IngestionResult(
            ticker=ticker,
            status="no_remote_data",
            previous_last_date=previous_last_date,
            latest_date=previous_last_date,
            rows_before=rows_before,
            rows_after=rows_before,
            changed=False,
        )

    fetched_daily = unify_to_daily(fetched)
    merged = fetched_daily
    if not existing.empty:
        base = existing[["date", "price"]].copy()
        merged = pd.concat([base, fetched_daily[["date", "price"]]], ignore_index=True)
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
    )


def build_summary(results: list[IngestionResult], prices_dir: Path, skip_remote: bool, run_id: str) -> dict[str, Any]:
    updated = [r for r in results if r.status == "updated"]
    unchanged = [r for r in results if r.status == "unchanged"]
    no_remote_data = [r for r in results if r.status == "no_remote_data"]
    skipped_remote = [r for r in results if r.status == "skipped_remote"]
    failed = [r for r in results if r.status == "failed" or r.error]
    latest_dates = [r.latest_date for r in results if r.latest_date]
    refreshed_assets = len(updated) + len(unchanged)
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
    if skip_remote:
        fatal_reason = "skip_remote_enabled"
    elif failed:
        fatal_reason = "fetch_failed"
    elif refreshed_assets == 0:
        fatal_reason = "no_assets_refreshed"
    elif not max_latest_date:
        fatal_reason = "missing_latest_date"
    elif stale_days is not None and stale_days > 4:
        fatal_reason = "stale_price_history"

    if no_remote_data:
        warning_reasons.append("assets_without_remote_data")
    if skipped_remote:
        warning_reasons.append("assets_skipped_remote")
    if stale_days is not None and stale_days > 2 and fatal_reason is None:
        warning_reasons.append("data_getting_stale")

    status = "fail" if fatal_reason else ("warn" if warning_reasons else "ok")
    return {
        "status": status,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prices_dir": str(prices_dir),
        "skip_remote": skip_remote,
        "attempted_assets": len(results),
        "updated_assets": len(updated),
        "unchanged_assets": len(unchanged),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Atualiza preços diários crus antes do recálculo do motor.")
    parser.add_argument("--prices-dir", default=str(DEFAULT_PRICES_DIR))
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--max-assets", type=int, default=0)
    parser.add_argument("--skip-remote", action="store_true")
    args = parser.parse_args()

    prices_dir = Path(args.prices_dir)
    csv_paths = sorted(prices_dir.glob("*.csv"))
    if args.max_assets and args.max_assets > 0:
        csv_paths = csv_paths[: args.max_assets]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
                    error=str(exc),
                )
            )

    summary = build_summary(results, prices_dir=prices_dir, skip_remote=args.skip_remote, run_id=run_id)
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_ROOT / "latest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if summary.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
