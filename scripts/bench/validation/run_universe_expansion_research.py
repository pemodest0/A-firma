#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true&limit=9999"
DEFAULT_OUTDIR = ROOT / "results" / "validation" / "universe_expansion"
DEFAULT_CACHE = ROOT / "results" / "validation" / "universe_expansion" / "_cache" / "nasdaq_screener.json"

BASELINE_GROUPS = ROOT / "data" / "asset_groups_global_plus.csv"
BASELINE_METADATA = ROOT / "data" / "asset_metadata_global_plus.csv"
PRICES_DIR = ROOT / "data" / "raw" / "finance" / "yfinance_daily"
POLICY_PATH = ROOT / "config" / "lab_corr_policy_shadow_global_plus.json"

GOOD_ARTIFACTS = [
    {
        "name": "profit_shadow_best",
        "run_dir": ROOT / "results" / "ops" / "profit_shadow" / "runs" / "20260306T053930Z",
        "summary_json": ROOT / "results" / "ops" / "profit_shadow" / "runs" / "20260306T053930Z" / "summary.json",
    },
    {
        "name": "profit_attack_validation",
        "run_dir": ROOT / "results" / "validation" / "profit_attack_validation" / "20260306T062520Z",
        "summary_json": ROOT / "results" / "validation" / "profit_attack_validation" / "20260306T062520Z" / "summary.json",
    },
]

MANUAL_LOCAL_GROUPS = {
    "KRE": "financials",
    "XLB": "materials",
    "XLF": "financials",
    "XLI": "industrials",
    "XLK": "technology",
    "XLP": "consumer_staples",
    "XLRE": "real_estate",
    "XLU": "utilities",
    "XLV": "health_care",
    "XLY": "consumer_discretionary",
    "XOP": "energy",
    "^VIX": "vol_regime",
}

SECTOR_MAP = {
    "Basic Materials": "materials",
    "Consumer Discretionary": "consumer_discretionary",
    "Consumer Staples": "consumer_staples",
    "Energy": "energy",
    "Finance": "financials",
    "Health Care": "health_care",
    "Industrials": "industrials",
    "Miscellaneous": "miscellaneous",
    "Real Estate": "real_estate",
    "Technology": "technology",
    "Telecommunications": "telecommunications",
    "Utilities": "utilities",
}

BANNED_NAME_TOKENS = (
    "warrant",
    "rights",
    "right",
    "preferred",
    "depositary share",
    "depositary shares",
    "note",
    "notes",
    "debenture",
    "unit",
    "units",
    "trust",
    "etf",
    "etn",
    "fund",
)


@dataclass(frozen=True)
class CandidateAsset:
    ticker: str
    group: str
    source: str
    sector_raw: str
    market_cap: float | None
    country: str | None
    name: str


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol).strip().upper()


def _safe_market_cap(value: Any) -> float | None:
    try:
        num = float(str(value).replace(",", "").strip())
    except Exception:
        return None
    if not np.isfinite(num) or num <= 0:
        return None
    return num


def is_research_equity_name(name: Any) -> bool:
    text = str(name).strip().lower()
    if not text:
        return False
    return not any(token in text for token in BANNED_NAME_TOKENS)


def normalize_group(sector: Any) -> str | None:
    raw = str(sector).strip()
    if not raw:
        return None
    if raw in SECTOR_MAP:
        return SECTOR_MAP[raw]
    snake = raw.lower().replace("&", "and").replace("/", " ").replace("-", " ")
    snake = "_".join(part for part in snake.split() if part)
    return snake or None


def _download_text_via_curl(url: str, out_path: Path, headers: dict[str, str] | None = None) -> None:
    cmd = ["curl", "-L", "--fail", url, "-o", str(out_path)]
    for key, value in (headers or {}).items():
        cmd.extend(["-H", f"{key}: {value}"])
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)


def fetch_screener(cache_path: Path, refresh: bool) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache_path.exists():
        _download_text_via_curl(
            NASDAQ_SCREENER_URL,
            cache_path,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain, */*"},
        )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    rows = payload.get("data", {}).get("rows", [])
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("nasdaq screener returned no rows")
    df["symbol"] = df["symbol"].map(_normalize_symbol)
    df["sector"] = df.get("sector", "").astype(str).str.strip()
    df["country"] = df.get("country", "").astype(str).str.strip()
    df["name"] = df.get("name", "").astype(str).str.strip()
    df["market_cap_num"] = df.get("marketCap", "").map(_safe_market_cap)
    df = df.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    return df


def _price_file_rows(prices_dir: Path, ticker: str) -> int:
    path = prices_dir / f"{ticker}.csv"
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path, usecols=["date"])
    except Exception:
        return 0
    return int(df.shape[0])


def _load_local_symbols(prices_dir: Path) -> set[str]:
    return {_normalize_symbol(p.stem) for p in prices_dir.glob("*.csv")}


def _round_robin_by_sector(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or df.empty:
        return df.head(0).copy()
    by_sector: dict[str, list[dict[str, Any]]] = {}
    for sector, sub in df.groupby("group", dropna=False):
        rows = sub.sort_values(["market_cap_num", "symbol"], ascending=[False, True]).to_dict("records")
        by_sector[str(sector)] = rows
    picked: list[dict[str, Any]] = []
    while len(picked) < limit:
        moved = False
        for sector in sorted(by_sector):
            rows = by_sector[sector]
            if not rows:
                continue
            picked.append(rows.pop(0))
            moved = True
            if len(picked) >= limit:
                break
        if not moved:
            break
    return pd.DataFrame(picked)


def build_candidate_universe(
    baseline_groups: pd.DataFrame,
    screener: pd.DataFrame,
    prices_dir: Path,
    target_total_assets: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_symbols = set(baseline_groups["asset"].map(_normalize_symbol))
    local_symbols = _load_local_symbols(prices_dir)
    extra_local = sorted(local_symbols.difference(current_symbols))
    screener_index = screener.set_index("symbol", drop=False)

    local_quality_assets: list[CandidateAsset] = []
    local_full_assets: list[CandidateAsset] = []
    for ticker in extra_local:
        if ticker in MANUAL_LOCAL_GROUPS:
            group = MANUAL_LOCAL_GROUPS[ticker]
            local_full_assets.append(CandidateAsset(ticker, group, "manual_local", group, None, None, ticker))
            local_quality_assets.append(CandidateAsset(ticker, group, "manual_local", group, None, None, ticker))
            continue
        if ticker not in screener_index.index:
            continue
        row = screener_index.loc[ticker]
        group = normalize_group(row.get("sector"))
        if not group:
            continue
        asset = CandidateAsset(
            ticker=ticker,
            group=group,
            source="local_screener",
            sector_raw=str(row.get("sector", "")),
            market_cap=_safe_market_cap(row.get("market_cap_num")),
            country=str(row.get("country", "")) or None,
            name=str(row.get("name", "")),
        )
        local_full_assets.append(asset)
        if is_research_equity_name(asset.name):
            local_quality_assets.append(asset)

    local_quality_df = pd.DataFrame([a.__dict__ for a in local_quality_assets])
    local_full_df = pd.DataFrame([a.__dict__ for a in local_full_assets])

    quality_groups = pd.concat(
        [
            baseline_groups[["asset", "group"]].copy(),
            local_quality_df.rename(columns={"ticker": "asset"})[["asset", "group"]].copy() if not local_quality_df.empty else pd.DataFrame(columns=["asset", "group"]),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["asset"], keep="first")

    need = max(0, int(target_total_assets) - int(quality_groups.shape[0]))
    screener_quality = screener.copy()
    screener_quality["group"] = screener_quality["sector"].map(normalize_group)
    screener_quality = screener_quality[
        screener_quality["group"].notna()
        & screener_quality["market_cap_num"].notna()
        & screener_quality["name"].map(is_research_equity_name)
    ].copy()
    screener_quality = screener_quality[~screener_quality["symbol"].isin(local_symbols | current_symbols)].copy()
    topup_pool = _round_robin_by_sector(screener_quality, limit=max(0, need + 40))

    topup_assets = pd.DataFrame(
        [
            {
                "ticker": _normalize_symbol(row["symbol"]),
                "group": str(row["group"]),
                "source": "remote_topup",
                "sector_raw": str(row["sector"]),
                "market_cap": _safe_market_cap(row["market_cap_num"]),
                "country": str(row["country"]) or None,
                "name": str(row["name"]),
            }
            for _, row in topup_pool.iterrows()
        ]
    )
    return quality_groups, local_quality_df, local_full_df, topup_assets


def _normalize_daily_stooq(raw_path: Path, out_path: Path) -> tuple[bool, int, str | None, str | None]:
    df = pd.read_csv(raw_path)
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return False, 0, None, None
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], errors="coerce"),
            "price": pd.to_numeric(df["Close"], errors="coerce"),
        }
    ).dropna()
    out = out[out["price"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        return False, 0, None, None
    out["log_price"] = np.log(out["price"].astype(float))
    out["r"] = out["log_price"].diff()
    out = out.dropna(subset=["r"]).reset_index(drop=True)
    if out.empty:
        return False, 0, None, None
    out["date"] = out["date"].dt.date.astype(str)
    out.to_csv(out_path, index=False)
    return True, int(out.shape[0]), str(out["date"].iloc[0]), str(out["date"].iloc[-1])


def _download_one_stooq(ticker: str, prices_dir: Path) -> dict[str, Any]:
    symbol = _normalize_symbol(ticker)
    out_path = prices_dir / f"{symbol}.csv"
    if out_path.exists() and _price_file_rows(prices_dir, symbol) >= 252:
        return {"ticker": symbol, "status": "cached", "rows": _price_file_rows(prices_dir, symbol)}
    stooq_symbol = symbol.replace(".", "-").lower()
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}.us&i=d"
    with tempfile.TemporaryDirectory(prefix="assyntrax-stooq-") as tmp:
        raw_path = Path(tmp) / f"{symbol}.csv"
        try:
            _download_text_via_curl(url, raw_path)
            ok, rows, start, end = _normalize_daily_stooq(raw_path, out_path)
            if ok:
                return {"ticker": symbol, "status": "ok", "rows": rows, "start": start, "end": end}
            return {"ticker": symbol, "status": "error", "error": "empty_after_normalize"}
        except subprocess.CalledProcessError as exc:
            return {"ticker": symbol, "status": "error", "error": exc.stderr.strip() or exc.stdout.strip() or "curl_failed"}
        except Exception as exc:  # pragma: no cover - defensive
            return {"ticker": symbol, "status": "error", "error": str(exc)}


def download_topups(topup_assets: pd.DataFrame, prices_dir: Path, target_success: int, workers: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if target_success <= 0 or topup_assets.empty:
        return topup_assets.head(0).copy(), []
    prices_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    selected: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_download_one_stooq, str(row["ticker"]), prices_dir): str(row["ticker"]) for _, row in topup_assets.iterrows()}
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            if rec.get("status") in {"ok", "cached"}:
                selected.append(str(rec["ticker"]))
            if len(selected) >= int(target_success):
                break
    selected_df = topup_assets[topup_assets["ticker"].isin(selected)].copy()
    if int(selected_df.shape[0]) > int(target_success):
        selected_df = selected_df.head(int(target_success)).copy()
    return selected_df.reset_index(drop=True), records


def build_metadata(groups_df: pd.DataFrame, prices_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in groups_df.iterrows():
        ticker = _normalize_symbol(row["asset"])
        group = str(row["group"]).strip()
        rows.append(
            {
                "asset_id": ticker,
                "ticker": ticker,
                "sector_gics": group,
                "sector_internal": group,
                "liquidity_proxy": _price_file_rows(prices_dir, ticker),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["asset_id"], keep="first")
    out = out.sort_values(["sector_gics", "ticker"]).reset_index(drop=True)
    return out


def _write_candidate_files(groups_df: pd.DataFrame, metadata_df: pd.DataFrame, groups_path: Path, metadata_path: Path) -> None:
    groups_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    groups_df.to_csv(groups_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)


def _extract_last_json_line(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        raw = line.strip()
        if not raw.startswith("{") or not raw.endswith("}"):
            continue
        try:
            return json.loads(raw)
        except Exception:
            continue
    return {}


def run_build_pack(asset_groups: Path, results_dir: Path) -> dict[str, Any]:
    cmd = [
        "python3",
        "scripts/lab/build_local_finance_pack.py",
        "--prices-dir",
        str(PRICES_DIR.relative_to(ROOT)),
        "--asset-groups",
        str(asset_groups.relative_to(ROOT)),
        "--results-dir",
        str(results_dir.relative_to(ROOT)),
        "--business-days-only",
        "1",
        "--min-rows",
        "252",
        "--min-date-coverage",
        "0.90",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    payload = _extract_last_json_line(proc.stdout)
    if not payload:
        raise RuntimeError(f"failed to parse build_local_finance_pack output for {asset_groups}")
    return payload


def run_macro(panel_path: Path, universe_path: Path, metadata_path: Path, out_base: Path, max_core_assets: int, n_global: int, n_sector: int) -> Path:
    cmd = [
        "python3",
        "scripts/lab/run_corr_macro_offline.py",
        "--policy-path",
        str(POLICY_PATH.relative_to(ROOT)),
        "--panel-path",
        str(panel_path),
        "--universe-path",
        str(universe_path),
        "--out-base",
        str(out_base.relative_to(ROOT)),
        "--max-core-assets",
        str(int(max_core_assets)),
        "--official-window",
        "120",
        "--windows",
        "120",
        "--noise-step",
        "10",
        "--overlap-step",
        "5",
        "--enable-hierarchical",
        "1",
        "--asset-metadata-path",
        str(metadata_path.relative_to(ROOT)),
        "--n-global",
        str(int(n_global)),
        "--n-sector",
        str(int(n_sector)),
        "--min-coverage-global",
        "0.90",
        "--min-coverage-sector",
        "0.80",
        "--enable-internal-sectors",
        "1",
        "--save-v1",
        "1",
        "--update-release-pointer",
        "1",
    ]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    latest = _read_json(out_base / "latest_release.json")
    run_dir = Path(str(latest.get("run_dir", ""))).resolve() if latest.get("run_dir") else None
    if run_dir and run_dir.exists():
        return run_dir
    runs = sorted([p for p in out_base.iterdir() if p.is_dir() and p.name[:8].isdigit()], key=lambda p: p.name, reverse=True)
    if not runs:
        raise RuntimeError(f"macro run dir not found under {out_base}")
    return runs[0]


def summarize_macro(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    gate = _read_json(run_dir / "deployment_gate.json")
    core_by_sector = pd.read_csv(run_dir / "universe_core_by_sector.csv") if (run_dir / "universe_core_by_sector.csv").exists() else pd.DataFrame()
    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "status": summary.get("status"),
        "n_core": summary.get("n_core"),
        "deployment_blocked": gate.get("blocked"),
        "deployment_reasons": gate.get("reasons", []),
        "eigvec_overlap_mean_60d": ((gate.get("checks") or {}).get("eigvec_overlap_mean_60d")),
        "joint_majority_60d": ((gate.get("checks") or {}).get("joint_majority_60d")),
        "sector_count_core": None,
        "largest_core_sector": None,
        "largest_core_sector_count": None,
        "largest_core_sector_share": None,
    }
    if not core_by_sector.empty:
        core_by_sector = core_by_sector.sort_values(["n_tickers", "sector"], ascending=[False, True]).reset_index(drop=True)
        top = core_by_sector.iloc[0]
        total = float(core_by_sector["n_tickers"].sum())
        out["sector_count_core"] = int(core_by_sector.shape[0])
        out["largest_core_sector"] = str(top["sector"])
        out["largest_core_sector_count"] = int(top["n_tickers"])
        out["largest_core_sector_share"] = float(top["n_tickers"]) / total if total > 0 else None
    return out


def write_good_artifact_lock(outdir: Path) -> dict[str, Any]:
    lock_dir = outdir / "artifact_lock"
    lock_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for item in GOOD_ARTIFACTS:
        summary = _read_json(item["summary_json"])
        snapshot_path = lock_dir / f"{item['name']}_summary.json"
        snapshot_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        entries.append(
            {
                "name": item["name"],
                "run_dir": str(item["run_dir"]),
                "summary_json": str(item["summary_json"]),
                "snapshot_json": str(snapshot_path),
                "exists": bool(item["run_dir"].exists() and item["summary_json"].exists()),
            }
        )
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": entries}
    (lock_dir / "good_artifacts_lock.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Expand research universes beyond global_plus and benchmark broader candidates.")
    ap.add_argument("--run-id", type=str, default=_run_id())
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--screener-cache", type=str, default=str(DEFAULT_CACHE))
    ap.add_argument("--refresh-screener", type=int, default=0)
    ap.add_argument("--target-total-assets", type=int, default=800)
    ap.add_argument("--download-missing", type=int, default=1)
    ap.add_argument("--download-workers", type=int, default=10)
    ap.add_argument("--benchmark-macro", type=int, default=1)
    ap.add_argument("--max-core-assets", type=int, default=360)
    ap.add_argument("--n-global", type=int, default=300)
    ap.add_argument("--n-sector", type=int, default=90)
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_root = Path(str(args.outdir)).resolve() / str(args.run_id).strip()
    run_root.mkdir(parents=True, exist_ok=True)

    baseline_groups = pd.read_csv(BASELINE_GROUPS)
    screener = fetch_screener(Path(str(args.screener_cache)), refresh=bool(int(args.refresh_screener)))

    lock_payload = write_good_artifact_lock(run_root)
    quality_groups, local_quality_df, local_full_df, topup_assets = build_candidate_universe(
        baseline_groups=baseline_groups,
        screener=screener,
        prices_dir=PRICES_DIR,
        target_total_assets=int(args.target_total_assets),
    )

    need = max(0, int(args.target_total_assets) - int(quality_groups.shape[0]))
    topup_success_df = topup_assets.head(0).copy()
    download_records: list[dict[str, Any]] = []
    if bool(int(args.download_missing)) and need > 0:
        topup_success_df, download_records = download_topups(
            topup_assets=topup_assets,
            prices_dir=PRICES_DIR,
            target_success=need,
            workers=int(args.download_workers),
        )

    candidates: dict[str, dict[str, Any]] = {}
    for name, groups_df in {
        "baseline_global_plus_528": baseline_groups[["asset", "group"]].copy(),
        "local_quality_expanded": quality_groups[["asset", "group"]].copy(),
        "target_800": pd.concat(
            [
                quality_groups[["asset", "group"]].copy(),
                topup_success_df.rename(columns={"ticker": "asset"})[["asset", "group"]].copy() if not topup_success_df.empty else pd.DataFrame(columns=["asset", "group"]),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["asset"], keep="first"),
    }.items():
        groups_df = groups_df.sort_values(["group", "asset"]).reset_index(drop=True)
        metadata_df = build_metadata(groups_df, PRICES_DIR)
        groups_path = ROOT / "data" / f"asset_groups_{name}.csv"
        metadata_path = ROOT / "data" / f"asset_metadata_{name}.csv"
        _write_candidate_files(groups_df, metadata_df, groups_path, metadata_path)
        candidates[name] = {
            "name": name,
            "asset_groups_csv": str(groups_path),
            "asset_metadata_csv": str(metadata_path),
            "n_assets": int(groups_df.shape[0]),
            "n_groups": int(groups_df["group"].nunique()),
            "group_counts": groups_df["group"].value_counts().sort_index().to_dict(),
        }

    (run_root / "local_quality_candidates.csv").write_text(local_quality_df.to_csv(index=False), encoding="utf-8")
    (run_root / "local_full_candidates.csv").write_text(local_full_df.to_csv(index=False), encoding="utf-8")
    if not topup_assets.empty:
        (run_root / "remote_topup_pool.csv").write_text(topup_assets.to_csv(index=False), encoding="utf-8")
    if download_records:
        pd.DataFrame(download_records).to_csv(run_root / "download_records.csv", index=False)

    macro_results: dict[str, Any] = {}
    if bool(int(args.benchmark_macro)):
        for name, meta in candidates.items():
            build_root = run_root / "packs" / name
            pack_payload = run_build_pack(Path(meta["asset_groups_csv"]), build_root)
            outdir = Path(str(pack_payload["outdir"]))
            if not outdir.is_absolute():
                outdir = (ROOT / outdir).resolve()
            panel_path = outdir / "panel_long_sector.csv"
            universe_path = outdir / "universe_fixed.csv"
            macro_out_base = run_root / "macro" / name
            macro_run_dir = run_macro(
                panel_path=panel_path,
                universe_path=universe_path,
                metadata_path=Path(meta["asset_metadata_csv"]),
                out_base=macro_out_base,
                max_core_assets=int(args.max_core_assets),
                n_global=int(args.n_global),
                n_sector=int(args.n_sector),
            )
            macro_results[name] = {
                "pack": pack_payload,
                "macro": summarize_macro(macro_run_dir),
            }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(args.run_id),
        "artifact_lock": lock_payload,
        "screener_cache": str(Path(str(args.screener_cache)).resolve()),
        "baseline_assets": int(baseline_groups.shape[0]),
        "local_quality_added": int(local_quality_df.shape[0]),
        "local_full_added": int(local_full_df.shape[0]),
        "target_total_assets": int(args.target_total_assets),
        "download_target_need": int(need),
        "download_success": int(topup_success_df.shape[0]),
        "download_attempts": int(len(download_records)),
        "candidates": candidates,
        "macro_results": macro_results,
    }
    if macro_results:
        rows = []
        for name, rec in macro_results.items():
            pack = rec.get("pack", {})
            macro = rec.get("macro", {})
            rows.append(
                {
                    "universe": name,
                    "assets_ok": pack.get("assets_ok"),
                    "panel_rows": pack.get("panel_rows"),
                    "n_core": macro.get("n_core"),
                    "sector_count_core": macro.get("sector_count_core"),
                    "largest_core_sector": macro.get("largest_core_sector"),
                    "largest_core_sector_share": macro.get("largest_core_sector_share"),
                    "eigvec_overlap_mean_60d": macro.get("eigvec_overlap_mean_60d"),
                    "joint_majority_60d": macro.get("joint_majority_60d"),
                    "deployment_blocked": macro.get("deployment_blocked"),
                }
            )
        pd.DataFrame(rows).to_csv(run_root / "macro_comparison.csv", index=False)
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
