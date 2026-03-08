#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GROUP_CAPS = {
    "industrials": 40,
    "technology": 40,
    "financials": 40,
    "consumer_discretionary": 35,
    "energy": 30,
    "health_care": 30,
    "consumer_staples": 25,
    "real_estate": 25,
    "materials": 25,
    "utilities": 20,
    "telecommunications": 15,
    "equities_br_bluechips": 24,
    "equities_ex_us": 12,
    "equities_us_broad": 10,
    "equities_us_other": 12,
}

DEFAULT_EXCLUDED_GROUPS = [
    "bonds_credit",
    "bonds_rates",
    "commodities",
    "crypto",
    "fx",
    "miscellaneous",
    "vol_regime",
]

DEFAULT_FORCE_INCLUDE_ASSETS = [
    "DIA",
    "EEM",
    "EFA",
    "EWJ",
    "EWZ",
    "IWM",
    "QQQ",
    "RSP",
    "SPY",
    "VT",
    "VTI",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLP",
    "XLK",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a cleaner execution universe from a larger observation universe.")
    ap.add_argument("--asset-groups-csv", type=str, default="data/asset_groups_target_800.csv")
    ap.add_argument("--asset-metadata-csv", type=str, default="data/asset_metadata_target_800.csv")
    ap.add_argument("--out-groups-csv", type=str, default="data/asset_groups_target_800_clean.csv")
    ap.add_argument("--out-metadata-csv", type=str, default="data/asset_metadata_target_800_clean.csv")
    ap.add_argument("--summary-json", type=str, default="results/ops/execution_universe/target_800_clean_summary.json")
    ap.add_argument("--min-liquidity-proxy", type=int, default=756)
    ap.add_argument("--excluded-groups", type=str, default=",".join(DEFAULT_EXCLUDED_GROUPS))
    ap.add_argument("--force-include-assets", type=str, default=",".join(DEFAULT_FORCE_INCLUDE_ASSETS))
    ap.add_argument("--group-caps-json", type=str, default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    groups_path = (ROOT / str(args.asset_groups_csv)).resolve()
    metadata_path = (ROOT / str(args.asset_metadata_csv)).resolve()
    out_groups_path = (ROOT / str(args.out_groups_csv)).resolve()
    out_metadata_path = (ROOT / str(args.out_metadata_csv)).resolve()
    summary_path = (ROOT / str(args.summary_json)).resolve()

    groups = pd.read_csv(groups_path)
    meta = pd.read_csv(metadata_path)
    groups["asset"] = groups["asset"].astype(str).str.strip()
    groups["group"] = groups["group"].astype(str).str.strip()
    meta["asset_id"] = meta["asset_id"].astype(str).str.strip()
    meta["ticker"] = meta["ticker"].astype(str).str.strip()
    meta["sector_internal"] = meta["sector_internal"].astype(str).str.strip()
    meta["liquidity_proxy"] = pd.to_numeric(meta["liquidity_proxy"], errors="coerce")

    merged = groups.merge(meta, left_on="asset", right_on="asset_id", how="left")
    merged["liquidity_proxy"] = pd.to_numeric(merged["liquidity_proxy"], errors="coerce")
    merged = merged.dropna(subset=["asset", "group", "liquidity_proxy"]).copy()

    excluded_groups = {x.strip() for x in str(args.excluded_groups).split(",") if x.strip()}
    force_include_assets = {x.strip().upper() for x in str(args.force_include_assets).split(",") if x.strip()}
    merged = merged[~merged["group"].isin(excluded_groups)].copy()
    forced = merged[merged["asset"].astype(str).str.upper().isin(force_include_assets)].copy()
    filtered = merged[merged["liquidity_proxy"] >= int(args.min_liquidity_proxy)].copy()

    group_caps = DEFAULT_GROUP_CAPS.copy()
    if str(args.group_caps_json).strip():
        group_caps.update({str(k): _safe_int(v, 0) for k, v in _read_json((ROOT / str(args.group_caps_json)).resolve()).items()})

    keep_parts: list[pd.DataFrame] = []
    for group, sub in filtered.groupby("group", dropna=False):
        cap = int(group_caps.get(str(group), max(1, int(sub.shape[0]))))
        part = (
            sub.sort_values(["liquidity_proxy", "asset"], ascending=[False, True])
            .head(cap)
            .copy()
        )
        keep_parts.append(part)
    kept = pd.concat(keep_parts, ignore_index=True) if keep_parts else filtered.head(0).copy()
    if not forced.empty:
        kept = pd.concat([kept, forced], ignore_index=True)
    kept = kept.sort_values(["group", "asset"]).drop_duplicates(subset=["asset"], keep="first").reset_index(drop=True)

    out_groups = kept[["asset", "group"]].copy()
    out_meta = kept[["asset_id", "ticker", "sector_gics", "sector_internal", "liquidity_proxy"]].copy()

    out_groups_path.parent.mkdir(parents=True, exist_ok=True)
    out_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_groups.to_csv(out_groups_path, index=False)
    out_meta.to_csv(out_metadata_path, index=False)

    counts = out_groups["group"].value_counts().sort_values(ascending=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": _run_id(),
        "source_groups_csv": str(groups_path),
        "source_metadata_csv": str(metadata_path),
        "out_groups_csv": str(out_groups_path),
        "out_metadata_csv": str(out_metadata_path),
        "min_liquidity_proxy": int(args.min_liquidity_proxy),
        "excluded_groups": sorted(excluded_groups),
        "force_include_assets": sorted(force_include_assets),
        "forced_included_assets_present": sorted(
            set(kept["asset"].astype(str).str.upper()).intersection(force_include_assets)
        ),
        "n_assets": int(out_groups.shape[0]),
        "n_groups": int(out_groups["group"].nunique()),
        "largest_group": str(counts.index[0]) if not counts.empty else "",
        "largest_group_count": int(counts.iloc[0]) if not counts.empty else 0,
        "group_counts": {str(k): int(v) for k, v in counts.sort_index().to_dict().items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
