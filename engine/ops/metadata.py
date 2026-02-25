from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["asset_id", "ticker", "sector_gics", "sector_internal"]
OPTIONAL_COLUMNS = ["liquidity_proxy"]


def load_asset_metadata(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"asset metadata not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"asset metadata is empty: {p}")

    miss = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"asset metadata missing columns: {miss}")

    out = df.copy()
    out["asset_id"] = out["asset_id"].astype(str).str.strip()
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["sector_gics"] = out["sector_gics"].astype(str).str.strip()
    out["sector_internal"] = out["sector_internal"].astype(str).str.strip()
    if "liquidity_proxy" not in out.columns:
        out["liquidity_proxy"] = pd.NA
    out["liquidity_proxy"] = pd.to_numeric(out["liquidity_proxy"], errors="coerce")

    if (out["asset_id"] == "").any():
        raise ValueError("asset metadata has empty asset_id rows")
    if out["asset_id"].duplicated().any():
        dup = out.loc[out["asset_id"].duplicated(keep=False), "asset_id"].astype(str).tolist()
        dup_u = sorted(set(dup))
        raise ValueError(f"asset metadata asset_id must be unique; duplicates={dup_u[:10]}")

    keep_cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    return out[keep_cols].sort_values("asset_id").reset_index(drop=True)

