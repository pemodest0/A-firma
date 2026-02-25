from __future__ import annotations

from typing import Any

import pandas as pd


def _prepare_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata is None or metadata.empty:
        return pd.DataFrame(columns=["asset_id", "ticker", "sector_gics", "sector_internal", "liquidity_proxy"])
    m = metadata.copy()
    if "asset_id" not in m.columns:
        if "ticker" in m.columns:
            m["asset_id"] = m["ticker"]
        else:
            raise ValueError("metadata must include asset_id or ticker")
    if "ticker" not in m.columns:
        m["ticker"] = m["asset_id"]
    if "sector_gics" not in m.columns:
        m["sector_gics"] = "unknown"
    if "sector_internal" not in m.columns:
        m["sector_internal"] = m["sector_gics"]
    if "liquidity_proxy" not in m.columns:
        m["liquidity_proxy"] = pd.NA
    m["asset_id"] = m["asset_id"].astype(str).str.strip()
    m["ticker"] = m["ticker"].astype(str).str.strip()
    m["sector_gics"] = m["sector_gics"].fillna("unknown").astype(str).str.strip()
    m["sector_internal"] = m["sector_internal"].fillna(m["sector_gics"]).astype(str).str.strip()
    m["liquidity_proxy"] = pd.to_numeric(m["liquidity_proxy"], errors="coerce")
    m = m[m["asset_id"] != ""].copy()
    m = m.drop_duplicates(subset=["asset_id"], keep="first")
    return m[["asset_id", "ticker", "sector_gics", "sector_internal", "liquidity_proxy"]].copy()


def _coverage_liquidity_table(
    returns_df: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    eligible_assets: list[str] | None = None,
) -> pd.DataFrame:
    if returns_df is None or returns_df.empty:
        return pd.DataFrame(columns=["asset_id", "coverage", "liquidity_proxy"])
    m = _prepare_metadata(metadata)
    cols = [str(c).strip() for c in returns_df.columns]
    if eligible_assets is not None:
        allow = {str(x).strip() for x in eligible_assets if str(x).strip()}
        cols = [c for c in cols if c in allow]
    if not cols:
        return pd.DataFrame(columns=["asset_id", "coverage", "liquidity_proxy"])
    cov = (
        pd.DataFrame(
            {
                "asset_id": cols,
                "coverage": [float(returns_df[c].notna().mean()) for c in cols],
            }
        )
        .sort_values("asset_id")
        .reset_index(drop=True)
    )
    if m.empty:
        cov["liquidity_proxy"] = cov["coverage"]
        return cov

    out = cov.merge(m[["asset_id", "liquidity_proxy"]], on="asset_id", how="left")
    out["liquidity_proxy"] = pd.to_numeric(out["liquidity_proxy"], errors="coerce").fillna(out["coverage"])
    return out


def _rank_assets(df: pd.DataFrame, n_target: int) -> list[str]:
    if df.empty:
        return []
    x = df.copy()
    x["coverage"] = pd.to_numeric(x["coverage"], errors="coerce")
    x["liquidity_proxy"] = pd.to_numeric(x["liquidity_proxy"], errors="coerce")
    x = x.sort_values(["coverage", "liquidity_proxy", "asset_id"], ascending=[False, False, True]).reset_index(drop=True)
    n = int(max(1, n_target))
    return x["asset_id"].astype(str).head(n).tolist()


def select_global_universe(
    returns_df: pd.DataFrame,
    metadata: pd.DataFrame,
    n_global: int = 250,
    min_coverage: float = 0.98,
) -> list[str]:
    tbl = _coverage_liquidity_table(returns_df=returns_df, metadata=metadata)
    if tbl.empty:
        return []
    x = tbl[pd.to_numeric(tbl["coverage"], errors="coerce") >= float(min_coverage)].copy()
    if x.empty:
        x = tbl.copy()
    return _rank_assets(x, n_target=int(n_global))


def select_sector_universe(
    returns_df: pd.DataFrame,
    metadata: pd.DataFrame,
    sector_name: str,
    n_sector: int = 60,
    min_coverage: float = 0.95,
    sector_col: str = "sector_gics",
) -> list[str]:
    m = _prepare_metadata(metadata)
    if m.empty:
        return []
    sc = str(sector_col).strip() or "sector_gics"
    if sc not in m.columns:
        raise ValueError(f"metadata missing sector column: {sc}")
    sname = str(sector_name).strip()
    if not sname:
        return []
    mm = m[m[sc].astype(str) == sname].copy()
    if mm.empty:
        return []
    eligible = mm["asset_id"].astype(str).tolist()
    tbl = _coverage_liquidity_table(returns_df=returns_df, metadata=mm, eligible_assets=eligible)
    if tbl.empty:
        return []
    x = tbl[pd.to_numeric(tbl["coverage"], errors="coerce") >= float(min_coverage)].copy()
    if x.empty:
        x = tbl.copy()
    return _rank_assets(x, n_target=int(n_sector))
