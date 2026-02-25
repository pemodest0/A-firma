from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd


def _as_vector_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "asset_id", "weight"])
    x = df.copy()
    for col in ("date", "asset_id", "weight"):
        if col not in x.columns:
            return pd.DataFrame(columns=["date", "asset_id", "weight"])
    x["date"] = x["date"].astype(str)
    x["asset_id"] = x["asset_id"].astype(str)
    x["weight"] = pd.to_numeric(x["weight"], errors="coerce")
    x = x.dropna(subset=["date", "asset_id", "weight"]).copy()
    return x[["date", "asset_id", "weight"]]


def _normalize_square_by_group(df: pd.DataFrame, *, group_cols: list[str], weight_col: str, out_col: str) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out[out_col] = pd.Series(dtype=float)
        return out
    x = df.copy()
    w2 = np.square(pd.to_numeric(x[weight_col], errors="coerce").astype(float))
    x["_w2"] = w2
    den = x.groupby(group_cols, dropna=False)["_w2"].transform("sum")
    den = pd.to_numeric(den, errors="coerce").astype(float)
    x[out_col] = np.where(den > 1e-12, x["_w2"] / den, 0.0)
    x[out_col] = pd.to_numeric(x[out_col], errors="coerce").fillna(0.0).astype(float)
    return x.drop(columns=["_w2"])


def compute_asset_global_impact(v1_global: pd.DataFrame) -> pd.DataFrame:
    g = _as_vector_df(v1_global)
    if g.empty:
        return pd.DataFrame(columns=["date", "asset_id", "impact_global"])
    g = _normalize_square_by_group(g, group_cols=["date"], weight_col="weight", out_col="impact_global")
    return g[["date", "asset_id", "impact_global"]].copy().sort_values(["date", "asset_id"]).reset_index(drop=True)


def compute_asset_sector_impact(sector_vectors: dict[Any, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key, raw in sector_vectors.items():
        df = _as_vector_df(raw)
        if df.empty:
            continue
        kind = "gics"
        sector = str(key)
        if isinstance(key, tuple) and len(key) == 2:
            kind = str(key[0]).strip().lower() or "gics"
            sector = str(key[1]).strip()
        elif isinstance(key, str) and ":" in key:
            a, b = key.split(":", 1)
            kind = str(a).strip().lower() or "gics"
            sector = str(b).strip()
        x = _normalize_square_by_group(df, group_cols=["date"], weight_col="weight", out_col="impact_sector")
        x["sector_kind"] = str(kind)
        x["sector"] = str(sector)
        rows.append(x[["date", "asset_id", "sector_kind", "sector", "impact_sector"]])
    if not rows:
        return pd.DataFrame(columns=["date", "asset_id", "sector_kind", "sector", "impact_sector"])
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["date", "sector_kind", "sector", "asset_id"])
        .reset_index(drop=True)
    )


def merge_asset_sector_global_impacts(
    *,
    asset_global: pd.DataFrame,
    asset_sector: pd.DataFrame,
    cross_daily: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    s = asset_sector.copy()
    if s.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "asset_id",
                "ticker",
                "sector_kind",
                "sector",
                "impact_global",
                "impact_sector",
                "sector_loading",
                "overlap_sector_global",
                "global_share_within_sector",
            ]
        )
    g = asset_global.copy()
    if g.empty:
        g = pd.DataFrame(columns=["date", "asset_id", "impact_global"])
    c = cross_daily.copy()
    if c.empty:
        c = pd.DataFrame(columns=["date", "sector_kind", "sector", "sector_loading", "overlap_sector_global"])

    s["date"] = s["date"].astype(str)
    s["asset_id"] = s["asset_id"].astype(str)
    s["sector_kind"] = s["sector_kind"].astype(str).str.lower()
    s["sector"] = s["sector"].astype(str)
    s["impact_sector"] = pd.to_numeric(s["impact_sector"], errors="coerce").fillna(0.0)

    g["date"] = g["date"].astype(str)
    g["asset_id"] = g["asset_id"].astype(str)
    g["impact_global"] = pd.to_numeric(g["impact_global"], errors="coerce").fillna(0.0)

    c["date"] = c["date"].astype(str)
    if "kind" in c.columns and "sector_kind" not in c.columns:
        c["sector_kind"] = c["kind"]
    c["sector_kind"] = c.get("sector_kind", "gics").astype(str).str.lower()
    c["sector"] = c["sector"].astype(str)
    c["sector_loading"] = pd.to_numeric(c.get("loading_sector_on_global"), errors="coerce")
    c["overlap_sector_global"] = pd.to_numeric(c.get("overlap_sector_global"), errors="coerce")
    c = c[["date", "sector_kind", "sector", "sector_loading", "overlap_sector_global"]].drop_duplicates(
        subset=["date", "sector_kind", "sector"], keep="last"
    )

    out = s.merge(g[["date", "asset_id", "impact_global"]], on=["date", "asset_id"], how="left")
    out = out.merge(c, on=["date", "sector_kind", "sector"], how="left")
    out["impact_global"] = pd.to_numeric(out["impact_global"], errors="coerce").fillna(0.0)
    out["sector_loading"] = pd.to_numeric(out["sector_loading"], errors="coerce")
    out["overlap_sector_global"] = pd.to_numeric(out["overlap_sector_global"], errors="coerce")
    out["global_share_within_sector"] = np.where(
        out["sector_loading"].fillna(0.0) > 1e-12,
        out["impact_global"] / out["sector_loading"].clip(lower=1e-12),
        np.nan,
    )

    if metadata is not None and (not metadata.empty):
        md = metadata.copy()
        if "asset_id" in md.columns:
            md["asset_id"] = md["asset_id"].astype(str)
        else:
            md["asset_id"] = md.get("ticker", "").astype(str)
        if "ticker" not in md.columns:
            md["ticker"] = md["asset_id"]
        md["ticker"] = md["ticker"].astype(str)
        out = out.merge(md[["asset_id", "ticker"]].drop_duplicates(subset=["asset_id"], keep="first"), on="asset_id", how="left")
    else:
        out["ticker"] = out["asset_id"]

    return (
        out[
            [
                "date",
                "asset_id",
                "ticker",
                "sector_kind",
                "sector",
                "impact_global",
                "impact_sector",
                "sector_loading",
                "overlap_sector_global",
                "global_share_within_sector",
            ]
        ]
        .sort_values(["date", "sector_kind", "sector", "asset_id"])
        .reset_index(drop=True)
    )


def compute_sector_pair_overlap(sector_vectors: dict[Any, pd.DataFrame]) -> pd.DataFrame:
    normalized: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for key, raw in sector_vectors.items():
        x = _as_vector_df(raw)
        if x.empty:
            continue
        kind = "gics"
        sector = str(key)
        if isinstance(key, tuple) and len(key) == 2:
            kind = str(key[0]).strip().lower() or "gics"
            sector = str(key[1]).strip()
        elif isinstance(key, str) and ":" in key:
            a, b = key.split(":", 1)
            kind = str(a).strip().lower() or "gics"
            sector = str(b).strip()
        kind = str(kind).strip().lower() or "gics"
        sector = str(sector).strip()
        if not sector:
            continue
        if kind not in normalized:
            normalized[kind] = {}
        per_date: dict[str, dict[str, float]] = {}
        for d, grp in x.groupby("date", sort=True):
            weights = pd.to_numeric(grp["weight"], errors="coerce").astype(float).to_numpy()
            assets = grp["asset_id"].astype(str).tolist()
            norm = float(np.linalg.norm(weights))
            if norm <= 1e-12:
                continue
            per_date[str(d)] = {a: float(w / norm) for a, w in zip(assets, weights)}
        if per_date:
            normalized[kind][sector] = per_date

    rows: list[dict[str, Any]] = []
    for kind, sectors in normalized.items():
        names = sorted(sectors.keys())
        if len(names) < 2:
            continue
        all_dates = sorted(set(itertools.chain.from_iterable([set(sectors[s].keys()) for s in names])))
        for d in all_dates:
            for a, b in itertools.combinations(names, 2):
                va = sectors[a].get(d)
                vb = sectors[b].get(d)
                if not va or not vb:
                    continue
                common = sorted(set(va.keys()).intersection(vb.keys()))
                dot = float(sum(float(va[k]) * float(vb[k]) for k in common)) if common else 0.0
                overlap = float(abs(dot))
                rows.append(
                    {
                        "date": str(d),
                        "sector_kind": str(kind),
                        "sector_a": str(a),
                        "sector_b": str(b),
                        "overlap_ab": overlap,
                        "n_common_assets": int(len(common)),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["date", "sector_kind", "sector_a", "sector_b", "overlap_ab", "n_common_assets"])
    return pd.DataFrame(rows).sort_values(["date", "sector_kind", "sector_a", "sector_b"]).reset_index(drop=True)
