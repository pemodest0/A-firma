#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.impact import compute_asset_global_impact  # noqa: E402
from engine.structural.run_manifest import write_run_manifest  # noqa: E402


def _safe_float(x: Any) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def _latest_lab_run() -> Path:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        raise FileNotFoundError(f"missing base dir: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for run_dir in runs:
        hier = run_dir / "hierarchical"
        if (hier / "vectors" / "v1_global.csv").exists() and (hier / "diagnostics_global_score_daily.csv").exists():
            return run_dir
    raise FileNotFoundError("no hierarchical run with vectors/global diagnostics found")


def _read_vector(path_no_suffix: Path) -> pd.DataFrame:
    p_parquet = path_no_suffix.with_suffix(".parquet")
    p_csv = path_no_suffix.with_suffix(".csv")
    if p_parquet.exists():
        try:
            return pd.read_parquet(p_parquet)
        except Exception:
            pass
    if p_csv.exists():
        return pd.read_csv(p_csv)
    return pd.DataFrame(columns=["date", "asset_id", "weight"])


def _load_metadata(run_dir: Path) -> pd.DataFrame:
    p1 = run_dir / "hierarchical" / "asset_metadata_used.csv"
    p2 = ROOT / "data" / "asset_metadata.csv"
    p = p1 if p1.exists() else p2
    if not p.exists():
        return pd.DataFrame(columns=["asset_id", "ticker", "sector_gics", "sector_internal"])
    md = pd.read_csv(p)
    if md.empty:
        return pd.DataFrame(columns=["asset_id", "ticker", "sector_gics", "sector_internal"])
    if "asset_id" not in md.columns:
        md["asset_id"] = md.get("ticker", "").astype(str)
    if "ticker" not in md.columns:
        md["ticker"] = md["asset_id"]
    if "sector_gics" not in md.columns:
        md["sector_gics"] = "unknown"
    if "sector_internal" not in md.columns:
        md["sector_internal"] = md["sector_gics"]
    md["asset_id"] = md["asset_id"].astype(str)
    md["ticker"] = md["ticker"].astype(str)
    md["sector_gics"] = md["sector_gics"].astype(str)
    md["sector_internal"] = md["sector_internal"].astype(str)
    return md[["asset_id", "ticker", "sector_gics", "sector_internal"]].drop_duplicates(subset=["asset_id"], keep="first")


def _load_global_state(hier_dir: Path) -> pd.DataFrame:
    score_path = hier_dir / "diagnostics_global_score_daily.csv"
    daily_path = hier_dir / "diagnostics_global_daily.csv"
    p = score_path if score_path.exists() else daily_path
    if not p.exists():
        return pd.DataFrame(columns=["date", "score", "phi", "deff", "Q", "N_used"])
    d = pd.read_csv(p)
    if d.empty or ("date" not in d.columns):
        return pd.DataFrame(columns=["date", "score", "phi", "deff", "Q", "N_used"])
    if "score" not in d.columns:
        d["score"] = np.nan
    if "phi" not in d.columns:
        d["phi"] = np.nan
    if "deff" not in d.columns:
        d["deff"] = np.nan
    if "Q" not in d.columns:
        d["Q"] = np.nan
    if "N_used" not in d.columns:
        d["N_used"] = np.nan
    out = d[["date", "score", "phi", "deff", "Q", "N_used"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["phi"] = pd.to_numeric(out["phi"], errors="coerce")
    out["deff"] = pd.to_numeric(out["deff"], errors="coerce")
    out["Q"] = pd.to_numeric(out["Q"], errors="coerce")
    out["N_used"] = pd.to_numeric(out["N_used"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _load_sector_overlap(hier_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for kind in ("gics", "internal"):
        p = hier_dir / f"cross_sector_global_{kind}_daily.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if d.empty or ("date" not in d.columns) or ("sector" not in d.columns):
            continue
        d["sector_kind"] = str(kind)
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        d["overlap_sector_global"] = pd.to_numeric(d.get("overlap_sector_global"), errors="coerce")
        frames.append(d[["date", "sector_kind", "sector", "overlap_sector_global"]].copy())
    if not frames:
        return pd.DataFrame(columns=["date", "sector_kind", "sector", "overlap_sector_global"])
    out = pd.concat(frames, ignore_index=True)
    out["sector"] = out["sector"].astype(str)
    return out.dropna(subset=["date"]).sort_values(["date", "sector_kind", "sector"]).reset_index(drop=True)


def _attach_metadata(asset_global: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if asset_global.empty:
        return pd.DataFrame(columns=["date", "asset_id", "ticker", "impact_global", "sector_gics", "sector_internal"])
    x = asset_global.copy()
    x["date"] = x["date"].astype(str)
    x["asset_id"] = x["asset_id"].astype(str)
    x["impact_global"] = pd.to_numeric(x["impact_global"], errors="coerce").fillna(0.0)
    if metadata is None or metadata.empty:
        x["ticker"] = x["asset_id"]
        x["sector_gics"] = "unknown"
        x["sector_internal"] = "unknown"
        return x[["date", "asset_id", "ticker", "impact_global", "sector_gics", "sector_internal"]].sort_values(["date", "asset_id"]).reset_index(drop=True)
    md = metadata.copy()
    md["asset_id"] = md["asset_id"].astype(str)
    out = x.merge(md, on="asset_id", how="left")
    out["ticker"] = out["ticker"].fillna(out["asset_id"]).astype(str)
    out["sector_gics"] = out["sector_gics"].fillna("unknown").astype(str)
    out["sector_internal"] = out["sector_internal"].fillna("unknown").astype(str)
    return out[["date", "asset_id", "ticker", "impact_global", "sector_gics", "sector_internal"]].sort_values(["date", "asset_id"]).reset_index(drop=True)


def _sector_impact_rows(day_assets: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind_col, kind_name in [("sector_gics", "gics"), ("sector_internal", "internal")]:
        g = (
            day_assets.groupby(kind_col, dropna=False)["impact_global"]
            .sum()
            .reset_index()
            .rename(columns={kind_col: "sector", "impact_global": "impact"})
        )
        if g.empty:
            continue
        g["sector_kind"] = str(kind_name)
        g["sector"] = g["sector"].fillna("unknown").astype(str)
        rows.append(g[["sector", "sector_kind", "impact"]])
    if not rows:
        return []
    out = pd.concat(rows, ignore_index=True).sort_values(["impact", "sector_kind", "sector"], ascending=[False, True, True])
    out["impact"] = pd.to_numeric(out["impact"], errors="coerce")
    return out.head(int(max(1, top_k))).to_dict(orient="records")


def _overlap_rows(day_overlap: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    if day_overlap is None or day_overlap.empty:
        return []
    x = day_overlap.copy()
    x["sector"] = x["sector"].astype(str)
    x["sector_kind"] = x["sector_kind"].astype(str)
    x["overlap"] = pd.to_numeric(x["overlap_sector_global"], errors="coerce")
    x = x.dropna(subset=["overlap"]).sort_values(["overlap", "sector_kind", "sector"], ascending=[False, True, True])
    return x[["sector", "sector_kind", "overlap"]].head(int(max(1, top_k))).to_dict(orient="records")


def _build_payloads(
    *,
    asset_global_daily: pd.DataFrame,
    overlap_daily: pd.DataFrame,
    global_state_daily: pd.DataFrame,
    top_assets: int,
    top_sectors: int,
) -> list[dict[str, Any]]:
    if global_state_daily.empty:
        return []
    payloads: list[dict[str, Any]] = []
    for d in sorted(global_state_daily["date"].dropna().astype(str).unique().tolist()):
        g_row = global_state_daily[global_state_daily["date"] == d].tail(1)
        day_assets = asset_global_daily[asset_global_daily["date"] == d].copy()
        day_assets = day_assets.sort_values(["impact_global", "asset_id"], ascending=[False, True])
        top_assets_rows = (
            day_assets[["asset_id", "ticker", "impact_global", "sector_gics", "sector_internal"]]
            .head(int(max(1, top_assets)))
            .rename(columns={"impact_global": "impact"})
            .to_dict(orient="records")
        )

        top_sector_rows = _sector_impact_rows(day_assets, top_k=int(max(1, top_sectors)))
        day_overlap = overlap_daily[overlap_daily["date"] == d].copy() if not overlap_daily.empty else pd.DataFrame()
        top_overlap_rows = _overlap_rows(day_overlap, top_k=int(max(1, top_sectors)))

        payloads.append(
            {
                "date": str(d),
                "top_assets_global_mode": top_assets_rows,
                "top_sectors_global_mode": top_sector_rows,
                "sector_global_overlap": top_overlap_rows,
                "global_state": {
                    "score": _safe_float(g_row["score"].iloc[-1]) if not g_row.empty else float("nan"),
                    "phi": _safe_float(g_row["phi"].iloc[-1]) if not g_row.empty else float("nan"),
                    "deff": _safe_float(g_row["deff"].iloc[-1]) if not g_row.empty else float("nan"),
                    "q": _safe_float(g_row["Q"].iloc[-1]) if not g_row.empty else float("nan"),
                    "n_used": _safe_float(g_row["N_used"].iloc[-1]) if not g_row.empty else float("nan"),
                },
            }
        )
    return payloads


def main() -> None:
    ap = argparse.ArgumentParser(description="Build daily hierarchical rankings for platform/UI consumption.")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--hierarchical-dir", type=str, default="")
    ap.add_argument("--outdir", type=str, default="results/platform")
    ap.add_argument("--top-assets", type=int, default=10)
    ap.add_argument("--top-sectors", type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else _latest_lab_run()
    hier_dir = Path(args.hierarchical_dir).resolve() if str(args.hierarchical_dir).strip() else (run_dir / "hierarchical")
    if not hier_dir.exists():
        raise SystemExit(f"missing hierarchical dir: {hier_dir}")

    outdir = (ROOT / str(args.outdir).strip()) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    v1_global = _read_vector(hier_dir / "vectors" / "v1_global")
    if v1_global.empty:
        raise SystemExit(f"missing v1 global vectors in {hier_dir / 'vectors'}")
    metadata = _load_metadata(run_dir)
    global_state = _load_global_state(hier_dir)
    overlap = _load_sector_overlap(hier_dir)

    asset_global = compute_asset_global_impact(v1_global)
    asset_global_daily = _attach_metadata(asset_global=asset_global, metadata=metadata)
    payloads = _build_payloads(
        asset_global_daily=asset_global_daily,
        overlap_daily=overlap,
        global_state_daily=global_state,
        top_assets=int(max(1, args.top_assets)),
        top_sectors=int(max(1, args.top_sectors)),
    )
    latest = payloads[-1] if payloads else {}

    daily_path = outdir / "rankings_daily.jsonl"
    latest_path = outdir / "rankings_latest.json"
    summary_path = outdir / "rankings_summary.json"
    manifest_path = outdir / "RUN_MANIFEST.json"

    daily_path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in payloads) + ("\n" if payloads else ""),
        encoding="utf-8",
    )
    latest_path.write_text(json.dumps(latest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "hierarchical_dir": str(hier_dir),
        "counts": {
            "daily_rows": int(len(payloads)),
            "asset_rows": int(asset_global_daily.shape[0]),
            "overlap_rows": int(overlap.shape[0]),
        },
        "files": {
            "rankings_daily_jsonl": str(daily_path),
            "rankings_latest_json": str(latest_path),
            "run_manifest_json": str(manifest_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/build_hierarchical_rankings.py",
        params={
            "run_dir": str(run_dir),
            "hierarchical_dir": str(hier_dir),
            "top_assets": int(max(1, args.top_assets)),
            "top_sectors": int(max(1, args.top_sectors)),
        },
        paths={
            "rankings_daily_jsonl": str(daily_path),
            "rankings_latest_json": str(latest_path),
            "rankings_summary_json": str(summary_path),
        },
        gates={
            "global_state_nonempty": bool(global_state.shape[0] > 0),
            "asset_global_nonempty": bool(asset_global_daily.shape[0] > 0),
            "daily_written": bool(daily_path.exists()),
            "latest_written": bool(latest_path.exists()),
        },
        extra={
            "run_dir": str(run_dir),
            "daily_rows": int(len(payloads)),
        },
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "run_dir": str(run_dir),
                "outdir": str(outdir),
                "daily_rows": int(len(payloads)),
                "latest_date": str(latest.get("date", "")) if isinstance(latest, dict) else "",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
