#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.csd import rolling_ac1
from engine.structural.forman_ricci import forman_edge_curvature, forman_summary
from engine.structural.graph import corr_to_graph
from engine.structural.run_manifest import write_run_manifest
from engine.structural.score import fit_normalizer, structural_score, transform
from scripts.structural._common import ROOT as PROJECT_ROOT
from scripts.structural._common import resolve_outdir


def _latest_lab_run_dir() -> Path | None:
    base = PROJECT_ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in runs:
        if (d / "macro_timeseries_T120.csv").exists():
            return d
    return None


def _synthetic_daily(n: int = 600, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    idx = pd.date_range("2020-01-01", periods=int(n), freq="D")
    phi = 0.45 + 0.05 * np.sin(np.linspace(0, 12.0, n)) + 0.02 * rng.normal(size=n)
    deff = 20.0 - 3.0 * np.sin(np.linspace(0, 10.0, n)) + 0.4 * rng.normal(size=n)
    forman = -0.2 + 0.15 * np.sin(np.linspace(0, 9.0, n) + 0.3) + 0.05 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "date": idx,
            "phi": phi,
            "deff": deff,
            "forman_mean": forman,
        }
    )


def _load_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    else:
        first = str(df.columns[0])
        df[first] = pd.to_datetime(df[first], errors="coerce")
        df = df.dropna(subset=[first]).set_index(first)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()


def _forman_series_from_returns(
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    corr_window: int,
    coverage: float,
    min_assets: int,
    topk: int,
) -> pd.Series:
    if returns.empty or len(dates) == 0:
        return pd.Series(dtype=float)
    index_map = {pd.Timestamp(d): i for i, d in enumerate(returns.index)}
    out = []
    for d in dates:
        i = index_map.get(pd.Timestamp(d))
        if i is None or i < int(corr_window) - 1:
            out.append(np.nan)
            continue
        block = returns.iloc[i - int(corr_window) + 1 : i + 1].copy()
        cov = block.notna().mean(axis=0)
        keep = cov[cov >= float(coverage)].index.to_list()
        if len(keep) < int(min_assets):
            out.append(np.nan)
            continue
        block = block[keep].dropna(how="any")
        if block.shape[0] < max(30, int(corr_window) // 2) or block.shape[1] < int(min_assets):
            out.append(np.nan)
            continue
        corr = np.corrcoef(block.to_numpy(dtype=float), rowvar=False)
        if not np.all(np.isfinite(corr)):
            out.append(np.nan)
            continue
        edges = corr_to_graph(corr, method="topk", k=int(topk), abs_weights=True)
        curv = forman_edge_curvature(edges, n_nodes=int(corr.shape[0]))
        s = forman_summary(curv)
        out.append(float(s["mean"]) if np.isfinite(s["mean"]) else np.nan)
    return pd.Series(out, index=dates, dtype=float)


def _load_or_build_daily(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    run_dir = _latest_lab_run_dir()
    if run_dir is None:
        df = _synthetic_daily(seed=int(args.seed))
        return df, {"source": "synthetic", "run_dir": ""}

    ts_path = run_dir / f"macro_timeseries_T{int(args.official_window)}.csv"
    if not ts_path.exists():
        df = _synthetic_daily(seed=int(args.seed))
        return df, {"source": "synthetic", "run_dir": ""}

    ts = pd.read_csv(ts_path)
    if ("date" not in ts.columns) or ("p1" not in ts.columns) or ("deff" not in ts.columns):
        df = _synthetic_daily(seed=int(args.seed))
        return df, {"source": "synthetic", "run_dir": ""}

    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts["p1"] = pd.to_numeric(ts["p1"], errors="coerce")
    ts["deff"] = pd.to_numeric(ts["deff"], errors="coerce")
    ts = ts.dropna(subset=["date", "p1", "deff"]).sort_values("date")
    if ts.empty:
        df = _synthetic_daily(seed=int(args.seed))
        return df, {"source": "synthetic", "run_dir": ""}

    out = pd.DataFrame({"date": ts["date"], "phi": ts["p1"], "deff": ts["deff"]})
    if "forman_mean" in ts.columns:
        out["forman_mean"] = pd.to_numeric(ts["forman_mean"], errors="coerce")
    else:
        out["forman_mean"] = np.nan

    if out["forman_mean"].notna().sum() == 0:
        ret_path = run_dir / "returns_wide_core.csv"
        if ret_path.exists():
            returns = _load_returns(ret_path)
            f = _forman_series_from_returns(
                returns,
                pd.DatetimeIndex(out["date"]),
                corr_window=int(args.official_window),
                coverage=float(args.coverage_window),
                min_assets=int(args.min_assets),
                topk=int(args.graph_topk),
            )
            out["forman_mean"] = f.to_numpy(dtype=float)

    return out.reset_index(drop=True), {"source": "latest_lab_corr_macro", "run_dir": str(run_dir)}


def _resolve_train_mask(dates: pd.Series, train_end: str) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    if train_end.strip():
        cutoff = pd.Timestamp(train_end.strip())
        m = d <= cutoff
        if m.sum() > 5:
            return pd.Series(m, index=dates.index)
    pos = int(max(5, min(len(d) - 1, int(0.7 * len(d)))))
    cutoff = pd.Timestamp(d.iloc[pos])
    return pd.Series(d <= cutoff, index=dates.index)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build structural daily diagnostics and fusion score.")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--official-window", type=int, default=120)
    ap.add_argument("--coverage-window", type=float, default=0.98)
    ap.add_argument("--min-assets", type=int, default=25)
    ap.add_argument("--graph-topk", type=int, default=10)
    ap.add_argument("--csd-window", type=int, default=60)
    ap.add_argument("--train-end", type=str, default="2019-12-31")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    outdir = resolve_outdir(args.outdir)
    base, meta = _load_or_build_daily(args)

    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base["phi"] = pd.to_numeric(base["phi"], errors="coerce")
    base["deff"] = pd.to_numeric(base["deff"], errors="coerce")
    base["forman_mean"] = pd.to_numeric(base["forman_mean"], errors="coerce")
    base = base.dropna(subset=["date", "phi", "deff"]).sort_values("date").reset_index(drop=True)

    ac1 = rolling_ac1(base["phi"], window=int(args.csd_window))
    base["ac1_phi"] = pd.to_numeric(ac1, errors="coerce")
    base["flags_valid"] = (
        base[["phi", "deff", "ac1_phi", "forman_mean"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).astype(int)
    )

    train_mask = _resolve_train_mask(base["date"], train_end=str(args.train_end))
    train_dict = {
        "phi": base.loc[train_mask, "phi"],
        "deff": base.loc[train_mask, "deff"],
        "ac1_phi": base.loc[train_mask, "ac1_phi"],
        "neg_kappa_mean": -base.loc[train_mask, "forman_mean"].fillna(0.0),
    }
    normalizer = fit_normalizer(train_dict)

    z_phi = transform(base["phi"], normalizer, key="phi")
    z_deff = transform(base["deff"], normalizer, key="deff")
    z_ac1 = transform(base["ac1_phi"], normalizer, key="ac1_phi")
    z_neg_kappa = transform(-base["forman_mean"].fillna(0.0), normalizer, key="neg_kappa_mean")
    score = structural_score(
        {
            "phi": z_phi,
            "deff": z_deff,
            "ac1_phi": z_ac1,
            "neg_kappa_mean": z_neg_kappa,
        }
    )

    daily = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(base["date"]).strftime("%Y-%m-%d"),
            "phi": base["phi"].to_numpy(dtype=float),
            "deff": base["deff"].to_numpy(dtype=float),
            "ac1_phi": base["ac1_phi"].to_numpy(dtype=float),
            "forman_mean": base["forman_mean"].to_numpy(dtype=float),
            "flags_valid": base["flags_valid"].to_numpy(dtype=int),
            "score": pd.to_numeric(score, errors="coerce").to_numpy(dtype=float),
        }
    )
    score_csv = outdir / "diagnostics_structural_score_daily.csv"
    daily.to_csv(score_csv, index=False)

    diag_csv = outdir / "diagnostics_structural_daily.csv"
    daily[["date", "phi", "deff", "ac1_phi", "forman_mean", "flags_valid"]].to_csv(diag_csv, index=False)

    meta_payload = {
        "source_meta": meta,
        "normalizer": normalizer,
        "params": {
            "official_window": int(args.official_window),
            "coverage_window": float(args.coverage_window),
            "min_assets": int(args.min_assets),
            "graph_topk": int(args.graph_topk),
            "csd_window": int(args.csd_window),
            "train_end": str(args.train_end),
        },
    }
    (outdir / "structural_score_meta.json").write_text(
        json.dumps(meta_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_run_manifest(
        outdir,
        script="scripts/structural/run_structural_score_demo.py",
        params={
            "official_window": int(args.official_window),
            "coverage_window": float(args.coverage_window),
            "min_assets": int(args.min_assets),
            "graph_topk": int(args.graph_topk),
            "csd_window": int(args.csd_window),
            "train_end": str(args.train_end),
            "seed": int(args.seed),
        },
        paths={
            "diagnostics_structural_daily_csv": str(diag_csv),
            "diagnostics_structural_score_daily_csv": str(score_csv),
            "structural_score_meta_json": str(outdir / "structural_score_meta.json"),
        },
        gates={
            "daily_rows_nonempty": bool(daily.shape[0] > 0),
            "score_has_values": bool(np.isfinite(daily["score"]).sum() > 0),
            "required_columns_present": bool(
                {"date", "score", "phi", "deff", "ac1_phi", "forman_mean", "flags_valid"}.issubset(set(daily.columns))
            ),
        },
        extra={"source_meta": meta},
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "n_rows": int(daily.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
