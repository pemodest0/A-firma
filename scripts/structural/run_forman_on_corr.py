#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.forman_ricci import forman_edge_curvature, forman_summary
from engine.structural.graph import corr_to_graph
from engine.structural.run_manifest import write_run_manifest
from scripts.structural._common import load_corr_matrix_or_synthetic, resolve_outdir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Forman-Ricci diagnostics from one correlation matrix.")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--min-assets", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--method", type=str, default="topk", choices=["topk", "dense"])
    args = ap.parse_args()

    outdir = resolve_outdir(args.outdir)
    corr, meta = load_corr_matrix_or_synthetic(
        window=int(args.window),
        min_assets=int(args.min_assets),
        seed=int(args.seed),
    )
    edges = corr_to_graph(corr, method=str(args.method), k=int(args.k), abs_weights=True)
    curv = forman_edge_curvature(edges, n_nodes=int(corr.shape[0]))
    summary = forman_summary(curv)
    payload = {**summary, "meta": meta, "n_nodes": int(corr.shape[0]), "k": int(args.k), "method": str(args.method)}

    out_json = outdir / "forman_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if edges:
        edges_df = pd.DataFrame(edges, columns=["u", "v", "w"])
        edges_df["curvature"] = curv
        edges_df.to_csv(outdir / "forman_edges.csv", index=False)

    write_run_manifest(
        outdir,
        script="scripts/structural/run_forman_on_corr.py",
        params={
            "window": int(args.window),
            "min_assets": int(args.min_assets),
            "k": int(args.k),
            "seed": int(args.seed),
            "method": str(args.method),
        },
        paths={
            "forman_summary_json": str(out_json),
            "forman_edges_csv": str(outdir / "forman_edges.csv"),
        },
        gates={
            "graph_nonempty": bool(len(edges) > 0),
            "curvature_finite": bool(np.isfinite(curv).sum() > 0),
            "report_written": bool(out_json.exists()),
        },
        extra={"source_meta": meta},
    )
    print(
        json.dumps(
            {"status": "ok", "outdir": str(outdir), "edges": len(edges), "share_negative": payload["share_negative"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
