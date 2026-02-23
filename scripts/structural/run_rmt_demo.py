#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.rmt import rmt_report
from engine.structural.run_manifest import write_run_manifest
from scripts.structural._common import load_corr_matrix_or_synthetic, resolve_outdir


def _load_corr_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    arr = df.to_numpy(dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"corr csv must be square: {path}")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RMT demo report from latest corr matrix or synthetic fallback.")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--corr-csv", type=str, default="")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--min-assets", type=int, default=20)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    outdir = resolve_outdir(args.outdir)
    meta: dict[str, Any]
    if str(args.corr_csv).strip():
        corr = _load_corr_csv(Path(args.corr_csv))
        meta = {"source": "user_corr_csv", "corr_csv": str(Path(args.corr_csv))}
    else:
        corr, meta = load_corr_matrix_or_synthetic(
            window=int(args.window),
            min_assets=int(args.min_assets),
            seed=int(args.seed),
        )

    eigs = np.real(np.sort(np.linalg.eigvalsh(corr))[::-1])
    n = int(corr.shape[0])
    t = int(meta.get("n_rows", meta.get("window", args.window)))
    rep = rmt_report(eigs=eigs, T=t, N=n, sigma=1.0)
    payload = {
        **rep,
        "meta": meta,
    }

    out_json = outdir / "rmt_report.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "corr_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_manifest(
        outdir,
        script="scripts/structural/run_rmt_demo.py",
        params={
            "corr_csv": str(args.corr_csv),
            "window": int(args.window),
            "min_assets": int(args.min_assets),
            "seed": int(args.seed),
        },
        paths={
            "rmt_report_json": str(out_json),
            "corr_meta_json": str(outdir / "corr_meta.json"),
        },
        gates={
            "corr_square": bool(corr.shape[0] == corr.shape[1]),
            "enough_assets": bool(corr.shape[0] >= int(args.min_assets)),
            "report_written": bool(out_json.exists()),
        },
        extra={"source_meta": meta},
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "count_sig": payload["count_sig"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
