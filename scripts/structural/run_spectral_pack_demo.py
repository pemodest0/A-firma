#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest
from engine.structural.spectral import spectral_pack
from scripts.structural._common import load_corr_matrix_or_synthetic, resolve_outdir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run spectral pack demo from latest corr matrix or synthetic fallback.")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--min-assets", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    outdir = resolve_outdir(args.outdir)
    corr, meta = load_corr_matrix_or_synthetic(
        window=int(args.window),
        min_assets=int(args.min_assets),
        seed=int(args.seed),
    )
    eigs = np.real(np.sort(np.linalg.eigvalsh(corr))[::-1])
    pack = spectral_pack(eigs=eigs, topk=int(args.topk))
    payload = {**pack, "meta": meta}

    out_json = outdir / "spectral_pack_demo.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_manifest(
        outdir,
        script="scripts/structural/run_spectral_pack_demo.py",
        params={
            "window": int(args.window),
            "min_assets": int(args.min_assets),
            "topk": int(args.topk),
            "seed": int(args.seed),
        },
        paths={"spectral_pack_demo_json": str(out_json)},
        gates={
            "corr_square": bool(corr.shape[0] == corr.shape[1]),
            "pack_finite_phi": bool(np.isfinite(payload.get("phi", np.nan))),
            "report_written": bool(out_json.exists()),
        },
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "phi": payload["phi"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
