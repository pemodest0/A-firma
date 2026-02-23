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

from engine.structural.csd import ews_pack
from engine.structural.run_manifest import write_run_manifest
from scripts.structural._common import load_phi_series_from_latest, resolve_outdir


def _resolve_train_end(idx: pd.DatetimeIndex, train_end_str: str) -> pd.Timestamp:
    if train_end_str.strip():
        return pd.Timestamp(train_end_str.strip())
    if len(idx) == 0:
        return pd.Timestamp("1970-01-01")
    pos = int(max(0, min(len(idx) - 1, int(0.7 * len(idx)))))
    return pd.Timestamp(idx[pos])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CSD indicators on phi(t) with causal rolling windows.")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--train-end", type=str, default="")
    args = ap.parse_args()

    outdir = resolve_outdir(args.outdir)
    phi, meta = load_phi_series_from_latest()
    train_end = _resolve_train_end(pd.DatetimeIndex(phi.index), str(args.train_end))
    pack = ews_pack(phi, window=int(args.window), train_end=train_end)

    out_df = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(phi.index).strftime("%Y-%m-%d"),
            "phi": pd.to_numeric(phi, errors="coerce").to_numpy(dtype=float),
            "var_phi": pd.to_numeric(pack["var"], errors="coerce").to_numpy(dtype=float),
            "ac1_phi": pd.to_numeric(pack["ac1"], errors="coerce").to_numpy(dtype=float),
            "z_var_phi": pd.to_numeric(pack["z_var"], errors="coerce").to_numpy(dtype=float),
            "z_ac1_phi": pd.to_numeric(pack["z_ac1"], errors="coerce").to_numpy(dtype=float),
        }
    )
    out_csv = outdir / "ews_phi.csv"
    out_df.to_csv(out_csv, index=False)
    (outdir / "ews_phi_meta.json").write_text(
        json.dumps(
            {
                "source_meta": meta,
                "window": int(args.window),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "n_rows": int(out_df.shape[0]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_run_manifest(
        outdir,
        script="scripts/structural/run_csd_on_phi.py",
        params={"window": int(args.window), "train_end": train_end.strftime("%Y-%m-%d")},
        paths={
            "ews_phi_csv": str(out_csv),
            "ews_phi_meta_json": str(outdir / "ews_phi_meta.json"),
        },
        gates={
            "phi_series_nonempty": bool(out_df.shape[0] > 0),
            "ac1_has_values": bool(np.isfinite(out_df["ac1_phi"]).sum() > 0),
            "report_written": bool(out_csv.exists()),
        },
        extra={"source_meta": meta},
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "n_rows": int(out_df.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
