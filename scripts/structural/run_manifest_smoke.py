#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest
from scripts.structural._common import resolve_outdir


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke script for RUN_MANIFEST generation.")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    outdir = resolve_outdir(args.outdir)
    payload = {"status": "ok", "seed": int(args.seed)}
    (outdir / "manifest_smoke.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_manifest(
        outdir,
        script="scripts/structural/run_manifest_smoke.py",
        params={"seed": int(args.seed)},
        paths={"manifest_smoke_json": str(outdir / "manifest_smoke.json")},
        gates={"smoke_file_written": True},
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
