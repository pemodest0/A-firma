#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch Energia Brasil sources (ONS one-shot sync wrapper)."
    )
    ap.add_argument("--download-once", type=int, default=1, help="1=nao redownload; 0=forca redownload")
    ap.add_argument(
        "--datasets",
        type=str,
        default="ons_carga_diaria,ons_ear_subsistema_di,ons_cmo_semanal",
        help="Datasets do ONS separados por virgula.",
    )
    ap.add_argument("--from-year", type=int, default=2018)
    ap.add_argument("--to-year", type=int, default=datetime.now(timezone.utc).year)
    ap.add_argument("--run-adequacy", type=int, default=1)
    ap.add_argument("--results-dir", type=str, default="results/energy_download")
    args = ap.parse_args()

    force = 0 if bool(int(args.download_once)) else 1
    cmd = [
        sys.executable,
        "scripts/data/sync_energy_ons_one_shot.py",
        "--source",
        "ONS",
        "--datasets",
        str(args.datasets),
        "--download",
        "1",
        "--force",
        str(int(force)),
        "--build-pack",
        "1",
        "--run-adequacy",
        str(int(args.run_adequacy)),
        "--from-year",
        str(int(args.from_year)),
        "--to-year",
        str(int(args.to_year)),
        "--results-dir",
        str(args.results_dir),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)  # noqa: S603

    latest_sync = ROOT / str(args.results_dir) / "latest_sync.json"
    payload = {}
    if latest_sync.exists():
        payload = json.loads(latest_sync.read_text(encoding="utf-8"))
    outdir = ROOT / "results" / "energy_br" / f"fetch_energy_br_{_utc_id()}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sync_cmd": cmd,
        "latest_sync": payload,
    }
    out_json = outdir / "fetch_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary_json": str(out_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
