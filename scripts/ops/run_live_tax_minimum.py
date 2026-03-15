#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.live_tax import build_live_tax_summary, write_json, write_monthly_summary_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera a camada fiscal minima em cima do ledger live.")
    ap.add_argument("--ledger-csv", default="results/ops/execution_live/execution_ledger.csv")
    ap.add_argument("--outdir-root", default="results/ops/execution_live")
    ap.add_argument("--net-assumptions-config", default="config/profit_net_assumptions.json")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary = build_live_tax_summary(
        ledger_csv=ROOT / args.ledger_csv,
        net_assumptions_config=ROOT / args.net_assumptions_config,
    )
    write_json(outdir / "latest_tax_summary.json", summary)
    write_monthly_summary_csv(outdir / "latest_tax_monthly_summary.csv", summary.get("monthly_rows", []))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
