#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import argparse
import json
from pathlib import Path

from engine.temporal import (
    TemporalConfig,
    build_temporal_report,
    compare_models,
    load_yearly_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera veredito de previsibilidade a partir de CSVs anuais.")
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "CSV_PATH"), required=True)
    parser.add_argument("--year-col", default="year")
    parser.add_argument("--model-col", default="model_error")
    parser.add_argument("--baseline-col", default="baseline_error")
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--required-win-rate", type=float, default=0.6)
    parser.add_argument("--min-years", type=int, default=5)
    parser.add_argument("--out", default="results/temporal_report.json")
    args = parser.parse_args()

    cfg = TemporalConfig(
        min_improvement_pct=args.min_improvement,
        required_win_rate=args.required_win_rate,
        required_years_min=args.min_years,
    )

    models = {}
    for name, csv_path in args.model:
        path = Path(csv_path)
        years = load_yearly_csv(path, args.year_col, args.model_col, args.baseline_col)
        models[name] = {"default": years}

    summaries = compare_models(models, cfg)
    report = build_temporal_report(summaries, cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Veredito salvo em {out_path}")


if __name__ == "__main__":
    main()

