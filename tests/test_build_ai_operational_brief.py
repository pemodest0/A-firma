from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_ai_operational_brief_outputs_latest_pointer(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    impact_dir = run_dir / "hierarchical" / "impact_learning_2015_2026_compare"
    impact_dir.mkdir(parents=True, exist_ok=True)

    (impact_dir / "historical_structure_summary.json").write_text(
        json.dumps(
            {
                "data_last_date": "2026-02-12",
                "verdict_counts_expanding": {"forte": 5, "moderado": 3},
                "stress_prealert_summary_top10": [
                    {"scope": "global", "before_5d_rate": 0.154}
                ],
            }
        ),
        encoding="utf-8",
    )
    (impact_dir / "historical_structure_next_month_indication.json").write_text(
        json.dumps(
            {
                "as_of_date": "2026-02-12",
                "data_last_date": "2026-02-12",
                "risk_level_next_month": "alto",
            }
        ),
        encoding="utf-8",
    )

    platform_dir = run_dir / "platform"
    platform_dir.mkdir(parents=True, exist_ok=True)
    (platform_dir / "rankings_latest.json").write_text(
        json.dumps(
            {
                "top_sectors_global_mode": [{"sector": "industrials", "impact": 0.22}],
                "top_assets_global_mode": [{"asset_id": "AAA", "impact": 0.04}],
                "sector_global_overlap": [{"sector": "industrials", "overlap": 0.83}],
                "global_state": {"score": 1.2, "phi": 0.3, "deff": 12.5, "q": 1.0},
            }
        ),
        encoding="utf-8",
    )

    sweep_dir = run_dir / "hierarchical" / "epistemic_horizon_sweep_global"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "ground_truth": "regime_entry",
                "horizon_days": 10,
                "model": "combined_or",
                "f1_mean": 0.63,
                "recall_mean": 0.55,
                "lift_precision_mean": 2.1,
                "lead_time_median": 5.0,
            },
            {
                "ground_truth": "drawdown",
                "horizon_days": 20,
                "model": "score_only",
                "f1_mean": 0.25,
                "recall_mean": 0.18,
                "lift_precision_mean": 1.4,
                "lead_time_median": 3.0,
            },
        ]
    ).to_csv(sweep_dir / "horizon_winners_global.csv", index=False)

    outdir = tmp_path / "ai_knowledge"
    subprocess.run(
        [
            sys.executable,
            "scripts/ops/build_ai_operational_brief.py",
            "--run-dir",
            str(run_dir),
            "--impact-dir",
            str(impact_dir),
            "--outdir",
            str(outdir),
        ],
        check=True,
    )

    latest_path = outdir / "latest_operational_brief.json"
    assert latest_path.exists()
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["status"] == "ok"
    assert latest["data_last_date"] == "2026-02-12"

    brief_path = Path(latest["operational_brief_path"])
    assert brief_path.exists()
    brief = json.loads(brief_path.read_text(encoding="utf-8"))
    assert brief["operational_signal"]["operational_state"] == "defensivo"
    assert brief["freshness"]["status"] in {"fresh", "attention", "stale", "unknown"}
    assert brief["state_snapshot"]["top_sectors_global_mode"][0]["sector"] == "industrials"
