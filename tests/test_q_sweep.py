from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.structural.run_q_sweep import _objective, _q_stats


def test_objective_penalizes_low_q_and_rewards_signal() -> None:
    weak = _objective(best_f1=0.10, best_lift_precision=1.00, q_median=0.30)
    strong = _objective(best_f1=0.25, best_lift_precision=1.40, q_median=0.95)
    assert strong > weak


def test_q_stats_uses_q_column_when_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "N_used": [100, 120, 125],
            "Q": [1.20, 1.00, 0.96],
            "insufficient_universe": [False, False, False],
        }
    )
    df.to_csv(run_dir / "macro_timeseries_T120.csv", index=False)

    s = _q_stats(run_dir=run_dir, official_window=120)
    assert int(s["n_rows"]) == 3
    assert abs(float(s["q_min"]) - 0.96) < 1e-9
    assert abs(float(s["q_max"]) - 1.20) < 1e-9

