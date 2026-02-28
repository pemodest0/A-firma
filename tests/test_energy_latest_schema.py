from __future__ import annotations

import json

from scripts.energy.run_energy_br_daily_pipeline import validate_latest_artifacts


def test_validate_latest_artifacts_ok(tmp_path):
    latest = tmp_path / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    (latest / "hierarchical_state_latest_energy_br.json").write_text(
        json.dumps(
            {
                "date": "2026-02-28",
                "global_score": 0.1,
                "top_sectors_by_score": [],
                "top_sectors_by_loading": [],
                "top_sectors_by_overlap": [],
            }
        ),
        encoding="utf-8",
    )
    (latest / "rankings_latest_energy_br.json").write_text(
        json.dumps(
            {
                "date": "2026-02-28",
                "top_assets_global_mode": [],
                "top_sectors_global_mode": [],
                "sector_global_overlap": [],
                "global_state": {},
            }
        ),
        encoding="utf-8",
    )
    (latest / "historical_structure_summary_energy_br.json").write_text(
        json.dumps(
            {
                "schema_version": "historical_structure_summary_energy_br_v1",
                "status": "ok",
                "last_date": "2026-02-28",
                "evidence": {},
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    checks = validate_latest_artifacts(latest)
    assert checks["all_ok"] is True


def test_validate_latest_artifacts_missing_key(tmp_path):
    latest = tmp_path / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    (latest / "hierarchical_state_latest_energy_br.json").write_text(
        json.dumps({"date": "2026-02-28"}),
        encoding="utf-8",
    )
    (latest / "rankings_latest_energy_br.json").write_text(
        json.dumps({"date": "2026-02-28"}),
        encoding="utf-8",
    )
    (latest / "historical_structure_summary_energy_br.json").write_text(
        json.dumps({"status": "ok"}),
        encoding="utf-8",
    )

    checks = validate_latest_artifacts(latest)
    assert checks["all_ok"] is False
    assert checks["hierarchical_state_latest_energy_br.json"]["ok"] is False

