from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from execution.shadow_health import build_shadow_health_summary


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_marks_stale_legacy_shadow_as_archive_or_retire(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    summary_path = repo / "results/ops/profit_shadow/latest_summary.json"
    _write(
        summary_path,
        {
            "shadow_name": "global_plus_profit_shadow",
            "status": "ok",
            "run_id": "20260306T061125Z",
            "generated_at_utc": "2026-03-06T06:33:30+00:00",
            "best_by_profit": {
                "profile": "profit_attack",
                "daily_edge_vs_benchmark": 0.4,
                "systematic_monthly_alpha_prob_positive_vs_eqw": 0.53,
            },
        },
    )
    _write(repo / "results/ops/profit_shadow_target_800_attack/latest_summary.json", {"status": "ok", "generated_at_utc": "2026-03-06T18:16:58+00:00", "best_by_profit": {"daily_edge_vs_benchmark": -0.2, "systematic_monthly_alpha_prob_positive_vs_eqw": 0.48}})
    _write(repo / "results/ops/profit_shadow_target_800_attack_ensemble/latest_summary.json", {"status": "ok", "generated_at_utc": "2026-03-06T18:17:15+00:00", "best_by_profit": {"daily_edge_vs_benchmark": -0.06, "systematic_monthly_alpha_prob_positive_vs_eqw": 0.49}})
    _write(repo / "results/ops/invest_shadow/latest_summary.json", {"status": "ok", "generated_at_utc": "2026-03-06T04:32:28+00:00", "best_by_profit": {}})
    _write(
        repo / "config/live_shadow_watchlist.json",
        {
            "policy": {"retire_legacy_shadows_when_stale_days_gte": 3},
            "active_watchlist": [{"candidate_id": "criticality_free_energy_attack", "alias": "Apollo", "role": "structural_anchor", "status": "shadow"}],
        },
    )
    summary = build_shadow_health_summary(
        repo,
        watchlist_path=repo / "config/live_shadow_watchlist.json",
        launch_agents_dir=repo / "launchd",
        now_utc=datetime(2026, 3, 15, tzinfo=UTC),
    )
    profit_shadow = next(item for item in summary["legacy_shadows"] if item["shadow_key"] == "profit_shadow")
    attack_shadow = next(item for item in summary["legacy_shadows"] if item["shadow_key"] == "profit_shadow_target_800_attack")
    assert profit_shadow["recommendation"] == "archive_only"
    assert attack_shadow["recommendation"] == "retire"
    assert summary["summary"]["all_legacy_stale"] is True


def test_returns_recommended_active_watchlist(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    for rel in [
        "results/ops/profit_shadow/latest_summary.json",
        "results/ops/profit_shadow_target_800_attack/latest_summary.json",
        "results/ops/profit_shadow_target_800_attack_ensemble/latest_summary.json",
        "results/ops/invest_shadow/latest_summary.json",
    ]:
        _write(repo / rel, {"status": "ok", "generated_at_utc": "2026-03-14T00:00:00+00:00", "best_by_profit": {"daily_edge_vs_benchmark": 0.1, "systematic_monthly_alpha_prob_positive_vs_eqw": 0.51}})
    _write(
        repo / "config/live_shadow_watchlist.json",
        {
            "policy": {"retire_legacy_shadows_when_stale_days_gte": 3},
            "active_watchlist": [
                {"candidate_id": "criticality_free_energy_attack", "alias": "Apollo", "role": "structural_anchor", "status": "shadow"},
                {"candidate_id": "meta_mode_selector", "alias": "Zeus", "role": "context_selector", "status": "disabled"},
            ],
        },
    )
    summary = build_shadow_health_summary(
        repo,
        watchlist_path=repo / "config/live_shadow_watchlist.json",
        launch_agents_dir=repo / "launchd",
        now_utc=datetime(2026, 3, 15, tzinfo=UTC),
    )
    assert len(summary["recommended_active_watchlist"]) == 1
    assert summary["recommended_active_watchlist"][0]["alias"] == "Apollo"
