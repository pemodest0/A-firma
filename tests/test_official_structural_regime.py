from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.official_structural_regime import load_official_structural_regime_series


def test_load_official_structural_regime_series_prefers_release_pointer(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "lab_corr_macro" / "20260301T000000Z"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "regime_series_T120.csv").write_text(
        "date,regime\n2026-03-10,stable\n2026-03-11,stress\n",
        encoding="utf-8",
    )
    latest_release = tmp_path / "results" / "lab_corr_macro" / "latest_release.json"
    latest_release.parent.mkdir(parents=True, exist_ok=True)
    latest_release.write_text(
        json.dumps({"run_dir": str(run_dir)}),
        encoding="utf-8",
    )

    other_run = tmp_path / "results" / "lab_corr_macro" / "20260302T000000Z"
    other_run.mkdir(parents=True, exist_ok=True)
    (other_run / "regime_series_T120.csv").write_text(
        "date,regime\n2026-03-10,transition\n",
        encoding="utf-8",
    )

    series, meta = load_official_structural_regime_series(tmp_path, official_window=120)

    assert str(meta.get("source")) == "results_lab_corr_latest_release"
    assert str(meta.get("run_dir")) == str(run_dir)
    assert str(series.loc["2026-03-10"]) == "stable"


def test_load_official_structural_regime_series_extends_with_live_nowcast(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "lab_corr_macro" / "20260301T000000Z"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "regime_series_T120.csv").write_text(
        "date,regime\n2026-03-10,stable\n2026-03-11,transition\n",
        encoding="utf-8",
    )
    latest_release = tmp_path / "results" / "lab_corr_macro" / "latest_release.json"
    latest_release.parent.mkdir(parents=True, exist_ok=True)
    latest_release.write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")

    live_regime = tmp_path / "results" / "ops" / "official_structural_regime" / "latest_structural_regime.json"
    live_regime.parent.mkdir(parents=True, exist_ok=True)
    live_regime.write_text(
        json.dumps(
            {
                "as_of_date": "2026-03-14",
                "regime": "stress",
            }
        ),
        encoding="utf-8",
    )

    series, meta = load_official_structural_regime_series(tmp_path, official_window=120)

    assert str(meta.get("live_extension_end_date")) == "2026-03-14"
    assert str(meta.get("live_extension_regime")) == "stress"
    assert str(series.loc["2026-03-12"]) == "stress"
    assert str(series.loc["2026-03-14"]) == "stress"
