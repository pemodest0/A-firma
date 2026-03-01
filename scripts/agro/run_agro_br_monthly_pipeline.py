#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_CATALOG = ROOT / "config" / "event_catalog_agro_br.json"


def _latest_pack(results_dir: Path) -> Path:
    if not results_dir.exists():
        raise FileNotFoundError(f"missing results dir: {results_dir}")
    packs = sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("local_pack_")],
        key=lambda p: p.name,
        reverse=True,
    )
    for p in packs:
        if (p / "panel_long_sector.csv").exists() and (p / "universe_fixed.csv").exists() and (p / "asset_metadata_agro_br.csv").exists():
            return p
    raise FileNotFoundError("no local_pack_* with panel/universe/metadata found")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)  # noqa: S603


def _run_optional(cmd: list[str]) -> tuple[bool, str]:
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)  # noqa: S603
        return True, ""
    except subprocess.CalledProcessError as exc:
        return False, str(exc)


def _resolve_latest_run(out_base: Path) -> Path:
    pointer = out_base / "latest_release.json"
    if pointer.exists():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        run_dir = str(payload.get("run_dir", "")).strip()
        if run_dir and Path(run_dir).exists():
            return Path(run_dir)
        run_id = str(payload.get("run_id", "")).strip()
        if run_id and (out_base / run_id).exists():
            return out_base / run_id
    runs = sorted([p for p in out_base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for r in runs:
        if (r / "summary.json").exists():
            return r
    raise FileNotFoundError(f"unable to resolve latest run in {out_base}")


def _validate_json(path: Path, required_keys: list[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_file"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - runtime parsing edge
        return False, f"invalid_json:{exc}"
    if not isinstance(obj, dict):
        return False, "not_object"
    missing = [k for k in required_keys if k not in obj]
    if missing:
        return False, f"missing_keys:{','.join(missing)}"
    return True, "ok"


def _safe_float(value: Any, default: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(n):
        return float(default)
    return float(n)


def validate_latest_artifacts(latest_dir: Path) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    specs = {
        "hierarchical_state_latest_agro_br.json": ["date", "global_score", "top_sectors_by_score"],
        "rankings_latest_agro_br.json": ["date", "top_assets_global_mode", "top_sectors_global_mode", "global_state"],
        "historical_structure_summary_agro_br.json": ["schema_version", "status", "last_date", "evidence", "events"],
    }
    for name, required in specs.items():
        ok, reason = _validate_json(latest_dir / name, required)
        checks[name] = {"ok": bool(ok), "reason": str(reason)}
    checks["all_ok"] = bool(all(v.get("ok") for v in checks.values() if isinstance(v, dict)))
    return checks


def _write_rankings_fallback(latest_dir: Path) -> None:
    state_path = latest_dir / "hierarchical_state_latest.json"
    date = ""
    global_score = float("nan")
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            date = str(state.get("date", ""))
            global_score = float(state.get("global_score", float("nan")))
        except Exception:
            pass
    payload = {
        "date": date,
        "top_assets_global_mode": [],
        "top_sectors_global_mode": [],
        "sector_global_overlap": [],
        "global_state": {
            "score": global_score,
            "phi": float("nan"),
            "deff": float("nan"),
            "q": float("nan"),
            "n_used": float("nan"),
        },
        "status": "fallback_no_v1_vectors",
    }
    (latest_dir / "rankings_latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Agro Brasil monthly Eigen Engine pipeline and publish latest artifacts.")
    ap.add_argument("--pack-dir", type=str, default="")
    ap.add_argument("--pack-results-dir", type=str, default="results/agro_br")
    ap.add_argument("--out-base", type=str, default="results/agro_br/runs")
    ap.add_argument("--latest-dir", type=str, default="results/agro_br/latest")
    ap.add_argument("--event-catalog", type=str, default=str(DEFAULT_EVENT_CATALOG))
    ap.add_argument("--start", type=str, default="2010-01-01")
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--windows", type=str, default="24,36,60")
    ap.add_argument("--official-window", type=int, default=36)
    ap.add_argument("--coverage-core", type=float, default=0.85)
    ap.add_argument("--coverage-window", type=float, default=0.90)
    ap.add_argument("--min-assets", type=int, default=10)
    ap.add_argument("--n-global", type=int, default=120)
    ap.add_argument("--n-sector", type=int, default=40)
    ap.add_argument("--min-coverage-global", type=float, default=0.85)
    ap.add_argument("--min-coverage-sector", type=float, default=0.80)
    ap.add_argument("--structural-csd-window", type=int, default=12)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--score-z-threshold", type=float, default=1.0)
    ap.add_argument("--phi-z-threshold", type=float, default=1.0)
    ap.add_argument("--deff-z-threshold", type=float, default=-1.0)
    ap.add_argument("--auto-thresholds-from-tuning", type=int, default=1)
    ap.add_argument("--target-alert-rate", type=float, default=0.20)
    ap.add_argument("--max-alert-rate", type=float, default=0.35)
    ap.add_argument("--run-temporal-validation", type=int, default=1)
    ap.add_argument("--random-iters", type=int, default=300)
    ap.add_argument("--run-epistemic-diagnostics", type=int, default=1)
    ap.add_argument("--epistemic-random-iters", type=int, default=300)
    args = ap.parse_args()

    if str(args.pack_dir).strip():
        pack_dir = Path(args.pack_dir)
    else:
        base_pack = Path(args.pack_results_dir)
        if not base_pack.is_absolute():
            base_pack = ROOT / str(args.pack_results_dir)
        pack_dir = _latest_pack(base_pack)
    if not pack_dir.is_absolute():
        pack_dir = ROOT / str(pack_dir)
    if not pack_dir.exists():
        raise SystemExit(f"pack dir not found: {pack_dir}")

    panel_path = pack_dir / "panel_long_sector.csv"
    universe_path = pack_dir / "universe_fixed.csv"
    metadata_path = pack_dir / "asset_metadata_agro_br.csv"
    for p in [panel_path, universe_path, metadata_path]:
        if not p.exists():
            raise SystemExit(f"missing pack artifact: {p}")

    out_base = Path(args.out_base)
    if not out_base.is_absolute():
        out_base = ROOT / str(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    latest_dir = Path(args.latest_dir)
    if not latest_dir.is_absolute():
        latest_dir = ROOT / str(args.latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/lab/run_corr_macro_offline.py",
        "--apply-policy",
        "0",
        "--panel-path",
        str(panel_path),
        "--universe-path",
        str(universe_path),
        "--asset-metadata-path",
        str(metadata_path),
        "--out-base",
        str(out_base),
        "--start",
        str(args.start),
        "--windows",
        str(args.windows),
        "--official-window",
        str(int(args.official_window)),
        "--coverage-core",
        str(float(args.coverage_core)),
        "--coverage-window",
        str(float(args.coverage_window)),
        "--min-assets",
        str(int(args.min_assets)),
        "--business-days-only",
        "0",
        "--noise-step",
        "1",
        "--bootstrap-block",
        "6",
        "--overlap-step",
        "1",
        "--seed",
        str(int(args.seed)),
        "--enable-structural-v1",
        "1",
        "--structural-csd-window",
        str(int(args.structural_csd_window)),
        "--structural-train-end",
        "2022-12-31",
        "--enable-hierarchical",
        "1",
        "--n-global",
        str(int(args.n_global)),
        "--n-sector",
        str(int(args.n_sector)),
        "--min-coverage-global",
        str(float(args.min_coverage_global)),
        "--min-coverage-sector",
        str(float(args.min_coverage_sector)),
        "--enable-internal-sectors",
        "1",
        "--save-v1",
        "1",
        "--strict-checks",
        "0",
        "--freeze-baseline",
        "0",
        "--update-release-pointer",
        "1",
    ]
    if str(args.end).strip():
        cmd.extend(["--end", str(args.end).strip()])
    _run(cmd)

    run_dir = _resolve_latest_run(out_base=out_base)

    _run(
        [
            sys.executable,
            "scripts/ops/run_platform_hierarchical_state.py",
            "--run-dir",
            str(run_dir),
            "--outdir",
            str(latest_dir),
            "--latest-pointer",
            "results/agro_br/latest/latest_hierarchical_state_agro_br.json",
            "--top-k",
            "8",
        ]
    )
    ranking_ok, ranking_err = _run_optional(
        [sys.executable, "scripts/build_hierarchical_rankings.py", "--run-dir", str(run_dir), "--outdir", str(latest_dir)]
    )
    if not ranking_ok:
        _write_rankings_fallback(latest_dir=latest_dir)

    selected_score = float(args.score_z_threshold)
    selected_phi = float(args.phi_z_threshold)
    selected_deff = float(args.deff_z_threshold)
    tuning_meta: dict[str, Any] = {
        "auto_enabled": bool(int(args.auto_thresholds_from_tuning)),
        "recommendation_json": "",
        "used_recommendation": False,
        "target_alert_rate": float(args.target_alert_rate),
        "max_alert_rate": float(args.max_alert_rate),
    }
    if bool(int(args.auto_thresholds_from_tuning)):
        tuning_outdir = latest_dir / f"threshold_tuning_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        _run(
            [
                sys.executable,
                "scripts/agro/tune_agro_event_thresholds.py",
                "--run-dir",
                str(run_dir),
                "--event-catalog",
                str(args.event_catalog),
                "--target-alert-rate",
                str(float(args.target_alert_rate)),
                "--max-alert-rate",
                str(float(args.max_alert_rate)),
                "--outdir",
                str(tuning_outdir),
            ]
        )
        rec_path = tuning_outdir / "threshold_sweep_recommendation.json"
        tuning_meta["recommendation_json"] = str(rec_path)
        if rec_path.exists():
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            best = rec.get("best", {}) if isinstance(rec, dict) else {}
            selected_score = _safe_float(best.get("score_z_threshold"), selected_score)
            selected_phi = _safe_float(best.get("phi_z_threshold"), selected_phi)
            selected_deff = _safe_float(best.get("deff_z_threshold"), selected_deff)
            tuning_meta["used_recommendation"] = True

    event_catalog = Path(args.event_catalog)
    if not event_catalog.is_absolute():
        event_catalog = ROOT / str(args.event_catalog)

    _run(
        [
            sys.executable,
            "scripts/agro/build_agro_event_evidence.py",
            "--run-dir",
            str(run_dir),
            "--event-catalog",
            str(event_catalog),
            "--outdir",
            str(latest_dir),
            "--score-z-threshold",
            str(float(selected_score)),
            "--phi-z-threshold",
            str(float(selected_phi)),
            "--deff-z-threshold",
            str(float(selected_deff)),
        ]
    )

    temporal_summary_json = ""
    if bool(int(args.run_temporal_validation)):
        temporal_outdir = latest_dir / f"temporal_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_auto"
        _run(
            [
                sys.executable,
                "scripts/agro/validate_agro_temporal_blocks.py",
                "--run-dir",
                str(run_dir),
                "--event-catalog",
                str(event_catalog),
                "--score-z-threshold",
                str(float(selected_score)),
                "--phi-z-threshold",
                str(float(selected_phi)),
                "--deff-z-threshold",
                str(float(selected_deff)),
                "--random-iters",
                str(int(args.random_iters)),
                "--seed",
                str(int(args.seed)),
                "--outdir",
                str(temporal_outdir),
            ]
        )
        temporal_summary_json = str(temporal_outdir / "temporal_validation_summary.json")

    epistemic_summary_json = ""
    epistemic_summary_csv = ""
    epistemic_ok = False
    epistemic_err = ""
    if bool(int(args.run_epistemic_diagnostics)):
        ep_outdir = latest_dir / "epistemic_global"
        epistemic_ok, epistemic_err = _run_optional(
            [
                sys.executable,
                "scripts/structural/run_epistemic_diagnostics.py",
                "--run-dir",
                str(run_dir),
                "--universe",
                "global",
                "--horizons",
                "3,6",
                "--quantiles",
                "0.75,0.80,0.85,0.90,0.95",
                "--split-cutoffs",
                "2020-12-31,2022-12-31",
                "--random-iters",
                str(int(args.epistemic_random_iters)),
                "--outdir",
                str(ep_outdir),
            ]
        )
        if epistemic_ok:
            epistemic_summary_json = str(ep_outdir / "epistemic_diagnostics_summary.json")
            epistemic_summary_csv = str(ep_outdir / "epistemic_summary.csv")

    state_src = latest_dir / "hierarchical_state_latest.json"
    rank_src = latest_dir / "rankings_latest.json"
    state_dst = latest_dir / "hierarchical_state_latest_agro_br.json"
    rank_dst = latest_dir / "rankings_latest_agro_br.json"
    if state_src.exists():
        shutil.copy2(state_src, state_dst)
    if rank_src.exists():
        shutil.copy2(rank_src, rank_dst)

    schema_checks = validate_latest_artifacts(latest_dir=latest_dir)
    (latest_dir / "latest_schema_checks_agro_br.json").write_text(
        json.dumps(schema_checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    release = {
        "status": "ok" if bool(schema_checks.get("all_ok")) else "partial",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "pack_dir": str(pack_dir),
        "latest_dir": str(latest_dir),
        "artifacts": {
            "state": str(state_dst),
            "rankings": str(rank_dst),
            "evidence": str(latest_dir / "historical_structure_summary_agro_br.json"),
            "schema_checks": str(latest_dir / "latest_schema_checks_agro_br.json"),
        },
        "schema_checks": schema_checks,
        "ranking_builder": {
            "ok": bool(ranking_ok),
            "error": str(ranking_err),
            "mode": "full" if ranking_ok else "fallback_no_v1_vectors",
        },
        "calibration": {
            "selected_thresholds": {
                "score_z_threshold": float(selected_score),
                "phi_z_threshold": float(selected_phi),
                "deff_z_threshold": float(selected_deff),
            },
            "tuning": tuning_meta,
        },
        "validation": {
            "temporal_validation_summary_json": str(temporal_summary_json),
            "random_baseline_iters": int(args.random_iters),
            "epistemic": {
                "ok": bool(epistemic_ok),
                "error": str(epistemic_err),
                "summary_json": str(epistemic_summary_json),
                "summary_csv": str(epistemic_summary_csv),
                "random_iters": int(args.epistemic_random_iters),
            },
        },
    }
    release_path = ROOT / "results" / "agro_br" / "latest_release_agro_br.json"
    release_path.parent.mkdir(parents=True, exist_ok=True)
    release_path.write_text(json.dumps(release, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": release["status"], "release_json": str(release_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
