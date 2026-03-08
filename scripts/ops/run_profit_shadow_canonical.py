#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_path(raw: str | Path | None) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _candidate_run_id(base_run_id: str, candidate_name: str) -> str:
    return f"{base_run_id}_{str(candidate_name).strip()}"


def _run(cmd: list[str], *, timeout_sec: float) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_sec)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _load_candidate_node(lock: dict[str, Any], key: str) -> dict[str, Any]:
    node = lock.get(key, {})
    return node if isinstance(node, dict) else {}


def _load_config(config_path: Path) -> dict[str, Any]:
    cfg = _read_json(config_path)
    if not cfg:
        raise SystemExit(f"invalid config: {config_path}")
    return cfg


def _run_candidate(
    *,
    candidate_key: str,
    candidate_node: dict[str, Any],
    base_run_id: str,
    resume: int,
    step_timeout_sec: float,
    reuse_latest: bool,
    prices_dir: Path,
) -> dict[str, Any]:
    config_path = _resolve_path(candidate_node.get("config_path"))
    if config_path is None or not config_path.exists():
        raise SystemExit(f"missing config path for {candidate_key}")
    cfg = _load_config(config_path)
    shadow_outdir = _resolve_path(cfg.get("shadow_outdir"))
    if shadow_outdir is None:
        raise SystemExit(f"missing shadow_outdir in {config_path}")
    run_id = _candidate_run_id(base_run_id, str(candidate_node.get("run_id", candidate_key)))

    if not reuse_latest:
        cmd = [
            PY,
            "scripts/ops/run_profit_shadow_suite.py",
            "--config-path",
            str(config_path),
            "--run-id",
            run_id,
            "--resume",
            str(int(resume)),
            "--step-timeout-sec",
            str(float(step_timeout_sec)),
        ]
        code, out, err = _run(cmd, timeout_sec=float(step_timeout_sec))
        if code != 0:
            raise SystemExit(f"{candidate_key} run failed: {err or out}")

    summary_path = shadow_outdir / "runs" / run_id / "summary.json"
    if not summary_path.exists():
        latest_run = _read_json(shadow_outdir / "latest_run.json")
        summary_path = _resolve_path(latest_run.get("summary_path")) or summary_path
    summary = _read_json(summary_path)
    if not summary:
        raise SystemExit(f"missing summary for {candidate_key}: {summary_path}")
    profiles = summary.get("profiles", [])
    first_profile = profiles[0] if isinstance(profiles, list) and profiles else {}
    profile_dir = _resolve_path(first_profile.get("run_dir"))
    if profile_dir is None or not profile_dir.exists():
        raise SystemExit(f"missing profile dir for {candidate_key}")

    bench_cmd = [
        PY,
        "scripts/ops/build_profit_shadow_benchmarks.py",
        "--profile-dir",
        str(profile_dir),
        "--prices-dir",
        str(prices_dir),
    ]
    code, out, err = _run(bench_cmd, timeout_sec=min(float(step_timeout_sec), 1800.0))
    if code != 0:
        raise SystemExit(f"benchmark build failed for {candidate_key}: {err or out}")
    bench_report = _read_json(profile_dir / "hard_benchmarks_report.json")
    profiles = summary.get("profiles", [])
    first_profile = profiles[0] if isinstance(profiles, list) and profiles else {}
    bench_metrics = ((bench_report.get("metrics") or {}) if isinstance(bench_report.get("metrics"), dict) else {})
    return {
        "candidate_key": candidate_key,
        "run_id": run_id,
        "config_path": str(config_path),
        "shadow_outdir": str(shadow_outdir),
        "summary_path": str(summary_path),
        "profile_dir": str(profile_dir),
        "benchmark_report_json": str(profile_dir / "hard_benchmarks_report.json"),
        "profile_name": str(first_profile.get("profile", "")),
        "daily_total_return": first_profile.get("daily_total_return"),
        "daily_ann_return": first_profile.get("daily_ann_return"),
        "daily_sharpe": first_profile.get("daily_sharpe"),
        "daily_max_drawdown": first_profile.get("daily_max_drawdown"),
        "latest_signal": first_profile.get("latest_signal", {}),
        "benchmarks": {
            "winner_by_ann_return": str(bench_report.get("winner_by_ann_return", "")),
            "spy_ann_return": ((bench_metrics.get("spy_buy_hold") or {}) if isinstance(bench_metrics.get("spy_buy_hold"), dict) else {}).get("ann_return"),
            "sixty_forty_ann_return": ((bench_metrics.get("sixty_forty") or {}) if isinstance(bench_metrics.get("sixty_forty"), dict) else {}).get("ann_return"),
            "sector_equal_weight_ann_return": (
                (bench_metrics.get("sector_equal_weight") or {}) if isinstance(bench_metrics.get("sector_equal_weight"), dict) else {}
            ).get("ann_return"),
            "momentum_global_top3_ann_return": (
                (bench_metrics.get("momentum_global_top3") or {}) if isinstance(bench_metrics.get("momentum_global_top3"), dict) else {}
            ).get("ann_return"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the canonical profit shadow candidates and build aggregate artifacts.")
    ap.add_argument("--lock-path", required=True)
    ap.add_argument("--run-state-file", default="")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--step-timeout-sec", type=float, default=14400.0)
    ap.add_argument("--prices-dir", default=str(ROOT / "data" / "raw" / "finance" / "yfinance_daily"))
    ap.add_argument("--reuse-latest", type=int, default=0)
    args = ap.parse_args()

    lock_path = Path(args.lock_path).resolve()
    lock = _read_json(lock_path)
    if not lock:
        raise SystemExit(f"missing canonical lock: {lock_path}")

    lock_root = lock_path.parent
    run_state_file = _resolve_path(args.run_state_file) or (lock_root / "canonical_current_run_id.txt")
    if run_state_file.exists():
        base_run_id = run_state_file.read_text(encoding="utf-8").strip()
    else:
        base_run_id = ""
    if not base_run_id:
        base_run_id = _run_id()
        run_state_file.parent.mkdir(parents=True, exist_ok=True)
        run_state_file.write_text(base_run_id + "\n", encoding="utf-8")

    run_dir = lock_root / "canonical_runs" / base_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prices_dir = _resolve_path(args.prices_dir)
    if prices_dir is None or not prices_dir.exists():
        raise SystemExit(f"missing prices dir: {args.prices_dir}")

    candidates: list[dict[str, Any]] = []
    for key in ["main", "challenger"]:
        candidates.append(
            _run_candidate(
                candidate_key=key,
                candidate_node=_load_candidate_node(lock, key),
                base_run_id=base_run_id,
                resume=int(args.resume),
                step_timeout_sec=float(args.step_timeout_sec),
                reuse_latest=bool(int(args.reuse_latest)),
                prices_dir=prices_dir,
            )
        )

    scorecard_cmd = [PY, "scripts/ops/build_profit_shadow_scorecard.py", "--lock-path", str(lock_path)]
    code, out, err = _run(scorecard_cmd, timeout_sec=600.0)
    if code != 0:
        raise SystemExit(f"scorecard build failed: {err or out}")
    scorecard = _read_json(lock_root / "latest_scorecard.json")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": str(lock_path),
        "base_run_id": base_run_id,
        "candidates": candidates,
        "scorecard_path": str(lock_root / "latest_scorecard.json"),
        "scorecard": scorecard,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(lock_root / "canonical_latest_run.json", summary)
    run_state_file.write_text(_run_id() + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
