#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def apply_profile_defaults(
    args: argparse.Namespace,
    *,
    defaults: dict[str, object],
) -> dict[str, object]:
    profile_meta: dict[str, object] = {
        "profile_applied": False,
        "profile_file": "",
        "profile_version": "",
    }
    if bool(getattr(args, "ignore_profile", False)):
        return profile_meta
    profile_path_raw = str(getattr(args, "profile_file", "")).strip()
    if not profile_path_raw:
        return profile_meta
    profile_path = ROOT / profile_path_raw
    if not profile_path.exists():
        return profile_meta
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return profile_meta

    params = payload.get("params", payload) if isinstance(payload, dict) else {}
    if not isinstance(params, dict):
        params = {}
    for k, default in defaults.items():
        if k not in params or not hasattr(args, k):
            continue
        cur = getattr(args, k)
        if cur != default:
            continue
        raw = params[k]
        try:
            if isinstance(default, bool):
                val = bool(raw)
            elif isinstance(default, int):
                val = int(raw)
            elif isinstance(default, float):
                val = float(raw)
            else:
                val = str(raw)
            setattr(args, k, val)
        except (TypeError, ValueError):
            continue

    profile_meta = {
        "profile_applied": True,
        "profile_file": str(profile_path),
        "profile_version": str(payload.get("profile_version", "")) if isinstance(payload, dict) else "",
    }
    return profile_meta


def run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "").strip()
    if out:
        print(out)
    err = (proc.stderr or "").strip()
    if err:
        print(err, file=sys.stderr)
    return out


def run_retry(
    cmd: list[str],
    *,
    label: str,
    retries: int,
    retry_delay_sec: float,
    step_log: list[dict[str, object]],
) -> str:
    last_exc: subprocess.CalledProcessError | OSError | None = None
    attempts = max(1, int(retries) + 1)
    for i in range(1, attempts + 1):
        t0 = time.time()
        try:
            out = run(cmd)
            step_log.append(
                {
                    "step": label,
                    "status": "ok",
                    "attempt": i,
                    "duration_sec": round(time.time() - t0, 3),
                }
            )
            return out
        except (subprocess.CalledProcessError, OSError) as exc:
            last_exc = exc
            step_log.append(
                {
                    "step": label,
                    "status": "error",
                    "attempt": i,
                    "duration_sec": round(time.time() - t0, 3),
                    "error": str(exc),
                }
            )
            if i < attempts:
                time.sleep(max(0.0, float(retry_delay_sec)))
    assert last_exc is not None
    raise last_exc


def read_levels_csv(path: Path) -> list[dict[str, object]]:
    def parse_float(x: str | None) -> float | None:
        if x is None:
            return None
        try:
            n = float(x)
        except (TypeError, ValueError):
            return None
        return n if math.isfinite(n) else None

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            level = str(row.get("alert_level", "verde")).strip().lower()
            rows.append(
                {
                    "sector": str(row.get("sector", "unknown")).strip(),
                    "date": str(row.get("date", "")).strip(),
                    "n_assets": int(float(row.get("n_assets", 0) or 0)),
                    "alert_level": level,
                    "sector_score": parse_float(row.get("sector_score")),
                    "share_unstable": parse_float(row.get("share_unstable")),
                    "share_transition": parse_float(row.get("share_transition")),
                    "mean_confidence": parse_float(row.get("mean_confidence")),
                }
            )
    return rows


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY,
          generated_at_utc TEXT NOT NULL,
          outdir TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sector_snapshots (
          run_id TEXT NOT NULL,
          generated_at_utc TEXT NOT NULL,
          sector TEXT NOT NULL,
          asof_date TEXT,
          alert_level TEXT,
          sector_score REAL,
          share_unstable REAL,
          share_transition REAL,
          mean_confidence REAL,
          n_assets INTEGER,
          PRIMARY KEY (run_id, sector)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sector_snapshots_sector_time ON sector_snapshots(sector, generated_at_utc)"
    )
    conn.commit()
    return conn


def upsert_run_and_snapshots(
    conn: sqlite3.Connection,
    run_id: str,
    generated_at_utc: str,
    outdir: str,
    levels: list[dict[str, object]],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO runs(run_id, generated_at_utc, outdir) VALUES(?,?,?)",
        (run_id, generated_at_utc, outdir),
    )
    for row in levels:
        conn.execute(
            """
            INSERT OR REPLACE INTO sector_snapshots(
              run_id, generated_at_utc, sector, asof_date, alert_level, sector_score,
              share_unstable, share_transition, mean_confidence, n_assets
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                generated_at_utc,
                row.get("sector"),
                row.get("date"),
                row.get("alert_level"),
                row.get("sector_score"),
                row.get("share_unstable"),
                row.get("share_transition"),
                row.get("mean_confidence"),
                row.get("n_assets"),
            ),
        )
    conn.commit()


def fetch_levels_by_run(conn: sqlite3.Connection, run_id: str) -> dict[str, str]:
    q = "SELECT sector, alert_level FROM sector_snapshots WHERE run_id=?"
    out: dict[str, str] = {}
    for sec, lvl in conn.execute(q, (run_id,)).fetchall():
        out[str(sec)] = str(lvl or "verde").lower()
    return out


def previous_run_id(conn: sqlite3.Connection, current_run_id: str) -> str | None:
    q = """
    SELECT run_id
    FROM runs
    WHERE run_id <> ?
    ORDER BY generated_at_utc DESC
    LIMIT 1
    """
    row = conn.execute(q, (current_run_id,)).fetchone()
    return str(row[0]) if row else None


def weekly_reference_run_id(conn: sqlite3.Connection, current_generated_at: str) -> str | None:
    try:
        current_dt = datetime.fromisoformat(current_generated_at.replace("Z", "+00:00"))
        ref_dt = current_dt.timestamp() - 7 * 24 * 3600
        ref_iso = datetime.fromtimestamp(ref_dt, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        ref_iso = current_generated_at
    q = """
    SELECT run_id
    FROM runs
    WHERE generated_at_utc <= ?
    ORDER BY generated_at_utc DESC
    LIMIT 1
    """
    row = conn.execute(q, (ref_iso,)).fetchone()
    if row:
        return str(row[0])
    q2 = "SELECT run_id FROM runs ORDER BY generated_at_utc DESC LIMIT 2"
    rows = conn.execute(q2).fetchall()
    if len(rows) >= 2:
        return str(rows[1][0])
    return None


def compute_weekly_compare(
    conn: sqlite3.Connection,
    current_run_id: str,
    ref_run_id: str | None,
) -> dict[str, object]:
    now_q = """
    SELECT sector, alert_level, sector_score, n_assets
    FROM sector_snapshots
    WHERE run_id=?
    """
    now_rows = conn.execute(now_q, (current_run_id,)).fetchall()
    now_map = {str(r[0]): r for r in now_rows}
    ref_map: dict[str, tuple] = {}
    if ref_run_id:
        ref_q = """
        SELECT sector, alert_level, sector_score, n_assets
        FROM sector_snapshots
        WHERE run_id=?
        """
        ref_map = {str(r[0]): r for r in conn.execute(ref_q, (ref_run_id,)).fetchall()}

    severity = {"verde": 0, "amarelo": 1, "vermelho": 2}
    rows: list[dict[str, object]] = []
    up = 0
    down = 0
    same = 0
    for sector, row in sorted(now_map.items()):
        now_lvl = str(row[1] or "verde").lower()
        now_score = float(row[2]) if row[2] is not None and math.isfinite(float(row[2])) else None
        n_assets = int(row[3] or 0)
        ref = ref_map.get(sector)
        prev_lvl = str(ref[1]).lower() if ref else None
        prev_score = float(ref[2]) if ref and ref[2] is not None and math.isfinite(float(ref[2])) else None
        delta_score = (float(now_score) - float(prev_score)) if (now_score is not None and prev_score is not None) else None
        trend = "same"
        if prev_lvl is not None:
            d = severity.get(now_lvl, 0) - severity.get(prev_lvl, 0)
            if d > 0:
                trend = "piorou"
                up += 1
            elif d < 0:
                trend = "melhorou"
                down += 1
            else:
                same += 1
        rows.append(
            {
                "sector": sector,
                "n_assets": n_assets,
                "level_now": now_lvl,
                "level_prev_week": prev_lvl,
                "score_now": float(now_score) if now_score is not None else None,
                "score_prev_week": prev_score,
                "delta_score_week": delta_score,
                "trend": trend,
                "changed": bool(prev_lvl is not None and prev_lvl != now_lvl),
            }
        )
    return {
        "reference_run_id": ref_run_id,
        "summary": {
            "sectors_total": len(rows),
            "changed_up": up,
            "changed_down": down,
            "unchanged": same,
        },
        "rows": rows,
    }


def notify_if_needed(
    out_root: Path,
    run_id: str,
    current_levels: dict[str, str],
    prev_levels: dict[str, str],
    *,
    webhook_url: str,
    force_send: bool = False,
) -> dict[str, object]:
    exited_green = sorted(
        [
            sec
            for sec, now_lvl in current_levels.items()
            if prev_levels.get(sec, "verde") == "verde" and now_lvl in {"amarelo", "vermelho"}
        ]
    )
    payload = {
        "status": "ok",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exited_green": exited_green,
        "n_exited_green": len(exited_green),
    }
    alerts_dir = out_root / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    (alerts_dir / f"alert_{run_id}.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (alerts_dir / "latest_alert.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    webhook = webhook_url.strip()
    sent = False
    should_send = bool(webhook) and (bool(exited_green) or bool(force_send))
    if should_send:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        req = urllib.request.Request(webhook, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10):
                sent = True
        except (urllib.error.URLError, TimeoutError, OSError):
            sent = False
    payload["webhook_sent"] = sent
    return payload


def main() -> None:
    defaults: dict[str, object] = {
        "lookbacks": "1,5,10,20",
        "n_random": 300,
        "random_baseline_method": "both",
        "random_block_size": 10,
        "min_sector_assets": 10,
        "min_cal_days": 252,
        "min_test_days": 252,
        "q_unstable": 0.80,
        "q_transition": 0.80,
        "q_confidence": 0.50,
        "q_confidence_guarded": 0.60,
        "q_score_balanced": 0.70,
        "q_score_guarded": 0.80,
        "confirm_n": 2,
        "confirm_m": 3,
        "min_alert_gap_days": 2,
        "two_layer_mode": "on",
        "auto_candidates": "regime_entry_confirm,regime_balanced,regime_guarded",
        "alert_policy": "regime_entry_confirm",
        "drift_baseline_days": 30,
        "drift_min_history_runs": 7,
        "drift_warn_zscore": 2.0,
        "drift_block_zscore": 3.0,
    }
    ap = argparse.ArgumentParser(description="Run daily sector alert package for website/API.")
    ap.add_argument("--lookbacks", type=str, default=str(defaults["lookbacks"]))
    ap.add_argument("--n-random", type=int, default=int(defaults["n_random"]))
    ap.add_argument(
        "--random-baseline-method",
        type=str,
        default=str(defaults["random_baseline_method"]),
        choices=["iid", "block", "both"],
    )
    ap.add_argument("--random-block-size", type=int, default=int(defaults["random_block_size"]))
    ap.add_argument("--min-sector-assets", type=int, default=int(defaults["min_sector_assets"]))
    ap.add_argument("--min-cal-days", type=int, default=int(defaults["min_cal_days"]))
    ap.add_argument("--min-test-days", type=int, default=int(defaults["min_test_days"]))
    ap.add_argument("--q-unstable", type=float, default=float(defaults["q_unstable"]))
    ap.add_argument("--q-transition", type=float, default=float(defaults["q_transition"]))
    ap.add_argument("--q-confidence", type=float, default=float(defaults["q_confidence"]))
    ap.add_argument("--q-confidence-guarded", type=float, default=float(defaults["q_confidence_guarded"]))
    ap.add_argument("--q-score-balanced", type=float, default=float(defaults["q_score_balanced"]))
    ap.add_argument("--q-score-guarded", type=float, default=float(defaults["q_score_guarded"]))
    ap.add_argument("--confirm-n", type=int, default=int(defaults["confirm_n"]))
    ap.add_argument("--confirm-m", type=int, default=int(defaults["confirm_m"]))
    ap.add_argument("--min-alert-gap-days", type=int, default=int(defaults["min_alert_gap_days"]))
    ap.add_argument("--two-layer-mode", type=str, default=str(defaults["two_layer_mode"]), choices=["on", "off"])
    ap.add_argument("--auto-candidates", type=str, default=str(defaults["auto_candidates"]))
    ap.add_argument("--drift-baseline-days", type=int, default=int(defaults["drift_baseline_days"]))
    ap.add_argument("--drift-min-history-runs", type=int, default=int(defaults["drift_min_history_runs"]))
    ap.add_argument("--drift-warn-zscore", type=float, default=float(defaults["drift_warn_zscore"]))
    ap.add_argument("--drift-block-zscore", type=float, default=float(defaults["drift_block_zscore"]))
    ap.add_argument(
        "--alert-policy",
        type=str,
        default=str(defaults["alert_policy"]),
        choices=[
            "regime_entry",
            "regime_entry_confirm",
            "regime_balanced",
            "regime_guarded",
            "regime_auto",
            "score_q80",
            "score_q90",
        ],
    )
    ap.add_argument("--out-root", type=str, default="results/event_study_sectors")
    ap.add_argument(
        "--diagnostics-csv",
        type=str,
        default="results/latest_graph_universe470_batch/diagnostics_assets_daily.csv",
    )
    ap.add_argument("--sector-pack-outdir", type=str, default="results/latest_graph_universe470_batch/sector_pack")
    ap.add_argument("--db-path", type=str, default="results/event_study_sectors/sector_alerts.db")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--retry-delay-sec", type=float, default=2.0)
    ap.add_argument("--profile-file", type=str, default="config/sector_alerts_profile.json")
    ap.add_argument("--ignore-profile", action="store_true")
    ap.add_argument("--webhook-url", type=str, default=os.environ.get("SECTOR_ALERT_WEBHOOK_URL", ""))
    ap.add_argument("--test-webhook", action="store_true", help="Force webhook send even without exited_green sectors.")
    args = ap.parse_args()
    profile_meta = apply_profile_defaults(args=args, defaults=defaults)
    t_start = time.time()
    step_log: list[dict[str, object]] = []

    cmd_validate = [
        sys.executable,
        "scripts/bench/event_study_validate_sectors.py",
        "--lookbacks",
        str(args.lookbacks),
        "--n-random",
        str(int(args.n_random)),
        "--random-baseline-method",
        str(args.random_baseline_method),
        "--random-block-size",
        str(int(args.random_block_size)),
        "--q-unstable",
        str(float(args.q_unstable)),
        "--q-transition",
        str(float(args.q_transition)),
        "--q-confidence",
        str(float(args.q_confidence)),
        "--q-confidence-guarded",
        str(float(args.q_confidence_guarded)),
        "--q-score-balanced",
        str(float(args.q_score_balanced)),
        "--q-score-guarded",
        str(float(args.q_score_guarded)),
        "--confirm-n",
        str(int(args.confirm_n)),
        "--confirm-m",
        str(int(args.confirm_m)),
        "--min-alert-gap-days",
        str(int(args.min_alert_gap_days)),
        "--auto-candidates",
        str(args.auto_candidates),
        "--min-sector-assets",
        str(int(args.min_sector_assets)),
        "--min-cal-days",
        str(int(args.min_cal_days)),
        "--min-test-days",
        str(int(args.min_test_days)),
        "--alert-policy",
        str(args.alert_policy),
        "--two-layer-mode",
        str(args.two_layer_mode),
        "--out-root",
        str(args.out_root),
    ]
    out_validate = run_retry(
        cmd_validate,
        label="event_study_validate_sectors",
        retries=int(args.retries),
        retry_delay_sec=float(args.retry_delay_sec),
        step_log=step_log,
    )
    validate_json = json.loads(out_validate.splitlines()[-1])
    outdir = Path(validate_json["outdir"])
    run_id = outdir.name
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    levels = read_levels_csv(outdir / "sector_alert_levels_latest.csv")

    cmd_pack = [
        sys.executable,
        "scripts/bench/organize_diagnostics_by_sector.py",
        "--diagnostics-csv",
        str(args.diagnostics_csv),
        "--outdir",
        str(args.sector_pack_outdir),
    ]
    out_pack = run_retry(
        cmd_pack,
        label="organize_diagnostics_by_sector",
        retries=int(args.retries),
        retry_delay_sec=float(args.retry_delay_sec),
        step_log=step_log,
    )
    pack_payload = None
    try:
        pack_payload = ast.literal_eval(out_pack.splitlines()[-1])
    except (ValueError, SyntaxError, IndexError):
        pack_payload = {"raw": out_pack}

    db_path = ROOT / str(args.db_path)
    out_root = ROOT / str(args.out_root)
    conn = init_db(db_path)
    upsert_run_and_snapshots(
        conn=conn,
        run_id=run_id,
        generated_at_utc=generated_at_utc,
        outdir=str(outdir),
        levels=levels,
    )
    prev_id = previous_run_id(conn, current_run_id=run_id)
    current_map = {str(x["sector"]): str(x["alert_level"]).lower() for x in levels}
    prev_map = fetch_levels_by_run(conn, prev_id) if prev_id else {}
    notification_payload = notify_if_needed(
        out_root=out_root,
        run_id=run_id,
        current_levels=current_map,
        prev_levels=prev_map,
        webhook_url=str(args.webhook_url),
        force_send=bool(args.test_webhook),
    )

    ref_week_id = weekly_reference_run_id(conn, current_generated_at=generated_at_utc)
    weekly_compare = compute_weekly_compare(conn, current_run_id=run_id, ref_run_id=ref_week_id)
    weekly_compare_path = outdir / "weekly_compare.json"
    weekly_compare_path.write_text(json.dumps(weekly_compare, indent=2, allow_nan=False), encoding="utf-8")
    conn.close()

    drift_monitor_path = outdir / "drift_monitor.json"
    drift_payload: dict[str, object] = {
        "status": "error",
        "message": "not_run",
        "out_json": str(drift_monitor_path),
    }
    cmd_drift = [
        sys.executable,
        "scripts/ops/monitor_sector_alerts_drift.py",
        "--db-path",
        str(args.db_path),
        "--current-run-id",
        str(run_id),
        "--baseline-days",
        str(int(args.drift_baseline_days)),
        "--min-history-runs",
        str(int(args.drift_min_history_runs)),
        "--warn-zscore",
        str(float(args.drift_warn_zscore)),
        "--block-zscore",
        str(float(args.drift_block_zscore)),
        "--profile-file",
        str(args.profile_file),
        "--out-root",
        str(Path(args.out_root) / "drift"),
        "--out-json",
        str(drift_monitor_path),
    ]
    if bool(args.ignore_profile):
        cmd_drift.append("--ignore-profile")
    try:
        out_drift = run_retry(
            cmd_drift,
            label="monitor_sector_alerts_drift",
            retries=int(args.retries),
            retry_delay_sec=float(args.retry_delay_sec),
            step_log=step_log,
        )
        drift_payload = json.loads(out_drift.splitlines()[-1])
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError, IndexError) as exc:
        step_log.append(
            {
                "step": "monitor_sector_alerts_drift",
                "status": "error",
                "attempt": 1,
                "duration_sec": 0.0,
                "error": str(exc),
            }
        )
        drift_payload = {
            "status": "error",
            "message": str(exc),
            "out_json": str(drift_monitor_path),
        }

    structural_report_path = outdir / "sector_structural_report.json"
    structural_payload: dict[str, object] = {
        "status": "error",
        "message": "not_run",
        "out_json": str(structural_report_path),
    }
    cmd_structural = [
        sys.executable,
        "scripts/ops/build_sector_structural_report.py",
        "--run-id",
        str(run_id),
        "--levels-csv",
        str(outdir / "sector_alert_levels_latest.csv"),
        "--weekly-compare-json",
        str(weekly_compare_path),
        "--drift-json",
        str(drift_monitor_path),
        "--out-json",
        str(structural_report_path),
    ]
    try:
        out_struct = run_retry(
            cmd_structural,
            label="build_sector_structural_report",
            retries=int(args.retries),
            retry_delay_sec=float(args.retry_delay_sec),
            step_log=step_log,
        )
        _ = json.loads(out_struct.splitlines()[-1])
        structural_payload = json.loads(structural_report_path.read_text(encoding="utf-8"))
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError, IndexError) as exc:
        step_log.append(
            {
                "step": "build_sector_structural_report",
                "status": "error",
                "attempt": 1,
                "duration_sec": 0.0,
                "error": str(exc),
            }
        )
        structural_payload = {
            "status": "error",
            "message": str(exc),
            "out_json": str(structural_report_path),
        }

    required_files = [
        outdir / "sector_alert_levels_latest.csv",
        outdir / "sector_rank_l5.csv",
        outdir / "weekly_compare.json",
        outdir / "drift_monitor.json",
        outdir / "sector_structural_report.json",
        ROOT / str(args.sector_pack_outdir) / "sector_overview.csv",
    ]
    missing = [str(p) for p in required_files if not p.exists()]

    counts = structural_payload.get("counts") if isinstance(structural_payload.get("counts"), dict) else {}
    clarity = (
        structural_payload.get("structural_clarity")
        if isinstance(structural_payload.get("structural_clarity"), dict)
        else {}
    )
    drift_block = structural_payload.get("drift") if isinstance(structural_payload.get("drift"), dict) else {}

    latest = {
        "status": "ok",
        "generated_at_utc": generated_at_utc,
        "event_study_outdir": str(outdir),
        "event_study_run_id": run_id,
        "alert_policy": str(args.alert_policy),
        "lookbacks": str(args.lookbacks),
        "n_random": int(args.n_random),
        "random_baseline_method": str(args.random_baseline_method),
        "random_block_size": int(args.random_block_size),
        "q_unstable": float(args.q_unstable),
        "q_transition": float(args.q_transition),
        "q_confidence": float(args.q_confidence),
        "q_confidence_guarded": float(args.q_confidence_guarded),
        "q_score_balanced": float(args.q_score_balanced),
        "q_score_guarded": float(args.q_score_guarded),
        "confirm_n": int(args.confirm_n),
        "confirm_m": int(args.confirm_m),
        "min_alert_gap_days": int(args.min_alert_gap_days),
        "auto_candidates": str(args.auto_candidates),
        "two_layer_mode": str(args.two_layer_mode),
        "profile_applied": bool(profile_meta.get("profile_applied", False)),
        "profile_file": str(profile_meta.get("profile_file", "")),
        "profile_version": str(profile_meta.get("profile_version", "")),
        "min_sector_assets": int(args.min_sector_assets),
        "min_cal_days": int(args.min_cal_days),
        "min_test_days": int(args.min_test_days),
        "drift_baseline_days": int(args.drift_baseline_days),
        "drift_min_history_runs": int(args.drift_min_history_runs),
        "drift_warn_zscore": float(args.drift_warn_zscore),
        "drift_block_zscore": float(args.drift_block_zscore),
        "db_path": str(db_path),
        "previous_run_id": prev_id,
        "weekly_compare_file": str(weekly_compare_path),
        "drift_monitor": drift_payload,
        "structural_report_file": str(structural_report_path),
        "structural_report": structural_payload,
        "counts": counts,
        "structural_clarity": clarity,
        "drift_level": str(drift_block.get("level", drift_payload.get("drift_level", "unknown"))),
        "notification": notification_payload,
        "sector_pack": pack_payload,
    }

    latest_path = ROOT / args.out_root / "latest_run.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(latest, indent=2, allow_nan=False), encoding="utf-8")

    health = {
        "status": "ok" if not missing else "warn",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "duration_sec": round(time.time() - t_start, 3),
        "steps": step_log,
        "missing_files": missing,
        "drift_level": str(drift_payload.get("drift_level", "unknown")),
        "drift_score": drift_payload.get("drift_score", None),
        "clarity_score": clarity.get("score"),
        "clarity_label": clarity.get("label"),
        "structural_gate_hint": (
            (structural_payload.get("gate_hint") or {}).get("status")
            if isinstance(structural_payload.get("gate_hint"), dict)
            else "unknown"
        ),
        "notification": {
            "n_exited_green": int(notification_payload.get("n_exited_green", 0)),
            "webhook_sent": bool(notification_payload.get("webhook_sent", False)),
        },
    }
    health_dir = ROOT / args.out_root / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / f"health_{run_id}.json").write_text(json.dumps(health, indent=2, allow_nan=False), encoding="utf-8")
    (health_dir / "latest_health.json").write_text(json.dumps(health, indent=2, allow_nan=False), encoding="utf-8")

    audit_path = ROOT / args.out_root / "audit_trail.jsonl"
    audit_event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": health["status"],
        "outdir": str(outdir),
        "n_exited_green": int(notification_payload.get("n_exited_green", 0)),
        "weekly_changed_up": int(weekly_compare.get("summary", {}).get("changed_up", 0)),
        "weekly_changed_down": int(weekly_compare.get("summary", {}).get("changed_down", 0)),
        "drift_level": str(drift_payload.get("drift_level", "unknown")),
        "drift_score": drift_payload.get("drift_score", None),
        "clarity_score": clarity.get("score"),
        "clarity_label": clarity.get("label"),
        "structural_gate_hint": (
            (structural_payload.get("gate_hint") or {}).get("status")
            if isinstance(structural_payload.get("gate_hint"), dict)
            else "unknown"
        ),
        "duration_sec": health["duration_sec"],
    }
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit_event, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "latest_run_json": str(latest_path),
                "event_study_outdir": str(outdir),
                "health_json": str(health_dir / "latest_health.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
