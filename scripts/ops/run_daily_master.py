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


def _ts_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _run(cmd: list[str], *, cwd: Path, timeout_sec: float) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_sec)
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        timeout_msg = f"timeout_after_{int(timeout_sec)}s"
        return 124, (out + "\n" + timeout_msg).strip(), (err + "\n" + timeout_msg).strip()


def _as_float(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _latest_run_id(runs_root: Path, current_run_id: str) -> str | None:
    if not runs_root.exists():
        return None
    dirs = sorted([d.name for d in runs_root.iterdir() if d.is_dir() and d.name != current_run_id])
    return dirs[-1] if dirs else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Rotina diaria unica: execucao, sanidade, gate e relatorio.")
    ap.add_argument("--run-id", type=str, default=_ts_id())
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--max-assets", type=int, default=80)
    ap.add_argument("--profile-file", type=str, default="config/sector_alerts_profile.json")
    ap.add_argument("--with-heavy", action="store_true", help="Inclui diagnostico motor 470 e suite de crise.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-root", type=str, default="results/ops/runs")
    ap.add_argument("--step-timeout-sec", type=float, default=900.0, help="Timeout por etapa em segundos")
    ap.add_argument("--asset-timeout-sec", type=float, default=180.0, help="Timeout por ativo na validacao")
    ap.add_argument("--max-points", type=int, default=1200, help="Maximo de pontos por ativo na validacao")
    args = ap.parse_args()

    run_id = str(args.run_id).strip() or _ts_id()
    outdir = ROOT / args.out_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []

    def do_step(name: str, cmd: list[str], *, required: bool = True) -> dict[str, Any]:
        row: dict[str, Any] = {"step": name, "cmd": cmd, "required": required}
        if args.dry_run:
            row.update({"status": "skipped_dry_run", "code": 0})
            steps.append(row)
            return row
        code, out, err = _run(cmd, cwd=ROOT, timeout_sec=float(args.step_timeout_sec))
        row.update(
            {
                "status": "ok" if code == 0 else "fail",
                "code": code,
                "stdout_tail": out[-4000:],
                "stderr_tail": err[-4000:],
            }
        )
        steps.append(row)
        if required and code != 0:
            raise RuntimeError(f"step_failed:{name}")
        return row

    heavy_outputs: dict[str, Any] = {}
    failed = False
    fail_reason = ""
    try:
        do_step(
            "daily_validation",
            [
                PY,
                "scripts/ops/run_daily_validation.py",
                "--seed",
                str(args.seed),
                "--max-assets",
                str(args.max_assets),
                "--run-id",
                run_id,
                "--step-timeout-sec",
                str(args.step_timeout_sec),
                "--asset-timeout-sec",
                str(args.asset_timeout_sec),
                "--max-points",
                str(args.max_points),
            ],
        )
        do_step("build_snapshot", [PY, "scripts/ops/build_daily_snapshot.py", "--run-id", run_id])
        do_step(
            "validate_output_contract",
            [
                PY,
                "scripts/ops/validate_output_contract.py",
                "--snapshot",
                f"results/ops/snapshots/{run_id}/api_snapshot.jsonl",
                "--out",
                f"results/ops/runs/{run_id}/contract_check.json",
            ],
        )
        do_step(
            "prediction_truth_daily",
            [PY, "scripts/ops/update_prediction_truth_daily.py", "--run-id", run_id],
            required=False,
        )
        do_step("daily_diff", [PY, "scripts/ops/daily_diff_report.py", "--outdir", f"results/ops/runs/{run_id}/diff"])
        do_step(
            "daily_sector_alerts",
            [PY, "scripts/ops/run_daily_sector_alerts.py", "--profile-file", str(args.profile_file)],
        )

        if args.with_heavy:
            r1 = do_step("motor_470_diagnostics", [PY, "scripts/bench/run_motor_470_diagnostics.py"], required=False)
            r2 = do_step(
                "sector_crisis_suite",
                [PY, "scripts/bench/run_sector_and_crisis_suite.py", "--alert-policy", "regime_entry"],
                required=False,
            )
            for key, row in [("motor_470_diagnostics", r1), ("sector_crisis_suite", r2)]:
                text = str(row.get("stdout_tail", "")).strip().splitlines()
                if text:
                    try:
                        heavy_outputs[key] = json.loads(text[-1])
                    except json.JSONDecodeError:
                        heavy_outputs[key] = {"status": row.get("status"), "note": "no_json_payload"}
    except RuntimeError as exc:
        failed = True
        fail_reason = str(exc)

    (outdir / "steps.json").write_text(json.dumps(steps, indent=2, ensure_ascii=False), encoding="utf-8")

    validation_summary = _read_json(ROOT / "results" / "ops" / "daily" / run_id / "summary.json")
    snapshot_summary = _read_json(ROOT / "results" / "ops" / "snapshots" / run_id / "summary.json")
    contract_check = _read_json(outdir / "contract_check.json")
    prediction_truth = _read_json(outdir / "prediction_truth_summary.json")
    diff_summary = _read_json(outdir / "diff" / "summary.json")
    latest_sector = _read_json(ROOT / "results" / "event_study_sectors" / "latest_run.json")

    sanity = {
        "status": "ok",
        "checks": {
            "pipeline_steps_ok": (not failed),
            "validation_status_ok": str(validation_summary.get("status", "")).lower() == "ok",
            "snapshot_status_ok": str(snapshot_summary.get("status", "")).lower() == "ok",
            "contract_ok": str(contract_check.get("status", "")).lower() == "ok",
            "diff_exists": bool(diff_summary),
            "diff_gate_not_blocked": not bool((diff_summary.get("deployment_gate") or {}).get("blocked", False)),
            "sector_run_available": bool(latest_sector),
        },
        "failure_reason": fail_reason,
    }
    sanity_ok = all(bool(v) for v in sanity["checks"].values())
    if not sanity_ok:
        sanity["status"] = "fail"
    (outdir / "sanity.json").write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")

    reasons: list[str] = []
    if failed:
        reasons.append(f"pipeline_failed:{fail_reason or 'unknown'}")
    for k, v in sanity["checks"].items():
        if not v:
            reasons.append(f"check_failed:{k}")

    publish_allowed = (not reasons) and sanity_ok
    gate = {
        "status": "ok" if publish_allowed else "blocked",
        "run_id": run_id,
        "publish_allowed": publish_allowed,
        "blocked_reasons": reasons,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (outdir / "publish_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    if not publish_allowed:
        (outdir / "PUBLISH_BLOCKED").write_text("\n".join(reasons) + "\n", encoding="utf-8")

    runs_root = ROOT / args.out_root
    prev_run_id = _latest_run_id(runs_root, run_id)
    prev_sum = _read_json(runs_root / prev_run_id / "daily_master_summary.json") if prev_run_id else {}

    cur_metrics = {
        "stability_score": _as_float((validation_summary.get("metrics") or {}).get("stability_score")),
        "universe_success_rate": _as_float((validation_summary.get("metrics") or {}).get("universe_success_rate")),
        "validated_ratio": _as_float(snapshot_summary.get("validated_ratio")),
        "prediction_accuracy": _as_float((prediction_truth.get("metrics") or {}).get("accuracy")),
        "prediction_precision_risk": _as_float((prediction_truth.get("metrics") or {}).get("precision_risk")),
        "sector_vermelho": _as_float((latest_sector.get("counts") or {}).get("vermelho")),
        "sector_amarelo": _as_float((latest_sector.get("counts") or {}).get("amarelo")),
        "sector_verde": _as_float((latest_sector.get("counts") or {}).get("verde")),
        "sector_clarity_score": _as_float((latest_sector.get("structural_clarity") or {}).get("score")),
        "sector_drift_score": _as_float((latest_sector.get("structural_report") or {}).get("drift", {}).get("score")),
        "sector_weekly_change_rate": _as_float((latest_sector.get("structural_report") or {}).get("weekly_change", {}).get("change_rate")),
    }
    sector_drift_level = str(
        (latest_sector.get("structural_report") or {}).get("drift", {}).get(
            "level", latest_sector.get("drift_level", "unknown")
        )
    )
    sector_clarity_label = str((latest_sector.get("structural_clarity") or {}).get("label", "unknown"))
    sector_gate_hint = str((latest_sector.get("structural_report") or {}).get("gate_hint", {}).get("status", "unknown"))
    prev_metrics = (prev_sum.get("metrics") or {}) if isinstance(prev_sum, dict) else {}
    history_compare = {
        "current_run_id": run_id,
        "previous_run_id": prev_run_id,
        "delta": {
            k: (
                None
                if cur_metrics.get(k) is None or _as_float(prev_metrics.get(k)) is None
                else float(cur_metrics[k] - float(prev_metrics[k]))
            )
            for k in cur_metrics.keys()
        },
    }
    (outdir / "history_compare.json").write_text(json.dumps(history_compare, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"Relatorio diario - {run_id}")
    lines.append("")
    lines.append(f"Publicacao: {'LIBERADA' if publish_allowed else 'BLOQUEADA'}")
    if reasons:
        lines.append("Motivos do bloqueio:")
        for r in reasons:
            lines.append(f"- {r}")
    lines.append("")
    lines.append("Resumo rapido:")
    lines.append(f"- estabilidade: {cur_metrics['stability_score'] if cur_metrics['stability_score'] is not None else '--'}")
    lines.append(
        f"- sucesso_universo: {cur_metrics['universe_success_rate'] if cur_metrics['universe_success_rate'] is not None else '--'}"
    )
    lines.append(f"- validated_ratio: {cur_metrics['validated_ratio'] if cur_metrics['validated_ratio'] is not None else '--'}")
    lines.append(
        f"- acerto_previsao: {cur_metrics['prediction_accuracy'] if cur_metrics['prediction_accuracy'] is not None else '--'}"
    )
    lines.append(
        f"- precisao_alerta_risco: {cur_metrics['prediction_precision_risk'] if cur_metrics['prediction_precision_risk'] is not None else '--'}"
    )
    lines.append(f"- setores vermelho: {int(cur_metrics['sector_vermelho']) if cur_metrics['sector_vermelho'] is not None else '--'}")
    lines.append(f"- setores amarelo: {int(cur_metrics['sector_amarelo']) if cur_metrics['sector_amarelo'] is not None else '--'}")
    lines.append(f"- setores verde: {int(cur_metrics['sector_verde']) if cur_metrics['sector_verde'] is not None else '--'}")
    lines.append(
        f"- clareza_setorial: {cur_metrics['sector_clarity_score'] if cur_metrics['sector_clarity_score'] is not None else '--'} ({sector_clarity_label})"
    )
    lines.append(
        f"- drift_setorial: {cur_metrics['sector_drift_score'] if cur_metrics['sector_drift_score'] is not None else '--'} ({sector_drift_level})"
    )
    lines.append(
        f"- rotacao_semanal_setorial: {cur_metrics['sector_weekly_change_rate'] if cur_metrics['sector_weekly_change_rate'] is not None else '--'}"
    )
    lines.append(f"- gate_hint_setorial: {sector_gate_hint}")
    lines.append("")
    lines.append("Comparacao com execucao anterior:")
    if prev_run_id:
        lines.append(f"- run anterior: {prev_run_id}")
        for k, v in history_compare["delta"].items():
            lines.append(f"- delta_{k}: {v if v is not None else '--'}")
    else:
        lines.append("- sem run anterior para comparar")
    lines.append("")
    lines.append("Arquivos principais:")
    lines.append(f"- sanity: results/ops/runs/{run_id}/sanity.json")
    lines.append(f"- gate: results/ops/runs/{run_id}/publish_gate.json")
    lines.append(f"- historico: results/ops/runs/{run_id}/history_compare.json")
    (outdir / "daily_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "status": "ok",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "publish_allowed": publish_allowed,
        "metrics": cur_metrics,
        "checks": sanity["checks"],
        "previous_run_id": prev_run_id,
        "prediction_truth": prediction_truth,
        "sector_structural": {
            "clarity_label": sector_clarity_label,
            "drift_level": sector_drift_level,
            "gate_hint": sector_gate_hint,
        },
        "heavy_outputs": heavy_outputs,
    }
    (outdir / "daily_master_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / args.out_root / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "path": str(outdir)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {"status": "ok", "outdir": str(outdir), "publish_allowed": publish_allowed, "dry_run": bool(args.dry_run)},
            ensure_ascii=False,
        )
    )
    if (not args.dry_run) and (not publish_allowed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
