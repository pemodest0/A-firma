#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_POINTS_DEFAULT = 200
OUTDIR_DEFAULT = ROOT / "results" / "validation" / "universe_mini"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _min_points_for_timeframe(cfg: dict[str, Any], timeframe: str) -> int:
    by_tf = cfg.get("min_points_by_timeframe")
    if isinstance(by_tf, dict):
        key = str(timeframe or "").strip().lower()
        if key in by_tf:
            try:
                return max(1, int(by_tf[key]))
            except Exception:
                pass
    try:
        return max(1, int(cfg.get("min_points_default", MIN_POINTS_DEFAULT)))
    except Exception:
        return MIN_POINTS_DEFAULT


def _discover_datasets(max_assets: int) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []

    finance = sorted((ROOT / "data" / "raw" / "finance" / "yfinance_daily").glob("*.csv"))
    for p in finance:
        assets.append({"asset_id": p.stem, "path": p, "value_col": "close", "date_col": "date", "timeframe": "daily"})

    ons = sorted((ROOT / "data" / "raw" / "ONS" / "ons_carga_diaria").glob("*.csv"))
    for p in ons:
        assets.append(
            {
                "asset_id": p.stem,
                "path": p,
                "value_col": "val_cargaenergiamwmed",
                "date_col": "din_instante",
                "timeframe": "daily",
            }
        )

    realestate = sorted((ROOT / "data" / "realestate" / "normalized").glob("*.csv"))
    for p in realestate:
        assets.append({"asset_id": f"RE_{p.stem}", "path": p, "value_col": "value", "date_col": "date", "timeframe": "monthly"})

    # Extra raw macro series tied to real estate diagnostics.
    re_raw = sorted((ROOT / "data" / "raw" / "realestate" / "bcb").glob("*.csv"))
    for p in re_raw:
        assets.append({"asset_id": f"RE_RAW_{p.stem}", "path": p, "value_col": "valor", "date_col": "data", "timeframe": "monthly"})

    return assets[:max_assets]


def _resolve_col(df: pd.DataFrame, col: str | None) -> str | None:
    if col is None:
        return None
    if col in df.columns:
        return col
    target = col.lower()
    for c in df.columns:
        if c.lower() == target:
            return c
    return None


def _auto_value_col(df: pd.DataFrame) -> str | None:
    for cand in ["close", "price", "value", "adj_close", "log_price", "r"]:
        found = _resolve_col(df, cand)
        if found is not None:
            return found
    return None


def _auto_date_col(df: pd.DataFrame) -> str | None:
    for cand in ["date", "datetime", "timestamp", "time"]:
        found = _resolve_col(df, cand)
        if found is not None:
            return found
    return None


def _run_one(
    row: dict[str, Any],
    outdir: Path,
    seed: int,
    cfg: dict[str, Any],
    engine_available: bool,
    engine_err: str | None,
) -> dict[str, Any]:
    asset_id = row["asset_id"]
    asset_dir = outdir / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    path = Path(row["path"])
    value_col = row["value_col"]
    date_col = row["date_col"]
    timeframe = row["timeframe"]

    result_line: dict[str, Any] = {
        "asset_id": asset_id,
        "domain": "realestate" if asset_id.startswith("RE_") else ("energy" if asset_id.startswith("ons_") else "finance"),
        "timeframe": timeframe,
        "dataset": str(path),
        "n_points": 0,
        "n_regimes": 0,
        "pct_transition": None,
        "mean_confidence": None,
        "mean_quality": None,
        "n_alerts": 0,
        "status": "fail",
        "reason": "",
    }

    if not path.exists():
        reason = f"dataset nÃ£o encontrado: {path}"
        _json_dump(asset_dir / "summary.json", {"status": "fail", "reason": reason})
        pd.DataFrame(columns=["t", "date", "regime_id", "regime_label", "confidence", "quality"]).to_csv(
            asset_dir / "regimes.csv", index=False
        )
        pd.DataFrame(columns=["t", "date", "alert_type", "severity", "score"]).to_csv(asset_dir / "alerts.csv", index=False)
        result_line["reason"] = reason
        return result_line

    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(path, sep=";")
    except Exception as exc:
        reason = f"falha ao ler CSV: {exc}"
        _json_dump(asset_dir / "summary.json", {"status": "fail", "reason": reason})
        pd.DataFrame(columns=["t", "date", "regime_id", "regime_label", "confidence", "quality"]).to_csv(
            asset_dir / "regimes.csv", index=False
        )
        pd.DataFrame(columns=["t", "date", "alert_type", "severity", "score"]).to_csv(asset_dir / "alerts.csv", index=False)
        result_line["reason"] = reason
        return result_line

    value_col_res = _resolve_col(df, value_col) or _auto_value_col(df)
    date_col_res = _resolve_col(df, date_col) or _auto_date_col(df)
    for c_name, c in [("value_col", value_col_res), ("date_col", date_col_res)]:
        if c is None:
            expected = value_col if c_name == "value_col" else date_col
            reason = f"coluna nÃ£o existe: {expected}"
            _json_dump(asset_dir / "summary.json", {"status": "fail", "reason": reason})
            pd.DataFrame(columns=["t", "date", "regime_id", "regime_label", "confidence", "quality"]).to_csv(
                asset_dir / "regimes.csv", index=False
            )
            pd.DataFrame(columns=["t", "date", "alert_type", "severity", "score"]).to_csv(asset_dir / "alerts.csv", index=False)
            result_line["reason"] = reason
            return result_line

    values = pd.to_numeric(df[value_col_res], errors="coerce").to_numpy()
    mask = np.isfinite(values)
    values = values[mask]
    dates = pd.to_datetime(df[date_col_res], errors="coerce").astype("datetime64[ns]").to_numpy()[mask]
    max_points = int(cfg.get("max_points") or 0)
    if max_points > 0 and values.shape[0] > max_points:
        # Keep the most recent history window only (causal, no future leakage).
        values = values[-max_points:]
        dates = dates[-max_points:]
    result_line["n_points"] = int(values.shape[0])

    min_points = _min_points_for_timeframe(cfg, timeframe=timeframe)
    if values.shape[0] < min_points:
        reason = f"pontos insuficientes (<{min_points})"
        _json_dump(asset_dir / "summary.json", {"status": "fail", "reason": reason})
        pd.DataFrame(columns=["t", "date", "regime_id", "regime_label", "confidence", "quality"]).to_csv(
            asset_dir / "regimes.csv", index=False
        )
        pd.DataFrame(columns=["t", "date", "alert_type", "severity", "score"]).to_csv(asset_dir / "alerts.csv", index=False)
        result_line["reason"] = reason
        return result_line

    if not engine_available:
        reason = f"dependÃªncia do motor indisponÃ­vel: {engine_err}"
        _json_dump(asset_dir / "summary.json", {"status": "fail", "reason": reason})
        pd.DataFrame(columns=["t", "date", "regime_id", "regime_label", "confidence", "quality"]).to_csv(
            asset_dir / "regimes.csv", index=False
        )
        pd.DataFrame(columns=["t", "date", "alert_type", "severity", "score"]).to_csv(asset_dir / "alerts.csv", index=False)
        result_line["reason"] = reason
        return result_line

    try:
        from engine.graph.core import run_graph_engine  # type: ignore
        from engine.graph.embedding import estimate_embedding_params  # type: ignore
    except Exception as exc:
        reason = f"dependÃªncia do motor indisponÃ­vel: {exc}"
        _json_dump(asset_dir / "summary.json", {"status": "fail", "reason": reason})
        pd.DataFrame(columns=["t", "date", "regime_id", "regime_label", "confidence", "quality"]).to_csv(
            asset_dir / "regimes.csv", index=False
        )
        pd.DataFrame(columns=["t", "date", "alert_type", "severity", "score"]).to_csv(asset_dir / "alerts.csv", index=False)
        result_line["reason"] = reason
        return result_line

    np.random.seed(seed)
    try:
        m, tau = estimate_embedding_params(values, max_tau=20, max_m=6)
        result = run_graph_engine(
            values,
            m=m,
            tau=tau,
            n_micro=cfg["n_micro"],
            n_regimes=cfg["n_regimes"],
            k_nn=cfg["k_nn"],
            theiler=cfg["theiler"],
            alpha=cfg["alpha"],
            seed=seed,
            timeframe="weekly" if timeframe == "monthly" else "daily",
        )
    except Exception as exc:
        reason = f"falha na execuÃ§Ã£o do motor: {exc}"
        _json_dump(asset_dir / "summary.json", {"status": "fail", "reason": reason})
        pd.DataFrame(columns=["t", "date", "regime_id", "regime_label", "confidence", "quality"]).to_csv(
            asset_dir / "regimes.csv", index=False
        )
        pd.DataFrame(columns=["t", "date", "alert_type", "severity", "score"]).to_csv(asset_dir / "alerts.csv", index=False)
        result_line["reason"] = reason
        return result_line

    n = int(result.state_labels.shape[0])
    if n <= 0:
        reason = "motor retornou sÃ©rie vazia"
        _json_dump(asset_dir / "summary.json", {"status": "fail", "reason": reason})
        pd.DataFrame(columns=["t", "date", "regime_id", "regime_label", "confidence", "quality"]).to_csv(
            asset_dir / "regimes.csv", index=False
        )
        pd.DataFrame(columns=["t", "date", "alert_type", "severity", "score"]).to_csv(asset_dir / "alerts.csv", index=False)
        result_line["reason"] = reason
        return result_line

    start = len(values) - n
    dd = pd.to_datetime(dates[start:], errors="coerce").strftime("%Y-%m-%d")
    labels = result.state_labels.astype(str).tolist()
    order = []
    for lbl in labels:
        if lbl not in order:
            order.append(lbl)
    rid = {k: i for i, k in enumerate(order)}
    quality = float((result.quality or {}).get("score", np.nan))
    regimes_df = pd.DataFrame(
        {
            "t": np.arange(n, dtype=int),
            "date": dd,
            "regime_id": [rid[x] for x in labels],
            "regime_label": labels,
            "confidence": result.confidence.astype(float),
            "quality": np.repeat(quality, n),
        }
    )
    regimes_df.to_csv(asset_dir / "regimes.csv", index=False)

    transitions = np.where(result.state_labels[1:] != result.state_labels[:-1])[0] + 1
    alerts_rows = [
        {
            "t": int(i),
            "date": str(dd.iloc[i]) if isinstance(dd, pd.Series) else str(dd[i]),
            "alert_type": "REGIME_TRANSITION",
            "severity": "medium",
            "score": 1.0,
        }
        for i in transitions.tolist()
    ]
    alerts_df = pd.DataFrame(alerts_rows, columns=["t", "date", "alert_type", "severity", "score"])
    alerts_df.to_csv(asset_dir / "alerts.csv", index=False)

    summary = {
        "status": "ok",
        "asset_id": asset_id,
        "dataset": str(path),
        "m": int(m),
        "tau": int(tau),
        "n_points": n,
        "n_regimes": int(len(np.unique(result.state_labels))),
        "pct_transition": float(len(transitions) / max(1, n - 1)),
        "mean_confidence": float(np.nanmean(result.confidence)),
        "mean_quality": quality if np.isfinite(quality) else None,
        "n_alerts_total": int(alerts_df.shape[0]),
    }
    _json_dump(asset_dir / "summary.json", summary)

    result_line.update(
        {
            "n_points": summary["n_points"],
            "n_regimes": summary["n_regimes"],
            "pct_transition": summary["pct_transition"],
            "mean_confidence": summary["mean_confidence"],
            "mean_quality": summary["mean_quality"],
            "n_alerts": summary["n_alerts_total"],
            "status": "ok",
            "reason": "",
        }
    )
    return result_line


def _build_fail_row(ds: dict[str, Any], reason: str) -> dict[str, Any]:
    asset_id = str(ds.get("asset_id", ""))
    return {
        "asset_id": asset_id,
        "domain": "realestate" if asset_id.startswith("RE_") else ("energy" if asset_id.startswith("ons_") else "finance"),
        "timeframe": str(ds.get("timeframe", "daily")),
        "dataset": str(ds.get("path", "")),
        "n_points": 0,
        "n_regimes": 0,
        "pct_transition": None,
        "mean_confidence": None,
        "mean_quality": None,
        "n_alerts": 0,
        "status": "fail",
        "reason": reason,
    }


def _write_fail_artifacts(outdir: Path, ds: dict[str, Any], reason: str) -> None:
    asset_dir = outdir / str(ds.get("asset_id", "unknown"))
    asset_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(asset_dir / "summary.json", {"status": "fail", "reason": reason})
    pd.DataFrame(columns=["t", "date", "regime_id", "regime_label", "confidence", "quality"]).to_csv(
        asset_dir / "regimes.csv", index=False
    )
    pd.DataFrame(columns=["t", "date", "alert_type", "severity", "score"]).to_csv(asset_dir / "alerts.csv", index=False)


def _run_one_worker(
    queue: mp.Queue[dict[str, Any]],
    ds: dict[str, Any],
    outdir_str: str,
    seed: int,
    cfg: dict[str, Any],
    engine_available: bool,
    engine_err: str | None,
) -> None:
    outdir = Path(outdir_str)
    try:
        queue.put(_run_one(ds, outdir, seed=seed, cfg=cfg, engine_available=engine_available, engine_err=engine_err))
    except Exception as exc:  # pragma: no cover
        queue.put(_build_fail_row(ds, f"worker_exception: {exc}"))


def _run_one_mp(
    ds: dict[str, Any],
    outdir: Path,
    seed: int,
    cfg: dict[str, Any],
    engine_available: bool,
    engine_err: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    q: mp.Queue[dict[str, Any]] = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_one_worker,
        args=(q, ds, str(outdir), int(seed), cfg, bool(engine_available), engine_err),
    )
    proc.start()
    proc.join(timeout=max(1.0, float(timeout_sec)))

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        reason = f"asset_timeout>{int(timeout_sec)}s"
        _write_fail_artifacts(outdir, ds, reason)
        return _build_fail_row(ds, reason)

    if proc.exitcode not in (0, None):
        reason = f"worker_exitcode:{proc.exitcode}"
        _write_fail_artifacts(outdir, ds, reason)
        return _build_fail_row(ds, reason)

    try:
        row = q.get_nowait()
    except Exception:
        reason = "worker_no_payload"
        _write_fail_artifacts(outdir, ds, reason)
        return _build_fail_row(ds, reason)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Universe mini de diagnÃ³stico por ativos.")
    parser.add_argument("--max-assets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT))
    parser.add_argument("--n-micro", type=int, default=80)
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--k-nn", type=int, default=5)
    parser.add_argument("--theiler", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--asset-timeout-sec", type=float, default=0.0, help="Timeout por ativo (0 = desabilitado)")
    parser.add_argument("--max-points", type=int, default=0, help="Limita pontos por ativo aos N mais recentes (0 = sem corte)")
    parser.add_argument("--min-points-default", type=int, default=MIN_POINTS_DEFAULT)
    parser.add_argument("--min-points-daily", type=int, default=200)
    parser.add_argument("--min-points-weekly", type=int, default=120)
    parser.add_argument("--min-points-monthly", type=int, default=60)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = _discover_datasets(args.max_assets)
    if not datasets:
        _json_dump(outdir / "universe_report.json", {"status": "fail", "reason": "nenhum dataset encontrado"})
        print("[fail] nenhum dataset encontrado")
        return

    engine_available = True
    engine_err = None
    try:
        import engine.graph.core  # noqa: F401
        import engine.graph.embedding  # noqa: F401
    except Exception as exc:
        engine_available = False
        engine_err = str(exc)

    cfg = {
        "n_micro": args.n_micro,
        "n_regimes": args.n_regimes,
        "k_nn": args.k_nn,
        "theiler": args.theiler,
        "alpha": args.alpha,
        "max_points": int(args.max_points),
        "min_points_default": int(args.min_points_default),
        "min_points_by_timeframe": {
            "daily": int(args.min_points_daily),
            "weekly": int(args.min_points_weekly),
            "monthly": int(args.min_points_monthly),
        },
    }

    rows = []
    failures = []
    for i, ds in enumerate(datasets):
        t0 = time.time()
        if float(args.asset_timeout_sec) > 0:
            row = _run_one_mp(
                ds,
                outdir,
                seed=args.seed + i,
                cfg=cfg,
                engine_available=engine_available,
                engine_err=engine_err,
                timeout_sec=float(args.asset_timeout_sec),
            )
        else:
            row = _run_one(ds, outdir, seed=args.seed + i, cfg=cfg, engine_available=engine_available, engine_err=engine_err)
        dt = float(time.time() - t0)
        rows.append(row)
        if row["status"] != "ok":
            failures.append({"asset_id": row["asset_id"], "reason": row["reason"]})
        print(
            f"[universe_mini] {i+1}/{len(datasets)} "
            f"asset={row.get('asset_id')} status={row.get('status')} "
            f"elapsed_sec={dt:.1f} reason={row.get('reason', '')}"
        )

    master_df = pd.DataFrame(
        rows,
        columns=[
            "asset_id",
            "domain",
            "timeframe",
            "dataset",
            "n_points",
            "n_regimes",
            "pct_transition",
            "mean_confidence",
            "mean_quality",
            "n_alerts",
            "status",
            "reason",
        ],
    )
    master_df.to_csv(outdir / "master_summary.csv", index=False)

    ok_df = master_df[master_df["status"] == "ok"].copy()
    if not ok_df.empty:
        top_instable = (
            ok_df.sort_values("pct_transition", ascending=False)
            .head(5)[["asset_id", "pct_transition"]]
            .to_dict(orient="records")
        )
        top_quality = (
            ok_df.sort_values("mean_quality", ascending=True)
            .head(5)[["asset_id", "mean_quality"]]
            .to_dict(orient="records")
        )
    else:
        top_instable = []
        top_quality = []

    n_ok = int((master_df["status"] == "ok").sum())
    n_fail = int((master_df["status"] == "fail").sum())
    report = {
        "status": "ok" if n_ok > 0 else "fail",
        "counts": {"ok": n_ok, "fail": n_fail, "total": int(master_df.shape[0])},
        "top_5_mais_instaveis": top_instable,
        "top_5_pior_qualidade": top_quality,
        "failures": failures,
    }
    _json_dump(outdir / "universe_report.json", report)

    print(f"universe_mini status={report['status']} ok={n_ok} fail={n_fail} total={master_df.shape[0]}")


if __name__ == "__main__":
    main()
