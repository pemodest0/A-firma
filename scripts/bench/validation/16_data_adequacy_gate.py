#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUTDIR_DEFAULT = ROOT / "results" / "validation" / "data_adequacy"
SOURCE_CONFIG_DEFAULT = ROOT / "config" / "data_sources.v1.json"
ADEQUACY_CONFIG_DEFAULT = ROOT / "config" / "data_adequacy.v1.json"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _scan_candidates() -> list[Path]:
    roots = [
        ROOT / "data" / "raw" / "finance" / "yfinance_daily",
        ROOT / "data" / "raw" / "realestate",
        ROOT / "data" / "raw" / "energy",
    ]
    files: list[Path] = []
    for r in roots:
        if r.exists():
            files.extend(sorted(r.rglob("*.csv")))
    return files


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def _infer_domain(path: Path) -> str:
    p = str(path).lower()
    if "realestate" in p or "imob" in p:
        return "realestate"
    if "energy" in p or "ons" in p:
        return "energy"
    return "finance"


def _profile(path: Path, min_points: int, min_years: float, max_gap_days: int) -> dict[str, Any]:
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(path, sep=";")
    except Exception as exc:
        return {"dataset": str(path), "status": "fail", "reason": f"read_error: {exc}"}

    dc = _find_col(df, ["date", "data", "datetime", "timestamp", "time", "din_instante"])
    vc = _find_col(
        df,
        [
            "close",
            "adj_close",
            "price",
            "value",
            "valor",
            "load",
            "consumo",
            "val_cargaenergiamwmed",
            "carga_mw",
        ],
    )
    if dc is None or vc is None:
        return {"dataset": str(path), "status": "fail", "reason": "missing_date_or_value_col"}

    d = pd.to_datetime(df[dc], errors="coerce")
    v = pd.to_numeric(df[vc], errors="coerce")
    raw_n = int(df.shape[0])
    valid = d.notna() & v.notna()
    d = d[valid].sort_values()
    n = int(d.shape[0])
    if n == 0:
        return {"dataset": str(path), "status": "fail", "reason": "no_valid_rows"}

    dates = pd.Series(d.reset_index(drop=True))
    min_d = pd.Timestamp(dates.iloc[0])
    max_d = pd.Timestamp(dates.iloc[-1])
    years = max(0.0, (max_d - min_d).days / 365.25)
    gaps = dates.diff().dt.days.dropna()
    max_gap = int(gaps.max()) if not gaps.empty else 0
    median_gap = float(gaps.median()) if not gaps.empty else 0.0
    inferred_freq = "daily" if median_gap <= 2 else "weekly" if median_gap <= 9 else "irregular"

    checks = {
        "min_points_ok": n >= min_points,
        "coverage_ok": years >= min_years,
        "max_gap_ok": max_gap <= max_gap_days,
    }
    status = "ok" if all(checks.values()) else "fail"

    return {
        "dataset": str(path),
        "asset_id": path.stem,
        "domain": _infer_domain(path),
        "status": status,
        "reason": "" if status == "ok" else "failed_adequacy_checks",
        "raw_rows": raw_n,
        "n_points": n,
        "n_missing": int(raw_n - n),
        "date_min": min_d.strftime("%Y-%m-%d"),
        "date_max": max_d.strftime("%Y-%m-%d"),
        "coverage_years": years,
        "median_gap_days": median_gap,
        "max_gap_days": max_gap,
        "inferred_freq": inferred_freq,
        "checks": checks,
        "thresholds": {
            "min_points": min_points,
            "min_years": min_years,
            "max_gap_days": max_gap_days,
        },
    }


def _source_meta(dataset_path: str, source_cfg: dict[str, Any]) -> tuple[str, str]:
    p = dataset_path.replace("\\", "/").lower()
    rules = source_cfg.get("rules") or []
    for rule in rules:
        pattern = str(rule.get("pattern", "")).lower()
        if pattern and pattern in p:
            return str(rule.get("source_type", "proxy")), str(rule.get("source_name", "unknown"))
    defaults = source_cfg.get("defaults") or {}
    return str(defaults.get("source_type", "proxy")), str(defaults.get("source_name", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Data adequacy gate before running motor.")
    parser.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT))
    parser.add_argument("--min-points", type=int, default=600)
    parser.add_argument("--min-years", type=float, default=3.0)
    parser.add_argument("--max-gap-days", type=int, default=14)
    parser.add_argument("--strict", action="store_true", default=False, help="Fail summary when any asset fails adequacy.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial pass even with failing assets.")
    parser.add_argument("--source-config", type=str, default=str(SOURCE_CONFIG_DEFAULT))
    parser.add_argument("--adequacy-config", type=str, default=str(ADEQUACY_CONFIG_DEFAULT))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    source_cfg = _read_json(Path(args.source_config))
    adequacy_cfg = _read_json(Path(args.adequacy_config))
    defaults = adequacy_cfg.get("defaults") or {}
    by_domain = adequacy_cfg.get("domains") or {}

    profiles = []
    for p in _scan_candidates():
        domain = _infer_domain(p)
        cfg = by_domain.get(domain, {})
        min_points = int(cfg.get("min_points", defaults.get("min_points", args.min_points)))
        min_years = float(cfg.get("min_years", defaults.get("min_years", args.min_years)))
        max_gap_days = int(cfg.get("max_gap_days", defaults.get("max_gap_days", args.max_gap_days)))
        profiles.append(_profile(p, min_points, min_years, max_gap_days))
    if not profiles:
        payload = {"status": "fail", "reason": "no_datasets_found"}
        _json_dump(outdir / "summary.json", payload)
        print("[data_adequacy] fail no datasets")
        return

    df = pd.DataFrame(profiles)
    if "dataset" in df.columns:
        source_meta = df["dataset"].apply(lambda s: _source_meta(str(s), source_cfg))
        df["source_type"] = source_meta.apply(lambda t: t[0])
        df["source_name"] = source_meta.apply(lambda t: t[1])
    df.to_csv(outdir / "data_adequacy_by_asset.csv", index=False)

    ok_df = df[df["status"] == "ok"]
    fail_df = df[df["status"] == "fail"]
    by_domain = (
        df.groupby("domain")["status"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        .to_dict(orient="records")
    )

    strict_mode = bool(args.strict and not args.allow_partial)
    has_failures = int(fail_df.shape[0]) > 0
    summary_status = "fail" if (strict_mode and has_failures) else "ok"

    summary = {
        "status": summary_status,
        "params": {
            "defaults": {
                "min_points": defaults.get("min_points", args.min_points),
                "min_years": defaults.get("min_years", args.min_years),
                "max_gap_days": defaults.get("max_gap_days", args.max_gap_days),
            },
            "domains": by_domain,
            "strict": strict_mode,
        },
        "counts": {
            "total": int(df.shape[0]),
            "ok": int(ok_df.shape[0]),
            "fail": int(fail_df.shape[0]),
        },
        "coverage": {
            "mean_years_ok": float(ok_df["coverage_years"].mean()) if not ok_df.empty else 0.0,
            "median_points_ok": float(ok_df["n_points"].median()) if not ok_df.empty else 0.0,
        },
        "by_domain": by_domain,
    }
    if summary_status == "fail":
        summary["reason"] = "data_adequacy_failed_for_one_or_more_assets"
    _json_dump(outdir / "summary.json", summary)
    print(
        f"[data_adequacy] status={summary_status} total={summary['counts']['total']} "
        f"ok={summary['counts']['ok']} fail={summary['counts']['fail']}"
    )
    if summary_status != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
