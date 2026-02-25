#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_lab_run() -> Path:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        raise FileNotFoundError(f"missing base dir: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in runs:
        hier = d / "hierarchical"
        if (hier / "diagnostics_global_score_daily.csv").exists():
            return d
    raise FileNotFoundError("no run with hierarchical diagnostics found")


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _read_sector_map(hier_dir: Path) -> dict[tuple[str, str], str]:
    index_path = hier_dir / "universes" / "sector_universe_index.csv"
    out: dict[tuple[str, str], str] = {}
    if not index_path.exists():
        return out
    df = pd.read_csv(index_path)
    if df.empty:
        return out
    for _, r in df.iterrows():
        kind = str(r.get("kind", "")).strip().lower()
        slug = str(r.get("slug", "")).strip()
        sector = str(r.get("sector", "")).strip()
        if kind and slug and sector:
            out[(kind, slug)] = sector
    return out


def _load_sector_scores(hier_dir: Path) -> pd.DataFrame:
    slug_map = _read_sector_map(hier_dir)
    rows: list[pd.DataFrame] = []

    def _load(pattern: str, kind: str, prefix: str) -> None:
        for p in sorted(hier_dir.glob(pattern)):
            slug = p.stem.replace(prefix, "")
            if slug.endswith("_score_daily"):
                slug = slug[: -len("_score_daily")]
            sector_name = slug_map.get((kind, slug), slug)
            d = pd.read_csv(p)
            if d.empty:
                continue
            if "date" not in d.columns or "score" not in d.columns:
                continue
            x = d.copy()
            x["date"] = x["date"].astype(str)
            x["score"] = pd.to_numeric(x["score"], errors="coerce")
            x["sector"] = str(sector_name)
            x["kind"] = str(kind)
            rows.append(x[["date", "score", "sector", "kind"]])

    _load("diagnostics_sector_gics_*_score_daily.csv", "gics", "diagnostics_sector_gics_")
    _load("diagnostics_sector_internal_*_score_daily.csv", "internal", "diagnostics_sector_internal_")
    if not rows:
        return pd.DataFrame(columns=["date", "score", "sector", "kind"])
    return pd.concat(rows, ignore_index=True)


def _load_cross(hier_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    gics_path = hier_dir / "cross_sector_global_gics_daily.csv"
    if gics_path.exists():
        g = pd.read_csv(gics_path)
        if not g.empty:
            g["kind"] = "gics"
            frames.append(g)
    internal_path = hier_dir / "cross_sector_global_internal_daily.csv"
    if internal_path.exists():
        i = pd.read_csv(internal_path)
        if not i.empty:
            i["kind"] = "internal"
            frames.append(i)
    if not frames:
        return pd.DataFrame(
            columns=["date", "sector", "kind", "loading_sector_on_global", "overlap_sector_global", "n_assets_sector"]
        )
    x = pd.concat(frames, ignore_index=True)
    x["date"] = x["date"].astype(str)
    x["loading_sector_on_global"] = pd.to_numeric(x.get("loading_sector_on_global"), errors="coerce")
    x["overlap_sector_global"] = pd.to_numeric(x.get("overlap_sector_global"), errors="coerce")
    return x


def _build_daily_payloads(
    global_scores: pd.DataFrame,
    sector_scores: pd.DataFrame,
    cross_scores: pd.DataFrame,
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    g = global_scores.copy()
    g["date"] = g["date"].astype(str)
    g["score"] = pd.to_numeric(g["score"], errors="coerce")
    g = g.dropna(subset=["date"]).sort_values("date")
    if g.empty:
        return [], {"status": "empty"}

    payloads: list[dict[str, Any]] = []
    for d in sorted(g["date"].unique().tolist()):
        g_row = g[g["date"] == d].sort_values("date").tail(1)
        global_score = _safe_float(g_row["score"].iloc[-1]) if not g_row.empty else float("nan")

        sec_d = sector_scores[sector_scores["date"] == d].copy() if not sector_scores.empty else pd.DataFrame()
        sec_d = sec_d.dropna(subset=["score"]) if not sec_d.empty else sec_d
        sec_top = (
            sec_d.sort_values("score", ascending=False)[["sector", "kind", "score"]].head(int(max(1, top_k))).to_dict(orient="records")
            if not sec_d.empty
            else []
        )

        cr_d = cross_scores[cross_scores["date"] == d].copy() if not cross_scores.empty else pd.DataFrame()
        top_load = (
            cr_d.sort_values("loading_sector_on_global", ascending=False)[["sector", "kind", "loading_sector_on_global"]]
            .head(int(max(1, top_k)))
            .to_dict(orient="records")
            if not cr_d.empty
            else []
        )
        top_overlap = (
            cr_d.sort_values("overlap_sector_global", ascending=False)[["sector", "kind", "overlap_sector_global"]]
            .head(int(max(1, top_k)))
            .to_dict(orient="records")
            if not cr_d.empty
            else []
        )

        high_priority: list[dict[str, Any]] = []
        if (not sec_d.empty) and (not cr_d.empty):
            joined = sec_d.merge(cr_d[["sector", "kind", "loading_sector_on_global"]], on=["sector", "kind"], how="left")
            if not joined.empty:
                score_thr = float(joined["score"].quantile(0.80))
                load_thr = float(joined["loading_sector_on_global"].quantile(0.80))
                hp = joined[(joined["score"] >= score_thr) & (joined["loading_sector_on_global"] >= load_thr)].copy()
                if not hp.empty:
                    high_priority = hp.sort_values(["score", "loading_sector_on_global"], ascending=[False, False])[
                        ["sector", "kind", "score", "loading_sector_on_global"]
                    ].head(int(max(1, top_k))).to_dict(orient="records")

        payload = {
            "date": str(d),
            "global_score": global_score,
            "top_sectors_by_score": sec_top,
            "top_sectors_by_loading": top_load,
            "top_sectors_by_overlap": top_overlap,
            "high_priority_sectors": high_priority,
        }
        payloads.append(payload)
    latest = payloads[-1] if payloads else {"status": "empty"}
    return payloads, latest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build platform-ready hierarchical daily/latest state JSON.")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--hierarchical-dir", type=str, default="")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--latest-pointer", type=str, default="results/platform/latest_hierarchical_state.json")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else _latest_lab_run()
    hier_dir = Path(args.hierarchical_dir).resolve() if str(args.hierarchical_dir).strip() else (run_dir / "hierarchical")
    if not hier_dir.exists():
        raise SystemExit(f"missing hierarchical dir: {hier_dir}")

    if str(args.outdir).strip():
        outdir = (ROOT / str(args.outdir).strip()) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    else:
        outdir = run_dir / "platform"
    outdir.mkdir(parents=True, exist_ok=True)

    global_path = hier_dir / "diagnostics_global_score_daily.csv"
    if not global_path.exists():
        raise SystemExit(f"missing global diagnostics: {global_path}")

    global_scores = pd.read_csv(global_path)
    if global_scores.empty or ("date" not in global_scores.columns) or ("score" not in global_scores.columns):
        raise SystemExit(f"invalid global diagnostics: {global_path}")

    sector_scores = _load_sector_scores(hier_dir=hier_dir)
    cross_scores = _load_cross(hier_dir=hier_dir)
    payloads, latest = _build_daily_payloads(
        global_scores=global_scores,
        sector_scores=sector_scores,
        cross_scores=cross_scores,
        top_k=int(max(1, args.top_k)),
    )

    daily_path = outdir / "hierarchical_state_daily.jsonl"
    latest_path = outdir / "hierarchical_state_latest.json"
    daily_path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in payloads) + ("\n" if payloads else ""),
        encoding="utf-8",
    )
    latest_path.write_text(json.dumps(latest, indent=2, ensure_ascii=False), encoding="utf-8")

    pointer_path = ROOT / str(args.latest_pointer).strip()
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir),
                "hierarchical_dir": str(hier_dir),
                "daily_jsonl": str(daily_path),
                "latest_json": str(latest_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "status": "ok",
        "run_dir": str(run_dir),
        "hierarchical_dir": str(hier_dir),
        "outdir": str(outdir),
        "daily_rows": int(len(payloads)),
        "latest_date": str(latest.get("date", "")) if isinstance(latest, dict) else "",
        "files": {
            "hierarchical_state_daily_jsonl": str(daily_path),
            "hierarchical_state_latest_json": str(latest_path),
            "latest_hierarchical_state_json": str(pointer_path),
        },
    }
    (outdir / "hierarchical_state_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/ops/run_platform_hierarchical_state.py",
        params={
            "run_dir": str(run_dir),
            "hierarchical_dir": str(hier_dir),
            "top_k": int(max(1, args.top_k)),
            "latest_pointer": str(pointer_path),
        },
        paths=summary["files"],
        gates={
            "global_diagnostics_exists": bool(global_path.exists()),
            "latest_json_written": bool(latest_path.exists()),
            "pointer_written": bool(pointer_path.exists()),
        },
        extra={"daily_rows": int(len(payloads))},
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
