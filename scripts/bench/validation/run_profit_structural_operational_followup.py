#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import build_official_mode_allocations  # noqa: E402
from scripts.bench.validation.run_profit_structural_extraction_audit import (  # noqa: E402
    _joint_signal_permutation,
    _load_cached_signal_frame,
    _load_cached_target_events,
    _operational_rollup,
    _operational_target_rows,
    _save_target_events,
    _target_catalog,
    _target_future_catalog,
    _build_target_event_matrix,
)


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_cached_rows(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else None


def _save_cached_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _load_cached_attack_mask(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "date" not in frame.columns or "attack" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    return pd.to_numeric(frame["attack"], errors="coerce").fillna(0.0).astype(int)


def _save_cached_attack_mask(path: Path, attack_mask: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"date": attack_mask.index, "attack": attack_mask.astype(int).to_numpy()})
    frame.to_csv(path, index=False)


def _load_attack_mask_from_historical_driver(repo_root: Path) -> pd.Series | None:
    path = repo_root / "results" / "ops" / "shadow_gods_historical" / "Apollo" / "200" / "recommendations_full.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if "driver_gross_exposure" in frame.columns:
        gross = pd.to_numeric(frame["driver_gross_exposure"], errors="coerce").fillna(0.0).astype(float)
    else:
        gross = pd.Series(0.0, index=frame.index, dtype=float)
    if "recommended_mode" in frame.columns:
        attack = frame["recommended_mode"].astype(str).str.lower().eq("ataque").astype(int)
    else:
        attack = (gross >= 0.30).astype(int)
    if "driver_regime" in frame.columns:
        stress = frame["driver_regime"].astype(str).str.lower().eq("stress")
        attack.loc[stress] = 0
    return pd.Series(attack.to_numpy(dtype=int), index=frame["date"], dtype=int).sort_index()


def _build_official_for_observer(
    *,
    prices_dir: Path,
    crypto_groups: Path,
    crypto_meta: Path,
    equity_groups: Path,
    equity_meta: Path,
    benchmark_crypto: str,
    benchmark_equity: str,
    observer_mode: str,
) -> dict[str, Any]:
    current = build_official_mode_allocations(
        prices_dir=prices_dir,
        crypto_groups=crypto_groups,
        crypto_meta=crypto_meta,
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        benchmark_crypto=str(benchmark_crypto),
        benchmark_equity=str(benchmark_equity),
    )
    if str(observer_mode) != "all22":
        return current
    return build_official_mode_allocations(
        prices_dir=prices_dir,
        crypto_groups=crypto_groups,
        crypto_meta=crypto_meta,
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        benchmark_crypto=str(benchmark_crypto),
        benchmark_equity=str(benchmark_equity),
        observer_context=current["context"],
        observer_crypto_tickers=list(current["context"]["crypto_tiers"]["crypto_all"]),
    )


def _build_subset_masks(index: pd.DatetimeIndex, attack_mask: pd.Series) -> dict[str, pd.Series]:
    base = pd.Series(1, index=index, dtype=int)
    attack = pd.to_numeric(attack_mask.reindex(index), errors="coerce").fillna(0.0).astype(int)
    masks: dict[str, pd.Series] = {"all": base, "attack_only": attack}
    for year in (2023, 2024, 2025):
        year_mask = pd.Series((index.year == int(year)).astype(int), index=index, dtype=int)
        masks[f"holdout_{year}"] = year_mask
        masks[f"attack_{year}"] = (attack & year_mask).astype(int)
    return masks


def _subset_stats(mask: pd.Series) -> dict[str, Any]:
    active = int(pd.to_numeric(mask, errors="coerce").fillna(0.0).astype(int).sum())
    return {"active_days": active, "coverage": float(active / max(len(mask), 1))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Follow-up operacional dos sinais estruturais por ano e dias de ataque.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--crypto-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-meta", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-meta", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--observer-mode", choices=["major8", "all22"], default="all22")
    ap.add_argument("--horizon-days", type=int, default=21)
    ap.add_argument("--null-permutations", type=int, default=16)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--turn-window", type=int, default=5)
    ap.add_argument("--operational-thresholds", default="0.6,0.7,0.8")
    ap.add_argument("--operational-min-run", type=int, default=2)
    ap.add_argument("--operational-cooldown", type=int, default=3)
    ap.add_argument("--outdir-root", default="results/validation/profit_structural_operational_followup")
    ap.add_argument("--cache-root", default="results/cache/profit_structural_extraction_audit")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()
    crypto_groups = (ROOT / args.crypto_groups).resolve()
    crypto_meta = (ROOT / args.crypto_meta).resolve()
    equity_groups = (ROOT / args.equity_groups).resolve()
    equity_meta = (ROOT / args.equity_meta).resolve()
    cache_root = (
        ROOT / args.cache_root / f"observer_{str(args.observer_mode)}__h{int(args.horizon_days)}__v2"
    ).resolve()
    followup_cache_root = cache_root / "operational_followup_v1"

    signal_frame = _load_cached_signal_frame(cache_root)
    if signal_frame is None:
        raise SystemExit("signal_frame cache not found; run profit_structural_extraction_audit first")

    cached_target_bundle = _load_cached_target_events(cache_root)
    if cached_target_bundle is None:
        targets = _target_catalog(prices_dir)
        target_events, target_event_summary = _build_target_event_matrix(targets, horizon=int(args.horizon_days))
        _save_target_events(cache_root, target_events, target_event_summary)
    else:
        target_events, target_event_summary = cached_target_bundle
        targets = _target_catalog(prices_dir)
    target_future = _target_future_catalog(targets, horizon=int(args.horizon_days))

    attack_cache_path = followup_cache_root / "attack_mask.csv"
    attack_mask = _load_cached_attack_mask(attack_cache_path)
    if attack_mask is None:
        attack_mask = _load_attack_mask_from_historical_driver(ROOT)
    if attack_mask is None:
        official = _build_official_for_observer(
            prices_dir=prices_dir,
            crypto_groups=crypto_groups,
            crypto_meta=crypto_meta,
            equity_groups=equity_groups,
            equity_meta=equity_meta,
            benchmark_crypto=str(args.benchmark_crypto),
            benchmark_equity=str(args.benchmark_equity),
            observer_mode=str(args.observer_mode),
        )
        attack_source = official["official_attack"].source.reindex(signal_frame.index).ffill()
        attack_mask = (attack_source.astype(str).str.lower() == "attack").astype(int)
    if attack_mask is not None and not attack_mask.empty:
        _save_cached_attack_mask(attack_cache_path, attack_mask)
    attack_mask = pd.to_numeric(attack_mask.reindex(signal_frame.index), errors="coerce").fillna(0.0).astype(int)
    subset_masks = _build_subset_masks(signal_frame.index, attack_mask)
    thresholds = tuple(float(x.strip()) for x in str(args.operational_thresholds).split(",") if x.strip())

    real_cache_path = followup_cache_root / "real_rows.json"
    real_rows = _load_cached_rows(real_cache_path)
    if real_rows is None:
        real_rows = []
        for subset_name, subset_mask in subset_masks.items():
            real_rows.extend(
                _operational_target_rows(
                    signal_frame,
                    target_events,
                    target_future,
                    thresholds=thresholds,
                    min_run=int(args.operational_min_run),
                    cooldown=int(args.operational_cooldown),
                    turn_window=int(args.turn_window),
                    date_mask=subset_mask,
                    subset_name=subset_name,
                    permutation_id=None,
                )
            )
        _save_cached_rows(real_cache_path, real_rows)
    _write_csv(outdir / "subset_operational_real.csv", real_rows)

    null_rows: list[dict[str, Any]] = []
    for permutation_id in range(int(args.null_permutations)):
        perm_cache_path = followup_cache_root / "null_rows" / f"perm_{int(permutation_id):03d}.json"
        cached_perm_rows = _load_cached_rows(perm_cache_path)
        if cached_perm_rows is None:
            permuted = _joint_signal_permutation(signal_frame, seed=int(args.seed) + permutation_id)
            cached_perm_rows = []
            for subset_name, subset_mask in subset_masks.items():
                cached_perm_rows.extend(
                    _operational_target_rows(
                        permuted,
                        target_events,
                        target_future,
                        thresholds=thresholds,
                        min_run=int(args.operational_min_run),
                        cooldown=int(args.operational_cooldown),
                        turn_window=int(args.turn_window),
                        date_mask=subset_mask,
                        subset_name=subset_name,
                        permutation_id=int(permutation_id),
                    )
                )
            _save_cached_rows(perm_cache_path, cached_perm_rows)
        null_rows.extend(cached_perm_rows)
    _write_csv(outdir / "subset_operational_null.csv", null_rows)

    rollup_rows: list[dict[str, Any]] = []
    real_frame = pd.DataFrame(real_rows)
    null_frame = pd.DataFrame(null_rows)
    if null_frame.empty:
        null_frame = pd.DataFrame(columns=["subset", "signal", "threshold", "permutation_id"])
    subset_summary_rows: list[dict[str, Any]] = []
    for subset_name, subset_mask in subset_masks.items():
        sub_real = real_frame[real_frame["subset"] == subset_name].to_dict(orient="records")
        sub_null = null_frame[null_frame["subset"] == subset_name].to_dict(orient="records")
        sub_rollup = _operational_rollup(sub_real, sub_null)
        for row in sub_rollup:
            row["subset"] = str(subset_name)
            rollup_rows.append(row)
        stats = _subset_stats(subset_mask)
        best_row = max(sub_rollup, key=lambda row: _safe_float(row.get("operational_score"), -1.0), default={})
        subset_summary_rows.append(
            {
                "subset": str(subset_name),
                "active_days": int(stats["active_days"]),
                "coverage": float(stats["coverage"]),
                "best_signal": str(best_row.get("signal") or ""),
                "best_threshold": _safe_float(best_row.get("threshold"), None),
                "best_score": int(_safe_float(best_row.get("operational_score"), 0)),
                "best_decision": str(best_row.get("operational_decision") or "cut"),
                "keep_count": int(sum(str(row.get("operational_decision")) == "keep" for row in sub_rollup)),
                "recalibrate_count": int(sum(str(row.get("operational_decision")) == "recalibrate" for row in sub_rollup)),
            }
        )
    _write_csv(outdir / "subset_operational_rollup.csv", rollup_rows)
    _write_csv(outdir / "subset_summary.csv", subset_summary_rows)

    best_all = max((row for row in subset_summary_rows if row["subset"] == "all"), key=lambda row: row["best_score"], default={})
    best_holdout = max(
        (row for row in subset_summary_rows if str(row["subset"]).startswith("holdout_")),
        key=lambda row: row["best_score"],
        default={},
    )
    best_attack = max(
        (row for row in subset_summary_rows if str(row["subset"]).startswith("attack")),
        key=lambda row: row["best_score"],
        default={},
    )
    summary = {
        "suite": "profit_structural_operational_followup",
        "generated_at": datetime.now(UTC).isoformat(),
        "observer_mode": str(args.observer_mode),
        "horizon_days": int(args.horizon_days),
        "null_permutations": int(args.null_permutations),
        "attack_days": int(attack_mask.sum()),
        "attack_coverage": float(attack_mask.mean()) if len(attack_mask) else 0.0,
        "best_all": best_all,
        "best_holdout": best_holdout,
        "best_attack_subset": best_attack,
        "subset_summary": subset_summary_rows,
        "verdict": {
            "improves_in_holdout": any(row["keep_count"] > 0 or row["recalibrate_count"] > 0 for row in subset_summary_rows if str(row["subset"]).startswith("holdout_")),
            "improves_in_attack_only": any(row["keep_count"] > 0 or row["recalibrate_count"] > 0 for row in subset_summary_rows if str(row["subset"]).startswith("attack")),
            "notes": [
                "Os subsets holdout_YYYY avaliam cada ano isoladamente, sem misturar 2023, 2024 e 2025.",
                "Os subsets attack_only/attack_YYYY filtram apenas os dias em que o modo official_attack estaria em ATTACK.",
                "A comparação continua contra nulo por permutação conjunta dos sinais estruturais.",
            ],
        },
        "artifacts": {
            "subset_operational_real_csv": str(outdir / "subset_operational_real.csv"),
            "subset_operational_null_csv": str(outdir / "subset_operational_null.csv"),
            "subset_operational_rollup_csv": str(outdir / "subset_operational_rollup.csv"),
            "subset_summary_csv": str(outdir / "subset_summary.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
