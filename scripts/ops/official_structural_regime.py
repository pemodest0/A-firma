from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _pointer_run_dirs(root: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []

    latest_release = _read_json(root / "results" / "lab_corr_macro" / "latest_release.json")
    run_dir = Path(str(latest_release.get("run_dir") or "").strip()) if str(latest_release.get("run_dir") or "").strip() else None
    if run_dir is not None:
        candidates.append(("results_lab_corr_latest_release", run_dir))

    public_release = _read_json(root / "website-ui" / "public" / "data" / "lab_corr_macro" / "latest" / "latest_release.json")
    run_dir = Path(str(public_release.get("run_dir") or "").strip()) if str(public_release.get("run_dir") or "").strip() else None
    if run_dir is not None:
        candidates.append(("public_lab_corr_latest_release", run_dir))

    finance_ready = _read_json(root / "results" / "ops" / "finance_product_ready" / "latest_finance_product_ready.json")
    run_dir = Path(str(finance_ready.get("run_dir") or "").strip()) if str(finance_ready.get("run_dir") or "").strip() else None
    if run_dir is not None:
        candidates.append(("finance_product_ready", run_dir))

    return candidates


def _fallback_run_dirs(root: Path) -> list[tuple[str, Path]]:
    base = root / "results" / "lab_corr_macro"
    if not base.exists():
        return []
    return [("results_lab_corr_scan", run_dir) for run_dir in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)]


def _run_dir_order_key(item: tuple[str, Path]) -> tuple[int, str]:
    source, run_dir = item
    priority = {
        "finance_product_ready": 0,
        "results_lab_corr_latest_release": 1,
        "public_lab_corr_latest_release": 2,
        "results_lab_corr_scan": 3,
    }.get(source, 9)
    return (-int("".join(ch for ch in str(run_dir.name) if ch.isdigit()) or "0"), priority)


def _regime_series_from_csv(regime_csv: Path) -> pd.Series:
    df = pd.read_csv(regime_csv, usecols=["date", "regime"])
    if df.empty:
        return pd.Series(dtype=object)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["regime"] = df["regime"].astype(str).str.lower()
    df = df.dropna(subset=["date", "regime"])
    if df.empty:
        return pd.Series(dtype=object)
    return df.drop_duplicates(subset=["date"], keep="last").set_index("date")["regime"].sort_index().astype(object)


def _resolve_regime_csv(run_dir: Path, *, official_window: int) -> Path | None:
    direct = run_dir / f"regime_series_T{int(official_window)}.csv"
    if direct.exists():
        return direct
    candidates = sorted(run_dir.glob("regime_series_T*.csv"))
    if candidates:
        return candidates[-1]
    return None


def _live_structural_regime_payload(root: Path) -> dict[str, Any]:
    payload = _read_json(root / "results" / "ops" / "official_structural_regime" / "latest_structural_regime.json")
    return payload if isinstance(payload, dict) else {}


def load_official_structural_regime_series(
    root: Path,
    *,
    official_window: int = 120,
) -> tuple[pd.Series, dict[str, Any]]:
    candidates: list[tuple[pd.Timestamp, int, str, Path, Path, pd.Series]] = []
    pointer_candidates = sorted(_pointer_run_dirs(root), key=_run_dir_order_key)
    for source, run_dir in pointer_candidates + _fallback_run_dirs(root):
        if not run_dir.exists():
            continue
        regime_csv = _resolve_regime_csv(run_dir, official_window=official_window)
        if regime_csv is None or not regime_csv.exists():
            continue
        try:
            series = _regime_series_from_csv(regime_csv)
        except Exception:
            continue
        if series.empty:
            continue
        max_date = pd.to_datetime(series.index.max(), errors="coerce")
        if pd.isna(max_date):
            continue
        priority = {
            "finance_product_ready": 0,
            "results_lab_corr_latest_release": 1,
            "public_lab_corr_latest_release": 2,
            "results_lab_corr_scan": 3,
        }.get(source, 9)
        candidates.append((max_date, -priority, source, run_dir, regime_csv, series))

    if not candidates:
        return pd.Series(dtype=object), {"source": "missing", "official_window": int(official_window)}

    max_date, _, source, run_dir, regime_csv, series = sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True)[0]
    meta: dict[str, Any] = {
        "source": source,
        "run_dir": str(run_dir),
        "regime_csv": str(regime_csv),
        "official_window": int(official_window),
        "base_series_end_date": str(max_date.date()),
    }
    live_payload = _live_structural_regime_payload(root)
    live_date_raw = str(live_payload.get("as_of_date") or "").strip()
    live_regime = str(live_payload.get("regime") or "").strip().lower()
    live_date = pd.to_datetime(live_date_raw, errors="coerce") if live_date_raw else pd.NaT
    if live_regime and not pd.isna(live_date) and live_date.normalize() > max_date.normalize():
        extension_index = pd.date_range(start=max_date + pd.Timedelta(days=1), end=live_date.normalize(), freq="D")
        if len(extension_index):
            extension = pd.Series(live_regime, index=extension_index, dtype=object)
            series = pd.concat([series, extension]).sort_index()
            meta.update(
                {
                    "live_extension_source": "results/ops/official_structural_regime/latest_structural_regime.json",
                    "live_extension_end_date": str(live_date.date()),
                    "live_extension_regime": live_regime,
                }
            )
    return series.astype(object), meta
