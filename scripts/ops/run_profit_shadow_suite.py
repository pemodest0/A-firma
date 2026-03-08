#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execution.returns import load_return_series_csv

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
RUN_ID_RE = re.compile(r"^\d{8}T.*$")
LOCAL_PACK_RE = re.compile(r"^local_pack_\d{8}T\d{6}Z$")
MACRO_CACHE_FILE = "_profit_shadow_macro_cache.json"


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return out


def _sanitize_json_value(x: Any) -> Any:
    if isinstance(x, float):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _sanitize_json_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_json_value(v) for v in x]
    return x


def _run(cmd: list[str], *, cwd: Path, timeout_sec: float) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_sec)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, out, (err + f"\ntimeout_after_{int(timeout_sec)}s").strip()


def _extract_last_json_line(text: str) -> dict[str, Any]:
    for line in reversed(str(text or "").splitlines()):
        s = line.strip()
        if not s.startswith("{") or not s.endswith("}"):
            continue
        try:
            payload = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _find_latest_timestamped_run(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = sorted((p for p in root.iterdir() if p.is_dir() and RUN_ID_RE.match(p.name)), key=lambda p: p.name)
    return runs[-1] if runs else None


def _find_latest_local_pack(root: Path) -> Path | None:
    if not root.exists():
        return None
    packs = sorted((p for p in root.iterdir() if p.is_dir() and LOCAL_PACK_RE.match(p.name)), key=lambda p: p.name)
    return packs[-1] if packs else None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_sanitize_json_value(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _macro_cache_index_path(root: Path) -> Path:
    return root / MACRO_CACHE_FILE


def _load_macro_cache(root: Path) -> dict[str, Any]:
    payload = _read_json(_macro_cache_index_path(root))
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
    return {"entries": entries}


def _write_macro_cache(root: Path, payload: dict[str, Any]) -> None:
    data = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries": payload.get("entries", {}) if isinstance(payload.get("entries", {}), dict) else {},
    }
    _write_json(_macro_cache_index_path(root), data)


def _macro_cache_key(
    *,
    finance_pack_dir: Path,
    lab_policy_path: Path,
    asset_metadata_csv: Path,
    cfg: dict[str, Any],
) -> str:
    policy_text = ""
    try:
        policy_text = lab_policy_path.read_text(encoding="utf-8")
    except OSError:
        policy_text = str(lab_policy_path)
    build_meta = _read_json(finance_pack_dir / "build_meta.json")
    payload = {
        "finance_pack_dir": str(finance_pack_dir.resolve()),
        "finance_pack_meta": build_meta,
        "lab_policy_path": str(lab_policy_path.resolve()),
        "lab_policy_hash": _sha256_text(policy_text),
        "asset_metadata_csv": str(asset_metadata_csv.resolve()),
        "macro_args": {
            "max_core_assets": _safe_int(cfg.get("max_core_assets", 300), 300),
            "macro_windows": str(cfg.get("macro_windows", "120")),
            "official_window": _safe_int(cfg.get("official_window", 120), 120),
            "noise_step": _safe_int(cfg.get("noise_step", 10), 10),
            "overlap_step": _safe_int(cfg.get("overlap_step", 5), 5),
            "enable_hierarchical": bool(cfg.get("enable_hierarchical", True)),
            "save_v1": bool(cfg.get("save_v1", True)),
            "n_global": _safe_int(cfg.get("n_global", 240), 240),
            "n_sector": _safe_int(cfg.get("n_sector", 70), 70),
            "min_coverage_global": _safe_float(cfg.get("min_coverage_global", 0.90), 0.90),
            "min_coverage_sector": _safe_float(cfg.get("min_coverage_sector", 0.80), 0.80),
        },
    }
    return _sha256_text(_canonical_json(payload))


def _find_macro_cache_hit(root: Path, key: str) -> Path | None:
    cache = _load_macro_cache(root)
    entry = cache.get("entries", {}).get(str(key), {})
    if not isinstance(entry, dict):
        return None
    run_dir = _resolve_path(entry.get("run_dir"))
    if run_dir is None or not run_dir.exists() or not _macro_run_complete(run_dir):
        return None
    return run_dir


def _register_macro_cache(root: Path, key: str, run_dir: Path) -> None:
    if not _macro_run_complete(run_dir):
        return
    cache = _load_macro_cache(root)
    entries = cache.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
    entries[str(key)] = {
        "run_dir": str(run_dir.resolve()),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_macro_cache(root, {"entries": entries})


def _has_required_files(base: Path, names: list[str]) -> bool:
    return all((base / name).exists() for name in names)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    safe = _sanitize_json_value(payload)
    _write_json(path, safe if isinstance(safe, dict) else {})


def _finance_pack_matches(
    *,
    pack_dir: Path,
    prices_dir: Path,
    asset_groups_csv: Path,
    business_days_only: int,
    min_rows: int,
    min_date_coverage: float,
) -> bool:
    meta = _read_json(pack_dir / "build_meta.json")
    if not meta:
        return False
    if not _has_required_files(pack_dir, ["panel_long_sector.csv", "universe_fixed.csv", "build_meta.json"]):
        return False
    if str(meta.get("stored_return_kind", "")).strip().lower() != "simple":
        return False
    prices_meta = str(meta.get("prices_dir", "")).strip()
    groups_meta = str(meta.get("asset_groups", "")).strip()
    if prices_meta and _resolve_path(prices_meta) != prices_dir.resolve():
        return False
    if groups_meta and _resolve_path(groups_meta) != asset_groups_csv.resolve():
        return False
    if int(bool(meta.get("business_days_only", False))) != int(bool(business_days_only)):
        return False
    if _safe_int(meta.get("min_rows"), -1) != int(min_rows):
        return False
    if not np.isclose(_safe_float(meta.get("min_date_coverage"), float("nan")), float(min_date_coverage), atol=1e-9, rtol=0.0):
        return False
    return True


def _macro_run_complete(run_dir: Path) -> bool:
    return _has_required_files(
        run_dir,
        [
            "summary.json",
            "returns_wide_core.csv",
            "universe_core.csv",
            "universe_core_by_sector.csv",
        ],
    )


def _impact_complete(outdir: Path) -> bool:
    return _has_required_files(outdir, ["impact_training_dataset.csv", "impact_summary.json"])


def _profile_complete(profile_dir: Path) -> bool:
    return _has_required_files(
        profile_dir,
        [
            "monthly_systematic_eval.csv",
            "simulation_summary.json",
            "systematic_summary.json",
        ],
    )


def _reuse_profile_row(
    *,
    profile_name: str,
    profile_description: str,
    profile_dir: Path,
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_returns: pd.Series | None,
    initial_capital: float,
) -> dict[str, Any]:
    monthly_path = profile_dir / "monthly_systematic_eval.csv"
    sim_summary_path = profile_dir / "simulation_summary.json"
    sys_summary_path = profile_dir / "systematic_summary.json"
    daily_history_path = profile_dir / "daily_replay.csv"
    daily_summary_path = profile_dir / "daily_replay_summary.json"

    monthly_eval = pd.read_csv(monthly_path)
    daily_history = build_daily_replay(
        monthly_eval=monthly_eval,
        returns_wide=returns_wide,
        benchmark_symbol=benchmark_symbol,
        benchmark_returns=benchmark_returns,
        initial_capital=initial_capital,
    )
    daily_history.to_csv(daily_history_path, index=False)
    daily_summary = summarize_daily_replay(daily_history)
    _write_json(daily_summary_path, daily_summary if isinstance(daily_summary, dict) else {})

    sim_summary = _read_json(sim_summary_path)
    sys_summary = _read_json(sys_summary_path)
    latest_row = monthly_eval.tail(1).to_dict(orient="records")[0] if not monthly_eval.empty else {}
    return _sanitize_json_value(
        {
            "profile": profile_name,
            "description": profile_description,
            "run_dir": str(profile_dir),
            "status": "ok",
            "daily_total_return": _safe_float((daily_summary.get("portfolio") or {}).get("total_return")),
            "daily_ann_return": _safe_float((daily_summary.get("portfolio") or {}).get("ann_return")),
            "daily_sharpe": _safe_float((daily_summary.get("portfolio") or {}).get("sharpe")),
            "daily_max_drawdown": _safe_float((daily_summary.get("portfolio") or {}).get("max_drawdown")),
            "daily_edge_vs_benchmark": _safe_float(daily_summary.get("edge_vs_benchmark_total_return")),
            "systematic_worth_it_rate_vs_eqw": _safe_float(sys_summary.get("worth_it_rate_vs_eqw")),
            "systematic_monthly_alpha_prob_positive_vs_eqw": _safe_float(sys_summary.get("monthly_alpha_prob_positive_vs_eqw")),
            "systematic_strategy_max_drop": _safe_float(sys_summary.get("strategy_max_drop")),
            "latest_signal": latest_row,
            "best_params": sim_summary.get("best_params", {}),
            "artifacts": {
                "monthly_eval_csv": str(monthly_path),
                "daily_replay_csv": str(daily_history_path),
                "daily_replay_summary_json": str(daily_summary_path),
                "simulation_summary_json": str(sim_summary_path),
                "systematic_summary_json": str(sys_summary_path),
                "latest_allocation_weights_csv": str(profile_dir / "latest_allocation_weights.csv"),
            },
        }
    )


def _perf_from_simple_returns(simple_returns: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(simple_returns, errors="coerce").dropna().astype(float)
    if x.empty:
        return {
            "total_return": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "positive_days_share": float("nan"),
        }
    eq = (1.0 + x).clip(lower=1e-9, upper=10.0).cumprod()
    ann_return = float(np.power(float(eq.iloc[-1]), 252.0 / max(int(x.shape[0]), 1)) - 1.0)
    ann_vol = float(x.std(ddof=0) * np.sqrt(252.0))
    drawdown = float((eq / eq.cummax() - 1.0).min())
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(ann_return / ann_vol) if ann_vol > 1e-12 else float("nan"),
        "max_drawdown": drawdown,
        "positive_days_share": float((x > 0).mean()),
    }


def _load_returns_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"returns_wide missing date column: {path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    asset_cols = [c for c in df.columns if c != "date"]
    for col in asset_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.set_index("date")[asset_cols].astype(float).sort_index()


def _load_price_returns(prices_dir: Path, ticker: str) -> pd.Series:
    path = prices_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing benchmark price returns: {path}")
    out = load_return_series_csv(path, source_kind="log", target_kind="simple", series_name=ticker)
    if out.empty:
        raise ValueError(f"empty benchmark price returns after cleaning: {path}")
    return out.astype(float).sort_index()


def _resolve_benchmark_series(
    *,
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_returns: pd.Series | None = None,
) -> pd.Series:
    daily = returns_wide.copy().sort_index()
    if benchmark_returns is not None:
        bench = pd.Series(benchmark_returns, copy=True)
        bench.index = pd.to_datetime(bench.index, errors="coerce")
        bench = pd.to_numeric(bench, errors="coerce")
        bench = bench[bench.index.notna()].sort_index().groupby(level=0).last()
        aligned = bench.reindex(daily.index)
        if not aligned.notna().any():
            raise ValueError(f"benchmark series {benchmark_symbol} does not overlap returns_wide index")
        return aligned.fillna(0.0).astype(float)
    if benchmark_symbol in daily.columns:
        return pd.to_numeric(daily[benchmark_symbol], errors="coerce").fillna(0.0).astype(float)
    raise ValueError(
        f"benchmark symbol {benchmark_symbol} not found in returns_wide and no external benchmark series was provided"
    )


def _arg_name(name: str) -> str:
    return "--" + str(name).strip().replace("_", "-")


def _json_weight_map(raw: Any) -> dict[str, float]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        w = _safe_float(value)
        if np.isfinite(w) and abs(w) > 1e-14:
            out[str(key)] = float(w)
    return out


def build_daily_replay(
    *,
    monthly_eval: pd.DataFrame,
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_returns: pd.Series | None = None,
    initial_capital: float,
) -> pd.DataFrame:
    if monthly_eval.empty or returns_wide.empty:
        return pd.DataFrame()
    daily = returns_wide.copy().sort_index()
    benchmark = _resolve_benchmark_series(
        returns_wide=daily,
        benchmark_symbol=benchmark_symbol,
        benchmark_returns=benchmark_returns,
    )
    month_rows = monthly_eval.copy()
    month_rows["ym"] = month_rows["ym"].astype(str)
    month_rows = month_rows.drop_duplicates(subset=["ym"], keep="last").sort_values("ym").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    capital = float(initial_capital)
    benchmark_capital = float(initial_capital)
    for _, month_row in month_rows.iterrows():
        ym = str(month_row["ym"])
        period = pd.Period(ym, freq="M")
        month_mask = daily.index.to_period("M") == period
        if not bool(month_mask.any()):
            continue
        weights = _json_weight_map(month_row.get("executed_weights_json", "{}"))
        cash_weight = _safe_float(month_row.get("cash_weight"), 0.0)
        hedge_weight = _safe_float(month_row.get("hedge_weight"), 0.0)
        selected_assets = str(month_row.get("executed_assets", "")).strip()
        risk_bucket = str(month_row.get("risk_bucket", "")).strip()
        for dt, ret_row in daily.loc[month_mask].iterrows():
            core_ret = 0.0
            for asset_id, weight in weights.items():
                if asset_id in ret_row.index:
                    core_ret += float(weight) * float(ret_row[asset_id])
            bench_ret = float(benchmark.loc[dt])
            day_ret = float(core_ret + hedge_weight * bench_ret + cash_weight * 0.0)
            capital *= 1.0 + day_ret
            benchmark_capital *= 1.0 + bench_ret
            rows.append(
                {
                    "date": dt.date().isoformat(),
                    "ym": ym,
                    "risk_bucket": risk_bucket,
                    "selected_assets": selected_assets,
                    "n_assets": int(len(weights)),
                    "cash_weight": float(cash_weight),
                    "hedge_weight": float(hedge_weight),
                    "gross_exposure": float(sum(abs(float(v)) for v in weights.values()) + abs(float(hedge_weight))),
                    "net_exposure": float(sum(float(v) for v in weights.values()) + float(hedge_weight)),
                    "portfolio_return": float(day_ret),
                    "benchmark_return": float(bench_ret),
                    "capital": float(capital),
                    "benchmark_capital": float(benchmark_capital),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["capital_peak"] = pd.to_numeric(out["capital"], errors="coerce").cummax()
    out["drawdown"] = pd.to_numeric(out["capital"], errors="coerce") / out["capital_peak"] - 1.0
    return out


def summarize_daily_replay(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {"status": "empty"}
    perf = _perf_from_simple_returns(pd.to_numeric(history["portfolio_return"], errors="coerce"))
    bench = _perf_from_simple_returns(pd.to_numeric(history["benchmark_return"], errors="coerce"))
    last = history.iloc[-1]
    return _sanitize_json_value(
        {
            "status": "ok",
            "start_date": str(history.iloc[0]["date"]),
            "end_date": str(last["date"]),
            "n_days": int(history.shape[0]),
            "capital_start": float(pd.to_numeric(history["capital"], errors="coerce").iloc[0] / max(1.0 + float(pd.to_numeric(history["portfolio_return"], errors="coerce").iloc[0]), 1e-9)),
            "capital_end": float(last["capital"]),
            "portfolio": perf,
            "benchmark": bench,
            "edge_vs_benchmark_total_return": _safe_float(perf.get("total_return")) - _safe_float(bench.get("total_return")),
            "latest_risk_bucket": str(last.get("risk_bucket", "")),
            "latest_n_assets": int(_safe_int(last.get("n_assets"), 0)),
            "latest_gross_exposure": _safe_float(last.get("gross_exposure")),
            "latest_net_exposure": _safe_float(last.get("net_exposure")),
        }
    )


def _total_return(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").fillna(0.0).astype(float)
    if x.empty:
        return float("nan")
    return float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0)


def _yearly_eval_from_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame(columns=["year", "strategy_total", "eqw_total", "market_total", "motor_total"])
    d = monthly.copy()
    if "year" not in d.columns:
        d["year"] = d["ym"].astype(str).str[:4].astype(int)
    out = (
        d.groupby("year", as_index=False)
        .agg(
            strategy_total=("ret", _total_return),
            eqw_total=("eqw_ret", _total_return),
            market_total=("mkt_ret", _total_return),
            motor_total=("motor_ret", _total_return),
        )
        .copy()
    )
    out["alpha_total_vs_eqw"] = out["strategy_total"] - out["eqw_total"]
    out["alpha_total_vs_market"] = out["strategy_total"] - out["market_total"]
    out["worth_it_vs_eqw"] = out["strategy_total"] > out["eqw_total"]
    out["worth_it_vs_market"] = out["strategy_total"] > out["market_total"]
    return out


def _combine_weight_maps(weight_maps: list[dict[str, float]], vote_threshold: float) -> dict[str, float]:
    if not weight_maps:
        return {}
    n_maps = len(weight_maps)
    threshold_count = int(max(1, np.ceil(float(max(0.0, min(1.0, vote_threshold))) * float(n_maps))))
    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    for wm in weight_maps:
        for asset, weight in wm.items():
            asset_key = str(asset)
            counts[asset_key] = counts.get(asset_key, 0) + 1
            sums[asset_key] = sums.get(asset_key, 0.0) + float(weight)
    kept = {asset: sums[asset] / float(n_maps) for asset, cnt in counts.items() if cnt >= threshold_count}
    if not kept:
        kept = {asset: sums[asset] / float(n_maps) for asset in sums}
    kept = {asset: float(weight) for asset, weight in kept.items() if np.isfinite(weight) and abs(weight) > 1e-14}
    gross = float(sum(max(0.0, w) for w in kept.values()))
    if gross <= 0.0:
        return {}
    return {asset: float(max(0.0, weight) / gross) for asset, weight in kept.items() if max(0.0, weight) > 1e-14}


def _ensemble_risk_bucket(values: list[str]) -> str:
    items = [str(v).strip().lower() for v in values if str(v).strip()]
    if not items:
        return "stable"
    counts = pd.Series(items).value_counts()
    return str(counts.index[0]) if not counts.empty else "stable"


def _build_ensemble_monthly_eval(member_monthlies: list[pd.DataFrame], vote_threshold: float) -> pd.DataFrame:
    valid = [m.copy() for m in member_monthlies if isinstance(m, pd.DataFrame) and not m.empty and "ym" in m.columns]
    if not valid:
        return pd.DataFrame()
    common_months = sorted(set(valid[0]["ym"].astype(str).tolist()).intersection(*[set(m["ym"].astype(str).tolist()) for m in valid[1:]]))
    if not common_months:
        return pd.DataFrame()
    parts = []
    for monthly in valid:
        part = monthly.copy()
        part["ym"] = part["ym"].astype(str)
        parts.append(part.set_index("ym", drop=False))
    rows: list[dict[str, Any]] = []
    numeric_mean_cols = [
        "ret",
        "eqw_ret",
        "mkt_ret",
        "motor_ret",
        "risk_budget",
        "risk_budget_trade",
        "turnover",
        "turnover_target",
        "cash_weight",
        "hedge_weight",
        "core_gross_exposure",
        "net_exposure",
        "effective_top_k",
    ]
    bool_mean_cols = [
        "rebalance_executed",
        "defense_active",
        "weekly_stress_active",
        "dd_guard_active",
        "auto_aggressive_active",
    ]
    for ym in common_months:
        member_rows = [p.loc[ym] for p in parts if ym in p.index]
        if not member_rows:
            continue
        weight_maps = [_json_weight_map(r.get("executed_weights_json", "{}")) for r in member_rows]
        merged_weights = _combine_weight_maps(weight_maps, vote_threshold=float(vote_threshold))
        merged_assets = ",".join(sorted(merged_weights.keys()))
        row: dict[str, Any] = {
            "ym": ym,
            "risk_bucket": _ensemble_risk_bucket([r.get("risk_bucket", "") for r in member_rows]),
            "selected_assets": merged_assets,
            "executed_assets": merged_assets,
            "executed_weights_json": json.dumps(merged_weights, ensure_ascii=False, sort_keys=True),
            "n_selected": int(len(merged_weights)),
            "ensemble_members": int(len(member_rows)),
        }
        for col in numeric_mean_cols:
            vals = [_safe_float(r.get(col)) for r in member_rows]
            vals = [v for v in vals if np.isfinite(v)]
            row[col] = float(np.mean(vals)) if vals else float("nan")
        for col in bool_mean_cols:
            vals = [bool(r.get(col)) for r in member_rows if r.get(col) is not None]
            row[col] = bool(np.mean(vals) >= 0.5) if vals else False
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["year"] = out["ym"].astype(str).str[:4].astype(int)
    return out.sort_values("ym").reset_index(drop=True)


def _write_ensemble_profile_artifacts(
    *,
    profile_dir: Path,
    profile_name: str,
    profile_description: str,
    member_profile_dirs: list[Path],
    member_profile_names: list[str],
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_returns: pd.Series | None,
    initial_capital: float,
    vote_threshold: float,
) -> None:
    member_monthlies = [pd.read_csv(p / "monthly_systematic_eval.csv") for p in member_profile_dirs]
    monthly_eval = _build_ensemble_monthly_eval(member_monthlies, vote_threshold=float(vote_threshold))
    if monthly_eval.empty:
        raise RuntimeError(f"ensemble profile {profile_name} has no overlapping monthly data")
    profile_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = profile_dir / "monthly_systematic_eval.csv"
    monthly_eval.to_csv(monthly_path, index=False)
    yearly_eval = _yearly_eval_from_monthly(monthly_eval)
    yearly_path = profile_dir / "yearly_systematic_eval.csv"
    yearly_eval.to_csv(yearly_path, index=False)
    subprocess.run(
        [PY, "scripts/ops/rebuild_systematic_summary.py", "--yearly-dir", str(profile_dir)],
        check=True,
        cwd=ROOT,
    )
    daily_history = build_daily_replay(
        monthly_eval=monthly_eval,
        returns_wide=returns_wide,
        benchmark_symbol=benchmark_symbol,
        benchmark_returns=benchmark_returns,
        initial_capital=initial_capital,
    )
    daily_history.to_csv(profile_dir / "daily_replay.csv", index=False)
    daily_summary = summarize_daily_replay(daily_history)
    _write_json(profile_dir / "daily_replay_summary.json", daily_summary if isinstance(daily_summary, dict) else {})
    latest_alloc_rows: list[dict[str, Any]] = []
    latest_row = monthly_eval.tail(1).to_dict(orient="records")[0] if not monthly_eval.empty else {}
    latest_weights = _json_weight_map(latest_row.get("executed_weights_json", "{}"))
    for asset_id, weight in sorted(latest_weights.items()):
        latest_alloc_rows.append(
            {
                "ym": str(latest_row.get("ym", "")),
                "asset_id": str(asset_id),
                "weight": float(weight),
                "cash_weight": float(_safe_float(latest_row.get("cash_weight"), 0.0)),
                "hedge_weight": float(_safe_float(latest_row.get("hedge_weight"), 0.0)),
                "risk_bucket": str(latest_row.get("risk_bucket", "")),
            }
        )
    pd.DataFrame(latest_alloc_rows).to_csv(profile_dir / "latest_allocation_weights.csv", index=False)
    sim_summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execution_universe_ensemble",
        "profile": profile_name,
        "description": profile_description,
        "ensemble_vote_threshold": float(vote_threshold),
        "ensemble_members": member_profile_names,
        "months_evaluated": int(monthly_eval.shape[0]),
        "best_params": {
            "mode": "ensemble",
            "members": member_profile_names,
            "vote_threshold": float(vote_threshold),
        },
    }
    _write_json(profile_dir / "simulation_summary.json", sim_summary)


def _load_profit_config(path: Path) -> dict[str, Any]:
    cfg = _read_json(path)
    if not cfg:
        raise FileNotFoundError(f"profit shadow config not found or invalid: {path}")
    return cfg


def _common_profile_cmd(
    *,
    impact_dir: Path,
    returns_csv: Path,
    prices_dir: Path,
    outdir: Path,
    benchmark_symbol: str,
    train_end: str,
    start_ym: str,
    max_assets_per_month: int,
    shadow_tail_months: int,
    profile_args: dict[str, Any],
) -> list[str]:
    cmd = [
        PY,
        "scripts/ops/run_canonical_systematic_eval.py",
        "--impact-dir",
        str(impact_dir),
        "--returns-csv",
        str(returns_csv),
        "--prices-dir",
        str(prices_dir),
        "--outdir",
        str(outdir),
        "--benchmark-symbol",
        str(benchmark_symbol),
        "--train-end",
        str(train_end),
        "--start-ym",
        str(start_ym),
        "--max-assets-per-month",
        str(int(max_assets_per_month)),
        "--shadow-tail-months",
        str(int(max(0, shadow_tail_months))),
    ]
    for key, value in profile_args.items():
        if value is None:
            continue
        cmd.extend([_arg_name(str(key)), str(value)])
    return cmd


def _normalized_execution_variants(profile: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, str]]:
    raw = profile.get("execution_universe_variants", [])
    out: list[dict[str, str]] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, str):
                text = str(item).strip()
                if text:
                    out.append(
                        {
                            "name": f"variant_{idx + 1}",
                            "execution_universe_csv": text,
                            "execution_liquidity_csv": str(cfg.get("execution_liquidity_csv", "")).strip(),
                        }
                    )
            elif isinstance(item, dict):
                csv_path = str(item.get("execution_universe_csv", "")).strip()
                if not csv_path:
                    continue
                out.append(
                    {
                        "name": str(item.get("name", f"variant_{idx + 1}")).strip() or f"variant_{idx + 1}",
                        "execution_universe_csv": csv_path,
                        "execution_liquidity_csv": str(item.get("execution_liquidity_csv", cfg.get("execution_liquidity_csv", ""))).strip(),
                    }
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Expanded-universe profit shadow suite with aggressive A/B profiles.")
    ap.add_argument("--config-path", type=str, default="config/profit_shadow_mode.json")
    ap.add_argument("--run-id", type=str, default=_run_id())
    ap.add_argument("--step-timeout-sec", type=float, default=7200.0)
    ap.add_argument("--resume", type=int, default=1, help="Reuse checkpointed stages and completed profile outputs when possible.")
    ap.add_argument("--finance-pack-dir", type=str, default="", help="Reuse an existing local finance pack dir and skip rebuilding it.")
    ap.add_argument("--macro-run-dir", type=str, default="", help="Reuse an existing macro run dir and skip rebuilding macro outputs.")
    ap.add_argument("--impact-dir", type=str, default="", help="Reuse an existing impact learning outdir and skip recomputing impact artifacts.")
    args = ap.parse_args()

    config_path = _resolve_path(args.config_path)
    if config_path is None:
        raise SystemExit("missing config path")
    cfg = _load_profit_config(config_path)

    asset_groups_csv = _resolve_path(cfg.get("asset_groups_csv"))
    asset_metadata_csv = _resolve_path(cfg.get("asset_metadata_csv"))
    prices_dir = _resolve_path(cfg.get("prices_dir"))
    finance_pack_out_root = _resolve_path(cfg.get("finance_pack_out_root"))
    lab_policy_path = _resolve_path(cfg.get("lab_policy_path"))
    lab_out_base = _resolve_path(cfg.get("lab_out_base"))
    impact_out_root = _resolve_path(cfg.get("impact_out_root"))
    shadow_outdir = _resolve_path(cfg.get("shadow_outdir"))
    if None in {asset_groups_csv, asset_metadata_csv, prices_dir, finance_pack_out_root, lab_policy_path, lab_out_base, impact_out_root, shadow_outdir}:
        raise SystemExit("profit shadow config has unresolved required paths")

    run_id = str(args.run_id).strip() or _run_id()
    run_dir = shadow_outdir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    profiles_root = run_dir / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = _read_checkpoint(checkpoint_path)
    resume_enabled = bool(int(args.resume))
    latest_summary_path = shadow_outdir / "latest_summary.json"
    latest_run_path = shadow_outdir / "latest_run.json"

    finance_pack_business_days_only = 1 if bool(cfg.get("finance_pack_business_days_only", True)) else 0
    finance_pack_min_rows = _safe_int(cfg.get("finance_pack_min_rows", 252), 252)
    finance_pack_min_date_coverage = _safe_float(cfg.get("finance_pack_min_date_coverage", 0.90), 0.90)
    reuse_latest_finance_pack = bool(cfg.get("reuse_latest_finance_pack", True))
    reuse_latest_macro = bool(cfg.get("reuse_latest_macro", True))
    reuse_latest_impact = bool(cfg.get("reuse_latest_impact", True))
    reuse_profile_outputs = bool(cfg.get("reuse_profile_outputs", True))

    finance_pack_dir = _resolve_path(args.finance_pack_dir)
    if finance_pack_dir is None and resume_enabled:
        finance_pack_dir = _resolve_path(checkpoint.get("finance_pack_dir"))
    if finance_pack_dir is None and reuse_latest_finance_pack:
        cand = _find_latest_local_pack(finance_pack_out_root)
        if cand is not None and _finance_pack_matches(
            pack_dir=cand,
            prices_dir=prices_dir,
            asset_groups_csv=asset_groups_csv,
            business_days_only=finance_pack_business_days_only,
            min_rows=finance_pack_min_rows,
            min_date_coverage=finance_pack_min_date_coverage,
        ):
            finance_pack_dir = cand.resolve()

    macro_code = 0
    macro_out = ""
    macro_err = ""
    macro_cache_key = ""
    macro_run_dir = _resolve_path(args.macro_run_dir)
    if macro_run_dir is None and resume_enabled:
        macro_run_dir = _resolve_path(checkpoint.get("macro_run_dir"))

    if macro_run_dir is None or (not macro_run_dir.exists()) or not _macro_run_complete(macro_run_dir):
        if finance_pack_dir is None or not _finance_pack_matches(
            pack_dir=finance_pack_dir,
            prices_dir=prices_dir,
            asset_groups_csv=asset_groups_csv,
            business_days_only=finance_pack_business_days_only,
            min_rows=finance_pack_min_rows,
            min_date_coverage=finance_pack_min_date_coverage,
        ):
            build_cmd = [
                PY,
                "scripts/lab/build_local_finance_pack.py",
                "--prices-dir",
                str(prices_dir),
                "--asset-groups",
                str(asset_groups_csv),
                "--results-dir",
                str(finance_pack_out_root),
                "--business-days-only",
                str(int(finance_pack_business_days_only)),
                "--min-rows",
                str(int(finance_pack_min_rows)),
                "--min-date-coverage",
                str(finance_pack_min_date_coverage),
            ]
            build_code, build_out, build_err = _run(build_cmd, cwd=ROOT, timeout_sec=float(args.step_timeout_sec))
            build_payload = _extract_last_json_line(build_out)
            if build_code != 0 or not build_payload:
                raise SystemExit(f"profit shadow finance pack build failed: {build_err or build_out}")
            finance_pack_dir = Path(str(build_payload.get("outdir", "")).strip())
            if not finance_pack_dir.is_absolute():
                finance_pack_dir = (ROOT / finance_pack_dir).resolve()
        macro_cache_key = _macro_cache_key(
            finance_pack_dir=finance_pack_dir,
            lab_policy_path=lab_policy_path,
            asset_metadata_csv=asset_metadata_csv,
            cfg=cfg,
        )
        if reuse_latest_macro:
            cand = _find_macro_cache_hit(lab_out_base, macro_cache_key)
            if cand is not None:
                macro_run_dir = cand.resolve()
        if (macro_run_dir is None or not _macro_run_complete(macro_run_dir)) and reuse_latest_macro:
            cand = _find_latest_timestamped_run(lab_out_base)
            if cand is not None and _macro_run_complete(cand):
                macro_run_dir = cand.resolve()
        if macro_run_dir is not None and _macro_run_complete(macro_run_dir):
            checkpoint["finance_pack_dir"] = str(finance_pack_dir)
            checkpoint["macro_run_dir"] = str(macro_run_dir)
            checkpoint["macro_cache_key"] = str(macro_cache_key)
            _write_checkpoint(checkpoint_path, checkpoint)
        else:
            checkpoint["finance_pack_dir"] = str(finance_pack_dir)
            checkpoint["macro_cache_key"] = str(macro_cache_key)
            _write_checkpoint(checkpoint_path, checkpoint)
        if macro_run_dir is not None and _macro_run_complete(macro_run_dir):
            pass
        else:
            panel_path = finance_pack_dir / "panel_long_sector.csv"
            universe_path = finance_pack_dir / "universe_fixed.csv"
            macro_cmd = [
                PY,
                "scripts/lab/run_corr_macro_offline.py",
                "--policy-path",
                str(lab_policy_path),
                "--panel-path",
                str(panel_path),
                "--universe-path",
                str(universe_path),
                "--out-base",
                str(lab_out_base),
                "--max-core-assets",
                str(_safe_int(cfg.get("max_core_assets", 300), 300)),
                "--windows",
                str(cfg.get("macro_windows", "120")),
                "--official-window",
                str(_safe_int(cfg.get("official_window", 120), 120)),
                "--noise-step",
                str(_safe_int(cfg.get("noise_step", 10), 10)),
                "--overlap-step",
                str(_safe_int(cfg.get("overlap_step", 5), 5)),
                "--enable-hierarchical",
                "1" if bool(cfg.get("enable_hierarchical", True)) else "0",
                "--asset-metadata-path",
                str(asset_metadata_csv),
                "--n-global",
                str(_safe_int(cfg.get("n_global", 240), 240)),
                "--n-sector",
                str(_safe_int(cfg.get("n_sector", 70), 70)),
                "--min-coverage-global",
                str(_safe_float(cfg.get("min_coverage_global", 0.90), 0.90)),
                "--min-coverage-sector",
                str(_safe_float(cfg.get("min_coverage_sector", 0.80), 0.80)),
                "--enable-internal-sectors",
                "1",
                "--save-v1",
                "1" if bool(cfg.get("save_v1", True)) else "0",
                "--update-release-pointer",
                "1",
            ]
            macro_code, macro_out, macro_err = _run(macro_cmd, cwd=ROOT, timeout_sec=float(args.step_timeout_sec))
            macro_ptr = _read_json(lab_out_base / "latest_release.json")
            macro_run_dir = _resolve_path(macro_ptr.get("run_dir"))
            if macro_run_dir is None or (not macro_run_dir.exists()):
                macro_run_dir = _find_latest_timestamped_run(lab_out_base)
            if macro_run_dir is None or (not macro_run_dir.exists()):
                raise SystemExit(f"profit shadow macro run failed before producing a run dir: {macro_err or macro_out}")
    else:
        if finance_pack_dir is None:
            finance_pack_dir = _find_latest_local_pack(finance_pack_out_root)
        checkpoint["finance_pack_dir"] = str(finance_pack_dir) if finance_pack_dir is not None else ""
        checkpoint["macro_run_dir"] = str(macro_run_dir)
        _write_checkpoint(checkpoint_path, checkpoint)
    checkpoint["macro_run_dir"] = str(macro_run_dir)
    if finance_pack_dir is not None and not macro_cache_key:
        macro_cache_key = _macro_cache_key(
            finance_pack_dir=finance_pack_dir,
            lab_policy_path=lab_policy_path,
            asset_metadata_csv=asset_metadata_csv,
            cfg=cfg,
        )
    if macro_cache_key:
        checkpoint["macro_cache_key"] = str(macro_cache_key)
    _write_checkpoint(checkpoint_path, checkpoint)
    macro_summary = _read_json(macro_run_dir / "summary.json")
    if not macro_summary:
        raise SystemExit(f"profit shadow macro run failed before producing summary.json: {macro_err or macro_out}")
    if macro_cache_key:
        _register_macro_cache(lab_out_base, macro_cache_key, macro_run_dir)

    impact_cfg = cfg.get("impact_learning", {}) if isinstance(cfg.get("impact_learning"), dict) else {}
    impact_code = 0
    impact_out = ""
    impact_err = ""
    impact_outdir = _resolve_path(args.impact_dir)
    if impact_outdir is None and resume_enabled:
        impact_outdir = _resolve_path(checkpoint.get("impact_outdir"))
    if (impact_outdir is None or not _impact_complete(impact_outdir)) and reuse_latest_impact:
        cand = _find_latest_timestamped_run(impact_out_root)
        if cand is not None and _impact_complete(cand):
            impact_outdir = cand.resolve()
    if impact_outdir is None or not _impact_complete(impact_outdir):
        impact_outdir = impact_out_root / run_id
        impact_cmd = [
            PY,
            "scripts/structural/run_structural_impact_learning.py",
            "--run-dir",
            str(macro_run_dir),
            "--outdir",
            str(impact_outdir),
            "--horizon-days",
            str(_safe_int(impact_cfg.get("horizon_days", 10), 10)),
            "--drawdown-threshold",
            str(_safe_float(impact_cfg.get("drawdown_threshold", 0.05), 0.05)),
            "--target-regimes",
            str(impact_cfg.get("target_regimes", "stress,transition")),
            "--split-date",
            str(impact_cfg.get("split_date", "2024-12-31")),
            "--train-end",
            str(impact_cfg.get("train_end", "2023-12-31")),
            "--walkforward-monthly",
            "1" if bool(impact_cfg.get("walkforward_monthly", True)) else "0",
            "--walkforward-start",
            str(impact_cfg.get("walkforward_start", "2024-01-01")),
            "--alert-quantile",
            str(_safe_float(impact_cfg.get("alert_quantile", 0.85), 0.85)),
            "--seed",
            str(_safe_int(impact_cfg.get("seed", 23), 23)),
            "--random-iters",
            str(_safe_int(impact_cfg.get("random_iters", 300), 300)),
            "--enable-gboost",
            "1" if bool(impact_cfg.get("enable_gboost", True)) else "0",
            "--enable-xgboost",
            "1" if bool(impact_cfg.get("enable_xgboost", False)) else "0",
        ]
        impact_code, impact_out, impact_err = _run(impact_cmd, cwd=ROOT, timeout_sec=float(args.step_timeout_sec))
        if impact_code != 0 or not _impact_complete(impact_outdir):
            raise SystemExit(f"profit shadow impact learning failed: {impact_err or impact_out}")
    checkpoint["impact_outdir"] = str(impact_outdir)
    _write_checkpoint(checkpoint_path, checkpoint)

    returns_wide = _load_returns_wide(macro_run_dir / "returns_wide_core.csv")
    benchmark_symbol = str(cfg.get("benchmark_symbol", "SPY")).strip() or "SPY"
    benchmark_returns: pd.Series | None = None
    try:
        benchmark_returns = _load_price_returns(prices_dir, benchmark_symbol)
    except FileNotFoundError:
        if benchmark_symbol not in returns_wide.columns:
            raise
    initial_capital = _safe_float(cfg.get("initial_capital", 10000.0), 10000.0)
    train_end = str(impact_cfg.get("train_end", "2023-12-31"))
    start_ym = str(cfg.get("systematic_start_ym", "2019-01"))
    max_assets_per_month = _safe_int(cfg.get("max_assets_per_month", 220), 220)
    shadow_tail_months = _safe_int(cfg.get("systematic_shadow_tail_months", 12), 12)

    def _run_or_reuse_profile(
        *,
        profile_name: str,
        profile_description: str,
        profile_dir: Path,
        profile_args: dict[str, Any],
    ) -> dict[str, Any]:
        profile_dir.mkdir(parents=True, exist_ok=True)
        if resume_enabled and reuse_profile_outputs and _profile_complete(profile_dir):
            return _reuse_profile_row(
                profile_name=profile_name,
                profile_description=profile_description,
                profile_dir=profile_dir,
                returns_wide=returns_wide,
                benchmark_symbol=benchmark_symbol,
                benchmark_returns=benchmark_returns,
                initial_capital=initial_capital,
            )
        cmd = _common_profile_cmd(
            impact_dir=impact_outdir,
            returns_csv=macro_run_dir / "returns_wide_core.csv",
            prices_dir=prices_dir,
            outdir=profile_dir,
            benchmark_symbol=benchmark_symbol,
            train_end=train_end,
            start_ym=start_ym,
            max_assets_per_month=max_assets_per_month,
            shadow_tail_months=shadow_tail_months,
            profile_args=profile_args,
        )
        code, out, err = _run(cmd, cwd=ROOT, timeout_sec=float(args.step_timeout_sec))
        if code != 0:
            raise SystemExit(f"profile run failed ({profile_name}): {err or out}")
        if not _profile_complete(profile_dir):
            raise SystemExit(f"profile artifacts missing for {profile_name}")
        return _reuse_profile_row(
            profile_name=profile_name,
            profile_description=profile_description,
            profile_dir=profile_dir,
            returns_wide=returns_wide,
            benchmark_symbol=benchmark_symbol,
            benchmark_returns=benchmark_returns,
            initial_capital=initial_capital,
        )

    profile_rows: list[dict[str, Any]] = []
    profiles = cfg.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        raise SystemExit("profit shadow config missing profiles")

    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        profile_name = str(profile.get("name", "")).strip()
        if not profile_name:
            continue
        profile_args = profile.get("args", {})
        if not isinstance(profile_args, dict):
            profile_args = {}
        profile_args = dict(profile_args)
        execution_universe_csv = str(cfg.get("execution_universe_csv", "")).strip()
        execution_liquidity_csv = str(cfg.get("execution_liquidity_csv", "")).strip()
        if execution_universe_csv and "execution_universe_csv" not in profile_args:
            profile_args["execution_universe_csv"] = execution_universe_csv
        if execution_liquidity_csv:
            profile_args["hybrid_liquidity_csv"] = execution_liquidity_csv
        profile_dir = profiles_root / profile_name
        profile_description = str(profile.get("description", ""))
        execution_variants = _normalized_execution_variants(profile, cfg)
        if execution_variants:
            member_rows: list[dict[str, Any]] = []
            member_dirs: list[Path] = []
            member_names: list[str] = []
            for variant in execution_variants:
                variant_name = str(variant.get("name", "")).strip() or "variant"
                variant_profile_name = f"{profile_name}__{variant_name}"
                variant_profile_dir = profiles_root / variant_profile_name
                variant_args = dict(profile_args)
                variant_csv = str(variant.get("execution_universe_csv", "")).strip()
                variant_liq = str(variant.get("execution_liquidity_csv", "")).strip()
                if variant_csv:
                    variant_args["execution_universe_csv"] = variant_csv
                if variant_liq:
                    variant_args["hybrid_liquidity_csv"] = variant_liq
                member_row = _run_or_reuse_profile(
                    profile_name=variant_profile_name,
                    profile_description=f"{profile_description} [{variant_name}]".strip(),
                    profile_dir=variant_profile_dir,
                    profile_args=variant_args,
                )
                member_rows.append(member_row)
                member_dirs.append(variant_profile_dir)
                member_names.append(variant_profile_name)
            ensemble_vote_threshold = _safe_float(profile.get("ensemble_vote_threshold", 0.5), 0.5)
            if not (resume_enabled and reuse_profile_outputs and _profile_complete(profile_dir)):
                _write_ensemble_profile_artifacts(
                    profile_dir=profile_dir,
                    profile_name=profile_name,
                    profile_description=profile_description,
                    member_profile_dirs=member_dirs,
                    member_profile_names=member_names,
                    returns_wide=returns_wide,
                    benchmark_symbol=benchmark_symbol,
                    benchmark_returns=benchmark_returns,
                    initial_capital=initial_capital,
                    vote_threshold=float(ensemble_vote_threshold),
                )
            profile_row = _reuse_profile_row(
                profile_name=profile_name,
                profile_description=profile_description,
                profile_dir=profile_dir,
                returns_wide=returns_wide,
                benchmark_symbol=benchmark_symbol,
                benchmark_returns=benchmark_returns,
                initial_capital=initial_capital,
            )
            profile_row["ensemble"] = {
                "enabled": True,
                "vote_threshold": float(ensemble_vote_threshold),
                "members": member_names,
            }
        else:
            profile_row = _run_or_reuse_profile(
                profile_name=profile_name,
                profile_description=profile_description,
                profile_dir=profile_dir,
                profile_args=profile_args,
            )
        profile_rows.append(profile_row)
        checkpoint_profiles = checkpoint.get("profiles", {}) if isinstance(checkpoint.get("profiles"), dict) else {}
        checkpoint_profiles[profile_name] = {
            "status": "ok",
            "run_dir": str(profile_dir),
        }
        checkpoint["profiles"] = checkpoint_profiles
        _write_checkpoint(checkpoint_path, checkpoint)

    profile_df = pd.DataFrame(profile_rows)
    if profile_df.empty:
        raise SystemExit("no successful profile rows produced")
    profile_df.to_csv(run_dir / "profile_comparison.csv", index=False)
    _write_json(run_dir / "profile_comparison.json", {"profiles": profile_rows})

    def _best_row(col: str) -> dict[str, Any]:
        x = profile_df.copy()
        x[col] = pd.to_numeric(x[col], errors="coerce")
        x = x.dropna(subset=[col]).sort_values(col, ascending=False).reset_index(drop=True)
        return {} if x.empty else next((r for r in profile_rows if r.get("profile") == str(x.iloc[0]["profile"])), {})

    best_by_profit = _best_row("daily_total_return")
    best_by_sharpe = _best_row("daily_sharpe")
    drawdown_df = profile_df.copy()
    drawdown_df["daily_max_drawdown"] = pd.to_numeric(drawdown_df["daily_max_drawdown"], errors="coerce")
    drawdown_df = drawdown_df.dropna(subset=["daily_max_drawdown"]).sort_values("daily_max_drawdown", ascending=False).reset_index(drop=True)
    best_by_drawdown = {} if drawdown_df.empty else next((r for r in profile_rows if r.get("profile") == str(drawdown_df.iloc[0]["profile"])), {})

    summary = _sanitize_json_value(
        {
            "status": "ok",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "mode": "profit_shadow_suite",
            "shadow_name": str(cfg.get("name", "profit_shadow")),
            "description": str(cfg.get("description", "")),
            "macro": {
                "run_dir": str(macro_run_dir),
                "status": str(macro_summary.get("status", "")),
                "gate_blocked": bool(((macro_summary.get("deployment_gate") or {}) if isinstance(macro_summary.get("deployment_gate"), dict) else {}).get("blocked", True)),
                "n_core": _safe_int(macro_summary.get("n_core", 0), 0),
                "official_window": _safe_int(macro_summary.get("official_window", 0), 0),
                "global_universe_size": _safe_int((((macro_summary.get("hierarchical") or {}) if isinstance(macro_summary.get("hierarchical"), dict) else {}).get("global_universe_size")), 0),
                "sector_universes_count": _safe_int((((macro_summary.get("hierarchical") or {}) if isinstance(macro_summary.get("hierarchical"), dict) else {}).get("sector_universes_count")), 0),
                "stderr": str(macro_err).strip(),
                "exit_code": int(macro_code),
            },
            "impact_learning": {
                "outdir": str(impact_outdir),
                "summary": _read_json(impact_outdir / "impact_summary.json"),
            },
            "benchmark_symbol": benchmark_symbol,
            "initial_capital": float(initial_capital),
            "profiles": profile_rows,
            "best_by_profit": best_by_profit,
            "best_by_sharpe": best_by_sharpe,
            "best_by_drawdown": best_by_drawdown,
            "artifacts": {
                "profile_comparison_csv": str(run_dir / "profile_comparison.csv"),
                "profile_comparison_json": str(run_dir / "profile_comparison.json"),
            },
        }
    )

    _write_json(run_dir / "summary.json", summary)
    _write_json(latest_summary_path, summary)
    _write_json(
        latest_run_path,
        {
            "status": "ok",
            "updated_at_utc": summary["generated_at_utc"],
            "run_id": run_id,
            "summary_path": str(run_dir / "summary.json"),
            "macro_run_dir": str(macro_run_dir),
            "impact_outdir": str(impact_outdir),
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
