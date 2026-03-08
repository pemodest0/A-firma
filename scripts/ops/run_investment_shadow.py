#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")

try:
    from scripts.finance.yf_fetch_or_load import fetch_yfinance, unify_to_daily

    YF_FETCH_OK = True
except Exception:
    YF_FETCH_OK = False


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


def _find_latest_timestamped_run(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = sorted((p for p in root.iterdir() if p.is_dir() and RUN_ID_RE.match(p.name)), key=lambda p: p.name)
    return runs[-1] if runs else None


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


def _load_shadow_config(path: Path) -> dict[str, Any]:
    cfg = _read_json(path)
    if not cfg:
        raise FileNotFoundError(f"shadow config not found or invalid: {path}")
    return cfg


def _load_returns_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "date" not in df.columns or "r" not in df.columns:
        raise ValueError(f"returns file missing date/r columns: {path}")
    out = df[["date", "r"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["r"] = pd.to_numeric(out["r"], errors="coerce")
    out = out.dropna(subset=["date", "r"]).sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        raise ValueError(f"returns file empty after cleanup: {path}")
    s = out.set_index("date")["r"].astype(float)
    s.index = pd.DatetimeIndex(s.index)
    return s


def next_available_date(dates: pd.DatetimeIndex, signal_date: str | pd.Timestamp) -> pd.Timestamp | None:
    if len(dates) <= 0:
        return None
    sd = pd.Timestamp(signal_date)
    later = dates[dates > sd]
    if len(later) <= 0:
        return None
    return pd.Timestamp(later[0])


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


def build_portfolio_history(
    *,
    risk_returns: pd.Series,
    defensive_returns: pd.Series,
    signals: pd.DataFrame,
    initial_capital: float,
    cost_bps: float,
    max_daily_turnover: float,
    initial_exposure: float,
) -> pd.DataFrame:
    rr = pd.to_numeric(risk_returns, errors="coerce").dropna().astype(float)
    dr = pd.to_numeric(defensive_returns, errors="coerce").dropna().astype(float)
    common = rr.index.intersection(dr.index).sort_values()
    if len(common) <= 0 or signals.empty:
        return pd.DataFrame()

    sig = signals.copy()
    sig["effective_date"] = pd.to_datetime(sig["effective_date"], errors="coerce")
    sig["signal_date"] = pd.to_datetime(sig["signal_date"], errors="coerce")
    sig["target_exposure"] = pd.to_numeric(sig["target_exposure"], errors="coerce")
    sig = sig.dropna(subset=["effective_date", "target_exposure"]).sort_values(["effective_date", "generated_at_utc"])
    if sig.empty:
        return pd.DataFrame()

    dates_df = pd.DataFrame({"date": common})
    mapped = pd.merge_asof(
        dates_df.sort_values("date"),
        sig[
            [
                "effective_date",
                "signal_date",
                "target_exposure",
                "regime",
                "lab_run_dir",
                "gate_blocked",
                "generated_at_utc",
            ]
        ].sort_values("effective_date"),
        left_on="date",
        right_on="effective_date",
        direction="backward",
        allow_exact_matches=True,
    )
    mapped = mapped.dropna(subset=["target_exposure"]).reset_index(drop=True)
    if mapped.empty:
        return pd.DataFrame()

    capital = float(initial_capital)
    prev_exec = float(np.clip(initial_exposure, 0.0, 1.0))
    first = True
    rows: list[dict[str, Any]] = []
    for _, row in mapped.iterrows():
        dt = pd.Timestamp(row["date"])
        desired = float(np.clip(float(row["target_exposure"]), 0.0, 1.0))
        if first:
            exec_exp = desired
            turnover = 0.0
            cost = 0.0
            first = False
        else:
            step = float(np.clip(desired - prev_exec, -float(max_daily_turnover), float(max_daily_turnover)))
            exec_exp = float(np.clip(prev_exec + step, 0.0, 1.0))
            turnover = abs(exec_exp - prev_exec)
            cost = turnover * (float(cost_bps) / 10000.0)
        risk_ret = float(rr.loc[dt])
        def_ret = float(dr.loc[dt])
        simple_ret = exec_exp * risk_ret + (1.0 - exec_exp) * def_ret - cost
        benchmark_ret = risk_ret
        capital = capital * (1.0 + simple_ret)
        rows.append(
            {
                "date": dt.date().isoformat(),
                "signal_date": pd.Timestamp(row["signal_date"]).date().isoformat() if pd.notna(row["signal_date"]) else "",
                "effective_date": pd.Timestamp(row["effective_date"]).date().isoformat(),
                "regime": str(row.get("regime", "")),
                "target_exposure": desired,
                "executed_exposure": exec_exp,
                "turnover": turnover,
                "cost": cost,
                "risk_return": risk_ret,
                "defensive_return": def_ret,
                "portfolio_return": simple_ret,
                "benchmark_return": benchmark_ret,
                "capital": capital,
                "lab_run_dir": str(row.get("lab_run_dir", "")),
                "gate_blocked": bool(row.get("gate_blocked", False)),
                "generated_at_utc": str(row.get("generated_at_utc", "")),
            }
        )
        prev_exec = exec_exp

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["capital_peak"] = pd.to_numeric(out["capital"], errors="coerce").cummax()
    out["drawdown"] = pd.to_numeric(out["capital"], errors="coerce") / out["capital_peak"] - 1.0
    out["benchmark_capital"] = float(initial_capital) * (1.0 + pd.to_numeric(out["benchmark_return"], errors="coerce")).cumprod()
    return out


def summarize_portfolio_history(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {"status": "empty"}
    d = history.copy()
    perf = _perf_from_simple_returns(pd.to_numeric(d["portfolio_return"], errors="coerce"))
    bench = _perf_from_simple_returns(pd.to_numeric(d["benchmark_return"], errors="coerce"))
    last = d.iloc[-1]
    start_capital = float(pd.to_numeric(d["capital"], errors="coerce").iloc[0] / max(1.0 + float(pd.to_numeric(d["portfolio_return"], errors="coerce").iloc[0]), 1e-9))
    return _sanitize_json_value(
        {
            "status": "ok",
            "start_date": str(d.iloc[0]["date"]),
            "end_date": str(last["date"]),
            "n_days": int(d.shape[0]),
            "capital_start": start_capital,
            "capital_end": float(last["capital"]),
            "latest_target_exposure": float(last["target_exposure"]),
            "latest_executed_exposure": float(last["executed_exposure"]),
            "latest_regime": str(last.get("regime", "")),
            "latest_signal_date": str(last.get("signal_date", "")),
            "portfolio": perf,
            "benchmark": bench,
            "edge_vs_benchmark_total_return": _safe_float(perf.get("total_return")) - _safe_float(bench.get("total_return")),
        }
    )


def _refresh_ticker_returns(
    *,
    ticker: str,
    prices_dir: Path,
    lookback_days: int,
    retries: int,
    sleep_ms: int,
) -> dict[str, Any]:
    if not YF_FETCH_OK:
        return {"ticker": ticker, "status": "skip", "reason": "yfinance_unavailable"}
    path = prices_dir / f"{ticker}.csv"
    existing_start = "2018-01-01"
    old = pd.DataFrame(columns=["date", "r"])
    if path.exists():
        try:
            old = pd.read_csv(path)
            if "date" in old.columns:
                dd = pd.to_datetime(old["date"], errors="coerce").dropna()
                if not dd.empty:
                    last_date = dd.max().date()
                    existing_start = (last_date - timedelta(days=max(5, int(lookback_days)))).isoformat()
        except Exception:
            old = pd.DataFrame(columns=["date", "r"])

    end = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    last_err = ""
    for attempt in range(max(1, int(retries))):
        try:
            df = fetch_yfinance(ticker, start=existing_start, end=end)
            if df is None or df.empty:
                raise RuntimeError("empty_remote_data")
            normalized = unify_to_daily(df)
            if normalized.empty:
                raise RuntimeError("empty_after_unify")
            fresh = normalized[["date", "r"]].copy()
            fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce").dt.date.astype(str)
            if old.empty:
                merged = fresh
            else:
                old2 = old.copy()
                if "date" not in old2.columns or "r" not in old2.columns:
                    old2 = pd.DataFrame(columns=["date", "r"])
                else:
                    old2 = old2[["date", "r"]].copy()
                    old2["date"] = pd.to_datetime(old2["date"], errors="coerce").dt.date.astype(str)
                    old2["r"] = pd.to_numeric(old2["r"], errors="coerce")
                    old2 = old2.dropna(subset=["date", "r"])
                merged = pd.concat([old2, fresh], ignore_index=True)
                merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
                merged["r"] = pd.to_numeric(merged["r"], errors="coerce")
                merged = merged.dropna(subset=["date", "r"]).sort_values("date").drop_duplicates("date", keep="last")
                merged["date"] = merged["date"].dt.date.astype(str)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(path, index=False)
            return {
                "ticker": ticker,
                "status": "ok",
                "rows": int(merged.shape[0]),
                "last_date": str(merged["date"].iloc[-1]) if not merged.empty else "",
            }
        except Exception as exc:
            last_err = str(exc)
            if attempt + 1 < max(1, int(retries)):
                time.sleep(max(0, int(sleep_ms)) / 1000.0)
    return {"ticker": ticker, "status": "fail", "reason": last_err}


def _load_frozen_universe(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "universe_core.csv"
    if not path.exists():
        raise FileNotFoundError(f"frozen universe not found: {path}")
    df = pd.read_csv(path)
    if "ticker" not in df.columns or "sector" not in df.columns:
        raise ValueError(f"universe_core missing ticker/sector columns: {path}")
    out = df[["ticker", "sector"]].copy()
    out["ticker"] = out["ticker"].astype(str)
    out["sector"] = out["sector"].astype(str)
    out = out.drop_duplicates(subset=["ticker"], keep="last").sort_values(["sector", "ticker"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"frozen universe empty: {path}")
    return out


def _build_signal_snapshot(
    *,
    run_dir: Path,
    official_window: int,
    common_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    regime_path = run_dir / f"regime_series_T{int(official_window)}.csv"
    if not regime_path.exists():
        raise FileNotFoundError(f"regime series not found: {regime_path}")
    regime = pd.read_csv(regime_path)
    if regime.empty:
        raise ValueError(f"regime series empty: {regime_path}")
    regime["date"] = pd.to_datetime(regime["date"], errors="coerce")
    regime = regime.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    last = regime.iloc[-1]
    signal_date = pd.Timestamp(last["date"])
    effective = next_available_date(common_dates, signal_date)
    gate = _read_json(run_dir / "deployment_gate.json")
    policy_used = _read_json(run_dir / "policy_used.json")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": signal_date.date().isoformat(),
        "effective_date": effective.date().isoformat() if effective is not None else "",
        "target_exposure": float(_safe_float(last.get("exposure"), 0.0)),
        "regime": str(last.get("regime", "")),
        "lab_run_dir": str(run_dir),
        "policy_path": str(policy_used.get("policy_path", "")),
        "gate_blocked": bool(gate.get("blocked", True)),
    }


def _load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "generated_at_utc",
                "signal_date",
                "effective_date",
                "target_exposure",
                "regime",
                "lab_run_dir",
                "policy_path",
                "gate_blocked",
            ]
        )
    df = pd.read_csv(path)
    return df if not df.empty else pd.DataFrame(columns=["generated_at_utc", "signal_date", "effective_date", "target_exposure", "regime", "lab_run_dir", "policy_path", "gate_blocked"])


def _append_signal(path: Path, signal: dict[str, Any]) -> pd.DataFrame:
    cur = _load_signals(path)
    row = pd.DataFrame([signal])
    out = pd.concat([cur, row], ignore_index=True)
    out["signal_date"] = out["signal_date"].astype(str)
    out["lab_run_dir"] = out["lab_run_dir"].astype(str)
    out = out.drop_duplicates(subset=["signal_date", "lab_run_dir"], keep="last").sort_values(["signal_date", "generated_at_utc"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _build_historical_signal_schedule(
    *,
    regime_path: Path,
    common_dates: pd.DatetimeIndex,
    lab_run_dir: str,
    gate_blocked: bool,
    policy_path: str,
) -> pd.DataFrame:
    regime = pd.read_csv(regime_path)
    if regime.empty:
        return pd.DataFrame()
    regime["date"] = pd.to_datetime(regime["date"], errors="coerce")
    regime["exposure"] = pd.to_numeric(regime["exposure"], errors="coerce")
    regime = regime.dropna(subset=["date", "exposure"]).sort_values("date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, row in regime.iterrows():
        eff = next_available_date(common_dates, pd.Timestamp(row["date"]))
        if eff is None:
            continue
        rows.append(
            {
                "generated_at_utc": "",
                "signal_date": pd.Timestamp(row["date"]).date().isoformat(),
                "effective_date": eff.date().isoformat(),
                "target_exposure": float(row["exposure"]),
                "regime": str(row.get("regime", "")),
                "lab_run_dir": lab_run_dir,
                "policy_path": policy_path,
                "gate_blocked": bool(gate_blocked),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Runs a local shadow portfolio for the Eigen Engine using investable proxies.")
    ap.add_argument("--config-path", type=str, default="config/investment_shadow_mode.json")
    ap.add_argument("--run-id", type=str, default=_run_id())
    ap.add_argument("--refresh-prices", type=int, default=-1, help="Override config refresh_prices (1/0).")
    ap.add_argument("--rebuild-historical-replay", type=int, default=-1, help="Override config replay rebuild flag (1/0).")
    ap.add_argument("--step-timeout-sec", type=float, default=2400.0)
    args = ap.parse_args()

    config_path = _resolve_path(args.config_path)
    if config_path is None:
        raise SystemExit("missing config path")
    cfg = _load_shadow_config(config_path)

    frozen_run_dir = _resolve_path(cfg.get("frozen_universe_run_dir"))
    policy_path = _resolve_path(cfg.get("policy_path"))
    prices_dir = _resolve_path(cfg.get("prices_dir"))
    finance_pack_out_root = _resolve_path(cfg.get("finance_pack_out_root"))
    lab_out_base = _resolve_path(cfg.get("lab_out_base"))
    shadow_outdir = _resolve_path(cfg.get("shadow_outdir"))
    if None in {frozen_run_dir, policy_path, prices_dir, finance_pack_out_root, lab_out_base, shadow_outdir}:
        raise SystemExit("shadow config has unresolved required paths")

    run_id = str(args.run_id).strip() or _run_id()
    run_dir = shadow_outdir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_asset_groups = run_dir / "asset_groups_shadow.csv"
    signals_csv = shadow_outdir / "signals.csv"
    live_history_csv = shadow_outdir / "live_history.csv"
    latest_summary_path = shadow_outdir / "latest_summary.json"
    latest_run_path = shadow_outdir / "latest_run.json"
    historical_replay_csv = shadow_outdir / "historical_proxy_replay.csv"
    historical_replay_summary_path = shadow_outdir / "historical_proxy_replay_summary.json"

    universe = _load_frozen_universe(frozen_run_dir)
    asset_groups = universe.rename(columns={"ticker": "asset", "sector": "group"}).copy()
    asset_groups.to_csv(runtime_asset_groups, index=False)

    refresh_default = bool(cfg.get("refresh_prices", True))
    refresh_prices = refresh_default if int(args.refresh_prices) < 0 else bool(int(args.refresh_prices))
    replay_default = bool(cfg.get("rebuild_historical_replay_each_run", False))
    rebuild_replay = replay_default if int(args.rebuild_historical_replay) < 0 else bool(int(args.rebuild_historical_replay))

    refresh_rows: list[dict[str, Any]] = []
    tickers_to_refresh = sorted(set(asset_groups["asset"].astype(str).tolist() + [str(cfg.get("risk_proxy", "SPY")), str(cfg.get("defensive_proxy", "SHY"))]))
    if refresh_prices:
        for ticker in tickers_to_refresh:
            refresh_rows.append(
                _refresh_ticker_returns(
                    ticker=ticker,
                    prices_dir=prices_dir,
                    lookback_days=int(cfg.get("refresh_lookback_days", 21)),
                    retries=int(cfg.get("refresh_retries", 2)),
                    sleep_ms=int(cfg.get("refresh_sleep_ms", 250)),
                )
            )

    build_cmd = [
        PY,
        "scripts/lab/build_local_finance_pack.py",
        "--prices-dir",
        str(prices_dir),
        "--asset-groups",
        str(runtime_asset_groups),
        "--results-dir",
        str(finance_pack_out_root),
        "--business-days-only",
        "1",
        "--min-rows",
        "252",
        "--min-date-coverage",
        "0.90",
    ]
    build_code, build_out, build_err = _run(build_cmd, cwd=ROOT, timeout_sec=float(args.step_timeout_sec))
    build_payload = _extract_last_json_line(build_out)
    if build_code != 0 or not build_payload:
        raise SystemExit(f"shadow finance pack build failed: {build_err or build_out}")
    finance_pack_dir = Path(str(build_payload.get("outdir", "")).strip())
    if not finance_pack_dir.is_absolute():
        finance_pack_dir = (ROOT / finance_pack_dir).resolve()
    panel_path = finance_pack_dir / "panel_long_sector.csv"
    universe_path = finance_pack_dir / "universe_fixed.csv"

    macro_cmd = [
        PY,
        "scripts/lab/run_corr_macro_offline.py",
        "--policy-path",
        str(policy_path),
        "--panel-path",
        str(panel_path),
        "--universe-path",
        str(universe_path),
        "--out-base",
        str(lab_out_base),
        "--max-core-assets",
        str(int(cfg.get("max_core_assets", 120))),
        "--windows",
        str(cfg.get("macro_windows", str(_safe_int(cfg.get("official_window", 120), 120)))),
        "--official-window",
        str(_safe_int(cfg.get("official_window", 120), 120)),
        "--noise-step",
        str(_safe_int(cfg.get("noise_step", 10), 10)),
        "--overlap-step",
        str(_safe_int(cfg.get("overlap_step", 5), 5)),
        "--update-release-pointer",
        "1",
    ]
    macro_code, macro_out, macro_err = _run(macro_cmd, cwd=ROOT, timeout_sec=float(args.step_timeout_sec))
    shadow_lab_ptr = _read_json(lab_out_base / "latest_release.json")
    shadow_lab_run_dir = _resolve_path(shadow_lab_ptr.get("run_dir"))
    if shadow_lab_run_dir is None or (not shadow_lab_run_dir.exists()):
        shadow_lab_run_dir = _find_latest_timestamped_run(lab_out_base)
    if shadow_lab_run_dir is None or (not shadow_lab_run_dir.exists()):
        raise SystemExit(f"shadow macro run failed before producing a run dir: {macro_err or macro_out}")
    macro_summary = _read_json(shadow_lab_run_dir / "summary.json")
    if not macro_summary:
        raise SystemExit(f"shadow macro run failed before producing summary.json: {macro_err or macro_out}")

    risk_proxy = str(cfg.get("risk_proxy", "SPY")).strip()
    defensive_proxy = str(cfg.get("defensive_proxy", "SHY")).strip()
    risk_returns = _load_returns_series(prices_dir / f"{risk_proxy}.csv")
    defensive_returns = _load_returns_series(prices_dir / f"{defensive_proxy}.csv")
    common_dates = risk_returns.index.intersection(defensive_returns.index).sort_values()

    policy_used = _read_json(shadow_lab_run_dir / "policy_used.json")
    signal_snapshot = _build_signal_snapshot(
        run_dir=shadow_lab_run_dir,
        official_window=int(policy_used.get("effective", {}).get("official_window", 120)),
        common_dates=common_dates,
    )
    signals = _append_signal(signals_csv, signal_snapshot)

    historical_regime_csv = shadow_lab_run_dir / f"regime_series_T{int(policy_used.get('effective', {}).get('official_window', 120))}.csv"
    historical_rebuild_needed = rebuild_replay or (not historical_replay_csv.exists()) or (not historical_replay_summary_path.exists())
    historical_proxy_summary: dict[str, Any] = _read_json(historical_replay_summary_path)
    if historical_rebuild_needed:
        gate = _read_json(shadow_lab_run_dir / "deployment_gate.json")
        hist_signals = _build_historical_signal_schedule(
            regime_path=historical_regime_csv,
            common_dates=common_dates,
            lab_run_dir=str(shadow_lab_run_dir),
            gate_blocked=bool(gate.get("blocked", True)),
            policy_path=str(policy_used.get("policy_path", "")),
        )
        hist = build_portfolio_history(
            risk_returns=risk_returns,
            defensive_returns=defensive_returns,
            signals=hist_signals,
            initial_capital=float(cfg.get("initial_capital", 10000.0)),
            cost_bps=float(cfg.get("cost_bps", 5.0)),
            max_daily_turnover=float(cfg.get("max_daily_turnover", 0.10)),
            initial_exposure=float(cfg.get("initial_exposure", 0.70)),
        )
        hist.to_csv(historical_replay_csv, index=False)
        historical_proxy_summary = summarize_portfolio_history(hist)
        _write_json(historical_replay_summary_path, historical_proxy_summary)

    live = build_portfolio_history(
        risk_returns=risk_returns,
        defensive_returns=defensive_returns,
        signals=signals,
        initial_capital=float(cfg.get("initial_capital", 10000.0)),
        cost_bps=float(cfg.get("cost_bps", 5.0)),
        max_daily_turnover=float(cfg.get("max_daily_turnover", 0.10)),
        initial_exposure=float(cfg.get("initial_exposure", 0.70)),
    )
    live.to_csv(live_history_csv, index=False)
    live_summary = summarize_portfolio_history(live)

    engine_backtest_summary = _read_json(
        shadow_lab_run_dir / f"backtest_summary_T{int(policy_used.get('effective', {}).get('official_window', 120))}.json"
    )
    latest_price_date = common_dates.max().date().isoformat() if len(common_dates) > 0 else ""
    latest_signal_date = str(signal_snapshot.get("signal_date", ""))
    freshness_days = None
    if latest_price_date:
        freshness_days = (datetime.now(timezone.utc).date() - date.fromisoformat(latest_price_date)).days

    summary = _sanitize_json_value(
        {
            "status": "ok",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "mode": "paper_trading_shadow",
            "shadow_name": str(cfg.get("name", "investment_shadow")),
            "policy_path": str(policy_path),
            "frozen_universe_run_dir": str(frozen_run_dir),
            "shadow_lab_run_dir": str(shadow_lab_run_dir),
            "shadow_lab_status": str(macro_summary.get("status", "")),
            "proxies": {
                "risk_proxy": risk_proxy,
                "defensive_proxy": defensive_proxy,
            },
            "latest": {
                "price_date": latest_price_date,
                "signal_date": latest_signal_date,
                "effective_date": str(signal_snapshot.get("effective_date", "")),
                "regime": str(signal_snapshot.get("regime", "")),
                "target_exposure": _safe_float(signal_snapshot.get("target_exposure")),
                "gate_blocked": bool(signal_snapshot.get("gate_blocked", True)),
                "freshness_days": freshness_days,
            },
            "live": live_summary,
            "historical_proxy_replay": historical_proxy_summary,
            "historical_engine_backtest": engine_backtest_summary,
            "refresh_prices": {
                "enabled": bool(refresh_prices),
                "attempted": int(len(refresh_rows)),
                "ok": int(sum(1 for r in refresh_rows if str(r.get("status")) == "ok")),
                "failed": int(sum(1 for r in refresh_rows if str(r.get("status")) == "fail")),
                "rows": refresh_rows,
            },
            "macro_execution": {
                "exit_code": int(macro_code),
                "stderr": str(macro_err).strip(),
                "stdout_tail": str(macro_out).splitlines()[-20:],
            },
            "artifacts": {
                "signals_csv": str(signals_csv),
                "live_history_csv": str(live_history_csv),
                "historical_proxy_replay_csv": str(historical_replay_csv),
                "historical_proxy_replay_summary_json": str(historical_replay_summary_path),
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
            "shadow_lab_run_dir": str(shadow_lab_run_dir),
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
