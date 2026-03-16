from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from execution.live_ops import PortfolioState, compile_target_order_tickets
from execution.shadow_gods import (
    _safe_float,
    _write_json,
    apply_shadow_fills,
    build_shadow_target_weights,
    build_target_notional,
    capital_block,
    infer_shadow_market_state,
    revalue_portfolio,
    simulate_shadow_fills,
    snapshot_holdings,
)


def _read_price_history(path: Path, *, fx_rates: dict[str, float]) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.Series(dtype=float)
    if frame.empty or "date" not in frame.columns or "price" not in frame.columns:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    values = pd.to_numeric(frame["price"], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=dates, dtype=float).dropna()
    series = series[~series.index.isna()]
    series = series.sort_index()
    if path.stem.endswith("-USD"):
        usd_brl = _safe_float(fx_rates.get("USD_BRL"), 0.0)
        if usd_brl > 0.0:
            series = series * float(usd_brl)
    return series


def load_price_histories(prices_dir: str | Path, tickers: list[str], *, fx_rates: dict[str, float]) -> dict[str, pd.Series]:
    base = Path(prices_dir).resolve()
    out: dict[str, pd.Series] = {}
    for ticker in sorted({str(t).strip() for t in tickers if str(t).strip() and str(t).strip() != "CASH-BRL"}):
        out[ticker] = _read_price_history(base / f"{ticker}.csv", fx_rates=fx_rates)
    return out


def snapshot_prices_as_of(price_histories: dict[str, pd.Series], as_of_date: str) -> dict[str, dict[str, float | None]]:
    ts = pd.Timestamp(str(as_of_date)).tz_localize(None)
    prices: dict[str, dict[str, float | None]] = {}
    for ticker, history in price_histories.items():
        if history.empty:
            prices[ticker] = {"price_native": None, "price_brl": None}
            continue
        try:
            price_brl = float(history.asof(ts))
        except Exception:
            price_brl = float("nan")
        if pd.isna(price_brl) or price_brl <= 0.0:
            prices[ticker] = {"price_native": None, "price_brl": None}
            continue
        prices[ticker] = {"price_native": price_brl, "price_brl": price_brl}
    return prices


def _proxy_operation_fields(*, regime: str, gross_exposure: float) -> dict[str, Any]:
    gross = float(max(0.0, min(1.0, gross_exposure)))
    if str(regime) == "stress" or gross < 0.12:
        recommended_mode = "proteção"
        vigilance_status = "warn"
        posture = "protecao"
    elif gross >= 0.60:
        recommended_mode = "ataque"
        vigilance_status = "ok"
        posture = "ataque_pleno"
    elif gross >= 0.30:
        recommended_mode = "ataque"
        vigilance_status = "ok"
        posture = "ataque_parcial"
    else:
        recommended_mode = "equilibrio"
        vigilance_status = "ok"
        posture = "protecao"
    confidence_score = min(0.95, max(0.05, 0.50 + 0.50 * gross))
    return {
        "recommended_mode": recommended_mode,
        "vigilance_status": vigilance_status,
        "posture": posture,
        "confidence_score": confidence_score,
        "gross_exposure": gross,
    }


def build_historical_operation_proxy(
    *,
    repo_root: Path,
    observer_mode: str = "all22",
) -> pd.DataFrame:
    from scripts.bench.validation.run_profit_marketmode_criticality_suite import (  # noqa: WPS433
        _classify_official_structural_regime,
        build_official_mode_allocations,
    )

    prices_dir = repo_root / "data" / "raw" / "finance" / "yfinance_daily"
    crypto_groups = repo_root / "data" / "asset_groups_crypto_top_liquid_plus.csv"
    crypto_meta = repo_root / "data" / "asset_metadata_crypto_top_liquid_plus.csv"
    equity_groups = repo_root / "data" / "asset_groups_target_800_clean_plus.csv"
    equity_meta = repo_root / "data" / "asset_metadata_target_800_clean_plus.csv"

    official = build_official_mode_allocations(
        prices_dir=prices_dir,
        crypto_groups=crypto_groups,
        crypto_meta=crypto_meta,
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        benchmark_crypto="BTC-USD",
        benchmark_equity="SPY",
    )
    structure_daily = pd.DataFrame(official["structure_daily"]).copy()
    weights = pd.DataFrame(official["official_attack"].weights).copy()
    idx = structure_daily.index.intersection(weights.index)
    structure_daily = structure_daily.reindex(idx)
    weights = weights.reindex(idx).fillna(0.0)

    gross = 1.0 - pd.to_numeric(weights.get("cash"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    driver_rows: list[dict[str, Any]] = []
    for dt in idx:
        regime_now = _classify_official_structural_regime(
            as_of_date=str(pd.Timestamp(dt).date()),
            criticality_value=_safe_float(structure_daily.loc[dt].get("criticality"), 0.5),
            structural_stress_value=_safe_float(structure_daily.loc[dt].get("structural_stress"), 0.5),
            market_mode_share_pct_value=_safe_float(structure_daily.loc[dt].get("market_mode_share_pct"), 0.5),
        )
        proxy = _proxy_operation_fields(regime=str(regime_now.get("regime") or "stable"), gross_exposure=_safe_float(gross.loc[dt], 0.0))
        driver_rows.append(
            {
                "date": pd.Timestamp(dt).tz_localize(None),
                "regime": str(regime_now.get("regime") or "stable"),
                "criticality": _safe_float(regime_now.get("criticality"), 0.5),
                "structural_stress": _safe_float(regime_now.get("structural_stress"), 0.5),
                "market_mode_share_pct": _safe_float(regime_now.get("market_mode_share_pct"), 0.5),
                "gross_exposure": _safe_float(proxy["gross_exposure"], 0.0),
                "recommended_mode": str(proxy["recommended_mode"]),
                "vigilance_status": str(proxy["vigilance_status"]),
                "posture": str(proxy["posture"]),
                "confidence_score": _safe_float(proxy["confidence_score"], 0.0),
                "observer_mode": str(observer_mode),
            }
        )
    driver = pd.DataFrame(driver_rows).sort_values("date").reset_index(drop=True)
    for column in [
        "regime",
        "criticality",
        "structural_stress",
        "market_mode_share_pct",
        "gross_exposure",
        "recommended_mode",
        "vigilance_status",
        "posture",
        "confidence_score",
        "observer_mode",
    ]:
        shifted = driver[column].shift(1)
        if column in {"criticality", "structural_stress", "market_mode_share_pct"}:
            shifted = shifted.fillna(0.5)
        elif column == "gross_exposure":
            shifted = shifted.fillna(0.0)
        elif column == "confidence_score":
            shifted = shifted.fillna(0.25)
        elif column == "recommended_mode":
            shifted = shifted.fillna("proteção")
        elif column == "vigilance_status":
            shifted = shifted.fillna("warn")
        elif column == "posture":
            shifted = shifted.fillna("protecao")
        else:
            shifted = shifted.fillna("stable")
        driver[f"signal_{column}"] = shifted
    return driver


def _daily_operation_payload(row: dict[str, Any], *, as_of_date: str) -> tuple[dict[str, Any], dict[str, Any]]:
    regime = str(row.get("signal_regime") or "stable")
    operation = {
        "as_of_date": str(as_of_date),
        "inputs_as_of_date": str(as_of_date),
        "official_structural_regime": {
            "as_of_date": str(as_of_date),
            "regime": regime,
            "criticality": _safe_float(row.get("signal_criticality"), 0.5),
            "structural_stress": _safe_float(row.get("signal_structural_stress"), 0.5),
            "market_mode_share_pct": _safe_float(row.get("signal_market_mode_share_pct"), 0.5),
        },
        "recommended_live_mode": {
            "mode": str(row.get("signal_recommended_mode") or "proteção"),
            "confidence_score": _safe_float(row.get("signal_confidence_score"), 0.25),
            "current_posture": {"posture": str(row.get("signal_posture") or "protecao")},
        },
        "mode_confidence": {
            "confidence_score": _safe_float(row.get("signal_confidence_score"), 0.25),
            "metrics": {"vigilance_status": str(row.get("signal_vigilance_status") or "warn")},
        },
        "mode_attack": {
            "gross_exposure": _safe_float(row.get("signal_gross_exposure"), 0.0),
        },
    }
    vigilance = {
        "as_of_date": str(as_of_date),
        "status": str(row.get("signal_vigilance_status") or "warn"),
    }
    return operation, vigilance


def _max_drawdown(nav_series: pd.Series) -> float:
    if nav_series.empty:
        return 0.0
    nav = pd.to_numeric(nav_series, errors="coerce").dropna().astype(float)
    if nav.empty:
        return 0.0
    peak = nav.cummax().replace(0.0, pd.NA)
    dd = (nav / peak - 1.0).fillna(0.0)
    return float(dd.min())


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


def _public_artifact_path(alias: str, capital_brl: float, year: int, kind: str) -> str:
    capital_label = str(int(round(capital_brl)))
    return f"/data/site/shadow_gods_historical/{alias}/{capital_label}/{year}_{kind}.csv"


def _write_public_csv(repo_root: Path, alias: str, capital_brl: float, year: int, kind: str, rows: list[dict[str, Any]]) -> str:
    rel = Path("website-ui") / "public" / "data" / "site" / "shadow_gods_historical" / alias / str(int(round(capital_brl))) / f"{year}_{kind}.csv"
    path = repo_root / rel
    _write_csv(path, rows)
    return _public_artifact_path(alias, capital_brl, year, kind)


def _year_slice(rows: list[dict[str, Any]], year: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        raw = str(row.get("as_of_date") or row.get("date") or "").strip()
        if raw.startswith(str(year)):
            out.append(row)
    return out


def _year_summary(*, year: int, history_rows: list[dict[str, Any]], recommendation_rows: list[dict[str, Any]], request_rows: list[dict[str, Any]], fill_rows: list[dict[str, Any]], alias: str, capital_brl: float, repo_root: Path, results_dir: Path) -> dict[str, Any]:
    history_year = _year_slice(history_rows, year)
    rec_year = _year_slice(recommendation_rows, year)
    req_year = _year_slice(request_rows, year)
    fills_year = _year_slice(fill_rows, year)

    nav_before = _safe_float(history_year[0].get("nav_before_brl"), capital_brl) if history_year else float(capital_brl)
    nav_after = _safe_float(history_year[-1].get("nav_after_brl"), nav_before) if history_year else float(capital_brl)
    nav_series = pd.Series([_safe_float(row.get("nav_after_brl"), nav_after) for row in history_year], dtype=float)
    total_return = nav_after / nav_before - 1.0 if nav_before > 0.0 else 0.0
    trade_days = len({str(row.get("as_of_date") or "") for row in fills_year if str(row.get("status") or "") == "filled"})
    recommendation_days = len({str(row.get("as_of_date") or "") for row in rec_year})
    no_trade_days = len([row for row in rec_year if int(_safe_float(row.get("order_count"), 0.0)) == 0])
    state_counter = Counter(str(row.get("market_state") or "unknown") for row in rec_year)
    request_tickers = Counter(str(row.get("ticker") or "") for row in req_year if str(row.get("ticker") or "").strip())
    fill_tickers = Counter(str(row.get("ticker") or "") for row in fills_year if str(row.get("status") or "") == "filled" and str(row.get("ticker") or "").strip())

    result_dir = results_dir / alias / str(int(round(capital_brl))) / str(year)
    history_csv = result_dir / "history.csv"
    recommendations_csv = result_dir / "recommendations.csv"
    requests_csv = result_dir / "requests.csv"
    fills_csv = result_dir / "fills.csv"
    _write_csv(history_csv, history_year)
    _write_csv(recommendations_csv, rec_year)
    _write_csv(requests_csv, req_year)
    _write_csv(fills_csv, fills_year)

    public_history_csv = _write_public_csv(repo_root, alias, capital_brl, year, "history", history_year)
    public_recommendations_csv = _write_public_csv(repo_root, alias, capital_brl, year, "recommendations", rec_year)
    public_requests_csv = _write_public_csv(repo_root, alias, capital_brl, year, "requests", req_year)
    public_fills_csv = _write_public_csv(repo_root, alias, capital_brl, year, "fills", fills_year)

    return {
        "year": int(year),
        "start_nav_brl": float(nav_before),
        "end_nav_brl": float(nav_after),
        "total_return": float(total_return),
        "max_drawdown": _max_drawdown(nav_series),
        "days_total": int(len(history_year)),
        "recommendation_days": int(recommendation_days),
        "trade_days": int(trade_days),
        "no_trade_days": int(no_trade_days),
        "order_count": int(len(req_year)),
        "fill_count": int(len([row for row in fills_year if str(row.get("status") or "") == "filled"])),
        "states_breakdown": dict(state_counter),
        "top_requested_tickers": [{"ticker": ticker, "count": int(count)} for ticker, count in request_tickers.most_common(6)],
        "top_filled_tickers": [{"ticker": ticker, "count": int(count)} for ticker, count in fill_tickers.most_common(6)],
        "history_tail": history_year[-7:],
        "recommendation_tail": rec_year[-7:],
        "request_tail": req_year[-12:],
        "fill_tail": fills_year[-12:],
        "artifacts": {
            "history_csv": str(history_csv),
            "recommendations_csv": str(recommendations_csv),
            "requests_csv": str(requests_csv),
            "fills_csv": str(fills_csv),
            "public_history_csv": public_history_csv,
            "public_recommendations_csv": public_recommendations_csv,
            "public_requests_csv": public_requests_csv,
            "public_fills_csv": public_fills_csv,
        },
    }


def replay_shadow_god_scenario(
    *,
    repo_root: Path,
    god: dict[str, Any],
    profile: dict[str, Any],
    capital_brl: float,
    driver: pd.DataFrame,
    price_histories: dict[str, pd.Series],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    alias = str(god.get("alias") or "unknown")
    role = str(god.get("role") or "")
    cap_block = capital_block(profile, capital_brl)
    scenario_id = f"{alias.lower()}_{int(round(capital_brl))}"

    portfolio = PortfolioState(
        as_of_date=str(start_date),
        base_currency=str(profile.get("base_currency") or "BRL"),
        cash_brl=float(capital_brl),
        fx_rates={str(k): float(v) for k, v in (profile.get("default_fx_rates") or {}).items()},
        positions=[],
        open_orders=[],
    )

    start_ts = pd.Timestamp(str(start_date)).tz_localize(None)
    end_ts = pd.Timestamp(str(end_date)).tz_localize(None)
    frame = driver[(pd.to_datetime(driver["date"]) >= start_ts) & (pd.to_datetime(driver["date"]) <= end_ts)].copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    history_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []

    for raw in frame.to_dict(orient="records"):
        as_of_date = str(pd.Timestamp(raw["date"]).date())
        operation, vigilance = _daily_operation_payload(raw, as_of_date=as_of_date)
        prices = snapshot_prices_as_of(price_histories, as_of_date)
        portfolio_before = revalue_portfolio(portfolio, prices, as_of_date=as_of_date)
        market_state, market_notes = infer_shadow_market_state(operation, vigilance, role=role)
        target_weights, target_notes = build_shadow_target_weights(
            god,
            market_state=market_state,
            max_assets=int(_safe_float(cap_block.get("max_assets"), 2)),
            prices=prices,
        )
        target_notional = build_target_notional(portfolio_before.nav_brl, target_weights)
        selected = {
            "mode_key": market_state,
            "candidate_id": str(god.get("candidate_id") or ""),
            "alias": alias,
            "label": str(god.get("thesis") or alias),
        }
        exec_profile = {
            "min_order_brl": _safe_float(cap_block.get("min_order_brl"), 40.0),
            "min_position_brl": _safe_float(cap_block.get("min_position_brl"), 80.0),
            "max_assets": int(_safe_float(cap_block.get("max_assets"), 2)),
        }
        compiled = compile_target_order_tickets(
            target=target_notional,
            selected=selected,
            exec_profile=exec_profile,
            guardrails={"max_turnover_fraction_of_nav": 1.0},
            portfolio=portfolio_before,
            prices=prices,
            reason_prefix=f"historical_{alias.lower()}",
            notes=[
                "Replay histórico sem olhar o futuro; a recomendação usa apenas o driver deslocado em um dia.",
            ],
        )
        fills = simulate_shadow_fills(
            compiled.get("tickets", []),
            profile=profile,
            portfolio=portfolio_before,
            prices=prices,
            as_of_date=as_of_date,
        )
        portfolio_after = apply_shadow_fills(portfolio_before, fills, as_of_date=as_of_date)
        portfolio_after = revalue_portfolio(portfolio_after, prices, as_of_date=as_of_date)
        portfolio = portfolio_after

        selected_assets = ",".join([row["ticker"] for row in snapshot_holdings(portfolio_after)])
        recommendation_rows.append(
            {
                "date": as_of_date,
                "as_of_date": as_of_date,
                "scenario_id": scenario_id,
                "alias": alias,
                "capital_brl": float(capital_brl),
                "market_state": market_state,
                "driver_regime": str(raw.get("signal_regime") or raw.get("regime") or ""),
                "driver_confidence_score": _safe_float(raw.get("signal_confidence_score"), 0.0),
                "driver_gross_exposure": _safe_float(raw.get("signal_gross_exposure"), 0.0),
                "recommended_mode": str(((operation.get("recommended_live_mode") or {}).get("mode")) or ""),
                "order_count": int(len(compiled.get("tickets", []))),
                "fill_count": int(len([row for row in fills if str(row.get("status") or "") == "filled"])),
                "selected_assets": selected_assets,
                "target_weights_json": json.dumps(target_weights, ensure_ascii=False, sort_keys=True),
                "notes": " | ".join([*market_notes, *target_notes]),
            }
        )
        for ticket in compiled.get("tickets", []):
            request_rows.append(
                {
                    "as_of_date": as_of_date,
                    "scenario_id": scenario_id,
                    "alias": alias,
                    "capital_brl": float(capital_brl),
                    "market_state": market_state,
                    "ticket_id": ticket.get("ticket_id"),
                    "ticker": ticket.get("ticker"),
                    "side": ticket.get("side"),
                    "notional_brl": ticket.get("notional_brl"),
                    "reason": ticket.get("reason"),
                }
            )
        for fill in fills:
            row = dict(fill)
            row.update(
                {
                    "as_of_date": as_of_date,
                    "scenario_id": scenario_id,
                    "alias": alias,
                    "capital_brl": float(capital_brl),
                    "market_state": market_state,
                }
            )
            fill_rows.append(row)
        history_rows.append(
            {
                "as_of_date": as_of_date,
                "scenario_id": scenario_id,
                "alias": alias,
                "capital_brl": float(capital_brl),
                "market_state": market_state,
                "nav_before_brl": float(portfolio_before.nav_brl),
                "nav_after_brl": float(portfolio_after.nav_brl),
                "cash_after_brl": float(portfolio_after.cash_brl),
                "order_count": int(len(compiled.get("tickets", []))),
                "fill_count": int(len([row for row in fills if str(row.get("status") or "") == "filled"])),
                "selected_assets": selected_assets,
            }
        )

    results_dir = repo_root / "results" / "ops" / "shadow_gods_historical"
    year_summaries = [
        _year_summary(
            year=year,
            history_rows=history_rows,
            recommendation_rows=recommendation_rows,
            request_rows=request_rows,
            fill_rows=fill_rows,
            alias=alias,
            capital_brl=capital_brl,
            repo_root=repo_root,
            results_dir=results_dir,
        )
        for year in (2023, 2024, 2025)
    ]
    nav_series = pd.Series([_safe_float(row.get("nav_after_brl"), capital_brl) for row in history_rows], dtype=float)
    result_dir = results_dir / alias / str(int(round(capital_brl)))
    _write_csv(result_dir / "history_full.csv", history_rows)
    _write_csv(result_dir / "recommendations_full.csv", recommendation_rows)
    _write_csv(result_dir / "requests_full.csv", request_rows)
    _write_csv(result_dir / "fills_full.csv", fill_rows)

    public_base = repo_root / "website-ui" / "public" / "data" / "site" / "shadow_gods_historical" / alias / str(int(round(capital_brl)))
    _write_csv(public_base / "history_full.csv", history_rows)
    _write_csv(public_base / "recommendations_full.csv", recommendation_rows)
    _write_csv(public_base / "requests_full.csv", request_rows)
    _write_csv(public_base / "fills_full.csv", fill_rows)

    return {
        "scenario_id": scenario_id,
        "capital_brl": float(capital_brl),
        "candidate_id": str(god.get("candidate_id") or ""),
        "overall": {
            "start_nav_brl": float(capital_brl),
            "end_nav_brl": float(history_rows[-1]["nav_after_brl"]) if history_rows else float(capital_brl),
            "total_return": float((_safe_float(history_rows[-1]["nav_after_brl"], capital_brl) / float(capital_brl)) - 1.0) if history_rows else 0.0,
            "max_drawdown": _max_drawdown(nav_series),
            "days_total": int(len(history_rows)),
            "recommendation_days": int(len(recommendation_rows)),
            "trade_days": int(len({str(row.get("as_of_date") or "") for row in fill_rows if str(row.get("status") or "") == "filled"})),
            "order_count": int(len(request_rows)),
            "fill_count": int(len([row for row in fill_rows if str(row.get("status") or "") == "filled"])),
        },
        "years": year_summaries,
        "artifacts": {
            "history_full_csv": str(result_dir / "history_full.csv"),
            "recommendations_full_csv": str(result_dir / "recommendations_full.csv"),
            "requests_full_csv": str(result_dir / "requests_full.csv"),
            "fills_full_csv": str(result_dir / "fills_full.csv"),
            "public_history_full_csv": f"/data/site/shadow_gods_historical/{alias}/{int(round(capital_brl))}/history_full.csv",
            "public_recommendations_full_csv": f"/data/site/shadow_gods_historical/{alias}/{int(round(capital_brl))}/recommendations_full.csv",
            "public_requests_full_csv": f"/data/site/shadow_gods_historical/{alias}/{int(round(capital_brl))}/requests_full.csv",
            "public_fills_full_csv": f"/data/site/shadow_gods_historical/{alias}/{int(round(capital_brl))}/fills_full.csv",
        },
    }


def build_shadow_gods_historical_summary(
    *,
    repo_root: Path,
    profile: dict[str, Any],
    prices_dir: Path,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
) -> dict[str, Any]:
    driver = build_historical_operation_proxy(repo_root=repo_root, observer_mode="all22")
    all_tickers = sorted(
        {
            ticker
            for god in profile.get("gods", [])
            if isinstance(god, dict)
            for allocations in (god.get("allocations") or {}).values()
            if isinstance(allocations, dict)
            for ticker in allocations.keys()
            if ticker != "CASH-BRL"
        }
    )
    price_histories = load_price_histories(
        prices_dir,
        all_tickers,
        fx_rates={str(k): float(v) for k, v in (profile.get("default_fx_rates") or {}).items()},
    )
    gods_out = []
    for god in profile.get("gods", []):
        if not isinstance(god, dict):
            continue
        scenarios = []
        for block in profile.get("capital_blocks", []):
            if not isinstance(block, dict):
                continue
            scenarios.append(
                replay_shadow_god_scenario(
                    repo_root=repo_root,
                    god=god,
                    profile=profile,
                    capital_brl=_safe_float(block.get("capital_brl"), 0.0),
                    driver=driver,
                    price_histories=price_histories,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        gods_out.append(
            {
                "alias": str(god.get("alias") or ""),
                "candidate_id": str(god.get("candidate_id") or ""),
                "role": str(god.get("role") or ""),
                "thesis": str(god.get("thesis") or ""),
                "scenarios": scenarios,
            }
        )

    overview = {
        "god_count": int(len(gods_out)),
        "scenario_count": int(sum(len(god.get("scenarios", [])) for god in gods_out)),
        "order_count_total": int(sum(_safe_float(((scenario.get("overall") or {}).get("order_count")), 0.0) for god in gods_out for scenario in god.get("scenarios", []))),
        "fill_count_total": int(sum(_safe_float(((scenario.get("overall") or {}).get("fill_count")), 0.0) for god in gods_out for scenario in god.get("scenarios", []))),
    }
    years_overview = []
    for year in (2023, 2024, 2025):
        year_rows = [annual for god in gods_out for scenario in god.get("scenarios", []) for annual in scenario.get("years", []) if int(annual.get("year", 0)) == year]
        years_overview.append(
            {
                "year": int(year),
                "scenario_count": int(len(year_rows)),
                "order_count_total": int(sum(_safe_float(row.get("order_count"), 0.0) for row in year_rows)),
                "fill_count_total": int(sum(_safe_float(row.get("fill_count"), 0.0) for row in year_rows)),
                "trade_days_total": int(sum(_safe_float(row.get("trade_days"), 0.0) for row in year_rows)),
            }
        )

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "as_of_date": str(end_date),
        "window_start": str(start_date),
        "window_end": str(end_date),
        "driver": {
            "observer_mode": "all22",
            "notes": [
                "Replay diário sem olhar o futuro: o sinal de cada dia usa o driver deslocado em um dia.",
                "Os deuses continuam congelados; muda só o estado diário e a execução simulada nas datas históricas.",
            ],
        },
        "overview": overview,
        "years_overview": years_overview,
        "gods": gods_out,
    }
    public_summary_path = repo_root / "website-ui" / "public" / "data" / "site" / "latest_shadow_gods_historical.json"
    _write_json(public_summary_path, summary)
    return summary
