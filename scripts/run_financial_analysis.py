from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dados.brutos.financial_loader import (
    PriceSeries,
    YFINANCE_AVAILABLE,
    download_price_series,
    generate_return_windows,
    load_price_csv,
    load_value_csv,
    prepare_returns,
    discretize_returns,
)
from financial_walk_model import (
    MODE_CLASSICAL,
    MODE_GROVER,
    MODE_HADAMARD,
    analyze_financial_windows,
    classify_phase,
    describe_phase,
    forecast_from_distribution,
)
RESULTS_DIR = Path("results_finance")
if TYPE_CHECKING:  # pragma: no cover
    from dados.brutos.financial_loader import ReturnWindow
    from financial_walk_model import WindowAnalysis
MODE_LABELS: Dict[str, str] = {
    MODE_CLASSICAL: "Classico",
    MODE_HADAMARD: "Quantico (Hadamard)",
    MODE_GROVER: "Quantico (Grover)",
}
MODE_COLORS: Dict[str, str] = {
    MODE_CLASSICAL: "black",
    MODE_HADAMARD: "#6A0DAD",
    MODE_GROVER: "#1f77b4",
}
SYMBOL_ALIASES = {
    "IBOV": "^BVSP",
    "IBOV.SA": "^BVSP",
    "^BVSP": "^BVSP",
}

PHASE_COLORS: Dict[str, str] = {
    "difusiva": "#2ca02c",
    "transicao": "#ff7f0e",
    "coerente": "#d62728",
    "indefinido": "#7f7f7f",
}
def _as_naive_datetime(series: pd.Series) -> pd.Series:
    tzinfo = getattr(series.dt, "tz", None)
    if tzinfo is not None:
        return series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series.astype("datetime64[ns]")
def _safe_label(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _return_units(return_mode: str, value_label: str) -> str:
    if return_mode in ("log", "simple"):
        return "%"
    return value_label


def _format_return_value(summary: ForecastSummary, value: float, value_label: str) -> float:
    if not np.isfinite(value):
        return float("nan")
    if summary.return_mode == "log":
        return float(np.expm1(value) * 100.0)
    if summary.return_mode == "simple":
        return float(value * 100.0)
    return float(value)
def _compute_phase_thresholds(df: pd.DataFrame, reference_mode: str, args: argparse.Namespace) -> Tuple[float, float]:
    if args.auto_phase:
        alphas = df[df["mode"] == reference_mode]["dispersion_alpha"].dropna()
        if alphas.empty:
            return args.phase_low, args.phase_high
        low = float(alphas.quantile(args.phase_lower_quantile))
        high = float(alphas.quantile(args.phase_upper_quantile))
        if low >= high:
            high = low + 1e-3
        return low, high
    return args.phase_low, args.phase_high
def _slice_price_series(series: PriceSeries, end_date: pd.Timestamp) -> PriceSeries:
    mask = series.data["date"] <= end_date
    if mask.sum() == 0:
        raise ValueError(f"No data available up to cutoff {end_date.date()}.")
    data = series.data.loc[mask].copy()
    return PriceSeries(data=data.reset_index(drop=True), price_column=series.price_column, return_column=series.return_column)
def _run_analysis_for_series(
    series: PriceSeries,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[List["ReturnWindow"], List[WindowAnalysis], pd.DataFrame]:
    windows = generate_return_windows(
        series,
        bin_edges,
        window=args.window,
        step=args.step,
    )
    if not windows:
        raise RuntimeError("Not enough windows generated; reduce cutoff or window size.")
    analyses = analyze_financial_windows(
        windows,
        bin_centers,
        steps=args.walk_steps,
        fit_min=args.fit_min,
        fit_max=args.fit_max,
        noise_level=args.noise,
    )
    df = _summaries_to_dataframe(analyses)
    df.sort_values(by=["window_end", "mode"], inplace=True)
    return windows, analyses, df
def _direction_from_return(total_return: float, threshold: float) -> str:
    if total_return > threshold:
        return "alta"
    if total_return < -threshold:
        return "queda"
    return "neutro"
def _forecast_quality(
    error_value: float,
    direction_pred: str,
    direction_real: str,
    phase_match: bool,
    return_mode: str,
) -> str:
    if not np.isfinite(error_value):
        return "indefinido"
    abs_error = abs(error_value)
    if return_mode == "diff":
        good_threshold = 50.0
        ok_threshold = 100.0
    else:
        good_threshold = 2.0
        ok_threshold = 5.0
    direction_match = direction_pred == direction_real or direction_real == "neutro"
    if direction_match and phase_match:
        if abs_error <= good_threshold:
            return "acerto pleno"
        if abs_error <= ok_threshold:
            return "tendencia certa"
        return "tendencia certa (erro alto)"
    if direction_match:
        if abs_error <= ok_threshold:
            return "direcao ok"
        return "direcao ok (erro alto)"
    if abs_error <= (good_threshold * 1.5):
        return "erro leve"
    return "erro"
def _total_return(price_start: float, price_end: float, return_mode: str) -> float:
    if not np.isfinite(price_start) or not np.isfinite(price_end):
        return 0.0
    if return_mode == "log":
        if price_start > 0 and price_end > 0:
            return float(np.log(price_end / price_start))
        return 0.0
    if return_mode == "simple":
        return float((price_end - price_start) / price_start) if price_start != 0 else 0.0
    return float(price_end - price_start)
def _run_historical_forecast(
    price_series: PriceSeries,
    price_df_full: pd.DataFrame,
    args: argparse.Namespace,
    cutoff: pd.Timestamp,
    forecast_horizon: int,
    label: str,
) -> Tuple[List[Dict[str, object]], Optional[Dict[str, object]], List[str]]:
    truncated_series = _slice_price_series(price_series, cutoff)
    bin_edges, bin_centers = discretize_returns(
        truncated_series.data[truncated_series.return_column],
        num_bins=args.bins,
        method=args.bin_method,
        clip=args.clip,
    )
    _, analyses_pred, df_pred = _run_analysis_for_series(truncated_series, bin_edges, bin_centers, args)
    reference_mode = MODE_HADAMARD if MODE_HADAMARD in df_pred["mode"].unique() else MODE_CLASSICAL
    low_thr, high_thr = _compute_phase_thresholds(df_pred, reference_mode, args)
    df_pred["phase"] = df_pred["dispersion_alpha"].apply(lambda alpha: classify_phase(alpha, low_thr, high_thr))
    phase_counts_pred = df_pred[df_pred["mode"] == reference_mode]["phase"].value_counts().to_dict()
    last_window_pred = analyses_pred[-1]
    phase_by_mode_pred = {
        mode: classify_phase(metrics.dispersion_alpha, low_thr, high_thr)
        for mode, metrics in last_window_pred.metrics.items()
    }
    last_price = float(
        price_df_full.loc[price_df_full["date"] <= cutoff, "price"].iloc[-1]
    )
    price_cutoff = last_price
    forecast_curves, forecast_summaries = forecast_from_distribution(
        last_window_pred.distribution,
        bin_centers,
        last_price=last_price,
        steps=forecast_horizon,
        phase_by_mode=phase_by_mode_pred,
        return_mode=args.return_method,
    )
    forecast_dates = pd.bdate_range(
        start=cutoff + pd.Timedelta(days=1),
        periods=forecast_horizon,
    )
    with_price = price_df_full.set_index("date").reindex(forecast_dates, method="ffill")
    actual_prices = with_price["price"]
    if actual_prices.isna().any():
        raise RuntimeError("Missing price data for forecast comparison window.")
    observed_end = forecast_dates[-1]
    observed_series = _slice_price_series(price_series, observed_end)
    _, analyses_obs, df_obs = _run_analysis_for_series(observed_series, bin_edges, bin_centers, args)
    df_obs["phase"] = df_obs["dispersion_alpha"].apply(lambda alpha: classify_phase(alpha, low_thr, high_thr))
    last_window_obs = analyses_obs[-1]
    metrics_real_map = last_window_obs.metrics
    phase_real_by_mode = {
        mode: classify_phase(metrics.dispersion_alpha, low_thr, high_thr)
        for mode, metrics in metrics_real_map.items()
    }
    curve_map = {curve.mode: curve for curve in forecast_curves}
    summary_map = {summary.mode: summary for summary in forecast_summaries}
    price_start = float(actual_prices.iloc[0])
    price_end = float(actual_prices.iloc[-1])
    actual_total_return = _total_return(price_start, price_end, args.return_method)
    direction_threshold = 0.002 if args.return_method in ("log", "simple") else 0.05
    direction_real = _direction_from_return(actual_total_return, direction_threshold)
    if args.return_method == "diff":
        scale = float(np.nanstd(actual_prices.values))
        if not np.isfinite(scale) or scale == 0.0:
            scale = float(max(abs(price_end), 1.0))
    else:
        scale = float(abs(price_end))
        if scale == 0.0:
            scale = 1.0
    metrics_rows: List[Dict[str, object]] = []
    console_lines: List[str] = []
    for mode, metrics_pred in last_window_pred.metrics.items():
        curve = curve_map.get(mode)
        summary = summary_map.get(mode)
        metrics_real = metrics_real_map.get(mode)
        predicted_price_end = float(curve.predicted_prices[-1]) if curve is not None and curve.predicted_prices.size else float("nan")
        rmse = float(np.sqrt(np.mean((curve.predicted_prices - actual_prices.values) ** 2))) if curve is not None else float("nan")
        if args.return_method == "diff":
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_errors = (
                    (curve.predicted_prices - actual_prices.values) / scale
                ) if curve is not None else np.array([np.nan])
            mape_pct = float("nan")
            rmse_pct = float(100.0 * rmse / scale) if curve is not None and scale != 0 else float("nan")
            error_metric = float(100.0 * (predicted_price_end - price_end) / scale) if scale != 0 else float("nan")
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_errors = np.where(
                    actual_prices.values != 0,
                    (curve.predicted_prices - actual_prices.values) / actual_prices.values,
                    0.0,
                ) if curve is not None else np.array([np.nan])
            mape_pct = float(np.nanmean(np.abs(rel_errors)) * 100.0) if curve is not None else float("nan")
            rmse_pct = float(100.0 * rmse / price_end) if curve is not None and price_end != 0 else float("nan")
            error_metric = float(100.0 * (predicted_price_end - price_end) / price_end) if price_end != 0 else float("nan")
        alpha_real = metrics_real.dispersion_alpha if metrics_real is not None else float("nan")
        entropy_real = metrics_real.entropy_final if metrics_real is not None else float("nan")
        phase_real = phase_real_by_mode.get(mode, "indefinido")
        phase_match = phase_real == phase_by_mode_pred.get(mode, "indefinido")
        quality = _forecast_quality(
            error_metric,
            summary.direction if summary is not None else "indefinido",
            direction_real,
            phase_match,
            args.return_method,
        )
        metrics_rows.append(
            {
                "cutoff_date": cutoff.date(),
                "forecast_days": forecast_horizon,
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "alpha_pred": metrics_pred.dispersion_alpha,
                "entropy_pred": metrics_pred.entropy_final,
                "phase_pred": phase_by_mode_pred.get(mode, "indefinido"),
                "alpha_real": alpha_real,
                "entropy_real": entropy_real,
                "phase_real": phase_real,
                "phase_match": phase_match,
                "price_cutoff": price_cutoff,
                "predicted_price": predicted_price_end,
                "actual_price": price_end,
                "error_pct": error_metric,
                "rmse_pct": rmse_pct,
                "mape_pct": mape_pct,
                "direction_pred": summary.direction if summary is not None else "indefinido",
                "direction_real": direction_real,
                "quality": quality,
                "error_scale": scale,
            }
        )
        error_label = f"{error_metric:.2f}%" if args.return_method != "diff" else f"{error_metric:.2f}%_sigma"
        console_lines.append(
            f"[HIST] {label} {cutoff.date()} [{MODE_LABELS.get(mode, mode)}] "
            f"fase_pred={phase_by_mode_pred.get(mode, 'indefinido')} "
            f"fase_real={phase_real} erro={error_label} rmse={rmse_pct:.2f}% "
            f"dir_pred={summary.direction if summary is not None else 'indefinido'} dir_real={direction_real} qualidade={quality}"
        )
    price_history = price_df_full[
        (price_df_full["date"] >= cutoff - pd.Timedelta(days=args.window * 2))
        & (price_df_full["date"] <= forecast_dates[-1])
    ].copy()
    plot_info = {
        "cutoff": cutoff,
        "reference_mode": reference_mode,
        "df_pred_mode": df_pred[df_pred["mode"] == reference_mode].copy(),
        "df_obs_mode": df_obs[df_obs["mode"] == reference_mode].copy(),
        "low_thr": low_thr,
        "high_thr": high_thr,
        "phase_counts": phase_counts_pred,
        "forecast_dates": forecast_dates,
        "curves": curve_map,
        "actual_prices": actual_prices,
        "price_history": price_history,
        "phase_pred": phase_by_mode_pred,
        "phase_real": phase_real_by_mode,
    }
    console_lines.append(
        f"[HIST] {label} {cutoff.date()} thresholds low={low_thr:.3f} high={high_thr:.3f} (ref {MODE_LABELS.get(reference_mode, reference_mode)})"
    )
    return metrics_rows, plot_info, console_lines
def _plot_historical_forecasts(plot_data: List[Dict[str, object]], output_path: Path) -> None:
    if not plot_data:
        return
    n_cols = len(plot_data)
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 9), sharex=False)
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(3, 1)
    for idx, info in enumerate(plot_data):
        ax_price = axes[0, idx]
        ax_alpha = axes[1, idx]
        ax_entropy = axes[2, idx]
        dataset_label = info.get("label", "dataset")
        cutoff = info["cutoff"]
        reference_mode = info["reference_mode"]
        price_history: pd.DataFrame = info["price_history"]
        forecast_dates: pd.DatetimeIndex = info["forecast_dates"]
        curves: Dict[str, object] = info["curves"]
        actual_prices: pd.Series = info["actual_prices"]
        df_pred_mode: pd.DataFrame = info["df_pred_mode"]
        df_obs_mode: pd.DataFrame = info["df_obs_mode"]
        low_thr: float = info["low_thr"]
        high_thr: float = info["high_thr"]
        ax_price.plot(price_history["date"], price_history["price"], color="black", linewidth=1.5, label="Preco real")
        ax_price.axvline(cutoff, color="gray", linestyle="--", linewidth=1.0, label="Cutoff" if idx == 0 else None)
        for mode, curve in curves.items():
            if curve.predicted_prices.size == 0:
                continue
            mode_label = f"{MODE_LABELS.get(mode, mode)} previsao"
            ax_price.plot(
                forecast_dates,
                curve.predicted_prices,
                linestyle="--",
                linewidth=1.8,
                color=MODE_COLORS.get(mode, "#444444"),
                label=mode_label,
            )
        ax_price.plot(forecast_dates, actual_prices.values, color="#ff7f0e", linewidth=1.6, label="Preco observado")
        ax_price.set_title(f"{dataset_label} - Preco e previsoes (cutoff {cutoff.date()})")
        ax_price.set_ylabel("Preco")
        ax_price.grid(True, linestyle="--", alpha=0.3)
        ax_price.legend(loc="upper left", fontsize=9)
        pred_mask = df_pred_mode["window_end"] <= cutoff
        obs_mask = df_obs_mode["window_end"] >= cutoff
        ax_alpha.plot(
            df_pred_mode.loc[pred_mask, "window_end"],
            df_pred_mode.loc[pred_mask, "dispersion_alpha"],
            color=MODE_COLORS.get(reference_mode, "#1f77b4"),
            linewidth=1.6,
            label="alpha pre-cutoff",
        )
        ax_alpha.plot(
            df_obs_mode.loc[obs_mask, "window_end"],
            df_obs_mode.loc[obs_mask, "dispersion_alpha"],
            color="#ff7f0e",
            linewidth=1.6,
            linestyle="--",
            label="alpha observado",
        )
        ax_alpha.axhline(low_thr, color="gray", linestyle=":", linewidth=1.0)
        ax_alpha.axhline(high_thr, color="gray", linestyle=":", linewidth=1.0)
        ax_alpha.axvline(cutoff, color="gray", linestyle="--", linewidth=1.0)
        ax_alpha.set_ylabel("alpha")
        ax_alpha.set_title(f"{dataset_label} - Disp. alpha ({MODE_LABELS.get(reference_mode, reference_mode)})")
        ax_alpha.grid(True, linestyle="--", alpha=0.3)
        ax_alpha.legend(loc="upper left", fontsize=9)
        ax_entropy.plot(
            df_pred_mode.loc[pred_mask, "window_end"],
            df_pred_mode.loc[pred_mask, "entropy_final"],
            color=MODE_COLORS.get(reference_mode, "#1f77b4"),
            linewidth=1.6,
            label="Entropia pre-cutoff",
        )
        ax_entropy.plot(
            df_obs_mode.loc[obs_mask, "window_end"],
            df_obs_mode.loc[obs_mask, "entropy_final"],
            color="#ff7f0e",
            linewidth=1.6,
            linestyle="--",
            label="Entropia observada",
        )
        ax_entropy.axvline(cutoff, color="gray", linestyle="--", linewidth=1.0)
        ax_entropy.set_ylabel("Entropia (bits)")
        ax_entropy.set_xlabel("Data")
        ax_entropy.set_title(f"{dataset_label} - Entropia ({MODE_LABELS.get(reference_mode, reference_mode)})")
        ax_entropy.grid(True, linestyle="--", alpha=0.3)
        ax_entropy.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
def _process_dataset(
    price_series: PriceSeries,
    args: argparse.Namespace,
    forecast_horizon: int,
    output_dir: Path,
    label: str,
) -> None:
    price_df_full = price_series.data.rename(columns={price_series.price_column: "price"}).copy()
    price_df_full["date"] = _as_naive_datetime(price_df_full["date"])
    price_df_full["price"] = price_df_full["price"].astype(float)
    cutoff_dates: List[pd.Timestamp] = []
    if args.cutoff:
        for cutoff_str in args.cutoff:
            ts = pd.Timestamp(pd.to_datetime(cutoff_str)).tz_localize(None)
            cutoff_dates.append(ts)
        cutoff_dates = sorted(set(cutoff_dates))
    if cutoff_dates:
        metrics_all: List[Dict[str, object]] = []
        plot_payloads: List[Dict[str, object]] = []
        console_lines: List[str] = []
        for cutoff in cutoff_dates:
            if cutoff <= price_df_full["date"].min() or cutoff >= price_df_full["date"].max():
                console_lines.append(f"[WARN] {label} cutoff {cutoff.date()} fora do intervalo de dados; ignorando.")
                continue
            try:
                rows, plot_info, lines = _run_historical_forecast(
                    price_series,
                    price_df_full,
                    args,
                    cutoff,
                    forecast_horizon,
                    label,
                )
            except Exception as exc:  # pragma: no cover
                console_lines.append(f"[WARN] {label} cutoff {cutoff.date()} falhou: {exc}")
                continue
            metrics_all.extend(rows)
            if plot_info is not None and args.compare_real:
                plot_payloads.append(plot_info)
            console_lines.extend(lines)
        if not metrics_all:
            raise RuntimeError(f"Nenhuma previsao historica produzida para {label}.")
        metrics_df = pd.DataFrame(metrics_all)
        metrics_df["alpha_delta"] = metrics_df["alpha_real"] - metrics_df["alpha_pred"]
        metrics_df["entropy_delta"] = metrics_df["entropy_real"] - metrics_df["entropy_pred"]
        metrics_path = output_dir / "historical_forecast_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        for line in console_lines:
            print(line)
        if len(cutoff_dates) == 1:
            display_cols = [
                "mode_label",
                "phase_pred",
                "phase_real",
                "quality",
                "alpha_pred",
                "alpha_real",
                "alpha_delta",
                "entropy_pred",
                "entropy_real",
                "price_cutoff",
                "predicted_price",
                "actual_price",
                "error_pct",
                "direction_pred",
                "direction_real",
            ]
            single_df = metrics_df[display_cols].copy()
            for col in [
                "alpha_pred",
                "alpha_real",
                "alpha_delta",
                "entropy_pred",
                "entropy_real",
                "price_cutoff",
                "predicted_price",
                "actual_price",
                "error_pct",
            ]:
                single_df[col] = pd.to_numeric(single_df[col], errors="coerce").map(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "nan"
                )
            print(f"\nResumo por modo ({label}):")
            print(single_df.to_string(index=False))
        else:
            pivot = metrics_df.pivot_table(
                index="cutoff_date",
                columns="mode_label",
                values="error_pct",
                aggfunc="mean",
            )
            print(f"\nErro medio por cutoff (%) - {label}:")
            print(pivot.round(2).to_string())
        print(f"[RESULT] [{label}] Historico salvo em: {metrics_path}")
        if args.compare_real and plot_payloads:
            plot_path = output_dir / "historical_forecast.png"
            _plot_historical_forecasts(plot_payloads, plot_path)
            print(f"[RESULT] [{label}] Grafico historico gerado em: {plot_path}")
        return
    bin_edges, bin_centers = discretize_returns(
        price_series.data[price_series.return_column],
        num_bins=args.bins,
        method=args.bin_method,
        clip=args.clip,
    )
    _, analyses, df = _run_analysis_for_series(price_series, bin_edges, bin_centers, args)
    reference_mode = MODE_HADAMARD if MODE_HADAMARD in df["mode"].unique() else MODE_CLASSICAL
    low_thr, high_thr = _compute_phase_thresholds(df, reference_mode, args)
    df["phase"] = df["dispersion_alpha"].apply(lambda alpha: classify_phase(alpha, low_thr, high_thr))
    phase_counts = df[df["mode"] == reference_mode]["phase"].value_counts().to_dict()
    csv_path = output_dir / "window_metrics.csv"
    df.to_csv(csv_path, index=False)
    phase_regions = _compute_phase_regions(df, reference_mode)
    df["window_start"] = _as_naive_datetime(df["window_start"])
    df["window_end"] = _as_naive_datetime(df["window_end"])
    last_window = analyses[-1]
    last_price = float(price_df_full["price"].iloc[-1])
    last_date = price_df_full["date"].iloc[-1]
    phase_by_mode = {
        mode: classify_phase(metrics.dispersion_alpha, low_thr, high_thr)
        for mode, metrics in last_window.metrics.items()
    }
    forecast_curves, forecast_summaries = forecast_from_distribution(
        last_window.distribution,
        bin_centers,
        last_price=last_price,
        steps=forecast_horizon,
        phase_by_mode=phase_by_mode,
        return_mode=args.return_method,
    )
    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
    )
    figure_path = output_dir / "market_overview.png"
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    _plot_price_with_forecast(axes[0], price_df_full, forecast_dates, forecast_curves)
    _plot_alpha(axes[1], df, phase_regions)
    _plot_entropy(axes[2], df)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    curve_map = {curve.mode: curve for curve in forecast_curves}
    summary_rows = []
    for summary in forecast_summaries:
        curve = curve_map.get(summary.mode)
        final_price = float(curve.predicted_prices[-1]) if curve and curve.predicted_prices.size else last_price
        return_units = _return_units(summary.return_mode, args.value_col)
        summary_rows.append(
            {
                "mode": summary.mode,
                "mode_label": MODE_LABELS.get(summary.mode, summary.mode),
                "return_mode": summary.return_mode,
                "phase": summary.phase,
                "direction": summary.direction,
                "return_units": return_units,
                "total_return_raw": summary.total_return,
                "total_return_display": _format_return_value(summary, summary.total_return, args.value_col),
                "avg_return_raw": summary.avg_return,
                "avg_return_display": _format_return_value(summary, summary.avg_return, args.value_col),
                "volatility_raw": summary.volatility,
                "volatility_display": _format_return_value(summary, summary.volatility, args.value_col),
                "predicted_price": final_price,
                "forecast_steps": forecast_horizon,
            }
        )
    forecast_summary_df = pd.DataFrame(summary_rows)
    forecast_summary_df.to_csv(output_dir / "market_forecast.csv", index=False)
    detail_rows = []
    for curve in forecast_curves:
        for step, (date, ret, cum, price, std) in enumerate(
            zip(forecast_dates, curve.expected_returns, curve.cumulative_returns, curve.predicted_prices, curve.std_per_step),
            start=1,
        ):
            detail_rows.append(
                {
                    "mode": curve.mode,
                    "mode_label": MODE_LABELS.get(curve.mode, curve.mode),
                    "step": step,
                    "date": date,
                    "expected_return": ret,
                    "cumulative_return": cum,
                    "predicted_price": price,
                    "std_return": std,
                }
            )
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(output_dir / "market_forecast_detail.csv", index=False)
    reference_metrics = last_window.metrics[reference_mode]
    phase_description = describe_phase(reference_metrics.dispersion_alpha, reference_metrics.entropy_final, low_thr, high_thr)
    print(f"[RESULT] [{label}] Fase atual ({MODE_LABELS.get(reference_mode, reference_mode)}): {phase_description}")
    reference_curve = curve_map.get(reference_mode)
    ref_summary = next((s for s in forecast_summaries if s.mode == reference_mode), None)
    if reference_curve is not None and ref_summary is not None:
        total_display = _format_return_value(ref_summary, ref_summary.total_return, args.value_col)
        volatility_display = _format_return_value(ref_summary, ref_summary.volatility, args.value_col)
        units = _return_units(ref_summary.return_mode, args.value_col)
        if units == "%":
            total_str = f"{total_display:.2f}%"
            vol_str = f"{volatility_display:.2f}%"
        else:
            total_str = f"{total_display:.2f} {units}"
            vol_str = f"{volatility_display:.2f} {units}"
        direction_label = ref_summary.direction
        print(
            f"[RESULT] [{label}] Previsao ({MODE_LABELS.get(reference_mode, reference_mode)}): "
            f"{direction_label} esperada ~ {total_str} +/- {vol_str} nos proximos {forecast_horizon} dias"
        )
    print(f"[RESULT] [{label}] Metricas consolidadas salvas em: {csv_path}")
    print(f"[RESULT] [{label}] Previsoes salvas em: {output_dir / 'market_forecast.csv'}")
    print(f"[RESULT] [{label}] Grafico gerado em: {figure_path}")
    print(f"[INFO] [{label}] Phase thresholds usados: low={low_thr:.3f}, high={high_thr:.3f}")
    print(f"[INFO] [{label}] Fases observadas ({MODE_LABELS.get(reference_mode, reference_mode)}): {phase_counts}")
def _parse_clip(value: Optional[str]) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("clip must be provided as min,max")
    low, high = map(float, parts)
    if low >= high:
        raise argparse.ArgumentTypeError("clip requires min < max")
    return low, high
def _resolve_symbol(symbol: str) -> str:
    key = symbol.upper()
    return SYMBOL_ALIASES.get(key, symbol)


def _load_series(args: argparse.Namespace, symbol: Optional[str] = None, csv_path: Optional[str] = None) -> PriceSeries:
    if csv_path is not None:
        frame = load_value_csv(Path(csv_path), date_column=args.date_col, value_column=args.value_col)
    else:
        if symbol is None:
            raise ValueError("Symbol must be provided when csv_path is not set.")
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance is not installed; use --csv to provide a local dataset.")
        resolved_symbol = _resolve_symbol(symbol)
        frame = download_price_series(
            resolved_symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
            price_column=args.price_column,
        )
    if args.month_zscore:
        frame["month"] = frame["date"].dt.month
        grouped = frame.groupby("month")["price"]

        def _zscore(series: pd.Series) -> pd.Series:
            std = series.std(ddof=0)
            if std == 0 or np.isnan(std):
                return series - series.mean()
            return (series - series.mean()) / std

        frame["price"] = grouped.transform(_zscore)
        frame.drop(columns=["month"], inplace=True)
    return prepare_returns(frame, method=args.return_method)
def _summaries_to_dataframe(analyses) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for window in analyses:
        for mode, metrics in window.metrics.items():
            rows.append(
                {
                    "window_start": pd.Timestamp(window.start),
                    "window_end": pd.Timestamp(window.end),
                    "mode": mode,
                    "entropy_initial": metrics.entropy_initial,
                    "entropy_final": metrics.entropy_final,
                    "entropy_change": metrics.entropy_change,
                    "dispersion_alpha": metrics.dispersion_alpha,
                    "dispersion_alpha_noisy": metrics.dispersion_alpha_noisy,
                    "dispersion_sensitivity": metrics.dispersion_sensitivity,
                }
            )
    return pd.DataFrame(rows)
def _compute_phase_regions(df: pd.DataFrame, mode: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    regions: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    mode_df = df[df["mode"] == mode].sort_values("window_start")
    if mode_df.empty:
        return regions
    current_phase = mode_df.iloc[0]["phase"]
    region_start = mode_df.iloc[0]["window_start"]
    for _, row in mode_df.iloc[1:].iterrows():
        phase = row["phase"]
        start = row["window_start"]
        if phase != current_phase:
            regions.append((region_start, row["window_start"], current_phase))
            region_start = start
            current_phase = phase
    regions.append((region_start, mode_df.iloc[-1]["window_end"], current_phase))
    return regions
def _plot_price_with_forecast(
    ax: plt.Axes,
    price_df: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    forecast_curves,
) -> None:
    ax.plot(price_df["date"], price_df["price"], color="black", linewidth=1.6, label="Preco real")
    last_price = price_df["price"].iloc[-1]
    ax.scatter([price_df["date"].iloc[-1]], [last_price], color="black")
    for curve in forecast_curves:
        if curve.predicted_prices.size == 0:
            continue
        label = f"{MODE_LABELS.get(curve.mode, curve.mode)} - previsao"
        ax.plot(
            forecast_dates,
            curve.predicted_prices,
            linestyle="--",
            linewidth=1.8,
            color=MODE_COLORS.get(curve.mode, "#444444"),
            label=label,
        )
    ax.set_ylabel("Preco")
    ax.set_title("Preco historico e projecoes")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
def _plot_alpha(
    ax: plt.Axes,
    df: pd.DataFrame,
    phase_regions: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
) -> None:
    plotted_phases = set()
    for start, end, phase in phase_regions:
        if not pd.isna(start) and not pd.isna(end):
            color = PHASE_COLORS.get(phase, "#dddddd")
            label = f"Fase: {phase}" if phase not in plotted_phases else None
            ax.axvspan(start, end, color=color, alpha=0.12, label=label)
            plotted_phases.add(phase)
    pivot = df.pivot_table(index="window_end", columns="mode", values="dispersion_alpha")
    pivot.sort_index(inplace=True)
    for mode, series in pivot.items():
        series = series.dropna()
        if series.empty:
            continue
        ax.plot(
            series.index,
            series.values,
            linewidth=2.0,
            color=MODE_COLORS.get(mode, "#444444"),
            label=f"alpha - {MODE_LABELS.get(mode, mode)}",
        )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.4)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.4)
    ax.set_ylabel("alpha (disp.)")
    ax.set_title("Evolucao do expoente de dispersao")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
def _plot_entropy(ax: plt.Axes, df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="window_end", columns="mode", values="entropy_final")
    pivot.sort_index(inplace=True)
    for mode, series in pivot.items():
        series = series.dropna()
        if series.empty:
            continue
        ax.plot(
            series.index,
            series.values,
            linewidth=2.0,
            color=MODE_COLORS.get(mode, "#444444"),
            label=f"Entropia final - {MODE_LABELS.get(mode, mode)}",
        )
    ax.set_xlabel("Data")
    ax.set_ylabel("Entropia (bits)")
    ax.set_title("Entropia final por janela")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
def _format_pct(x: float) -> float:
    if not np.isfinite(x):
        return float('nan')
    if abs(x) < 0.5:
        return float(np.expm1(x) * 100.0)
    return float(x * 100.0)
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quantum/classical walk analysis for financial time series.",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        help="Ticker symbol for yfinance download (repeat flag to add more).",
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV with columns date,value (generic series).")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD) for download.")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD) for download.")
    parser.add_argument("--interval", type=str, default="1d", help="Sampling interval for yfinance (default: 1d).")
    parser.add_argument("--price-column", type=str, default="Adj Close", help="Column to use as price.")
    parser.add_argument("--date-col", type=str, default="date", help="Date column for CSV input.")
    parser.add_argument("--value-col", type=str, default="price", help="Value column for CSV input.")
    parser.add_argument("--return-method", type=str, choices=("log", "simple", "diff"), default="log")
    parser.add_argument("--bins", type=int, default=51, help="Number of discretization bins / graph nodes.")
    parser.add_argument("--bin-method", type=str, choices=("quantile", "linear"), default="quantile")
    parser.add_argument("--clip", type=_parse_clip, default=None, help="Optional clipping bounds for returns (min,max).")
    parser.add_argument("--window", type=int, default=60, help="Sliding window size in days.")
    parser.add_argument("--step", type=int, default=10, help="Window hop size (days).")
    parser.add_argument("--walk-steps", type=int, default=40, help="Number of walk steps to simulate per window.")
    parser.add_argument("--fit-min", type=int, default=5, help="Minimum step for dispersion fit (alpha).")
    parser.add_argument("--fit-max", type=int, default=None, help="Maximum step for dispersion fit.")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level for alpha sensitivity (0-1).")
    parser.add_argument("--month-zscore", action="store_true", help="Normalize values by monthly z-score before computing retornos.")
    parser.add_argument("--phase-low", type=float, default=0.6, help="Threshold below which regime is difusive.")
    parser.add_argument("--phase-high", type=float, default=0.85, help="Threshold above which regime is coherent.")
    parser.add_argument("--forecast-steps", type=int, default=15, help="Forecast horizon (steps/days) for projection.")
    parser.add_argument("--forecast-days", type=int, default=None, help="Forecast horizon in business days (overrides --forecast-steps).")
    parser.add_argument("--auto-phase", action="store_true", help="Infer phase thresholds from alpha quantiles.")
    parser.add_argument(
        "--phase-lower-quantile",
        type=float,
        default=0.35,
        help="Quantile for lower alpha threshold when auto-phase is enabled.",
    )
    parser.add_argument(
        "--phase-upper-quantile",
        type=float,
        default=0.65,
        help="Quantile for upper alpha threshold when auto-phase is enabled.",
    )
    parser.add_argument(
        "--cutoff",
        action="append",
        default=None,
        help="Historical cutoff date (YYYY-MM-DD). Use multiple times for several tests.",
    )
    parser.add_argument(
        "--compare-real",
        action="store_true",
        help="Generate comparison plot for historical forecasts (requires --cutoff).",
    )
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR), help="Directory for results.")
    parser.add_argument("--outdir", dest="output", type=str, help="Alias for --output.")
    args = parser.parse_args(argv)
    forecast_horizon = args.forecast_days if args.forecast_days is not None else args.forecast_steps
    if forecast_horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    if args.csv and args.symbol:
        parser.error("Use --symbol (opcional) ou --csv, mas nao ambos.")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks: List[Tuple[Optional[str], Optional[str]]] = []
    if args.csv:
        tasks.append((None, args.csv))
    else:
        symbols = args.symbol if args.symbol else ["SPY", "^BVSP"]
        tasks.extend((symbol, None) for symbol in symbols)
    for symbol, csv_path in tasks:
        label = Path(csv_path).stem if csv_path else symbol or "dataset"
        print(f"\n=== {label} ===")
        price_series = _load_series(args, symbol=symbol, csv_path=csv_path)
        price_series.data["date"] = _as_naive_datetime(price_series.data["date"])
        price_series.data["price"] = price_series.data["price"].astype(float)
        sub_output = output_root / _safe_label(label)
        sub_output.mkdir(parents=True, exist_ok=True)
        _process_dataset(price_series, args, forecast_horizon, sub_output, label)


if __name__ == "__main__":
    main()
