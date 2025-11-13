#!/usr/bin/env python3
"""Gerar plot de fases do mercado e escolher bins automaticamente com base em MAE."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

from dados.brutos.financial_loader import download_price_series, prepare_returns, discretize_returns, generate_return_windows
from classical_walk import simulate_classical_walk
from graph_utils import line_graph
import src.quantum_walk as qw


OUT_DIR = Path('results_daily_ibov/_BVSP')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    p = OUT_DIR / 'daily_forecast_metrics.csv'
    if p.exists():
        return pd.read_csv(p, parse_dates=['date'])
    # fallback to smaller comparison file
    p2 = OUT_DIR / 'first10_oct2025_comparison.csv'
    if p2.exists():
        return pd.read_csv(p2, parse_dates=['date'])
    # else download series and compute basic metrics
    series = download_price_series('^BVSP', start='2015-01-01', end='2025-10-31')
    series = prepare_returns(series, method='log')
    df = series.data.copy()
    # fabricate simple metrics
    df['alpha'] = 0.0
    df['entropy'] = 3.0
    return df


def cluster_phases(df):
    # expect df to have 'alpha' and 'entropy'
    X = df[['alpha', 'entropy']].fillna(0.0).values
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    labels = kmeans.labels_
    df = df.copy()
    df['phase_label'] = labels
    # map labels to named phases by ordering mean alpha
    mapping = {}
    means = []
    for lab in sorted(np.unique(labels)):
        means.append((lab, df.loc[df['phase_label']==lab, 'alpha'].mean()))
    means_sorted = sorted(means, key=lambda x: x[1])
    name_map = {means_sorted[0][0]: 'Difusiva', means_sorted[1][0]: 'Caotica', means_sorted[2][0]: 'Interferencia quantica'}
    df['phase_name'] = df['phase_label'].map(name_map)
    return df


def plot_market_phases(df):
    # df must have date, price, phase_name
    fig, ax = plt.subplots(figsize=(16,5))
    df_sorted = df.sort_values('date').reset_index(drop=True)
    # smooth the price for visual clarity
    price_series = df_sorted['price'].astype(float)
    smooth = price_series.rolling(window=3, min_periods=1, center=True).median()
    ax.plot(df_sorted['date'], price_series, color='#111111', linewidth=1.0, alpha=0.6, label='Preço real (raw)', zorder=5)
    ax.plot(df_sorted['date'], smooth, color='#000000', linewidth=1.8, label='Preço (suavizado)', zorder=6)
    # plot model predictions if present: look for mode/price_pred or pred_cl/pred_q
    try:
        if 'mode' in df_sorted.columns and 'price_pred' in df_sorted.columns:
            # pivot to have columns per mode
            piv = df_sorted.pivot_table(index='date', columns='mode', values='price_pred', aggfunc='first')
            if 'classical' in piv.columns:
                ax.plot(piv.index, piv['classical'], linestyle='--', color='#2b7fb8', linewidth=1.2, label='Previsão clássico', zorder=7)
            # try common quantum mode labels
            q_modes = [c for c in piv.columns if 'quant' in str(c).lower() or 'grover' in str(c).lower() or 'hadamard' in str(c).lower()]
            if q_modes:
                # plot first quantum mode found
                ax.plot(piv.index, piv[q_modes[0]], linestyle='-.', color='#e07a5f', linewidth=1.2, label='Previsão quântico', zorder=7)
    except Exception:
        pass
    colors = {'Difusiva': '#c7e9c0', 'Caotica': '#fdd0a2', 'Interferencia quantica': '#f4b6c2'}
    # paint contiguous regions with an annotated legend-like box
    spans = []
    prev = None
    start = None
    last = None
    for _, row in df_sorted.iterrows():
        ph = row['phase_name']
        if prev is None:
            prev = ph
            start = row['date']
            last = row['date']
            continue
        if ph != prev:
            spans.append((start, last, prev))
            start = row['date']
            prev = ph
        last = row['date']
    spans.append((start, last, prev))
    # draw spans with subtle borders
    for s, e, name in spans:
        ax.axvspan(s, e, color=colors.get(name, '#ddd'), alpha=0.22, zorder=1)
    # add legend entry explaining background colors (phases)
    from matplotlib.patches import Patch
    phase_patches = [Patch(facecolor=colors['Difusiva'], alpha=0.5, label='Difusiva'), Patch(facecolor=colors['Caotica'], alpha=0.5, label='Caótica'), Patch(facecolor=colors['Interferencia quantica'], alpha=0.5, label='Interferência quântica')]
    # construct final legend with line artists and phase patches
    handles, labels = ax.get_legend_handles_labels()
    # append phase patches
    handles.extend(phase_patches)
    ax.legend(handles=handles, fontsize=9, frameon=True, loc='upper left')
    # annotate long spans (optional): only annotate spans longer than threshold
    total_days = (df_sorted['date'].max() - df_sorted['date'].min()).days if len(df_sorted)>1 else 1
    for s, e, name in spans:
        span_days = (e - s).days if hasattr(e, 'days') else 0
        if span_days >= max(3, total_days*0.02):
            mid = s + (e - s) / 2
            ax.text(mid, df_sorted['price'].max() * 0.985, name, ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_title('^BVSP — Preço real e fases de mercado')
    ax.set_ylabel('Preço')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_DIR / 'market_phases.png'
    fig.savefig(out, dpi=300)
    print('Saved', out)


def auto_choose_bins(series_df, dates_sample, bins_candidates=[7,9,11,15,21,31,51], window=30):
    # simple grid: choose bins with minimal MAE for classical model over dates_sample
    results = []
    # prepare returns
    ret = series_df.copy()
    ret = prepare_returns(ret, method='log')
    price_df = ret.data
    for bins in bins_candidates:
        be, bc = discretize_returns(price_df[ret.return_column], num_bins=bins, method='quantile')
        windows = generate_return_windows(ret, be, window=window, step=1)
        errors = []
        for d in dates_sample:
            # pick last window ending <= d
            win = None
            for w in reversed(windows):
                if pd.Timestamp(w.end) <= pd.Timestamp(d):
                    win = w
                    break
            if win is None:
                continue
            # price_today and real
            idx = price_df[price_df['date']==pd.Timestamp(win.end)].index
            if len(idx)==0:
                price_today = float(price_df[price_df['date']<=pd.Timestamp(d)]['price'].iloc[-1])
                idx = price_df[price_df['date']<=pd.Timestamp(d)].index[-1]
            else:
                idx = idx[0]
                price_today = float(price_df.loc[idx, 'price'])
            if idx+1 < len(price_df):
                price_real = float(price_df.loc[idx+1, 'price'])
            else:
                continue
            graph = line_graph(len(bc))
            classical_result = simulate_classical_walk(graph, 30, initial_distribution=win.distribution)
            exp_ret_cl = float(np.dot(classical_result.distributions[min(1, classical_result.distributions.shape[0]-1)], bc))
            pred_cl = price_today * math.exp(exp_ret_cl)
            errors.append(abs(pred_cl - price_real))
        mae = float(np.mean(errors)) if errors else float('nan')
        results.append({'bins': bins, 'mae_cl': mae})
    df = pd.DataFrame(results).sort_values('mae_cl')
    out = OUT_DIR / 'auto_bins_choice.json'
    # also save full candidates as JSONL for inspection
    df.to_json(OUT_DIR / 'auto_bins_candidates.json', orient='records')
    best = df.iloc[0].to_dict()
    print('Auto bins choice best:', best)
    with open(out, 'w') as f:
        json.dump(best, f)
    return best


def main():
    df = load_metrics()
    # require 'date' and 'price'
    if 'date' not in df.columns:
        # try transform
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
    # ensure price column (support multiple possible names)
    if 'price' not in df.columns:
        if 'price_today' in df.columns:
            df['price'] = df['price_today']
        elif 'price_real' in df.columns:
            df['price'] = df['price_real']
        elif 'close' in df.columns:
            df['price'] = df['close']
        elif 'adjclose' in df.columns:
            df['price'] = df['adjclose']

    # ensure alpha/entropy exist
    if 'alpha' not in df.columns:
        df['alpha'] = 0.0
    if 'entropy' not in df.columns:
        df['entropy'] = 3.0

    clustered = cluster_phases(df)
    plot_market_phases(clustered)

    # auto choose bins based on MAE over the available dates (sample first 30 days)
    series = download_price_series('^BVSP', start='2015-01-01', end='2025-10-31')
    series = prepare_returns(series, method='log')
    dates_sample = clustered['date'].sort_values().unique()[:30]
    best = auto_choose_bins(series.data, dates_sample)
    print('Best bins chosen and saved to results directory')


if __name__ == '__main__':
    main()
