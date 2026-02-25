from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'simulations'
OUT.mkdir(parents=True, exist_ok=True)

# Choose 10 assets from available data (fall back to those present in dados/brutos)
DEFAULT_ASSETS = ['AAPL', 'SPY', 'MSFT', 'GOOG', 'AMZN', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'TLT', 'GC_F']


def load_price(symbol: str) -> pd.Series:
    # Try results history folder first
    cand = ROOT / 'results' / 'today_forecast_eval' / symbol / 'history.csv'
    if cand.exists():
        df = pd.read_csv(cand, parse_dates=['date'])
        df = df.set_index('date').sort_index()
        for col in ('close','Close','price','adj_close','Adj Close'):
            if col in df.columns:
                return df[col].astype(float)
    # Try dados/brutos CSV (case-insensitive)
    br = ROOT / 'dados' / 'brutos'
    names = [f"{symbol}.csv", f"{symbol.lower()}.csv", f"{symbol.upper()}.csv"]
    for name in names:
        p = br / name
        if p.exists():
            df = pd.read_csv(p, parse_dates=['date'])
            df = df.set_index('date').sort_index()
            for col in ('close','Close','price','adj_close','Adj Close'):
                if col in df.columns:
                    return df[col].astype(float)
    # fallback: any file containing symbol
    for p in br.glob(f"*{symbol}*.csv"):
        try:
            df = pd.read_csv(p, parse_dates=['date']).set_index('date').sort_index()
            for col in ('close','Close','price','adj_close','Adj Close'):
                if col in df.columns:
                    return df[col].astype(float)
        except Exception:
            continue
    raise FileNotFoundError(f'No price series found for {symbol}')


def classify_phase(series: pd.Series, window: int = 21) -> pd.Series:
    # Simple phase classification using rolling volatility and rolling mean slope
    series = series.dropna().astype(float)
    ret = series.pct_change().fillna(0)
    vol = ret.rolling(window, min_periods=5).std()
    trend = series.pct_change(periods=window).rolling(window, min_periods=5).mean()
    # baseline thresholds based on long-run quantiles
    vq = vol.quantile(0.66)
    phase = pd.Series('neutral', index=series.index)
    phase[(vol > vq) & (trend < 0)] = 'chaotic'
    phase[(vol > vq) & (trend >= 0)] = 'volatile_up'
    phase[(vol <= vq) & (trend > 0)] = 'coherent_up'
    phase[(vol <= vq) & (trend <= 0)] = 'coherent_down'
    return phase.fillna('neutral')


def naive_forecast(series: pd.Series, start: datetime, periods: int) -> pd.Series:
    # Naive: last observed value repeated
    last = series[:start].dropna().iloc[-1]
    idx = pd.date_range(start=start, periods=periods, freq='D')
    return pd.Series(last, index=idx)


def run_for_asset(symbol: str, outdir: Path, years=(2023, 2024, 2025)) -> list[Path]:
    
    
    
    s = load_price(symbol)
    s = s.dropna()
    # resample to business days and forward-fill small gaps, ensure clean numeric
    s = s.resample('B').ffill().dropna()
    end = s.index.max()
    start_10y = end - pd.DateOffset(years=10)
    s10 = s[s.index >= start_10y]
    if s10.empty or len(s10) < 50:
        raise ValueError('Not enough history for ' + symbol)
    phase = classify_phase(s10)

    pics = []
    ml_rows = []
    # We'll consider requested years
    for y in years:
        sim_start = pd.Timestamp(f'{y}-01-01')
        # align to next available business day
        if sim_start not in s10.index:
            pos = s10.index.get_indexer([sim_start], method='bfill')[0]
            if pos == -1:
                continue
            sim_start = s10.index[pos]
        horizon = 365
        actual = s10[s10.index >= sim_start][:horizon]
        if actual.empty:
            continue
        # forecast using simple drift from last 252 trading days
        lookback = 252
        history = s10[s10.index < sim_start].iloc[-lookback:]
        if history.empty:
            forecast = naive_forecast(s10, sim_start, horizon)
        else:
            mu = history.pct_change().mean()
            last = history.iloc[-1]
            idx = pd.date_range(start=sim_start, periods=horizon, freq='D')
            rel = (1 + mu) ** np.arange(1, len(idx) + 1)
            forecast = pd.Series(last * rel, index=idx)

        # ensure alignment lengths
        idx_common = forecast.index.intersection(actual.index)
        if idx_common.empty:
            continue

        # build plot: top = actual series with phase underlay; bottom = forecast vs real
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})

        ax = axes[0]
        ax.plot(s10.index, s10.values, color='black', label='price')
        # shade phases with legend entries
        phase_colors = {'chaotic':'#ffcccc','volatile_up':'#ffe6b3','coherent_up':'#ccffcc','coherent_down':'#cce5ff','neutral':'#f0f0f0'}
        handles = []
        from matplotlib.patches import Patch
        for ph, col in phase_colors.items():
            mask = phase == ph
            if mask.any():
                ax.fill_between(phase.index, ax.get_ylim()[0], ax.get_ylim()[1], where=mask, facecolor=col, alpha=0.25)
                handles.append(Patch(facecolor=col, label=ph))
        ax.set_title(f'{symbol} — realized (last 10y) — start {y}')
        ax.legend(handles=handles, loc='upper left')

        ax2 = axes[1]
        actual_plot = actual.reindex(forecast.index).dropna()
        ax2.plot(forecast.index[:len(actual_plot)], forecast[:len(actual_plot)], label='model_pred', color='C1')
        ax2.plot(actual_plot.index, actual_plot.values, label='real', color='C0', alpha=0.8)
        ax2.set_title(f'Forecast started {y} — horizon {horizon} days')
        ax2.legend()

        fn = outdir / f'{symbol}_{y}_sim.png'
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)
        pics.append(fn)

        # investment comparison per phase: determine phase at sim_start
        ph_at_start = phase.get(sim_start, 'neutral')
        # buy-and-hold return over horizon (real)
        real_series = actual.reindex(forecast.index).dropna()
        if len(real_series) < 2:
            continue
        real_ret = real_series.pct_change().fillna(0)
        bh_cum = (1 + real_ret).cumprod()[-1] - 1
        # simple signal strategy from forecast: long when next-day forecast > today
        fc = forecast.reindex(real_series.index)
        sig = (fc.shift(-1) > fc).astype(int).fillna(0)
        strat_ret = sig * real_ret
        strat_cum = (1 + strat_ret).cumprod()[-1] - 1
        # save summary
        summary = {'symbol': symbol, 'year': y, 'phase_at_start': ph_at_start, 'bh_cumret': float(bh_cum), 'strat_cumret': float(strat_cum), 'n_days': int(len(real_series))}
        (outdir / f'{symbol}_{y}_summary.json').write_text(json.dumps(summary))

        # prepare ML row: features at sim_start
        feat_window = 30
        hist_feat = s10[:sim_start].iloc[-feat_window:]
        feat = {
            'symbol': symbol,
            'year': y,
            'price': float(hist_feat.iloc[-1]) if not hist_feat.empty else float(last),
            'ret30_mean': float(hist_feat.pct_change().mean()) if not hist_feat.empty else 0.0,
            'ret30_vol': float(hist_feat.pct_change().std()) if not hist_feat.empty else 0.0,
            'phase_at_start': ph_at_start,
            'target_1y_ret': float(bh_cum)
        }
        ml_rows.append(feat)

        # save backtest CSV
        bt_df = pd.DataFrame({'date': real_series.index, 'real_price': real_series.values, 'forecast': fc.reindex(real_series.index).values, 'signal': sig.values, 'real_ret': real_ret.values, 'strat_ret': strat_ret.values})
        bt_df.to_csv(outdir / f'{symbol}_{y}_backtest.csv', index=False)

    return pics, ml_rows


def main():
    # remove previous outputs to avoid stale files
    for p in OUT.glob('*'):
        try:
            if p.is_file():
                p.unlink()
            else:
                import shutil
                shutil.rmtree(p)
        except Exception:
            continue

    # build asset list: prefer DEFAULT_ASSETS but fill from dados/brutos if missing
    assets = []
    br = ROOT / 'dados' / 'brutos'
    # try defaults first
    for a in DEFAULT_ASSETS:
        try:
            _ = load_price(a)
            assets.append(a)
        except Exception:
            continue
    # supplement from brutos if less than 10
    if len(assets) < 10 and br.exists():
        for p in sorted(br.glob('*.csv')):
            name = p.stem
            if name.upper() not in [x.upper() for x in assets]:
                assets.append(name)
            if len(assets) >= 10:
                break
    out = OUT
    produced = []
    ml_rows_all = []
    for sym in assets:
        try:
            pics, mlrows = run_for_asset(sym, out)
            produced.extend(pics)
            ml_rows_all.extend(mlrows)
        except Exception as e:
            print('Skipping', sym, 'error', e)

    # save ML dataset
    if ml_rows_all:
        mldf = pd.DataFrame(ml_rows_all)
        mldf.to_csv(out / 'ml_dataset.csv', index=False)

    # create index JSON for easy viewing
    idx = {'images': [str(p.relative_to(ROOT)) for p in produced]}
    (out / 'index.json').write_text(json.dumps(idx, indent=2))
    print('Produced', len(produced), 'images in', out)


if __name__ == '__main__':
    main()
