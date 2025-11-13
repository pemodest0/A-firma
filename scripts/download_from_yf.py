#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def _ensure_series(candidate: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(candidate, pd.Series):
        return candidate
    if isinstance(candidate, pd.DataFrame):
        if candidate.shape[1] == 1:
            return candidate.iloc[:, 0]
        # Pegamos a primeira coluna como fallback para manter compatibilidade.
        return candidate.iloc[:, 0]
    raise TypeError(f"Objeto inesperado retornado ao extrair preços: {type(candidate)}")


def _extract_price_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Extrair a série de preços ajustados independente do formato retornado pelo yfinance."""
    if 'Adj Close' in df.columns:
        return _ensure_series(df['Adj Close'])

    # Quando o yfinance retorna MultiIndex, percorremos coluna a coluna em busca do campo.
    if isinstance(df.columns, pd.MultiIndex):
        price_candidate: pd.Series | None = None
        for col in df.columns:
            parts = col if isinstance(col, tuple) else (col,)
            if 'Adj Close' in parts:
                series = _ensure_series(df[col])
                if ticker in parts:
                    return series
                price_candidate = series
        if price_candidate is not None:
            return price_candidate

        # Fallback usando 'Close' caso não haja ajuste.
        close_candidate: pd.Series | None = None
        for col in df.columns:
            parts = col if isinstance(col, tuple) else (col,)
            if 'Close' in parts:
                series = _ensure_series(df[col])
                if ticker in parts:
                    return series
                close_candidate = series
        if close_candidate is not None:
            return close_candidate

    if 'Close' in df.columns:
        return _ensure_series(df['Close'])

    available = ', '.join(map(str, df.columns))
    raise KeyError(f"Coluna de preço não encontrada para {ticker} (colunas: {available})")


def download_ticker(ticker: str, start: str, end: str, output: Path) -> None:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {ticker}")
    price_series = _extract_price_series(df, ticker)
    frame = price_series.to_frame(name='price')
    frame.index.name = 'date'
    frame = frame.reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Salvo {output} ({frame.shape[0]} linhas)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa preços de ativos via yfinance em CSV padronizado.")
    parser.add_argument('--tickers', nargs='+', required=True, help='Lista de tickers conforme Yahoo Finance (ex.: AAPL ^BVSP SPY)')
    parser.add_argument('--start', type=str, default='2016-01-01', help='Data inicial (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-31', help='Data final (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='dados/brutos/yf', help='Diretório onde salvar CSVs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for ticker in args.tickers:
        safe_name = ticker.replace('^', '').replace(' ', '_').replace('/', '_').upper()
        output_path = output_dir / f"{safe_name}.csv"
        download_ticker(ticker, args.start, args.end, output_path)


if __name__ == '__main__':
    main()
