import io
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AssyntraxDailyIngestion/1.0; +https://assyntrax.vercel.app)",
    "Accept": "application/json,text/csv,text/plain,*/*",
}

STOOQ_URL_TEMPLATE = "https://stooq.com/q/d/l/?s={symbol}&i=d"
BRAPI_URL_TEMPLATE = "https://brapi.dev/api/quote/{symbol}?{query}"
BINANCE_URL_TEMPLATE = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
COINGECKO_URL_TEMPLATE = (
    "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency=usd&from={start_ts}&to={end_ts}"
)

CRYPTO_BINANCE_MAP = {
    "BTC-USD": ["BTCUSDT"],
    "ETH-USD": ["ETHUSDT"],
    "SOL-USD": ["SOLUSDT"],
    "XRP-USD": ["XRPUSDT"],
    "BNB-USD": ["BNBUSDT"],
    "ADA-USD": ["ADAUSDT"],
    "DOGE-USD": ["DOGEUSDT"],
    "LINK-USD": ["LINKUSDT"],
    "AVAX-USD": ["AVAXUSDT"],
    # Binance migrou a liquidez principal de Polygon para POL; mantemos o alias antigo
    # para não deixar o histórico de MATIC travado se o símbolo legado desaparecer.
    "MATIC-USD": ["MATICUSDT", "POLUSDT"],
    "DOT-USD": ["DOTUSDT"],
    "LTC-USD": ["LTCUSDT"],
    "BCH-USD": ["BCHUSDT"],
    "ETC-USD": ["ETCUSDT"],
    "ATOM-USD": ["ATOMUSDT"],
    "TRX-USD": ["TRXUSDT"],
    "XLM-USD": ["XLMUSDT"],
    "FIL-USD": ["FILUSDT"],
    "ALGO-USD": ["ALGOUSDT"],
    "ICP-USD": ["ICPUSDT"],
    "NEAR-USD": ["NEARUSDT"],
    "APT-USD": ["APTUSDT"],
    "AAVE-USD": ["AAVEUSDT"],
    "ARB-USD": ["ARBUSDT"],
    "CRO-USD": ["CROUSDT"],
    "FET-USD": ["FETUSDT"],
    "HBAR-USD": ["HBARUSDT"],
    "IMX-USD": ["IMXUSDT"],
    "INJ-USD": ["INJUSDT"],
    "MKR-USD": ["MKRUSDT"],
    "ONDO-USD": ["ONDOUSDT"],
    "OP-USD": ["OPUSDT"],
    "PEPE-USD": ["PEPEUSDT"],
    "RUNE-USD": ["RUNEUSDT"],
    "SHIB-USD": ["SHIBUSDT"],
    "SUI-USD": ["SUIUSDT"],
    "TAO-USD": ["TAOUSDT"],
    "TON-USD": ["TONUSDT"],
    "UNI-USD": ["UNIUSDT"],
    "XMR-USD": ["XMRUSDT"],
}

CRYPTO_COINGECKO_MAP = {
    "BTC-USD": ["bitcoin"],
    "ETH-USD": ["ethereum"],
    "SOL-USD": ["solana"],
    "XRP-USD": ["ripple"],
    "BNB-USD": ["binancecoin"],
    "ADA-USD": ["cardano"],
    "DOGE-USD": ["dogecoin"],
    "LINK-USD": ["chainlink"],
    "AVAX-USD": ["avalanche-2"],
    "MATIC-USD": ["matic-network", "polygon-ecosystem-token"],
    "DOT-USD": ["polkadot"],
    "LTC-USD": ["litecoin"],
    "BCH-USD": ["bitcoin-cash"],
    "ETC-USD": ["ethereum-classic"],
    "ATOM-USD": ["cosmos"],
    "TRX-USD": ["tron"],
    "XLM-USD": ["stellar"],
    "FIL-USD": ["filecoin"],
    "ALGO-USD": ["algorand"],
    "ICP-USD": ["internet-computer"],
    "NEAR-USD": ["near"],
    "APT-USD": ["aptos"],
    "AAVE-USD": ["aave"],
    "ARB-USD": ["arbitrum"],
    "CRO-USD": ["crypto-com-chain"],
    "FET-USD": ["artificial-superintelligence-alliance", "fetch-ai"],
    "HBAR-USD": ["hedera-hashgraph"],
    "IMX-USD": ["immutable-x"],
    "INJ-USD": ["injective-protocol"],
    "MKR-USD": ["maker"],
    "ONDO-USD": ["ondo-finance"],
    "OP-USD": ["optimism"],
    "PEPE-USD": ["pepe"],
    "RUNE-USD": ["thorchain"],
    "SHIB-USD": ["shiba-inu"],
    "SUI-USD": ["sui"],
    "TAO-USD": ["bittensor"],
    "TON-USD": ["the-open-network"],
    "UNI-USD": ["uniswap"],
    "XMR-USD": ["monero"],
}


def find_local_data(ticker, base_dir):
    ticker_upper = ticker.upper()
    candidates = []
    for root, _, files in os.walk(base_dir):
        if any(part in root for part in ["venv", "site-packages", "website", "results", ".git"]):
            continue
        for name in files:
            if not name.lower().endswith(".csv"):
                continue
            stem = Path(name).stem.upper()
            if stem == ticker_upper or stem.replace("_CLEANED", "") == ticker_upper:
                candidates.append(Path(root) / name)
    return candidates


def _detect_date_column(columns):
    candidates = ["date", "datetime", "timestamp", "time"]
    lower_map = {col.lower(): col for col in columns}
    for cand in candidates:
        for key, col in lower_map.items():
            if cand == key or key.endswith(cand):
                return col
    return None


def _detect_price_column(columns):
    candidates = [
        "adj close",
        "adj_close",
        "adjclose",
        "adjusted close",
        "adjusted_close",
        "close",
        "price",
    ]
    lower_map = {col.lower(): col for col in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _http_get_text(url: str, timeout_sec: int = 8) -> str:
    req = Request(url, headers=HTTP_HEADERS)
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        cmd = ["curl", "-L", "--fail", "--max-time", str(int(timeout_sec)), url]
        for key, value in HTTP_HEADERS.items():
            cmd.extend(["-H", f"{key}: {value}"])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return proc.stdout


def load_price_series(path):
    df = pd.read_csv(path)
    date_col = _detect_date_column(df.columns)
    price_col = _detect_price_column(df.columns)
    if not date_col or not price_col:
        return None
    df = df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col)
    df.rename(columns={date_col: "date", price_col: "price"}, inplace=True)
    return df


def load_existing_base(path):
    df = load_price_series(path)
    if df is not None and not df.empty:
        return df
    try:
        fallback = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["date", "price"])
    if "date" not in fallback.columns:
        return pd.DataFrame(columns=["date", "price"])
    out = fallback[["date"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["price"] = np.nan
    return out.dropna(subset=["date"]).sort_values("date")


def fetch_yfinance(ticker, start="2009-01-01", end=None):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance not installed; cannot fetch remote data.") from exc

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, group_by="column")
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    df = df.reset_index()
    price_col = "adj close" if "adj close" in df.columns else "close"
    date_col = "Date" if "Date" in df.columns else "date"
    out = df[[date_col, price_col]].copy()
    out.rename(columns={date_col: "date", price_col: "price"}, inplace=True)
    return out


def _to_stooq_symbol(symbol: str) -> str:
    return f"{symbol.strip().upper().replace('.', '-').lower()}.us"


def fetch_stooq(ticker: str) -> pd.DataFrame | None:
    if ticker.upper().endswith(".SA"):
        return None
    stooq_symbol = _to_stooq_symbol(ticker)
    txt = _http_get_text(STOOQ_URL_TEMPLATE.format(symbol=stooq_symbol))
    if not txt.strip() or txt.lstrip().startswith("No data"):
        return None
    df = pd.read_csv(io.StringIO(txt))
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    if "date" not in cols or "close" not in cols:
        return None
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[cols["date"]], errors="coerce"),
            "price": pd.to_numeric(df[cols["close"]], errors="coerce"),
        }
    ).dropna()
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch_brapi(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
    if not ticker.upper().endswith(".SA"):
        return None
    symbol = ticker.upper().replace(".SA", "")
    query = urlencode({"range": "max", "interval": "1d"})
    payload = json.loads(_http_get_text(BRAPI_URL_TEMPLATE.format(symbol=symbol, query=query)))
    results = payload.get("results") or []
    if not results:
        return None
    hist = results[0].get("historicalDataPrice") or []
    if not hist:
        return None
    out = pd.DataFrame(
        {
            "date": pd.to_datetime([row.get("date") for row in hist], unit="s", errors="coerce"),
            "price": pd.to_numeric([row.get("adjustedClose") or row.get("close") for row in hist], errors="coerce"),
        }
    ).dropna()
    if out.empty:
        return None
    if start:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch_binance_crypto(ticker: str, start: str | None = None) -> pd.DataFrame | None:
    symbols = CRYPTO_BINANCE_MAP.get(ticker.upper()) or []
    for symbol in symbols:
        txt = _http_get_text(BINANCE_URL_TEMPLATE.format(symbol=symbol, limit=1000))
        rows = json.loads(txt)
        if not isinstance(rows, list) or not rows:
            continue
        out = pd.DataFrame(
            {
                "date": pd.to_datetime([row[0] for row in rows], unit="ms", errors="coerce"),
                "price": pd.to_numeric([row[4] for row in rows], errors="coerce"),
            }
        ).dropna()
        if start:
            out = out[out["date"] >= pd.Timestamp(start)]
        if not out.empty:
            return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return None


def fetch_coingecko_crypto(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
    coin_ids = CRYPTO_COINGECKO_MAP.get(ticker.upper()) or []
    start_ts = int(pd.Timestamp(start or "2009-01-01").timestamp())
    end_ts = int(pd.Timestamp(end).timestamp()) if end else int(pd.Timestamp.utcnow().timestamp())
    for coin_id in coin_ids:
        txt = _http_get_text(COINGECKO_URL_TEMPLATE.format(coin_id=coin_id, start_ts=start_ts, end_ts=end_ts))
        payload = json.loads(txt)
        prices = payload.get("prices") or []
        if not prices:
            continue
        out = pd.DataFrame(
            {
                "date": pd.to_datetime([row[0] for row in prices], unit="ms", errors="coerce"),
                "price": pd.to_numeric([row[1] for row in prices], errors="coerce"),
            }
        ).dropna()
        if not out.empty:
            return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return None


def fetch_market_data(
    ticker: str,
    start: str = "2009-01-01",
    end: str | None = None,
    *,
    allow_yfinance: bool = True,
) -> tuple[pd.DataFrame | None, str | None]:
    upper = ticker.upper()
    fetchers: list[tuple[str, callable]] = []
    if upper.endswith("-USD"):
        fetchers = [
            ("binance", lambda: fetch_binance_crypto(upper, start=start)),
            ("coingecko", lambda: fetch_coingecko_crypto(upper, start=start, end=end)),
            ("stooq", lambda: fetch_stooq(upper)),
        ]
    elif upper.endswith(".SA"):
        fetchers = [
            ("brapi", lambda: fetch_brapi(upper, start=start, end=end)),
        ]
    else:
        fetchers = [
            ("stooq", lambda: fetch_stooq(upper)),
        ]

    if allow_yfinance:
        fetchers.append(("yfinance", lambda: fetch_yfinance(upper, start=start, end=end)))

    for provider, fn in fetchers:
        try:
            data = fn()
        except Exception:
            data = None
        if data is not None and not data.empty:
            return data, provider
    return None, None


def unify_to_daily(df):
    df = df.copy()
    if "date" not in df.columns:
        date_col = _detect_date_column(df.columns)
        if date_col:
            df.rename(columns={date_col: "date"}, inplace=True)
    if "price" not in df.columns:
        price_col = _detect_price_column(df.columns)
        if price_col:
            df.rename(columns={price_col: "price"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date")
    df = df.drop_duplicates("date", keep="last")
    df["log_price"] = (df["price"]).astype(float).apply(lambda x: np.nan if x <= 0 else x)
    df = df.dropna(subset=["log_price"])
    df["log_price"] = np.log(df["log_price"].astype(float))
    df["r"] = df["log_price"].diff()
    df = df.dropna(subset=["r"])
    return df


def save_cache(df, base_dir, ticker):
    out_dir = Path(base_dir) / "data" / "raw" / "finance" / "yfinance_daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.csv"
    df.to_csv(out_path, index=False)
    return out_path
