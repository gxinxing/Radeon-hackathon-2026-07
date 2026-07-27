"""Fetch historical OHLCV data from crypto exchanges via CCXT.

Falls back to synthetic data generation when exchange APIs are unreachable
(e.g., network-restricted cloud instances).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import ccxt
import numpy as np
import pandas as pd


# Exchange instances (lazy init)
_exchanges: dict[str, ccxt.Exchange] = {}

# Synthetic data cache
_synthetic_cache: dict[str, pd.DataFrame] = {}


def get_exchange(name: str = "binance") -> ccxt.Exchange:
    """Get or create a CCXT exchange instance."""
    if name not in _exchanges:
        exchange_class = getattr(ccxt, name)
        _exchanges[name] = exchange_class({"enableRateLimit": True})
    return _exchanges[name]


def fetch_ohlcv(
    pair: str = "BTC/USDT",
    timeframe: str = "1h",
    exchange_name: str = "binance",
    days: int = 180,
    since: int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch historical OHLCV data.

    Tries real exchange first; falls back to synthetic data on network error.
    """
    try:
        exchange = get_exchange(exchange_name)
        if since is None:
            since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        all_data: list[list] = []
        while True:
            ohlcv = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < limit:
                break

        if all_data:
            df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)
            df.drop(columns=["timestamp"], inplace=True)
            return df
    except Exception as e:
        print(f"[DataFetcher] Exchange API unreachable ({e.__class__.__name__}), using synthetic data")

    return _generate_synthetic_ohlcv(pair, timeframe, days)


def _generate_synthetic_ohlcv(
    pair: str,
    timeframe: str,
    days: int,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data using geometric Brownian motion.

    Uses pair-specific base price and volatility parameters to create
    realistic-looking candlestick data for backtesting when exchange APIs
    are unreachable.
    """
    cache_key = f"{pair}_{timeframe}_{days}"
    if cache_key in _synthetic_cache:
        return _synthetic_cache[cache_key]

    # Pair-specific parameters
    base_prices = {
        "BTC/USDT": 65000,
        "ETH/USDT": 3500,
        "BNB/USDT": 600,
        "SOL/USDT": 150,
    }
    base_price = base_prices.get(pair, 1000)

    # Volatility per timeframe (annualized, adjusted for timeframe)
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
    minutes = tf_minutes.get(timeframe, 60)
    # BTC-like annual volatility ~70%, per-candle vol
    annual_vol = 0.70
    periods_per_year = 525600 / minutes  # minutes per year / minutes per candle
    sigma = annual_vol / np.sqrt(periods_per_year)
    mu = 0.15 / periods_per_year  # 15% annual drift

    # Generate timestamps
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    freq = pd.Timedelta(minutes=minutes)
    timestamps = pd.date_range(start=start, end=end, freq=freq)
    n = len(timestamps)

    # Generate price path with GBM
    np.random.seed(42)  # Reproducible
    returns = np.random.normal(mu, sigma, n)
    prices = base_price * np.exp(np.cumsum(returns))

    # Add some trend cycles for realism
    cycle = np.sin(np.linspace(0, 4 * np.pi, n)) * base_price * 0.05
    prices = prices + cycle

    # Build OHLCV from close prices
    intraday_vol = sigma * base_price * 0.5
    opens = np.roll(prices, 1)
    opens[0] = prices[0]

    highs = np.maximum(opens, prices) + np.random.uniform(0, intraday_vol, n)
    lows = np.minimum(opens, prices) - np.random.uniform(0, intraday_vol, n)
    lows = np.maximum(lows, prices * 0.95)  # Ensure positive
    volumes = np.random.lognormal(mean=15, sigma=1.5, size=n)  # Realistic volume range

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=timestamps[:n])

    _synthetic_cache[cache_key] = df
    print(f"[DataFetcher] Generated {len(df)} synthetic candles for {pair} {timeframe}")
    return df


def get_market_summary(
    pair: str = "BTC/USDT",
    exchange_name: str = "binance",
) -> dict[str, Any]:
    """Get current market summary for a trading pair.

    Returns dict with last_price, 24h_change, volume, high, low.
    """
    try:
        exchange = get_exchange(exchange_name)
        ticker = exchange.fetch_ticker(pair)
        return {
            "pair": pair,
            "exchange": exchange_name,
            "last_price": ticker.get("last", 0),
            "change_pct": ticker.get("percentage", 0),
            "high_24h": ticker.get("high", 0),
            "low_24h": ticker.get("low", 0),
            "volume_24h": ticker.get("baseVolume", 0),
            "quote_volume_24h": ticker.get("quoteVolume", 0),
            "timestamp": ticker.get("timestamp", 0),
        }
    except Exception:
        # Fallback to synthetic data's last value
        df = fetch_ohlcv(pair, "1h", exchange_name, days=1)
        if df.empty:
            return {"pair": pair, "exchange": exchange_name, "last_price": 0,
                    "change_pct": 0, "high_24h": 0, "low_24h": 0,
                    "volume_24h": 0, "quote_volume_24h": 0, "timestamp": 0}
        last = df.iloc[-1]
        first = df.iloc[0]
        return {
            "pair": pair,
            "exchange": exchange_name,
            "last_price": float(last["close"]),
            "change_pct": float((last["close"] - first["open"]) / first["open"] * 100),
            "high_24h": float(df["high"].max()),
            "low_24h": float(df["low"].min()),
            "volume_24h": float(df["volume"].sum()),
            "quote_volume_24h": float(df["volume"].sum() * last["close"]),
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        }


def calculate_indicators(
    df: pd.DataFrame,
    indicators: list[dict],
) -> pd.DataFrame:
    """Calculate technical indicators on a price DataFrame.

    Args:
        df: DataFrame with OHLCV columns.
        indicators: List of indicator specs from the DSL.

    Returns:
        DataFrame with indicator columns added.
    """
    import talib

    for ind in indicators:
        name = ind["name"]
        ind_type = ind["type"]
        params = ind.get("params", {})
        field = params.get("field", "close")
        period = params.get("period", 14)
        data = df[field].values

        if ind_type == "SMA":
            df[name] = talib.SMA(data, timeperiod=period)
        elif ind_type == "EMA":
            df[name] = talib.EMA(data, timeperiod=period)
        elif ind_type == "RSI":
            df[name] = talib.RSI(data, timeperiod=period)
        elif ind_type == "MACD":
            fast = params.get("fast_period", 12)
            slow = params.get("slow_period", 26)
            signal = params.get("signal_period", 9)
            macd, signal_line, hist = talib.MACD(
                data, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            df[name] = macd
            df[f"{name}_signal"] = signal_line
            df[f"{name}_hist"] = hist
        elif ind_type == "ATR":
            df[name] = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values,
                timeperiod=period,
            )
        elif ind_type == "BollingerBands":
            std = params.get("std_dev", 2.0)
            upper, middle, lower = talib.BBANDS(
                data, timeperiod=period, nbdevup=std, nbdevdn=std,
            )
            df[f"{name}_upper"] = upper
            df[f"{name}_middle"] = middle
            df[f"{name}_lower"] = lower
        elif ind_type == "ADX":
            df[name] = talib.ADX(
                df["high"].values, df["low"].values, df["close"].values,
                timeperiod=period,
            )
        elif ind_type == "CCI":
            df[name] = talib.CCI(
                df["high"].values, df["low"].values, df["close"].values,
                timeperiod=period,
            )
        elif ind_type == "OBV":
            df[name] = talib.OBV(data, df["volume"].values)
        elif ind_type == "Stochastic":
            k, d = talib.STOCH(
                df["high"].values, df["low"].values, df["close"].values,
                fastk_period=period,
            )
            df[f"{name}_k"] = k
            df[f"{name}_d"] = d
        elif ind_type == "WMA":
            df[name] = talib.WMA(data, timeperiod=period)
        else:
            df[name] = talib.SMA(data, timeperiod=period)

    return df
