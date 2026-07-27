"""Fetch historical OHLCV data from crypto exchanges via CCXT.

Used by the backtest microservice to fetch historical data for
strategy validation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import ccxt
import pandas as pd


# Exchange instances (lazy init)
_exchanges: dict[str, ccxt.Exchange] = {}


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

    Args:
        pair: Trading pair, e.g. "BTC/USDT".
        timeframe: Candle timeframe, e.g. "1h", "4h", "1d".
        exchange_name: Exchange name (binance, okx, bybit, etc.).
        days: Number of days of history to fetch (if since is None).
        since: Timestamp in ms to start from (overrides days).
        limit: Number of candles per request.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    exchange = get_exchange(exchange_name)

    if since is None:
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_data: list[list] = []
    while True:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # Move past last candle
        if len(ohlcv) < limit:
            break

    if not all_data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)

    return df


def get_market_summary(
    pair: str = "BTC/USDT",
    exchange_name: str = "binance",
) -> dict[str, Any]:
    """Get current market summary for a trading pair.

    Returns dict with last_price, 24h_change, volume, high, low.
    """
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
