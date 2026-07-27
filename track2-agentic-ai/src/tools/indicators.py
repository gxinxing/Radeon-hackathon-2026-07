"""Technical indicator calculator tool.

Calculates indicators on raw OHLCV data, exposed as API endpoints
for Dify to call during strategy analysis.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/indicators", tags=["indicators"])


class IndicatorRequest(BaseModel):
    candles: list[dict[str, Any]]  # OHLCV dicts
    indicators: list[dict[str, Any]]  # Indicator specs from DSL


class IndicatorResponse(BaseModel):
    values: dict[str, list[float | None]]
    last_values: dict[str, float | None]


@router.post("/calculate", response_model=IndicatorResponse)
async def calculate(req: IndicatorRequest):
    """Calculate technical indicators on provided candle data."""
    import pandas as pd
    import talib

    df = pd.DataFrame(req.candles)
    if "datetime" in df.columns:
        df.drop(columns=["datetime"], inplace=True)

    values: dict[str, list[float | None]] = {}
    last_values: dict[str, float | None] = {}

    for ind in req.indicators:
        name = ind["name"]
        ind_type = ind["type"]
        params = ind.get("params", {})
        field = params.get("field", "close")
        period = params.get("period", 14)
        data = df[field].values

        result: list[float | None] = []

        if ind_type == "SMA":
            result = talib.SMA(data, timeperiod=period).tolist()
        elif ind_type == "EMA":
            result = talib.EMA(data, timeperiod=period).tolist()
        elif ind_type == "RSI":
            result = talib.RSI(data, timeperiod=period).tolist()
        elif ind_type == "MACD":
            fast = params.get("fast_period", 12)
            slow = params.get("slow_period", 26)
            signal = params.get("signal_period", 9)
            macd, _, _ = talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            result = macd.tolist()
        elif ind_type == "ATR":
            result = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values,
                timeperiod=period,
            ).tolist()
        elif ind_type == "BollingerBands":
            std = params.get("std_dev", 2.0)
            upper, middle, lower = talib.BBANDS(data, timeperiod=period, nbdevup=std, nbdevdn=std)
            values[f"{name}_upper"] = upper.tolist()
            values[f"{name}_middle"] = middle.tolist()
            values[f"{name}_lower"] = lower.tolist()
            last_values[f"{name}_upper"] = upper[-1] if len(upper) > 0 else None
            last_values[f"{name}_middle"] = middle[-1] if len(middle) > 0 else None
            last_values[f"{name}_lower"] = lower[-1] if len(lower) > 0 else None
            continue
        elif ind_type == "ADX":
            result = talib.ADX(
                df["high"].values, df["low"].values, df["close"].values,
                timeperiod=period,
            ).tolist()
        else:
            result = talib.SMA(data, timeperiod=period).tolist()

        # Replace NaN with None
        result = [None if x != x else x for x in result]  # NaN check
        values[name] = result
        last_values[name] = result[-1] if result else None

    return IndicatorResponse(values=values, last_values=last_values)
