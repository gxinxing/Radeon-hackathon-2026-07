"""Backtest runner — executes strategy on historical data.

This module runs backtests by:
1. Fetching historical OHLCV data via CCXT
2. Computing indicators
3. Evaluating entry/exit conditions from DSL
4. Simulating trades and computing performance metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data_fetcher import fetch_ohlcv, calculate_indicators


@dataclass
class BacktestResult:
    """Structured backtest result."""
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    final_balance: float = 0.0
    initial_balance: float = 0.0
    duration_days: int = 0
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    error: str | None = None


def run_backtest(
    strategy_dsl: dict[str, Any],
    days: int = 180,
    initial_balance: float = 10000.0,
    fee: float = 0.001,  # 0.1% per trade
) -> BacktestResult:
    """Run a backtest on historical data using the strategy DSL.

    Args:
        strategy_dsl: Parsed strategy DSL dict.
        days: Number of days of historical data to use.
        initial_balance: Starting balance in quote currency.
        fee: Trading fee per transaction (as decimal).

    Returns:
        BacktestResult with performance metrics.
    """
    strat = strategy_dsl["strategy"]
    market = strat["market"]
    indicators = strat["indicators"]
    entry = strat["entry"]
    exit_conf = strat.get("exit", {})
    risk = strat["risk"]

    # --- Fetch historical data ---
    try:
        df = fetch_ohlcv(
            pair=market["pair"],
            timeframe=market["timeframe"],
            exchange_name=market["exchange"],
            days=days,
        )
    except Exception as e:
        return BacktestResult(error=f"Data fetch error: {e}")

    if df.empty:
        return BacktestResult(error="No historical data available")

    # --- Calculate indicators ---
    try:
        df = calculate_indicators(df, indicators)
    except Exception as e:
        return BacktestResult(error=f"Indicator calculation error: {e}")

    # --- Evaluate entry/exit signals ---
    df = _evaluate_signals(df, entry, exit_conf, indicators)

    # --- Simulate trades ---
    result = _simulate_trades(
        df=df,
        initial_balance=initial_balance,
        fee=fee,
        stop_loss=risk.get("stop_loss", -0.03),
        take_profit=risk.get("take_profit"),
        max_open_trades=risk.get("max_open_trades", 3),
        stake_amount=risk.get("stake_amount", 0.1),
    )

    result.duration_days = days
    result.initial_balance = initial_balance
    return result


def _evaluate_signals(
    df: pd.DataFrame,
    entry: dict,
    exit_conf: dict,
    indicators: list[dict],
) -> pd.DataFrame:
    """Evaluate entry and exit boolean expressions on the DataFrame."""
    # Build entry signals
    long_entry_expr = entry.get("long")
    short_entry_expr = entry.get("short")

    if long_entry_expr:
        df["enter_long"] = _eval_expression(df, long_entry_expr)
    else:
        df["enter_long"] = False

    if short_entry_expr:
        df["enter_short"] = _eval_expression(df, short_entry_expr)
    else:
        df["enter_short"] = False

    # Build exit signals
    long_exit_expr = exit_conf.get("long")
    short_exit_expr = exit_conf.get("short")

    if long_exit_expr:
        df["exit_long"] = _eval_expression(df, long_exit_expr)
    else:
        df["exit_long"] = False

    if short_exit_expr:
        df["exit_short"] = _eval_expression(df, short_exit_expr)
    else:
        df["exit_short"] = False

    return df


def _eval_expression(df: pd.DataFrame, expr: str) -> pd.Series:
    """Evaluate a boolean expression against DataFrame columns.

    Translates DSL expressions (AND, OR, NOT) to pandas operations.
    """
    # Translate boolean operators
    py_expr = expr
    py_expr = py_expr.replace(" AND ", " & ")
    py_expr = py_expr.replace(" and ", " & ")
    py_expr = py_expr.replace(" OR ", " | ")
    py_expr = py_expr.replace(" or ", " | ")
    py_expr = py_expr.replace(" NOT ", " ~")
    py_expr = py_expr.replace(" not ", " ~")

    try:
        return df.eval(py_expr)
    except Exception:
        return pd.Series(False, index=df.index)


def _simulate_trades(
    df: pd.DataFrame,
    initial_balance: float,
    fee: float,
    stop_loss: float,
    take_profit: float | None,
    max_open_trades: int,
    stake_amount: float | str,
) -> BacktestResult:
    """Simulate trades based on entry/exit signals."""
    result = BacktestResult()
    balance = initial_balance
    position: float = 0.0
    entry_price: float = 0.0
    trades: list[dict] = []

    equity_curve: list[float] = []
    dates: list[str] = []

    for i, (timestamp, row) in enumerate(df.iterrows()):
        current_price = row["close"]

        # Check stop loss / take profit for open position
        if position > 0:
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= stop_loss:
                # Stop loss hit
                _close_position(
                    balance=balance, position=position, price=current_price,
                    fee=fee, timestamp=timestamp, trades=trades,
                    reason="stop_loss", entry_price=entry_price,
                )
                balance += position * current_price * (1 - fee)
                position = 0.0
                entry_price = 0.0
            elif take_profit and pnl_pct >= take_profit:
                # Take profit hit
                _close_position(
                    balance=balance, position=position, price=current_price,
                    fee=fee, timestamp=timestamp, trades=trades,
                    reason="take_profit", entry_price=entry_price,
                )
                balance += position * current_price * (1 - fee)
                position = 0.0
                entry_price = 0.0

        # Check exit signal
        if position > 0 and row.get("exit_long", False):
            _close_position(
                balance=balance, position=position, price=current_price,
                fee=fee, timestamp=timestamp, trades=trades,
                reason="signal_exit", entry_price=entry_price,
            )
            balance += position * current_price * (1 - fee)
            position = 0.0
            entry_price = 0.0

        # Check entry signal
        if position == 0 and row.get("enter_long", False):
            # Calculate position size
            if isinstance(stake_amount, str) and stake_amount == "unlimited":
                pos_size = balance * 0.95  # Leave 5% buffer
            else:
                pos_size = min(balance * stake_amount, balance * 0.95)

            if pos_size > 0 and current_price > 0:
                position = pos_size / current_price
                entry_price = current_price
                balance -= pos_size + pos_size * fee
                trades.append({
                    "type": "buy",
                    "datetime": str(timestamp),
                    "price": current_price,
                    "amount": position,
                    "cost": pos_size,
                    "fee": pos_size * fee,
                    "reason": "signal_entry",
                })

        # Track equity
        equity = balance + position * current_price
        equity_curve.append(round(equity, 2))
        dates.append(str(timestamp))

    # Close any remaining position
    if position > 0:
        last_price = df.iloc[-1]["close"]
        _close_position(
            balance=balance, position=position, price=last_price,
            fee=fee, timestamp=df.index[-1], trades=trades,
            reason="backtest_end", entry_price=entry_price,
        )
        balance += position * last_price * (1 - fee)
        position = 0.0

    # --- Calculate metrics ---
    result.final_balance = round(balance, 2)
    result.total_return = (balance - initial_balance) / initial_balance
    result.equity_curve = equity_curve
    result.dates = dates
    result.trades = trades

    # Trade-level metrics
    completed_trades = [t for t in trades if t["type"] == "sell"]
    result.total_trades = len(completed_trades)

    if completed_trades:
        profits = [t.get("profit", 0) for t in completed_trades]
        result.win_trades = sum(1 for p in profits if p > 0)
        result.loss_trades = sum(1 for p in profits if p <= 0)
        result.win_rate = result.win_trades / result.total_trades if result.total_trades > 0 else 0
        result.avg_win = np.mean([p for p in profits if p > 0]) if result.win_trades > 0 else 0
        result.avg_loss = np.mean([p for p in profits if p <= 0]) if result.loss_trades > 0 else 0

        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    if equity_curve:
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = round(max_dd, 4)

    # Sharpe ratio (simplified, annualized)
    if len(equity_curve) > 2:
        returns = pd.Series(equity_curve).pct_change().dropna()
        if returns.std() > 0:
            result.sharpe_ratio = round(
                (returns.mean() / returns.std()) * np.sqrt(252 * 24), 4  # Assuming 1h candles
            )

    return result


def _close_position(
    balance: float, position: float, price: float,
    fee: float, timestamp, trades: list[dict],
    reason: str, entry_price: float,
) -> None:
    """Record a closing trade."""
    cost = position * price
    profit = (price - entry_price) * position - cost * fee
    trades.append({
        "type": "sell",
        "datetime": str(timestamp),
        "price": price,
        "amount": position,
        "cost": cost,
        "fee": cost * fee,
        "profit": round(profit, 4),
        "reason": reason,
    })
