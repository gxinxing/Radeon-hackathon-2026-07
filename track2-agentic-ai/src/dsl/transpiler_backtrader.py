"""DSL → Backtrader Strategy transpiler.

Converts a validated strategy DSL dict into a Python source file
containing a Backtrader Strategy subclass, ready for backtesting
with backtrader.Cerebro.

Usage:
    from src.dsl.transpiler_backtrader import transpile_to_backtrader
    code = transpile_to_backtrader(dsl_dict)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


# Maps DSL indicator type → backtrader indicator class
_BT_INDICATOR_MAP: dict[str, str] = {
    "SMA": "SMA",
    "EMA": "EMA",
    "RSI": "RSI",
    "MACD": "MACD",
    "ATR": "ATR",
    "BollingerBands": "BollingerBands",
    "Stochastic": "Stochastic",
    "ADX": "ADX",
    "CCI": "CCI",
    "OBV": "OBV",
    "VWAP": "VWAP",
    "WMA": "WMA",
    "HMA": "HullMA",
    "ZLEMA": "ZLEMA",
    "Supertrend": "SuperTrend",
    "ICHIMOKU": "Ichimoku",
}

# Boolean operator translation
_EXPR_TRANSLATIONS = [
    (" AND ", " and "),
    (" and ", " and "),
    (" OR ", " or "),
    (" or ", " or "),
    (" NOT ", " not "),
    (" not ", " not "),
    ("True", "True"),
    ("False", "False"),
    ("null", "None"),
]


def transpile_to_backtrader(dsl: dict[str, Any]) -> str:
    """Transpile a validated DSL dict into Backtrader Strategy Python source.

    Args:
        dsl: Parsed strategy DSL dict (must pass validation).

    Returns:
        Python source code string for a Backtrader strategy file.
    """
    strat = dsl["strategy"]
    name = strat["name"]
    indicators = strat["indicators"]
    entry = strat["entry"]
    exit_conf = strat.get("exit", {})
    risk = strat["risk"]
    market = strat["market"]

    lines: list[str] = []
    lines.append('"""')
    lines.append(f"Auto-generated Backtrader strategy: {name}")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("Transpiled from trading strategy DSL.")
    lines.append('"""')
    lines.append("import backtrader as bt")
    lines.append("import pandas as pd")
    lines.append("")
    lines.append("")
    lines.append(f"class {name}(bt.Strategy):")
    lines.append(f'    """{name} — auto-generated from DSL."""')
    lines.append("")
    lines.append("    params = (")

    # --- Risk parameters as Backtrader params ---
    lines.append(f"        ('stop_loss', {risk['stop_loss']}),")
    if risk.get("take_profit"):
        lines.append(f"        ('take_profit', {risk['take_profit']}),")
    lines.append(f"        ('max_open_trades', {risk['max_open_trades']}),")
    stake = risk.get("stake_amount", 0.1)
    if isinstance(stake, str):
        lines.append("        ('stake_pct', 0.9),")
    else:
        lines.append(f"        ('stake_pct', {stake}),")
    lines.append(f"        ('trailing_stop', {risk.get('trailing_stop', False)}),")
    if risk.get("trailing_stop_positive"):
        lines.append(f"        ('trailing_offset', {risk['trailing_stop_positive']}),")
    else:
        lines.append("        ('trailing_offset', 0.0),")
    lines.append("    )")
    lines.append("")

    # --- __init__ ---
    lines.append("    def __init__(self):")
    lines.append("        self.open_trades = 0")
    lines.append("        self.order = None")
    lines.append("")

    for ind in indicators:
        ind_name = ind["name"]
        ind_type = ind["type"]
        params = ind.get("params", {})
        period = params.get("period", 14)
        field = params.get("field", "close")

        if ind_type == "SMA":
            lines.append(f"        self.{ind_name} = bt.indicators.SMA(self.data.{field}, period={period})")
        elif ind_type == "EMA":
            lines.append(f"        self.{ind_name} = bt.indicators.EMA(self.data.{field}, period={period})")
        elif ind_type == "RSI":
            lines.append(f"        self.{ind_name} = bt.indicators.RSI(self.data.{field}, period={period})")
        elif ind_type == "MACD":
            fast = params.get("fast_period", 12)
            slow = params.get("slow_period", 26)
            signal = params.get("signal_period", 9)
            lines.append(f"        self.macd_{ind_name} = bt.indicators.MACD(")
            lines.append(f"            self.data.close,")
            lines.append(f"            period_me1={fast}, period_me2={slow}, period_signal={signal})")
            lines.append(f"        self.{ind_name} = self.macd_{ind_name}.macd")
            lines.append(f"        self.{ind_name}_signal = self.macd_{ind_name}.signal")
        elif ind_type == "ATR":
            lines.append(f"        self.{ind_name} = bt.indicators.ATR(self.data, period={period})")
        elif ind_type == "BollingerBands":
            std = params.get("std_dev", 2.0)
            lines.append(f"        self.bb_{ind_name} = bt.indicators.BollingerBands(")
            lines.append(f"            self.data.close, period={period}, devfactor={std})")
            lines.append(f"        self.{ind_name}_upper = self.bb_{ind_name}.top")
            lines.append(f"        self.{ind_name}_lower = self.bb_{ind_name}.bot")
            lines.append(f"        self.{ind_name} = self.bb_{ind_name}.mid")
        elif ind_type == "Stochastic":
            lines.append(f"        self.stoch_{ind_name} = bt.indicators.Stochastic(")
            lines.append(f"            self.data, period={period})")
            lines.append(f"        self.{ind_name} = self.stoch_{ind_name}.percK")
            lines.append(f"        self.{ind_name}_d = self.stoch_{ind_name}.percD")
        elif ind_type == "ADX":
            lines.append(f"        self.{ind_name} = bt.indicators.ADX(self.data, period={period})")
        elif ind_type == "CCI":
            lines.append(f"        self.{ind_name} = bt.indicators.CCI(self.data, period={period})")
        elif ind_type == "OBV":
            lines.append(f"        self.{ind_name} = bt.indicators.OBV(self.data)")
        elif ind_type == "VWAP":
            lines.append(f"        # VWAP computed manually (cumulative)")
            lines.append(f"        typical = (self.data.high + self.data.low + self.data.close) / 3.0")
            lines.append(f"        self.{ind_name} = bt.indicators.SumN(typical * self.data.volume, period=9999) / bt.indicators.SumN(self.data.volume, period=9999)")
        elif ind_type == "WMA":
            lines.append(f"        self.{ind_name} = bt.indicators.WMA(self.data.{field}, period={period})")
        elif ind_type == "HMA":
            lines.append(f"        # HMA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))")
            lines.append(f"        half = max(1, {period} // 2)")
            lines.append(f"        sqrt_p = max(1, int({period} ** 0.5))")
            lines.append(f"        wma_half = bt.indicators.WMA(self.data.{field}, period=half)")
            lines.append(f"        wma_full = bt.indicators.WMA(self.data.{field}, period={period})")
            lines.append(f"        raw = 2 * wma_half - wma_full")
            lines.append(f"        self.{ind_name} = bt.indicators.WMA(raw, period=sqrt_p)")
        elif ind_type == "ZLEMA":
            lines.append(f"        # ZLEMA: EMA(2*price - price(delay))")
            lines.append(f"        delay = max(1, {period} // 2)")
            lines.append(f"        delayed = bt.Delay(self.data.{field}, delay)")
            lines.append(f"        adjusted = 2 * self.data.{field} - delayed")
            lines.append(f"        self.{ind_name} = bt.indicators.EMA(adjusted, period={period})")
        elif ind_type == "Supertrend":
            multiplier = params.get("multiplier", 3.0)
            lines.append(f"        self.{ind_name} = bt.indicators.SuperTrend(")
            lines.append(f"            self.data, period={period}, multiplier={multiplier})")
        elif ind_type == "ICHIMOKU":
            conv = params.get("fast_period", 9)
            base = params.get("slow_period", 26)
            lines.append(f"        self.ichi_{ind_name} = bt.indicators.Ichimoku(")
            lines.append(f"            self.data, tenkan={conv}, kijun={base}, senkou={period})")
            lines.append(f"        self.{ind_name} = self.ichi_{ind_name}.senkou_span_a")
            lines.append(f"        self.{ind_name}_tenkan = self.ichi_{ind_name}.tenkan_sen")
            lines.append(f"        self.{ind_name}_kijun = self.ichi_{ind_name}.kijun_sen")
            lines.append(f"        self.{ind_name}_spanA = self.ichi_{ind_name}.senkou_span_a")
            lines.append(f"        self.{ind_name}_spanB = self.ichi_{ind_name}.senkou_span_b")
        else:
            bt_class = _BT_INDICATOR_MAP.get(ind_type, "SMA")
            lines.append(f"        self.{ind_name} = bt.indicators.{bt_class}(self.data.{field}, period={period})")

    lines.append("")

    # --- next() ---
    lines.append("    def next(self):")
    lines.append("        if self.order:")
    lines.append("            return")
    lines.append("")
    lines.append("        price = self.data.close[0]")
    lines.append("        cash = self.broker.get_cash()")
    lines.append("")

    # Build entry/exit conditions
    long_entry = entry.get("long")
    short_entry = entry.get("short")
    long_exit = exit_conf.get("long")
    short_exit = exit_conf.get("short")

    # Long entry
    if long_entry:
        py_expr = _translate_expr(long_entry)
        lines.append(f"        # Long entry signal")
        lines.append(f"        long_entry = {py_expr}")
        lines.append(f"        if long_entry and self.open_trades < self.p.max_open_trades:")
        lines.append(f"            size = (cash * self.p.stake_pct) / price")
        lines.append(f"            self.order = self.buy(size=size)")
        lines.append(f"            self.open_trades += 1")
        lines.append("")

    # Long exit
    if long_exit:
        py_expr = _translate_expr(long_exit)
        lines.append(f"        # Long exit signal")
        lines.append(f"        long_exit = {py_expr}")
        lines.append(f"        if long_exit and self.position:")
        lines.append(f"            self.order = self.sell()")
        lines.append(f"            self.open_trades -= 1")
        lines.append("")

    # Short entry
    if short_entry:
        py_expr = _translate_expr(short_entry)
        lines.append(f"        # Short entry signal")
        lines.append(f"        short_entry = {py_expr}")
        lines.append(f"        if short_entry and self.open_trades < self.p.max_open_trades:")
        lines.append(f"            size = (cash * self.p.stake_pct) / price")
        lines.append(f"            self.order = self.sell_short(size=size)")
        lines.append(f"            self.open_trades += 1")
        lines.append("")

    # Short exit
    if short_exit:
        py_expr = _translate_expr(short_exit)
        lines.append(f"        # Short exit signal")
        lines.append(f"        short_exit = {py_expr}")
        lines.append(f"        if short_exit and self.position:")
        lines.append(f"            self.order = self.buy()")
        lines.append(f"            self.open_trades -= 1")
        lines.append("")

    # Stop-loss check
    lines.append(f"        # Stop-loss / take-profit")
    lines.append(f"        if self.position:")
    lines.append(f"            pnl_pct = (price - self.position.price) / self.position.price")
    lines.append(f"            if pnl_pct <= self.p.stop_loss:")
    lines.append(f"                self.order = self.close()")
    lines.append(f"                self.open_trades = 0")
    if risk.get("take_profit"):
        lines.append(f"            elif self.p.take_profit and pnl_pct >= self.p.take_profit:")
        lines.append(f"                self.order = self.close()")
        lines.append(f"                self.open_trades = 0")
    lines.append("")

    # --- notify_order ---
    lines.append("    def notify_order(self, order):")
    lines.append("        if order.status in [order.Completed, order.Canceled, order.Margin]:")
    lines.append("            self.order = None")
    lines.append("")

    return "\n".join(lines)


def _translate_expr(expr: str) -> str:
    """Translate a DSL boolean expression to a Python expression."""
    result = expr
    for dsl_op, py_op in _EXPR_TRANSLATIONS:
        result = result.replace(dsl_op, py_op)
    return result


def transpile_to_file(dsl: dict[str, Any], output_path: str) -> None:
    """Transpile DSL and write to a Python strategy file."""
    source = transpile_to_backtrader(dsl)
    with open(output_path, "w") as f:
        f.write(source)
