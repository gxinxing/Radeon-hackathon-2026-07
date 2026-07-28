"""Knowledge base entries for crypto trading RAG.

Each entry has:
- keywords: terms that trigger retrieval (English + Chinese)
- category: indicator | strategy | risk | market
- title: Short title
- content: Knowledge text injected into LLM prompt
- weight: Priority weight (higher = more relevant when scores tie)
"""

KNOWLEDGE_ENTRIES: list[dict] = [

    # ==================== INDICATORS ====================

    {
        "keywords": ["ema", "ma", "moving average", "均线", "移动平均", "sma", "wma"],
        "category": "indicator",
        "title": "Moving Averages (SMA/EMA/WMA/HMA)",
        "content": (
            "Moving averages smooth price data to identify trends.\n"
            "- SMA: Simple average, slowest response, best for long-term trends\n"
            "- EMA: Exponential weighting, faster response to recent prices, most popular\n"
            "- WMA: Linear weighting, middle ground between SMA and EMA\n"
            "- HMA: Hull MA, very responsive with minimal lag, good for short-term\n"
            "Common periods: 9/21 (fast), 50 (medium), 200 (slow)\n"
            "Strategy: Buy when fast MA crosses above slow MA (golden cross); "
            "sell on reverse cross (death cross).\n"
            "DSL tip: Use EMA for crypto (24/7 market, fast-moving). "
            "Typical params: {period: 20, field: close} for fast, {period: 50} for slow."
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["rsi", "relative strength", "超卖", "超买", "oversold", "overbought"],
        "category": "indicator",
        "title": "RSI (Relative Strength Index)",
        "content": (
            "RSI measures momentum on a 0-100 scale over a lookback period (default 14).\n"
            "- RSI < 30: Oversold (potential buy signal)\n"
            "- RSI > 70: Overbought (potential sell signal)\n"
            "- RSI divergence with price: Strong reversal signal\n"
            "- 50 line: Trend direction (above = bullish, below = bearish)\n"
            "DSL tip: Use RSI < 30 for entry, RSI > 70 for exit. "
            "Combine with volume confirmation to filter false signals. "
            "In strong trends, RSI can stay overbought/oversold for extended periods."
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["macd", "moving average convergence", "金叉", "死叉"],
        "category": "indicator",
        "title": "MACD (Moving Average Convergence Divergence)",
        "content": (
            "MACD shows relationship between two EMAs (default 12/26).\n"
            "- MACD line crosses above signal (9-period EMA): Bullish signal\n"
            "- MACD line crosses below signal: Bearish signal\n"
            "- Histogram > 0: Bullish momentum increasing\n"
            "- Zero-line crossover: Trend confirmation\n"
            "- Divergence with price: Reversal warning\n"
            "DSL tip: MACD requires fast_period (12), slow_period (26), signal_period (9). "
            "Entry: 'macd > 0' (above zero = bullish). Exit: 'macd < 0'."
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["bollinger", "bb", "布林带", "upper band", "lower band"],
        "category": "indicator",
        "title": "Bollinger Bands",
        "content": (
            "Bollinger Bands: SMA(20) ± 2 standard deviations.\n"
            "- Price at upper band: Overbought / strong uptrend\n"
            "- Price at lower band: Oversold / strong downtrend\n"
            "- Band squeeze (narrowing): Low volatility, breakout imminent\n"
            "- Band expansion: Volatility increasing, trend confirmed\n"
            "Strategy: Mean reversion — buy at lower band, sell at upper band.\n"
            "DSL tip: BB generates {name}_upper, {name}_middle, {name}_lower. "
            "Entry: 'close < bb_lower'. Exit: 'close > bb_upper'."
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["atr", "average true range", "volatility", "波动率"],
        "category": "indicator",
        "title": "ATR (Average True Range)",
        "content": (
            "ATR measures market volatility over a period (default 14).\n"
            "- High ATR: High volatility, wider stops needed\n"
            "- Low ATR: Low volatility, tighter stops acceptable\n"
            "Uses: 1) Dynamic stop-loss (1.5-2x ATR), 2) Position sizing, "
            "3) Market regime detection.\n"
            "DSL tip: ATR is best used for setting stop_loss dynamically. "
            "For 1h BTC: stop_loss = -0.03 (3%) is roughly 1.5x ATR in normal conditions. "
            "In high volatility (ATR > 3% of price), widen to -0.05."
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["supertrend", "趋势跟踪"],
        "category": "indicator",
        "title": "Supertrend Indicator",
        "content": (
            "Supertrend uses ATR + multiplier (default 3) to plot a trend-following line.\n"
            "- Price above line (green): Uptrend → go long\n"
            "- Price below line (red): Downtrend → go short or exit\n"
            "- Line flips: Trend reversal signal\n"
            "Works best in trending markets; expect whipsaws in ranging conditions.\n"
            "DSL tip: Supertrend with params {period: 10, multiplier: 3.0}. "
            "Entry: 'close > supertrend'. Exit: 'close < supertrend'."
        ),
        "weight": 0.9,
    },
    {
        "keywords": ["ichimoku", "一目均衡", "cloud"],
        "category": "indicator",
        "title": "Ichimoku Cloud",
        "content": (
            "Ichimoku provides trend direction, support/resistance, and momentum.\n"
            "- Price above cloud (SpanA > SpanB): Bullish\n"
            "- Price below cloud: Bearish\n"
            "- Price inside cloud: Sideways/neutral\n"
            "- Tenkan crosses Kijun: Short-term signal\n"
            "DSL tip: Generates {name}_tenkan, {name}_kijun, {name}_spanA, {name}_spanB. "
            "Entry: 'close > ichi_spanA AND close > ichi_spanB'. "
            "Params: {period: 52, fast_period: 9, slow_period: 26}."
        ),
        "weight": 0.9,
    },
    {
        "keywords": ["vwap", "volume weighted", "成交量加权"],
        "category": "indicator",
        "title": "VWAP (Volume Weighted Average Price)",
        "content": (
            "VWAP is the cumulative average price weighted by volume.\n"
            "- Price above VWAP: Buyers in control (bullish)\n"
            "- Price below VWAP: Sellers in control (bearish)\n"
            "- VWAP acts as dynamic support/resistance\n"
            "Most useful for intraday strategies on short timeframes (1m-15m).\n"
            "DSL tip: VWAP has no period param — it's cumulative. "
            "Entry: 'close > vwap'. Exit: 'close < vwap'."
        ),
        "weight": 0.8,
    },
    {
        "keywords": ["adx", "trend strength", "趋势强度"],
        "category": "indicator",
        "title": "ADX (Average Directional Index)",
        "content": (
            "ADX measures trend strength (not direction) on 0-100 scale.\n"
            "- ADX > 25: Strong trend (good for trend-following)\n"
            "- ADX < 20: Weak/no trend (range/sideways market)\n"
            "- ADX rising: Trend gaining strength\n"
            "Use as a filter: only take trend signals when ADX > 25.\n"
            "DSL tip: Entry: 'ema_fast > ema_slow AND adx > 25'. "
            "This filters out choppy market false signals."
        ),
        "weight": 0.8,
    },

    # ==================== STRATEGIES ====================

    {
        "keywords": ["crossover", "金叉", "cross", "突破", "breakout"],
        "category": "strategy",
        "title": "Trend Following: MA Crossover",
        "content": (
            "Strategy type: Trend following\n"
            "Logic: Buy when fast MA crosses above slow MA, sell on reverse cross.\n"
            "Best market: Strong trending (high ADX)\n"
            "Weakness: Whipsaws in ranging markets\n"
            "Improvement: Add volume confirmation and ADX filter.\n"
            "Typical params: EMA(20)/EMA(50) for 1h, EMA(9)/EMA(21) for 15m.\n"
            "Stop-loss: 2-3% for 1h, 1-1.5% for 15m.\n"
            "DSL example:\n"
            "  entry.long: ema_fast > ema_slow AND volume > vol_ma * 1.5\n"
            "  exit.long: ema_fast < ema_slow\n"
            "  risk.stop_loss: -0.03"
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["mean reversion", "超卖反弹", "oversold", "reversion", "回归", "反转"],
        "category": "strategy",
        "title": "Mean Reversion: RSI / Bollinger Bands",
        "content": (
            "Strategy type: Mean reversion\n"
            "Logic: Buy when price deviates significantly from mean, sell on return.\n"
            "Best market: Ranging/sideways (low ADX)\n"
            "Weakness: Losses in strong trends (price keeps going against you)\n"
            "Key: RSI < 30 + Bollinger lower band = strong oversold signal.\n"
            "Stop-loss: Wider (4-5%) because reversals can take time.\n"
            "DSL example:\n"
            "  entry.long: rsi < 30 AND close < bb_lower\n"
            "  exit.long: rsi > 70 OR close > bb_upper\n"
            "  risk.stop_loss: -0.05"
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["breakout", "突破", "volume breakout", "放量突破"],
        "category": "strategy",
        "title": "Volume-Confirmed Breakout",
        "content": (
            "Strategy type: Breakout\n"
            "Logic: Enter when price breaks key level with above-average volume.\n"
            "Best market: After consolidation (low ATR → ATR expansion)\n"
            "Key signal: Price > EMA + volume > 1.5x average + RSI not overbought.\n"
            "Stop-loss: Below breakout level or 2-3%.\n"
            "DSL example:\n"
            "  entry.long: ema_fast > ema_slow AND volume > vol_ma * 1.5 AND rsi < 70\n"
            "  exit.long: ema_fast < ema_slow\n"
            "  risk.stop_loss: -0.03, trailing_stop: true, trailing_stop_positive: 0.02"
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["momentum", "动量", "macd strategy"],
        "category": "strategy",
        "title": "Momentum: MACD Crossover",
        "content": (
            "Strategy type: Momentum\n"
            "Logic: Buy when MACD crosses above zero (bullish momentum), sell below zero.\n"
            "Best market: Trending with clear momentum\n"
            "Improvement: Combine with ADX > 25 for trend confirmation.\n"
            "Stop-loss: 3-5% depending on timeframe.\n"
            "DSL example:\n"
            "  entry.long: macd > 0 AND adx > 25\n"
            "  exit.long: macd < 0\n"
            "  risk.stop_loss: -0.04"
        ),
        "weight": 0.9,
    },
    {
        "keywords": ["confluence", "共振", "多指标", "multi-indicator"],
        "category": "strategy",
        "title": "Multi-Indicator Confluence",
        "content": (
            "Strategy type: Confluence (multiple signals align)\n"
            "Logic: Require 3+ indicators to agree before entering.\n"
            "Best market: All markets (versatile)\n"
            "Key: More conditions = fewer trades but higher win rate.\n"
            "Typical: EMA cross + RSI + volume + ADX.\n"
            "DSL example:\n"
            "  entry.long: ema_fast > ema_slow AND rsi < 35 AND volume > vol_ma * 1.5 AND adx > 25\n"
            "  exit.long: ema_fast < ema_slow OR rsi > 70\n"
            "  risk.stop_loss: -0.03, max_open_trades: 2"
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["short", "做空", "shorting", "bearish"],
        "category": "strategy",
        "title": "Short Selling Strategy",
        "content": (
            "Strategy type: Short / Bearish\n"
            "Logic: Sell when indicators turn bearish, buy back when they improve.\n"
            "Risk: Unlimited loss potential (price can rise indefinitely).\n"
            "Key: Always use stop-loss for shorts. Crypto pumps can be violent.\n"
            "DSL example:\n"
            "  entry.short: ema_fast < ema_slow AND rsi > 70\n"
            "  exit.short: ema_fast > ema_slow OR rsi < 30\n"
            "  risk.stop_loss: -0.04 (wider for shorts due to pump risk)"
        ),
        "weight": 0.9,
    },

    # ==================== RISK MANAGEMENT ====================

    {
        "keywords": ["stop loss", "止损", "stop", "风险"],
        "category": "risk",
        "title": "Stop-Loss Best Practices",
        "content": (
            "Stop-loss is the most important risk management tool.\n"
            "Guidelines:\n"
            "- 1h timeframe: 2-3% stop-loss\n"
            "- 4h timeframe: 3-5% stop-loss\n"
            "- 1d timeframe: 5-8% stop-loss\n"
            "- High volatility (ATR > 3% of price): Widen by 50%\n"
            "- Always use stop-loss — never trade without one\n"
            "- Trailing stop: Lock in profits when trade goes your way\n"
            "DSL: risk.stop_loss must be negative (e.g. -0.03 = 3% loss).\n"
            "For beginners: start with -0.03 and adjust based on ATR."
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["position sizing", "仓位", "stake", "position size", "资金管理"],
        "category": "risk",
        "title": "Position Sizing Rules",
        "content": (
            "Position sizing determines how much capital to risk per trade.\n"
            "Rules:\n"
            "- Risk no more than 1-2% of total capital per trade\n"
            "- With stop_loss=-0.03 and stake_amount=0.1: risk = 0.3% per trade\n"
            "- max_open_trades: 2-3 for conservative, 3-5 for aggressive\n"
            "- 'unlimited' stake = use all available capital (high risk!)\n"
            "DSL: risk.stake_amount = 0.1 (10% of balance per position).\n"
            "Recommended: stake_amount=0.1, max_open_trades=3 → 30% max deployed."
        ),
        "weight": 0.9,
    },
    {
        "keywords": ["trailing stop", "追踪止损", "trailing"],
        "category": "risk",
        "title": "Trailing Stop Configuration",
        "content": (
            "Trailing stops move with price to lock in profits.\n"
            "- trailing_stop_positive: How far behind price the stop trails (e.g. 0.02 = 2%)\n"
            "- trailing_stop_positive_offset: Price must move this far before trailing activates\n"
            "Best for: Strong trends where you want to ride the move.\n"
            "Not ideal for: Choppy markets (will get stopped out quickly).\n"
            "DSL example:\n"
            "  risk.trailing_stop: true\n"
            "  risk.trailing_stop_positive: 0.02\n"
            "  risk.trailing_stop_positive_offset: 0.04 (activate after 4% profit)"
        ),
        "weight": 0.8,
    },
    {
        "keywords": ["take profit", "止盈", "profit target"],
        "category": "risk",
        "title": "Take-Profit Strategy",
        "content": (
            "Take-profit closes a position when a target return is reached.\n"
            "- R:R ratio: Aim for at least 1:2 (risk $1 to make $2)\n"
            "- With stop_loss=-0.03, take_profit=0.06 gives 1:2 R:R\n"
            "- Higher R:R (1:3) = fewer wins needed to be profitable\n"
            "DSL: risk.take_profit = 0.06 (6% gain target).\n"
            "Alternative: Use trailing stop instead of fixed TP for trend riding."
        ),
        "weight": 0.8,
    },
    {
        "keywords": ["time in trade", "持仓时间", "max holding"],
        "category": "risk",
        "title": "Time-in-Trade Limits",
        "content": (
            "Time-in-trade limits force position closure after a set duration.\n"
            "Use cases:\n"
            "- Avoid stale positions that are going nowhere\n"
            "- Reduce overnight risk (close before daily close)\n"
            "- Day trading: max_hours = 8 (one trading session)\n"
            "- Swing trading: max_days = 3-7\n"
            "DSL example:\n"
            "  risk.time_in_trade:\n"
            "    max_hours: 48  # Close after 2 days regardless of PnL"
        ),
        "weight": 0.7,
    },

    # ==================== MARKET CONTEXT ====================

    {
        "keywords": ["btc", "bitcoin", "比特币"],
        "category": "market",
        "title": "BTC Market Characteristics",
        "content": (
            "Bitcoin-specific trading considerations:\n"
            "- 24/7 market, no opening/closing hours\n"
            "- Annual volatility: ~70% (much higher than stocks)\n"
            "- Base price range: $60K-$70K (as of 2026)\n"
            "- Typical 1h candle range: 0.5-2% of price\n"
            "- Volume highest during US/Asia overlap (UTC 13:00-16:00)\n"
            "- Weekend volume typically 30-50% lower\n"
            "For 1h timeframe: stop_loss=-0.03, take_profit=0.06 is reasonable.\n"
            "For 4h timeframe: stop_loss=-0.05, take_profit=0.10."
        ),
        "weight": 1.0,
    },
    {
        "keywords": ["eth", "ethereum", "以太坊"],
        "category": "market",
        "title": "ETH Market Characteristics",
        "content": (
            "Ethereum-specific considerations:\n"
            "- Higher beta than BTC (moves more % per unit of market move)\n"
            "- Annual volatility: ~80%\n"
            "- Base price range: $3K-$4K\n"
            "- Gas fees and DeFi activity affect price action\n"
            "- Correlation with BTC: ~0.7-0.8\n"
            "For ETH: use slightly wider stops than BTC (e.g. -0.04 vs -0.03)."
        ),
        "weight": 0.8,
    },
    {
        "keywords": ["timeframe", "时间周期", "1h", "4h", "1d", "15m"],
        "category": "market",
        "title": "Timeframe Selection Guide",
        "content": (
            "Timeframe affects strategy parameters significantly:\n"
            "- 15m: High noise, many false signals. Stop: 1-1.5%, fast exits.\n"
            "- 1h: Balanced, most popular for crypto. Stop: 2-3%, medium trades.\n"
            "- 4h: Lower noise, clearer trends. Stop: 3-5%, swing trades.\n"
            "- 1d: Long-term trends. Stop: 5-8%, position trades.\n"
            "Rule of thumb: Use longer timeframes for trend direction,\n"
            "shorter timeframes for entry timing.\n"
            "DSL: market.timeframe determines candle period and affects all metrics."
        ),
        "weight": 0.9,
    },
]
