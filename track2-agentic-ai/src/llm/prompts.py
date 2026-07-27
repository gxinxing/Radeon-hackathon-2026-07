"""LLM inference client and system prompts.

Wraps vLLM's OpenAI-compatible API for:
1. NL → DSL generation
2. Backtest report generation
3. Risk assessment
"""

from __future__ import annotations

import json
from typing import Any

import httpx


VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "qwen-trader-merged"  # Or the HuggingFace model name


# --- System Prompts ---

DSL_GENERATION_SYSTEM = """\
You are an expert crypto trading strategist. Your task is to convert \
natural language trading ideas into a YAML strategy DSL specification.

The DSL has the following structure:
```yaml
strategy:
  name: "StrategyName"           # Valid Python identifier
  market:
    exchange: binance            # binance|okx|bybit|kraken|gate
    pair: BTC/USDT               # Format: BASE/QUOTE
    timeframe: 1h                # 1m|5m|15m|30m|1h|4h|1d|1w
  indicators:
    - name: ema_fast             # snake_case variable name
      type: EMA                  # SMA|EMA|RSI|MACD|ATR|BollingerBands|...
      params:
        period: 20               # Integer 1-500
        field: close             # open|high|low|close|volume
  entry:
    long: "ema_fast > ema_slow AND volume > vol_ma * 1.5"
    short: null
  exit:
    long: "ema_fast < ema_slow"
    short: null
  risk:
    stop_loss: -0.03             # Negative ratio (e.g. -0.03 = 3% loss)
    take_profit: 0.06            # Positive ratio (optional)
    trailing_stop: false
    trailing_stop_positive: 0.02
    max_open_trades: 3
    stake_amount: 0.1

Available indicator types:
SMA, EMA, RSI, MACD, ATR, BollingerBands, Stochastic, ADX, CCI, OBV, VWAP, WMA, HMA, ZLEMA

Boolean expressions in entry/exit can use:
- Indicator names (e.g. ema_fast, rsi)
- Built-in columns: open, high, low, close, volume
- Operators: AND, OR, NOT, >, <, >=, <=, ==, !=
- Arithmetic: +, -, *, /

Rules:
1. Output ONLY valid YAML, no explanations
2. stop_loss must be negative
3. Indicator names must be snake_case
4. All referenced indicators must be defined in the indicators list
5. Keep strategy names short and descriptive
"""

REPORT_GENERATION_SYSTEM = """\
You are a professional crypto trading analyst. Given backtest results, \
generate a clear, actionable analysis report.

Format your report as:
1. **Strategy Summary**: One-paragraph overview of the strategy logic
2. **Backtest Performance**: Key metrics interpretation
3. **Risk Assessment**: Drawdown, Sharpe ratio, position sizing analysis
4. **Strengths & Weaknesses**: What works, what doesn't
5. **Recommendation**: Whether to deploy, and any suggested improvements

Use professional but accessible language. Include specific numbers.
Be honest about weaknesses — don't sugarcoat poor performance.
"""

RISK_ASSESSMENT_SYSTEM = """\
You are a risk management expert for crypto trading. Evaluate the \
following strategy's risk profile and provide recommendations.

Consider:
- Maximum drawdown vs. acceptable threshold (<15% is good)
- Sharpe ratio (>1.0 is acceptable, >2.0 is excellent)
- Win rate and profit factor
- Position sizing adequacy
- Stop loss appropriateness for the market's volatility

Output a structured risk assessment with:
- Risk Level: Low/Medium/High/Extreme
- Key Risk Factors: List
- Recommendations: List
- Verdict: APPROVE / APPROVE WITH MODIFICATIONS / REJECT
"""


class LLMClient:
    """Client for vLLM OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request to vLLM."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def generate_dsl(self, natural_language: str) -> str:
        """Generate strategy DSL from natural language input."""
        return await self.chat(
            system_prompt=DSL_GENERATION_SYSTEM,
            user_message=f"Convert this trading idea to DSL:\n\n{natural_language}",
            temperature=0.2,
            max_tokens=1024,
        )

    async def generate_report(
        self,
        strategy_name: str,
        metrics: dict[str, Any],
    ) -> str:
        """Generate a natural language report from backtest metrics."""
        metrics_str = json.dumps(metrics, indent=2)
        return await self.chat(
            system_prompt=REPORT_GENERATION_SYSTEM,
            user_message=(
                f"Strategy: {strategy_name}\n\n"
                f"Backtest Results:\n{metrics_str}\n\n"
                "Generate a comprehensive analysis report."
            ),
            temperature=0.4,
            max_tokens=1024,
        )

    async def assess_risk(
        self,
        strategy_name: str,
        metrics: dict[str, Any],
    ) -> str:
        """Assess risk of a strategy based on backtest metrics."""
        metrics_str = json.dumps(metrics, indent=2)
        return await self.chat(
            system_prompt=RISK_ASSESSMENT_SYSTEM,
            user_message=(
                f"Strategy: {strategy_name}\n\n"
                f"Metrics:\n{metrics_str}\n\n"
                "Provide a risk assessment."
            ),
            temperature=0.3,
            max_tokens=512,
        )
