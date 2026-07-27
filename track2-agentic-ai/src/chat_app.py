"""Gradio-based chat interface for the Crypto Trading Agent.

Replaces Dify with a lightweight Python web UI that connects to
vLLM (OpenAI-compatible API) and the backtest microservice.

Run:
    /opt/venv/bin/python src/chat_app.py

Access: http://localhost:7860
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import gradio as gr
import httpx
import yaml

# --- Configuration ---
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
BACKTEST_API_URL = os.environ.get("BACKTEST_API_URL", "http://localhost:8080")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen-trader-merged")

# --- System Prompts ---

SYSTEM_PROMPT_DSL = """You are an expert crypto trading strategist. Convert natural language trading ideas into a YAML strategy DSL specification.

Output ONLY valid YAML with this structure:
```yaml
strategy:
  name: "StrategyName"
  market:
    exchange: binance
    pair: BTC/USDT
    timeframe: 1h
  indicators:
    - name: ema_fast
      type: EMA
      params:
        period: 20
        field: close
  entry:
    long: "ema_fast > ema_slow AND volume > vol_ma * 1.5"
    short: null
  exit:
    long: "ema_fast < ema_slow"
    short: null
  risk:
    stop_loss: -0.03
    max_open_trades: 3
    stake_amount: 0.1
```

Available indicators: SMA, EMA, RSI, MACD, ATR, BollingerBands, Stochastic, ADX, CCI, OBV, VWAP, WMA
Boolean operators: AND, OR, NOT, >, <, >=, <=, ==, !=
Rules: stop_loss must be negative. Indicator names in snake_case. Output ONLY YAML."""

SYSTEM_PROMPT_REPORT = """You are a professional crypto trading analyst. Given backtest results, generate a clear analysis report in Chinese.

Format:
1. **策略概述** — One paragraph summary
2. **回测表现** — Key metrics interpretation
3. **风险评估** — Drawdown, Sharpe ratio analysis
4. **优势与不足** — Strengths and weaknesses
5. **建议** — Whether to deploy and improvements

Be honest about poor performance. Use specific numbers."""


def call_vllm(system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
    """Call vLLM's OpenAI-compatible chat API."""
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": temperature,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM Error] {e}"


def run_backtest(strategy_dsl: dict) -> dict:
    """Call the backtest API."""
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{BACKTEST_API_URL}/api/backtest",
                json={"strategy": strategy_dsl, "days": 180, "initial_balance": 10000},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_yaml(text: str) -> dict | None:
    """Extract YAML from LLM response."""
    # Try to find YAML block
    yaml_match = re.search(r"```yaml\s*(.*?)\s*```", text, re.DOTALL)
    if yaml_match:
        yaml_text = yaml_match.group(1)
    else:
        # Try to find strategy: at the start
        yaml_match = re.search(r"(strategy:\s*\n.*?)$", text, re.DOTALL)
        if yaml_match:
            yaml_text = yaml_match.group(1)
        else:
            yaml_text = text

    try:
        parsed = yaml.safe_load(yaml_text)
        if isinstance(parsed, dict) and "strategy" in parsed:
            return parsed
    except yaml.YAMLError:
        pass
    return None


def format_metrics(metrics: dict) -> str:
    """Format backtest metrics for display."""
    if not metrics:
        return "No metrics available"

    return f"""
| Metric | Value |
|--------|-------|
| Total Trades | {metrics.get('total_trades', 0)} |
| Win Rate | {metrics.get('win_rate', 0):.1%} |
| Total Return | {metrics.get('total_return', 0):.2%} |
| Max Drawdown | {metrics.get('max_drawdown', 0):.2%} |
| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |
| Profit Factor | {metrics.get('profit_factor', 'N/A')} |
| Final Balance | ${metrics.get('final_balance', 0):,.2f} |
| Win/Loss | {metrics.get('win_trades', 0)}/{metrics.get('loss_trades', 0)} |
"""


def process_user_message(message: str, history: list) -> str:
    """Full pipeline: NL → DSL → Backtest → Report."""
    yield "🔄 正在生成策略DSL..."

    # Step 1: Generate DSL via LLM
    dsl_text = call_vllm(SYSTEM_PROMPT_DSL, message, temperature=0.2)

    # Extract YAML
    strategy_dsl = extract_yaml(dsl_text)
    if strategy_dsl is None:
        yield f"❌ LLM无法生成有效的策略DSL。原始输出:\n\n{dsl_text}"
        return

    strategy_name = strategy_dsl.get("strategy", {}).get("name", "Unknown")
    yield f"✅ 策略DSL生成完成: **{strategy_name}**\n\n🔄 正在校验并执行回测..."

    # Step 2: Run backtest
    backtest_result = run_backtest(strategy_dsl)

    if not backtest_result.get("success"):
        error = backtest_result.get("error", "Unknown error")
        yield f"✅ 策略DSL生成完成: **{strategy_name}**\n\n❌ 回测失败: {error}"
        return

    metrics = backtest_result.get("metrics", {})
    metrics_str = format_metrics(metrics)

    yield f"✅ 策略DSL生成完成: **{strategy_name}**\n✅ 回测完成\n\n🔄 正在生成分析报告..."

    # Step 3: Generate report via LLM
    report = call_vllm(
        SYSTEM_PROMPT_REPORT,
        f"策略: {strategy_name}\n\n回测结果:\n{json.dumps(metrics, indent=2)}\n\n生成中文分析报告。",
        temperature=0.4,
    )

    # Final output
    dsl_yaml = yaml.dump(strategy_dsl, default_flow_style=False, sort_keys=False, allow_unicode=True)

    final_output = f"""## 策略分析报告

{report}

---

### 📊 回测指标

{metrics_str}

---

### 📝 策略DSL (YAML)

```yaml
{dsl_yaml}
```

---

### 🤖 技术栈
- **LLM推理**: Qwen2.5-7B (QLoRA微调) on AMD ROCm GPU via vLLM
- **回测引擎**: FastAPI + CCXT + TA-Lib
- **全链路**: 自然语言 → DSL生成 → Schema校验 → 历史回测 → AI分析报告
"""
    yield final_output


def create_app():
    """Create the Gradio chat interface."""
    with gr.Blocks(
        title="Crypto Trading Agent — AMD ROCm",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("""
# 🤖 Crypto Trading Agent on AMD Radeon GPU

**AMD AI DevMaster Hackathon 2026 — Track 2: Agentic AI**

用自然语言描述交易策略，AI自动生成策略DSL → 回测 → 分析报告。
Powered by Qwen2.5-7B (QLoRA fine-tuned) on AMD ROCm GPU.
""")

        with gr.Row():
            with gr.Column(scale=4):
                chat = gr.ChatInterface(
                    fn=process_user_message,
                    title="交易策略助手",
                    examples=[
                        "BTC放量突破前高，帮我做一个EMA突破策略，止损3%",
                        "RSI超卖反弹策略，BTC/USDT 1小时线，RSI低于30买入",
                        "布林带策略，ETH/USDT，价格触及下轨买入，上轨卖出",
                        "MACD金叉策略，BTC/USDT 4小时线，止损5%",
                        "多指标共振策略：EMA金叉 + RSI超卖 + 放量确认",
                    ],
                    type="messages",
                )
            with gr.Column(scale=1):
                gr.Markdown("""
### 🏗 架构

```
用户输入 (NL)
    ↓
LLM: Qwen2.5-7B
  (vLLM/ROCm GPU)
    ↓
策略DSL (YAML)
    ↓
JSON Schema 校验
    ↓
回测 (CCXT + TA-Lib)
    ↓
LLM: 分析报告
```

### 📊 评审对标

| 维度 | 分值 |
|------|------|
| 功能完整性 | 60 |
| ROCm优化 | 40 |
""")

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
