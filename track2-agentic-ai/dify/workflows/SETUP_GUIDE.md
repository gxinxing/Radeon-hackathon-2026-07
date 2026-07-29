# Dify Workflow Setup Guide — Crypto Trading Agent

Complete setup for the NL → DSL → Canonicalize → Validate → Backtest → Report pipeline
with optional paper trading confirmation branch.

## Prerequisites

1. **vLLM** serving fine-tuned Qwen2.5-7B on AMD GPU at `http://localhost:8000/v1`
   - Model name: `models/qwen-trader-merged`
2. **Trading API** (FastAPI) at `http://localhost:8080`
   - Endpoints: market summary, indicators, backtest, paper trade, walkforward
3. **Dify** running via Docker Compose
   - Dify containers access host services via `host.docker.internal`

## Docker Networking

```
┌──────────────────────────────────────────────────┐
│  Host (AMD GPU Instance)                         │
│                                                  │
│  vLLM :8000  ←─── host.docker.internal:8000      │
│  API  :8080  ←─── host.docker.internal:8080      │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │ Docker Network                            │    │
│  │  ┌──────────┐  ┌─────────┐  ┌────────┐ │    │
│  │  │ Dify API │  │ Dify Web │  │ Dify   │ │    │
│  │  │  :5001   │  │  :3000   │  │ Worker │ │    │
│  │  └──────────┘  └─────────┘  └────────┘ │    │
│  └─────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

### Quick Dify Start

```bash
cd /workspace/persistent
git clone https://github.com/langgenius/dify.git
cd dify/docker
cp .env.example .env

# Set host.docker.internal for Linux
echo 'DOCKER_HOST_NETWORK=dify-network' >> .env

docker compose up -d
```

Access Dify at `http://localhost:3000`

## Step 1: Configure Model Provider

1. Go to **Settings → Model Provider**
2. Install "OpenAI-API-compatible" plugin
3. Configure:
   - **API URL**: `http://host.docker.internal:8000/v1`
   - **API Key**: `EMPTY` (vLLM doesn't require auth)
   - **Model Name**: `models/qwen-trader-merged`
4. Save and verify connection

## Step 2: Import API Tools

1. Go to **Tools → Custom → Create Custom Tool**
2. Import `dify/tools/trading_api_openapi.yml`
3. Name it "Crypto Trading API"
4. Base URL is already set to `http://host.docker.internal:8080`
5. Available tools:
   - `getMarketSummary` — real-time price/volume
   - `getHistoricalData` — OHLCV history
   - `runBacktest` — strategy backtest
   - `calculateIndicators` — technical indicators
   - `executePaperTrade` — Binance Testnet (requires API keys)

## Step 3: Create Chatflow

Create a new **Chatflow** with these nodes:

### Node 1: Start
- **Input**: `user_message` (string) — user's trading idea

### Node 2: Knowledge Retrieval (Optional)
- **Knowledge Base**: Upload trading docs / indicator guides
- **Query**: `{{#start.user_message#}}`
- **Top K**: 3
- Provides RAG-enhanced context to LLM

### Node 3: LLM (NL → DSL Generation)
- **Model**: `models/qwen-trader-merged`
- **System Prompt**:
```
You are an expert crypto trading strategist. Convert the user's natural language
trading idea into a YAML strategy DSL specification.

Rules:
1. Output ONLY valid YAML, no explanations
2. stop_loss MUST be a negative number inside risk: (e.g. -0.03)
3. period MUST be an integer (not string)
4. indicators MUST be a non-empty list
5. entry/exit MUST only have 'long' and 'short' keys (no 'buy'/'sell')
6. All indicator names must be snake_case

DSL Structure:
strategy:
  name: "StrategyName"
  market:
    exchange: binance
    pair: BTC/USDT
    timeframe: 1h
  indicators:
    - name: ema_fast
      type: EMA
      params: {period: 20, field: close}
  entry:
    long: "ema_fast > ema_slow"
    short: null
  exit:
    long: "ema_fast < ema_slow"
    short: null
  risk:
    stop_loss: -0.03
    max_open_trades: 3
    stake_amount: 0.1
```
- **User Message**: `{{#start.user_message#}}` + RAG context if available
- **Output**: `dsl_yaml`

### Node 4: Code (Canonicalize + Validate)
- **Language**: Python
- **Input**: `dsl_yaml` from Node 3
- **Code**:
```python
import yaml, json

def main(dsl_yaml: str) -> dict:
    try:
        dsl = yaml.safe_load(dsl_yaml)
        if not isinstance(dsl, dict) or "strategy" not in dsl:
            return {"dsl_json": "{}", "is_valid": False, "error": "Missing strategy key"}

        strat = dsl["strategy"]

        # Fix common LLM errors (canonicalization)
        # 1. Coerce string numbers to int
        for ind in strat.get("indicators", []):
            params = ind.get("params", {})
            for k in ("period", "fast_period", "slow_period", "signal_period"):
                if k in params and isinstance(params[k], str):
                    try: params[k] = int(float(params[k]))
                    except: pass
            for k in ("std_dev", "multiplier"):
                if k in params and isinstance(params[k], str):
                    try: params[k] = float(params[k])
                    except: pass

        # 2. Fix stop_loss sign
        risk = strat.get("risk", {})
        if "stop_loss" in risk:
            sl = risk["stop_loss"]
            if isinstance(sl, (int, float)) and sl > 0:
                risk["stop_loss"] = -sl / 100 if sl > 1 else -sl

        # 3. Strip illegal entry/exit keys
        for section in ("entry", "exit"):
            if section in strat:
                allowed = {"long", "short"}
                for k in list(strat[section].keys()):
                    if k not in allowed:
                        del strat[section][k]
                for d in ("long", "short"):
                    if d not in strat[section]:
                        strat[section][d] = None

        # 4. Ensure risk exists
        if "risk" not in strat:
            strat["risk"] = {"stop_loss": -0.03, "max_open_trades": 3, "stake_amount": 0.1}

        return {
            "dsl_json": json.dumps(dsl),
            "is_valid": True,
            "strategy_name": strat.get("name", "Unknown"),
            "error": ""
        }
    except Exception as e:
        return {"dsl_json": "{}", "is_valid": False, "error": str(e)}
```
- **Output**: `dsl_json`, `is_valid`, `strategy_name`, `error`

### Node 5: IF/ELSE (Validation Check)
- **Condition**: `{{#code.is_valid}}` == `true`
- **True** → Node 6 (Backtest)
- **False** → Node 7 (Retry LLM)

### Node 6: HTTP Request (Backtest)
- **Method**: POST
- **URL**: `http://host.docker.internal:8080/api/backtest`
- **Headers**: `Content-Type: application/json`
- **Body**:
```json
{
  "strategy": {{#code.dsl_json#}},
  "days": 180,
  "initial_balance": 10000
}
```
- **Output**: `backtest_response`
→ Continue to Node 8

### Node 7: LLM (Retry — DSL Fix)
- **Model**: `models/qwen-trader-merged`
- **System Prompt**: Same as Node 3
- **User Message**:
```
Your previous output was invalid: {{#code.error#}}
Please fix and output ONLY valid YAML.
stop_loss MUST be negative number in risk: section.
indicators MUST be non-empty list.
entry/exit MUST only have long and short keys.
```
- **Output**: `retry_yaml`
→ Loop back to Node 4 (re-canonicalize) with max 1 retry

### Node 8: Code (Parse Metrics)
- **Input**: `backtest_response` from Node 6
- **Code**:
```python
import json

def main(backtest_response: str) -> dict:
    try:
        data = json.loads(backtest_response) if isinstance(backtest_response, str) else backtest_response
        metrics = data.get("metrics", {})
        return {
            "strategy_name": data.get("strategy_name", "Unknown"),
            "metrics_json": json.dumps(metrics, indent=2),
            "success": data.get("success", False),
            "error": data.get("error", ""),
            "trades": metrics.get("total_trades", 0),
            "win_rate": metrics.get("win_rate", 0),
            "sharpe": metrics.get("sharpe_ratio", 0),
            "return": metrics.get("total_return", 0),
            "drawdown": metrics.get("max_drawdown", 0),
            "alpha": metrics.get("alpha", 0),
        }
    except Exception as e:
        return {"error": str(e), "success": False}
```
- **Output**: `strategy_name`, `metrics_json`, `success`, `trades`, etc.

### Node 9: LLM (Risk Report Generation)
- **Model**: `models/qwen-trader-merged`
- **System Prompt**:
```
You are a professional crypto trading analyst. Given backtest results,
generate a clear analysis report in the user's language (Chinese if input is Chinese).

Format:
1. 策略概述 — Strategy logic summary
2. 回测表现 — Key metrics vs Buy&Hold benchmark
3. 风险分析 — Drawdown, Sharpe/Sortino, volatility
4. 优势与不足 — Strengths and weaknesses
5. 建议 — APPROVE / MODIFY / REJECT

Be honest. If alpha is negative, state underperformance vs buy-and-hold.
If max consecutive losses > 5, flag sustainability risk.
```
- **User Message**:
```
策略: {{#code2.strategy_name#}}
交易次数: {{#code2.trades#}}
胜率: {{#code2.win_rate#}}
Sharpe: {{#code2.sharpe#}}
收益率: {{#code2.return#}}
最大回撤: {{#code2.drawdown#}}
Alpha: {{#code2.alpha#}}

完整指标:
{{#code2.metrics_json#}}
```
- **Output**: `report`

### Node 10: Question Classifier (Paper Trading)
- **Question**: `策略分析报告已生成。是否要在 Binance Testnet 上执行模拟交易？`
- **Options**:
  - "Yes, execute paper trade" → Node 11
  - "No, just review" → Node 12

### Node 11: Tool (Paper Trade) — Optional
- **Tool**: `executePaperTrade`
- **Parameters**: `action=buy`, `pair=BTC/USDT`, `amount=0.01`
- **Note**: Requires `BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_API_SECRET`
- Paper trading is NEVER executed automatically without user confirmation

### Node 12: End
- **Output**: `{{#llm2.report#}}`

## Step 4: Test

Type in Dify chat:
> "BTC放量突破前高，帮我做一个EMA突破策略，止损3%"

Expected flow:
1. LLM generates DSL YAML (powered by AMD ROCm vLLM)
2. Canonicalizer fixes types and validates structure
3. API runs backtest on 180 days of historical data
4. LLM generates Chinese risk analysis report
5. User can optionally approve paper trading

## Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| vLLM URL | `http://host.docker.internal:8000/v1` | vLLM API (on host) |
| API URL | `http://host.docker.internal:8080` | Trading API (on host) |
| Model | `models/qwen-trader-merged` | vLLM registered model ID |
| Testnet Key | env var | Binance Testnet API key (optional) |

## Troubleshooting

- **Dify can't reach vLLM**: Ensure `host.docker.internal` resolves. On Linux, add `extra_hosts: ["host.docker.internal:host-gateway"]` to Dify's docker-compose.
- **Model name mismatch**: vLLM registers model as `models/qwen-trader-merged` (with path prefix). Use this exact name in Dify.
- **Backtest returns error**: Check that Trading API is running (`curl http://localhost:8080/health`).
- **Paper trade fails**: Binance Testnet keys must be set as environment variables on the API server.
