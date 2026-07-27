# Dify Workflow Setup Guide — Crypto Trading Agent

This document describes how to set up the Dify workflow for the
NL → DSL → Backtest → Report pipeline.

## Prerequisites

1. Dify running via Docker Compose (see `docker/docker-compose.yml`)
2. vLLM serving Qwen2.5-7B at `http://localhost:8000/v1`
3. Trading API server at `http://localhost:8080`

## Step 1: Configure Model Provider

1. Go to **Integrations > Model Provider**
2. Install "OpenAI-API-compatible" plugin
3. Configure:
   - **API URL**: `http://host.docker.internal:8000/v1`
   - **API Key**: `EMPTY` (or whatever vLLM requires)
   - **Model Name**: `qwen-trader-merged` (or the model name vLLM serves)
4. Add the model

## Step 2: Import API Tools

1. Go to **Integrations > Tools > Swagger API**
2. Import `dify/tools/trading_api_openapi.yml`
3. Name it "Crypto Trading API"
4. Set base URL to `http://host.docker.internal:8080`

## Step 3: Create Workflow

Create a new Chatflow with the following nodes:

### Node 1: Start
- **Input variable**: `user_message` (string) — the user's trading idea

### Node 2: LLM (NL → DSL)
- **Model**: qwen-trader-merged
- **System Prompt**: See `src/llm/prompts.py` → `DSL_GENERATION_SYSTEM`
- **User Message**: `{{#start.user_message#}}`
- **Output**: `dsl_yaml` (string)

### Node 3: Code (Validate & Parse DSL)
- **Language**: Python
- **Input**: `dsl_yaml` from Node 2
- **Code**:
```python
import yaml
import json

def main(dsl_yaml: str) -> dict:
    try:
        dsl = yaml.safe_load(dsl_yaml)
        return {
            "dsl_json": json.dumps(dsl),
            "is_valid": True,
            "error": ""
        }
    except Exception as e:
        return {
            "dsl_json": "{}",
            "is_valid": False,
            "error": str(e)
        }
```
- **Output**: `dsl_json`, `is_valid`, `error`

### Node 4: HTTP Request (Backtest)
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
- **Output**: `backtest_response` (object)

### Node 5: Code (Parse Metrics)
- **Language**: Python
- **Input**: `backtest_response` from Node 4
- **Code**:
```python
import json

def main(backtest_response: str) -> dict:
    try:
        data = json.loads(backtest_response) if isinstance(backtest_response, str) else backtest_response
        metrics = data.get("metrics", {})
        strategy_name = data.get("strategy_name", "Unknown")
        error = data.get("error", "")
        success = data.get("success", False)

        return {
            "strategy_name": strategy_name,
            "metrics_json": json.dumps(metrics, indent=2),
            "success": success,
            "error": error,
            "trades_count": data.get("metrics", {}).get("total_trades", 0)
        }
    except Exception as e:
        return {
            "strategy_name": "Error",
            "metrics_json": "{}",
            "success": False,
            "error": str(e),
            "trades_count": 0
        }
```
- **Output**: `strategy_name`, `metrics_json`, `success`, `error`

### Node 6: LLM (Report Generation)
- **Model**: qwen-trader-merged
- **System Prompt**: See `src/llm/prompts.py` → `REPORT_GENERATION_SYSTEM`
- **User Message**:
```
Strategy: {{#code2.strategy_name#}}

Backtest Results:
{{#code2.metrics_json#}}

Generate a comprehensive analysis report.
```
- **Output**: `report` (string)

### Node 7: End
- **Output**: `{{#llm2.report#}}`

## Step 4: Test

Type in the chat:
> "BTC放量突破前高，帮我做一个突破策略，止损3%"

The agent should:
1. Generate a strategy DSL (YAML)
2. Validate and run backtest
3. Return a natural language analysis report

## Step 5: Add Knowledge Base (Optional)

1. Go to **Knowledge** > Create
2. Upload trading experience documents, strategy guides, etc.
3. In the workflow, add a Knowledge Retrieval node before Node 2
4. This provides RAG-enhanced context to the LLM

## Environment Variables for Dify

Set these in Dify's Docker `.env` or the workflow variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://host.docker.internal:8000/v1` | vLLM API endpoint |
| `BACKTEST_API_URL` | `http://host.docker.internal:8080` | Trading API endpoint |
| `MODEL_NAME` | `qwen-trader-merged` | Model name in vLLM |
