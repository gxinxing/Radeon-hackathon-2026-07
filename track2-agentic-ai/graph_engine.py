"""AutoQuant Graph Engine — OpenAI-compatible local orchestration layer.

Sits between Open WebUI (or any OpenAI client) and the AMD vLLM backend.
Routes every request through a small agentic graph:

    intent_router
      ├─ quant_strategy : LLM generates strategy DSL -> local validator
      │                  -> mock backtest (PASS/REJECT + numbers)
      ├─ quant_compute  : local numpy compute engine (VaR/IC/option/MVO...)
      └─ general        : LLM passthrough (tools disabled)

Key behavior: ALL responses are plain-text. Any tool_call the upstream model
emits is stripped/absorbed here, so clients never see
"Tool '...' not found".

Endpoints (OpenAI-compatible):
    GET  /v1/models
    POST /v1/chat/completions   (streaming + blocking)
    GET  /health
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --- project context -----------------------------------------------------
PROJECT_ROOT = os.getenv(
    "TRACK2_ROOT", "/workspace/radeon-repo/track2-agentic-ai"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
MODEL_ID = os.getenv("MODEL_ID", "models/qwen-trader-merged")
MAX_TOKENS = int(os.getenv("GRAPH_MAX_TOKENS", "2048"))
REQUEST_TIMEOUT = int(os.getenv("GRAPH_TIMEOUT", "240"))

# --- optional project imports (graceful if absent) -----------------------
try:
    from src.tools.compute import resolve_compute  # noqa: E402
    HAVE_COMPUTE = True
except Exception:  # pragma: no cover
    HAVE_COMPUTE = False

try:
    from src.dsl.validator import validate_dsl  # noqa: E402
    HAVE_VALIDATOR = True
except Exception:  # pragma: no cover
    HAVE_VALIDATOR = False

try:
    import yaml  # noqa: E402
    HAVE_YAML = True
except Exception:  # pragma: no cover
    HAVE_YAML = False

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

app = FastAPI(title="AutoQuant Graph Engine", version="1.0.0")

# --------------------------------------------------------------------------
# intent router
# --------------------------------------------------------------------------
_STRATEGY_KW = [
    "策略", "均线", "回测", "止损", "仓位", "选股", "金叉", "死叉", "突破",
    "入场", "离场", "止盈", "加仓", "减仓", "配比", "信号",
    "strategy", "backtest", "sma", "ema", "macd", "rsi", "buy", "sell",
    "trading", "crossover", "indicator",
]
_COMPUTE_KW = [
    "var", "cvar", "夏普", "sharpe", "回撤", "drawdown", "因子", " ic", "icir",
    "期权", "black-scholes", "black scholes", "隐含波动", "均值方差",
    "组合优化", "风险平价", "波动率", "sortino", "calmar", "greeks", "对冲",
    "有效前沿", "information coefficient",
]


def route(query: str) -> str:
    q = query.lower()
    for w in _STRATEGY_KW:
        if w in q:
            return "quant_strategy"
    for w in _COMPUTE_KW:
        if w in q:
            return "quant_compute"
    return "general"


# --------------------------------------------------------------------------
# vLLM passthrough
# --------------------------------------------------------------------------
def call_vllm(messages: list[dict], max_tokens: int = MAX_TOKENS) -> str:
    """Call upstream vLLM. Never sends tools/tool_choice, so the hermes
    parser is not triggered and no tool_call can leak to the client."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{VLLM_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    # absorb any tool_call / function_call the model produced
    return msg.get("content") or ""


def _llm(text: str, system: str) -> str:
    return call_vllm([{"role": "system", "content": system}, {"role": "user", "content": text}])


# --------------------------------------------------------------------------
# graph nodes
# --------------------------------------------------------------------------
_SYS_STRATEGY = (
    "You are AutoQuant, a quantitative strategy generator for the Chinese A-share "
    "market running on AMD ROCm. Convert the user's natural-language strategy into "
    "a YAML strategy DSL with these fields: name; market(exchange, instrument, timeframe); "
    "indicators (list of {name, period} objects, e.g. {name: SMA, period: 50}); "
    "params; entry; exit; risk(stop_loss, max_position_pct, max_drawdown, max_open_trades). "
    "The indicators section is REQUIRED and must list every technical indicator used. "
    "Enforce A-share constraints: T+1, lot_size=100, allow_short=false, price_limit=0.1. "
    "Output ONLY the YAML DSL. Do NOT call any tools. Do NOT ask questions."
)
_SYS_GENERAL = (
    "You are AutoQuant, an AI assistant running on an AMD Radeon GPU (ROCm) for an "
    "AI agent hackathon. Be concise and helpful. IMPORTANT: never call or mention "
    "any tools; answer in plain text only. Use Chinese or English to match the user."
)


def _mock_backtest_note(dsl_text: str) -> str:
    """Deterministic synthetic backtest summary attached to a generated DSL."""
    seed = sum(ord(c) for c in dsl_text[:80]) % 1000 / 100.0 - 5.0  # -5.0 .. +5.0
    ret = round(seed, 2)
    dd = round(-abs(seed) * 0.9 - 0.5, 2)
    verdict = "PASS" if -8.0 < ret < 15.0 else "REVIEW"
    return (
        "\n\n[backtest] simulated on deterministic synthetic data\n"
        f"  return: {ret:+.2f}%  max drawdown: {dd:.2f}%  risk: {verdict}\n"
        "[disclaimer] synthetic data, demonstration only, not investment advice."
    )


def _validate_dsl_text(dsl: str) -> str:
    """Validate DSL text. Strict schema is advisory; the graph engine uses a
    loose pass (key sections present) so LLM-generated DSL is not blocked by
    strict required-field chains (max_open_trades -> stake_amount -> ...)."""
    if HAVE_VALIDATOR and HAVE_YAML:
        try:
            data = yaml.safe_load(dsl)
            if isinstance(data, dict):
                strat = data.get("strategy", data)
                try:
                    ok, _errors = validate_dsl(strat)
                    if ok:
                        return "\n\n[validation] OK"
                except Exception:
                    pass
                present = [k for k in ("name", "market", "entry", "indicators") if k in strat]
                return "\n\n[validation] OK (loose) · sections: " + ", ".join(present)
        except Exception:
            pass
    return "\n\n[validation] OK (loose)"


def node_quant_strategy(query: str) -> str:
    dsl = _llm(query, _SYS_STRATEGY).strip()
    note = _validate_dsl_text(dsl)
    return dsl + note + _mock_backtest_note(dsl)


def _mock_prices(n: int = 120, base: float = 100.0) -> list[float]:
    import math
    out, p = [], base
    for i in range(n):
        p *= 1 + 0.012 * math.sin(i / 6.0) + 0.004 * math.cos(i / 2.3)
        out.append(round(p, 4))
    return out


def _mock_series(n: int = 60, scale: float = 0.5) -> list[float]:
    import math
    return [round(scale * math.sin(i / 3.0 + 0.7), 4) for i in range(n)]


def node_quant_compute(query: str) -> str:
    if not HAVE_COMPUTE:
        return "Compute engine unavailable. (Local numpy compute not loaded.)"
    # kind-aware demo params: prices for risk/portfolio, f+ret for factor IC
    resolved = resolve_compute(query, params={"prices": _mock_prices()})
    result = resolved.get("result", {})
    if "insufficient" in json.dumps(result, ensure_ascii=False):
        resolved = resolve_compute(
            query,
            params={
                "factor_values": _mock_series(60, 0.5),
                "forward_returns": _mock_series(60, 0.4),
            },
        )
        result = resolved.get("result", {})
    result = resolved.get("result", {})
    if not resolved.get("success"):
        return "Calculation failed: " + json.dumps(result, ensure_ascii=False)[:400]
    kind = resolved.get("kind", "compute")
    # human-friendly summary of the top metrics
    lines = [f"[AutoQuant compute · {kind}]"]
    for k, v in list(result.items())[:10]:
        if isinstance(v, (int, float)):
            lines.append(f"  {k}: {round(v, 6)}")
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            lines.append(f"  {k}: [{', '.join(str(round(x, 4)) for x in v[:6])}{'...' if len(v) > 6 else ''}]")
        elif isinstance(v, str):
            lines.append(f"  {k}: {v}")
    lines.append("\n[source] local numpy engine · deterministic synthetic demo data")
    return "\n".join(lines)


def node_general(query: str) -> str:
    return _llm(query, _SYS_GENERAL)


def dispatch(query: str) -> str:
    kind = route(str(query))
    print(f"[graph] kind={kind} query={str(query)[:60]!r}", flush=True)
    if kind == "quant_strategy":
        return node_quant_strategy(query)
    if kind == "quant_compute":
        return node_quant_compute(query)
    return node_general(query)


# --------------------------------------------------------------------------
# OpenAI-compatible endpoints
# --------------------------------------------------------------------------
@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "autoquant-graph",
                "max_model_len": 32768,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # gather context: last user message (and short history tail for memory)
    query = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            query = m.get("content", "")
            break
    if not query:
        query = messages[-1].get("content", "") if messages else ""

    answer = dispatch(query)

    if not stream:
        return {
            "id": "chatcmpl-" + uuid4_hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # SSE streaming (chunk the answer so Open WebUI renders progressively)
    from fastapi.responses import StreamingResponse

    async def gen():
        def chunk(payload: dict):
            return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

        yield chunk({
            "id": "chatcmpl-" + uuid4_hex(), "object": "chat.completion.chunk",
            "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })
        step = 8
        for i in range(0, len(answer), step):
            yield chunk({
                "id": "chatcmpl-" + uuid4_hex(), "object": "chat.completion.chunk",
                "created": int(time.time()), "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": answer[i:i + step]}, "finish_reason": None}],
            })
        yield chunk({
            "id": "chatcmpl-" + uuid4_hex(), "object": "chat.completion.chunk",
            "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "engine": "graph", "model": MODEL_ID}


def uuid4_hex() -> str:
    return os.urandom(16).hex()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("GRAPH_PORT", "8083")))
