"""Unified FastAPI server combining all tools and backtest service.

Run:
    /opt/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8080

This single server exposes:
- /backtest         — Run strategy backtest
- /validate         — Validate strategy DSL
- /market/summary   — Get market data
- /market/history   — Get historical OHLCV
- /indicators/calc  — Calculate technical indicators
- /paper-trade/exec — Execute paper trade on Binance Testnet
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from .backtest.server import app as backtest_app, BacktestRequest, BacktestResponse
from .tools.market_data import router as market_router
from .tools.indicators import router as indicators_router
from .tools.paper_trade import router as paper_trade_router


app = FastAPI(
    title="AMD CN Market Quant Agent API",
    description="Domestic stock and ETF strategy generation, validation and simulation API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for Dify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(market_router, prefix="/api")
app.include_router(indicators_router, prefix="/api")
app.include_router(paper_trade_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "cn-market-quant-agent", "mode": "simulation"}


@app.post("/api/cn/backtest/report", response_class=PlainTextResponse)
async def cn_backtest_report(payload: dict[str, Any]):
    """Return an auditable Chinese report for a domestic-market DSL."""
    from .backtest.cn_runner import run_cn_demo_backtest

    strategy = payload.get("strategy", {})
    market = strategy.get("market", {})
    constraints = strategy.get("constraints", {})
    risk = strategy.get("risk", {})
    errors: list[str] = []
    instrument = str(market.get("instrument", ""))
    if market.get("exchange") != "cn_stock":
        errors.append("exchange 必须为 cn_stock")
    if not (instrument.endswith(".SH") or instrument.endswith(".SZ")):
        errors.append("instrument 必须为 .SH 或 .SZ 证券代码")
    if constraints.get("allow_short") is not False:
        errors.append("国内现货演示禁止裸卖空")
    if int(constraints.get("lot_size", 0) or 0) != 100:
        errors.append("lot_size 必须为100股")
    if float(risk.get("stop_loss", 0) or 0) >= 0:
        errors.append("stop_loss 必须为负数")
    if errors:
        return "# AMD 国内市场策略报告\n\n- 结论：❌ REJECT\n- 原因：" + "；".join(errors)

    result = run_cn_demo_backtest(payload, days=180, initial_balance=100000.0)
    verdict = "⚠️ REVIEW" if result.max_drawdown < -0.15 or result.total_trades < 2 else "✅ PASS"
    alpha = result.total_return - result.benchmark_return
    return (
        "# AMD 国内市场量化策略演示报告\n\n"
        f"- 策略：{strategy.get('name', '未命名策略')}\n"
        f"- 标的：{instrument}\n"
        f"- 周期：{market.get('timeframe', '1d')}\n"
        "- 运行模式：Paper Trading / 模拟回测\n"
        "- 行情来源：确定性合成历史行情（仅用于系统闭环演示）\n"
        "- 市场约束：T+1、100股整数手、禁止裸卖空、佣金、卖出印花税、滑点\n"
        f"- 初始资金：¥{result.initial_balance:,.2f}\n"
        f"- 最终资金：¥{result.final_balance:,.2f}\n"
        f"- 总收益率：{result.total_return:.2%}\n"
        f"- 最大回撤：{result.max_drawdown:.2%}\n"
        f"- 完成交易：{result.total_trades} 笔\n"
        f"- 胜率：{result.win_rate:.2%}\n"
        f"- 相对演示基准 Alpha：{alpha:.2%}\n\n"
        f"## 风控结论\n\n{verdict}\n\n"
        "> 本结果只验证 AMD GPU Agent 的策略生成、校验与模拟执行闭环，不构成投资建议。"
    )


# Re-export backtest endpoints
@app.post("/api/backtest", response_model=BacktestResponse)
async def backtest(req: BacktestRequest):
    """Run a strategy backtest from DSL specification."""
    from .backtest.server import backtest as _backtest
    return await _backtest(req)


@app.post("/api/backtest/report", response_class=PlainTextResponse)
async def backtest_report(req: BacktestRequest):
    """Run a strategy backtest and return a concise Chinese report."""
    from .backtest.server import backtest_report as _backtest_report
    return await _backtest_report(req)


@app.post("/api/validate")
async def validate(strategy: dict):
    """Validate a strategy DSL without running backtest."""
    from .dsl.validator import validate_dsl
    is_valid, errors = validate_dsl(strategy)
    return {"is_valid": is_valid, "errors": errors}


@app.post("/api/walkforward")
async def walk_forward(req: BacktestRequest):
    """Run walk-forward analysis (in-sample / out-of-sample)."""
    from .backtest.server import walk_forward as _wf
    return await _wf(req)


@app.get("/api/knowledge")
async def knowledge_retrieval(query: str = ""):
    """Retrieve trading knowledge from RAG knowledge base."""
    from .knowledge_base.cn_knowledge import retrieve_cn_knowledge
    if not query:
        return {"success": False, "error": "Query parameter 'query' is required"}
    context = retrieve_cn_knowledge(query)
    return {"success": True, "query": query, "context": context}
=======
    from .knowledge_base.cn_knowledge import retrieve_cn_knowledge
    if not query:
        return {"success": False, "error": "Query parameter 'query' is required"}
    context = retrieve_cn_knowledge(query)
    return {"success": True, "query": query, "context": context}


# ── Multi-Agent Endpoint (for Dify integration) ────────────────────


@app.post("/api/agent/run")
async def agent_run(payload: dict[str, Any]):
    """Run the multi-agent pipeline (Retrieval → Reasoning → Risk).

    This endpoint allows Dify (or any HTTP client) to invoke the full
    multi-agent system in a single request — no need to chain multiple
    HTTP calls or build a complex Chatflow.

    Request body:
        {
            "message": "BTC放量突破前高，帮我做一个EMA突破策略",
            "asset": "BTC-USDT",       // optional, default BTC-USDT
            "timeframe": "1h"           // optional, default 1h
        }

    Response:
        {
            "success": true,
            "retrieval": { ... },     // Retrieval agent output
            "reasoning": { ... },     // Reasoning agent output (trading intent)
            "risk": { ... },          // Risk agent output (allow/reject)
            "report": "..."           // Formatted markdown report
        }
    """
    from .agent.protocol import AgentMessage
    from .agent.retrieval_agent import run_retrieval_agent
    from .agent.reasoning_agent import run_reasoning_agent
    from .agent.risk_agent import run_risk_agent
    import httpx

    message = payload.get("message", "")
    asset = payload.get("asset", "BTC-USDT")
    timeframe = payload.get("timeframe", "1h")

    if not message:
        return {"success": False, "error": "Missing 'message' field"}

    # ── Fetch market data ───────────────────────────────────────
    market_data = "Market data unavailable"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"http://localhost:8080/api/market/summary",
                params={"pair": asset.replace("-", "/")},
            )
            if resp.status_code == 200:
                d = resp.json()
                market_data = (
                    f"{asset} = ${d.get('last_price', 0):,.2f}, "
                    f"24h change: {d.get('change_pct', 0):+.1f}%, "
                    f"volume: {d.get('volume_24h', 0):,.0f}"
                )
    except Exception:
        pass

    # ── Step 1: Retrieval Agent ────────────────────────────────
    retrieval_msg = AgentMessage(
        payload={"query": message, "filter_meta": {"asset": asset, "timeframe": timeframe}},
        source_agent="api",
        target_agent="retrieval_agent",
        asset=asset,
        timeframe=timeframe,
    )
    retrieval_result = run_retrieval_agent(retrieval_msg)

    # ── Step 2: Reasoning Agent ────────────────────────────────
    reasoning_msg = AgentMessage(
        payload={
            **retrieval_result.payload,
            "market_data": market_data,
            "user_request": message,
        },
        source_agent="retrieval_agent",
        target_agent="reasoning_agent",
        asset=asset,
        timeframe=timeframe,
    )
    reasoning_result = run_reasoning_agent(reasoning_msg)

    # ── Step 3: Risk Agent ─────────────────────────────────────
    risk_msg = AgentMessage(
        payload=reasoning_result.payload,
        source_agent="reasoning_agent",
        target_agent="risk_agent",
        asset=asset,
        timeframe=timeframe,
    )
    risk_result = run_risk_agent(risk_msg)

    # ── Build report ───────────────────────────────────────────
    intent = reasoning_result.payload
    risk_payload = risk_result.payload
    allow = risk_payload.get("allow_execute", False)

    checks_detail = "\n".join(
        f"  {'✅' if c['passed'] else '❌'} {c['name']}: {c['detail']}"
        for c in risk_payload.get("check_details", [])
    )

    report = f"""## 多Agent决策报告

### 用户请求
{message}

### 市场数据
{market_data}

### 检索Agent
- 有效文档: {'✅ 有' if retrieval_result.payload.get('has_valid_docs') else '❌ 无'}
- 最高置信度: {retrieval_result.payload.get('max_confidence_score', 0):.2f}

### 推理Agent (交易意向)
| 字段 | 值 |
|------|-----|
| 方向 | {intent.get('view', 'neutral')} |
| 置信度 | {intent.get('confidence', 0):.2f} |
| 建议仓位 | {intent.get('suggest_position_ratio', 0):.2%} |
| 止损价 | {intent.get('stop_loss_price', 'N/A')} |
| 理由 | {intent.get('reason', '')} |

### 风控Agent (最终决策)
- 执行许可: {'✅ 允许' if allow else '❌ 驳回'}
- 最终仓位: {risk_payload.get('final_position_ratio', 0):.2%}
- 审计说明: {risk_payload.get('audit_note', '')}

#### 风控检查明细
{checks_detail}
"""

    return {
        "success": True,
        "retrieval": retrieval_result.to_dict(),
        "reasoning": reasoning_result.to_dict(),
        "risk": risk_result.to_dict(),
        "report": report,
    }


@app.post("/api/agent/reward")
async def agent_reward(payload: dict[str, Any]):
    """Compute RL reward for a strategy based on backtest metrics.

    Request body:
        {
            "metrics": { ... backtest metrics ... },
            "walkforward": { ... }  // optional
        }

    Response:
        {
            "success": true,
            "reward": { "total": 0.42, "grade": "A", "feedback": "...", "components": {...} }
        }
    """
    from .agent.reward import compute_reward

    metrics = payload.get("metrics", {})
    walkforward = payload.get("walkforward")

    if not metrics:
        return {"success": False, "error": "Missing 'metrics' field"}

    reward = compute_reward(metrics, walkforward)

    return {
        "success": True,
        "reward": {
            "total": reward.total,
            "grade": reward.grade,
            "feedback": reward.feedback,
            "components": reward.components,
        },
    }
>>>>>>> track3-honest
