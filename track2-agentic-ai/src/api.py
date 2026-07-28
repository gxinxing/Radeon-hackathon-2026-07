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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .backtest.server import app as backtest_app, BacktestRequest, BacktestResponse
from .tools.market_data import router as market_router
from .tools.indicators import router as indicators_router
from .tools.paper_trade import router as paper_trade_router


app = FastAPI(
    title="Crypto Trading Agent API",
    description="Full-stack API for LLM-powered crypto trading: DSL validation, backtesting, market data, and paper trading.",
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
    return {"status": "ok", "service": "crypto-trading-agent"}


# Re-export backtest endpoints
@app.post("/api/backtest", response_model=BacktestResponse)
async def backtest(req: BacktestRequest):
    """Run a strategy backtest from DSL specification."""
    from .backtest.server import backtest as _backtest
    return await _backtest(req)


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
    from .knowledge_base.retriever import retrieve_knowledge
    if not query:
        return {"success": False, "error": "Query parameter 'query' is required"}
    context = retrieve_knowledge(query, max_results=5)
    return {"success": True, "query": query, "context": context}
