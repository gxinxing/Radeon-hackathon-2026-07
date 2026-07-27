"""FastAPI backtest microservice.

Accepts a strategy DSL, transpiles it, runs backtest, returns results.
This service is called by Dify's HTTP Request node.

Run:
    /opt/venv/bin/uvicorn src.backtest.server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..dsl.validator import validate_dsl
from ..dsl.transpiler import transpile_to_freqtrade
from .runner import run_backtest, BacktestResult


app = FastAPI(
    title="Crypto Trading Backtest Service",
    description="Backtest trading strategies from DSL specifications",
    version="1.0.0",
)


class BacktestRequest(BaseModel):
    """Request body for the backtest endpoint."""
    strategy: dict[str, Any]  # The full DSL dict
    days: int = 180
    initial_balance: float = 10000.0


class BacktestResponse(BaseModel):
    """Response from the backtest endpoint."""
    success: bool
    is_valid: bool
    validation_errors: list[str] = []
    strategy_name: str = ""
    strategy_code: str = ""  # Generated Freqtrade Python source
    metrics: dict[str, Any] = {}
    trades: list[dict] = []
    equity_curve: list[float] = []
    dates: list[str] = []
    error: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "backtest"}


@app.post("/backtest", response_model=BacktestResponse)
async def backtest(req: BacktestRequest) -> BacktestResponse:
    """Run a backtest from a strategy DSL.

    1. Validates the DSL against JSON Schema
    2. Transpiles to Freqtrade strategy code
    3. Runs the backtest on historical data
    4. Returns structured results
    """
    # --- Validate DSL ---
    is_valid, errors = validate_dsl(req.strategy)
    if not is_valid:
        return BacktestResponse(
            success=False,
            is_valid=False,
            validation_errors=errors,
            error="DSL validation failed",
        )

    # --- Transpile to Freqtrade code ---
    strategy_code = transpile_to_freqtrade(req.strategy)
    strategy_name = req.strategy["strategy"]["name"]

    # --- Run backtest ---
    result: BacktestResult = run_backtest(
        strategy_dsl=req.strategy,
        days=req.days,
        initial_balance=req.initial_balance,
    )

    if result.error:
        return BacktestResponse(
            success=False,
            is_valid=True,
            strategy_name=strategy_name,
            strategy_code=strategy_code,
            error=result.error,
        )

    return BacktestResponse(
        success=True,
        is_valid=True,
        strategy_name=strategy_name,
        strategy_code=strategy_code,
        metrics={
            "total_trades": result.total_trades,
            "win_rate": round(result.win_rate, 4),
            "total_return": round(result.total_return, 4),
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "profit_factor": round(result.profit_factor, 4) if result.profit_factor != float('inf') else None,
            "avg_win": round(result.avg_win, 2),
            "avg_loss": round(result.avg_loss, 2),
            "final_balance": result.final_balance,
            "initial_balance": result.initial_balance,
            "win_trades": result.win_trades,
            "loss_trades": result.loss_trades,
        },
        trades=result.trades[-20:],  # Last 20 trades for brevity
        equity_curve=result.equity_curve[::max(1, len(result.equity_curve) // 100)],  # Sample down
        dates=result.dates[::max(1, len(result.dates) // 100)],
    )


@app.post("/validate")
async def validate(strategy: dict[str, Any]):
    """Validate a strategy DSL without running a backtest."""
    is_valid, errors = validate_dsl(strategy)
    return {"is_valid": is_valid, "errors": errors}
