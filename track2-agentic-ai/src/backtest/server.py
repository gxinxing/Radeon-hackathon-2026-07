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
from .runner import run_backtest, run_walk_forward, BacktestResult, WalkForwardResult


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
    benchmark_curve: list[float] = []
    error: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "backtest"}


@app.post("/api/backtest", response_model=BacktestResponse)
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
            benchmark_curve=[],
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
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "volatility_annual": result.volatility_annual,
            "profit_factor": round(result.profit_factor, 4) if result.profit_factor != float('inf') else None,
            "avg_win": round(result.avg_win, 2),
            "avg_loss": round(result.avg_loss, 2),
            "final_balance": result.final_balance,
            "initial_balance": result.initial_balance,
            "win_trades": result.win_trades,
            "loss_trades": result.loss_trades,
            "max_consecutive_losses": result.max_consecutive_losses,
            "avg_trade_duration": result.avg_trade_duration,
            "benchmark_return": result.benchmark_return,
            "alpha": result.alpha,
        },
        trades=result.trades[-20:],  # Last 20 trades for brevity
        equity_curve=result.equity_curve[::max(1, len(result.equity_curve) // 100)],  # Sample down
        dates=result.dates[::max(1, len(result.dates) // 100)],
        benchmark_curve=result.benchmark_curve[::max(1, len(result.benchmark_curve) // 100)] if result.benchmark_curve else []
    )


@app.post("/api/validate")
async def validate(strategy: dict[str, Any]):
    """Validate a strategy DSL without running a backtest."""
    is_valid, errors = validate_dsl(strategy)
    return {"is_valid": is_valid, "errors": errors}


@app.post("/api/walkforward")
async def walk_forward(req: BacktestRequest):
    """Run walk-forward analysis (in-sample / out-of-sample split).

    Tests strategy robustness by splitting data 70/30 and comparing
    performance between the two segments. Detects overfitting.
    """
    is_valid, errors = validate_dsl(req.strategy)
    if not is_valid:
        return {"success": False, "is_valid": False, "validation_errors": errors}

    wf: WalkForwardResult = run_walk_forward(
        strategy_dsl=req.strategy,
        days=req.days,
        initial_balance=req.initial_balance,
    )

    if wf.error:
        return {"success": False, "error": wf.error}

    def _metrics(r: BacktestResult) -> dict:
        return {
            "total_trades": r.total_trades,
            "win_rate": round(r.win_rate, 4),
            "total_return": round(r.total_return, 4),
            "max_drawdown": r.max_drawdown,
            "sharpe_ratio": r.sharpe_ratio,
            "sortino_ratio": r.sortino_ratio,
            "benchmark_return": r.benchmark_return,
            "alpha": r.alpha,
        }

    return {
        "success": True,
        "in_sample": _metrics(wf.in_sample),
        "out_of_sample": _metrics(wf.out_of_sample),
        "overfitting_score": wf.overfitting_score,
        "is_robust": wf.is_robust,
        "split_ratio": wf.split_ratio,
    }
