"""Paper trading tool — Binance Testnet integration.

Executes simulated trades on Binance Testnet for risk-free
strategy validation in live market conditions.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

import ccxt
from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/paper-trade", tags=["paper-trade"])


# Binance Testnet exchange instance (lazy init)
_testnet_exchange: ccxt.Exchange | None = None


def get_testnet_exchange(
    api_key: str | None = None,
    api_secret: str | None = None,
) -> ccxt.Exchange:
    """Get or create Binance Testnet exchange instance."""
    global _testnet_exchange
    if _testnet_exchange is None:
        _testnet_exchange = ccxt.binance({
            "apiKey": api_key or "",
            "secret": api_secret or "",
            "enableRateLimit": True,
            "urls": {
                "api": {
                    "public": "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            },
        })
    return _testnet_exchange


class PaperTradeRequest(BaseModel):
    action: str  # "buy" | "sell" | "status" | "balance"
    pair: str = "BTC/USDT"
    amount: float | None = None
    price: float | None = None  # Limit order price, None for market


class PaperTradeResponse(BaseModel):
    success: bool
    action: str
    pair: str
    details: dict[str, Any] = {}
    error: str | None = None


@router.post("/execute", response_model=PaperTradeResponse)
async def execute_paper_trade(req: PaperTradeRequest):
    """Execute a paper trade on Binance Testnet.

    Requires API keys to be configured in environment variables:
    BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET
    """
    import os

    api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")

    if not api_key or not api_secret:
        return PaperTradeResponse(
            success=False,
            action=req.action,
            pair=req.pair,
            error="Binance Testnet API keys not configured. Set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET.",
        )

    try:
        exchange = get_testnet_exchange(api_key, api_secret)

        if req.action == "buy":
            order = exchange.create_market_buy_order(req.pair, req.amount)
            return PaperTradeResponse(
                success=True,
                action="buy",
                pair=req.pair,
                details={
                    "order_id": order.get("id"),
                    "amount": order.get("amount"),
                    "cost": order.get("cost"),
                    "filled": order.get("filled"),
                },
            )
        elif req.action == "sell":
            order = exchange.create_market_sell_order(req.pair, req.amount)
            return PaperTradeResponse(
                success=True,
                action="sell",
                pair=req.pair,
                details={
                    "order_id": order.get("id"),
                    "amount": order.get("amount"),
                    "cost": order.get("cost"),
                    "filled": order.get("filled"),
                },
            )
        elif req.action == "balance":
            balance = exchange.fetch_balance()
            return PaperTradeResponse(
                success=True,
                action="balance",
                pair=req.pair,
                details={
                    "total": balance.get("total", {}),
                    "free": balance.get("free", {}),
                    "used": balance.get("used", {}),
                },
            )
        elif req.action == "status":
            ticker = exchange.fetch_ticker(req.pair)
            return PaperTradeResponse(
                success=True,
                action="status",
                pair=req.pair,
                details={
                    "last_price": ticker.get("last"),
                    "bid": ticker.get("bid"),
                    "ask": ticker.get("ask"),
                    "volume": ticker.get("baseVolume"),
                },
            )
        else:
            return PaperTradeResponse(
                success=False,
                action=req.action,
                pair=req.pair,
                error=f"Unknown action: {req.action}",
            )
    except Exception as e:
        return PaperTradeResponse(
            success=False,
            action=req.action,
            pair=req.pair,
            error=str(e),
        )
