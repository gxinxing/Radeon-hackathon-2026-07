#!/bin/bash
# End-to-end verification for Track 2 — Crypto Trading Agent
#
# Tests the full pipeline:
# 1. DSL validation
# 2. DSL transpilation to Freqtrade strategy
# 3. Backtest execution
# 4. LLM inference (if vLLM is running)
# 5. Paper trading API (if configured)
#
# Usage:
#   bash scripts/verify_e2e.sh

set -euo pipefail

PYTHON="${VENV_PYTHON:-/opt/venv/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "============================================"
echo "  Track 2: End-to-End Verification"
echo "============================================"
echo ""

PASS=0
FAIL=0
SKIP=0

pass() { echo "  ✅ $1"; PASS=$((PASS+1)); }
fail() { echo "  ❌ $1"; FAIL=$((FAIL+1)); }
skip() { echo "  ⏭️  $1"; SKIP=$((SKIP+1)); }

# --- Test 1: DSL Validator ---
echo "[1/6] DSL Schema Validation..."
${PYTHON} -c "
from src.dsl.validator import validate_dsl

# Valid strategy
valid = {
    'strategy': {
        'name': 'TestEMA',
        'market': {'exchange': 'binance', 'pair': 'BTC/USDT', 'timeframe': '1h'},
        'indicators': [
            {'name': 'ema_fast', 'type': 'EMA', 'params': {'period': 20, 'field': 'close'}},
            {'name': 'ema_slow', 'type': 'EMA', 'params': {'period': 50, 'field': 'close'}},
        ],
        'entry': {'long': 'ema_fast > ema_slow', 'short': None},
        'exit': {'long': 'ema_fast < ema_slow', 'short': None},
        'risk': {'stop_loss': -0.03, 'max_open_trades': 3, 'stake_amount': 0.1},
    }
}
is_valid, errors = validate_dsl(valid)
assert is_valid, f'Valid strategy failed: {errors}'
print('  Valid strategy passed validation')

# Invalid strategy (positive stop_loss)
invalid = {'strategy': {**valid['strategy'], 'risk': {'stop_loss': 0.03, 'max_open_trades': 3, 'stake_amount': 0.1}}}
is_valid, errors = validate_dsl(invalid)
assert not is_valid, 'Invalid strategy should have failed'
print('  Invalid strategy correctly rejected')
" && pass "DSL validation" || fail "DSL validation"

# --- Test 2: DSL Transpiler ---
echo ""
echo "[2/6] DSL → Freqtrade Transpilation..."
${PYTHON} -c "
from src.dsl.transpiler import transpile_to_freqtrade

dsl = {
    'strategy': {
        'name': 'TestStrategy',
        'market': {'exchange': 'binance', 'pair': 'BTC/USDT', 'timeframe': '1h'},
        'indicators': [
            {'name': 'ema_fast', 'type': 'EMA', 'params': {'period': 20, 'field': 'close'}},
            {'name': 'ema_slow', 'type': 'EMA', 'params': {'period': 50, 'field': 'close'}},
            {'name': 'rsi', 'type': 'RSI', 'params': {'period': 14}},
        ],
        'entry': {'long': 'ema_fast > ema_slow AND rsi < 70', 'short': None},
        'exit': {'long': 'ema_fast < ema_slow', 'short': None},
        'risk': {'stop_loss': -0.03, 'max_open_trades': 3, 'stake_amount': 0.1, 'trailing_stop': True, 'trailing_stop_positive': 0.02},
    }
}
code = transpile_to_freqtrade(dsl)
assert 'class TestStrategy(IStrategy)' in code
assert 'def populate_indicators' in code
assert 'def populate_entry_trend' in code
assert 'def populate_exit_trend' in code
assert 'stoploss = -0.03' in code
print('  Generated strategy code contains all required methods')
" && pass "DSL transpilation" || fail "DSL transpilation"

# --- Test 3: Backtest API ---
echo ""
echo "[3/6] Backtest Microservice..."
# Check if API is running
if curl -s http://localhost:8080/health >/dev/null 2>&1; then
    # Run a backtest via API
    RESPONSE=$(curl -s -X POST http://localhost:8080/api/backtest \
        -H "Content-Type: application/json" \
        -d '{
            "strategy": {
                "strategy": {
                    "name": "VerifyEMA",
                    "market": {"exchange": "binance", "pair": "BTC/USDT", "timeframe": "1h"},
                    "indicators": [
                        {"name": "ema_fast", "type": "EMA", "params": {"period": 20, "field": "close"}},
                        {"name": "ema_slow", "type": "EMA", "params": {"period": 50, "field": "close"}}
                    ],
                    "entry": {"long": "ema_fast > ema_slow", "short": null},
                    "exit": {"long": "ema_fast < ema_slow", "short": null},
                    "risk": {"stop_loss": -0.03, "max_open_trades": 3, "stake_amount": 0.1}
                }
            },
            "days": 30
        }')

    SUCCESS=$(echo "$RESPONSE" | ${PYTHON} -c "import sys,json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "False")
    if [ "$SUCCESS" = "True" ]; then
        pass "Backtest API returned results"
    else
        fail "Backtest API returned error: $RESPONSE"
    fi
else
    skip "Backtest API not running (start with: uvicorn src.api:app --port 8080)"
fi

# --- Test 4: LLM Inference ---
echo ""
echo "[4/6] LLM Inference (vLLM)..."
if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:8000/v1/models | ${PYTHON} -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('data',[])))" 2>/dev/null || echo "0")
    if [ "$MODELS" -gt "0" ]; then
        # Test NL → DSL generation
        RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are a trading expert. Output ONLY valid YAML."},
                    {"role": "user", "content": "Create a simple EMA crossover strategy for BTC/USDT with EMA 20 and 50."}
                ],
                "max_tokens": 256
            }')
        HAS_CONTENT=$(echo "$RESPONSE" | ${PYTHON} -c "import sys,json; print(len(json.load(sys.stdin).get('choices',[{}])[0].get('message',{}).get('content','')))" 2>/dev/null || echo "0")
        if [ "$HAS_CONTENT" -gt "10" ]; then
            pass "LLM inference working (${HAS_CONTENT} chars generated)"
        else
            fail "LLM returned empty response"
        fi
    else
        fail "vLLM running but no models loaded"
    fi
else
    skip "vLLM not running (start with: bash training/scripts/serve_vllm.sh)"
fi

# --- Test 5: Market Data ---
echo ""
echo "[5/6] Market Data (CCXT)..."
${PYTHON} -c "
from src.backtest.data_fetcher import get_market_summary
summary = get_market_summary('BTC/USDT', 'binance')
assert summary['last_price'] > 0, 'BTC price should be positive'
print(f'  BTC/USDT: \${summary[\"last_price\"]:,.2f} ({summary[\"change_pct\"]:+.2f}%)')
" && pass "Market data fetch" || fail "Market data fetch"

# --- Test 6: Training Data ---
echo ""
echo "[6/6] Training Data..."
DATA_FILE="${PROJECT_ROOT}/training/data/processed/merged_train.jsonl"
if [ -f "$DATA_FILE" ]; then
    COUNT=$(wc -l < "$DATA_FILE")
    if [ "$COUNT" -gt "100" ]; then
        pass "Training data ready ($COUNT samples)"
    else
        fail "Training data too small ($COUNT samples)"
    fi
else
    skip "Training data not generated (run: python training/data/prepare_dsl_pairs.py)"
fi

# --- Summary ---
echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================"

if [ "$FAIL" -gt "0" ]; then
    exit 1
fi
