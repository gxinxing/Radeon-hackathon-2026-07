#!/bin/bash
# End-to-end verification for Track 2 — Domestic Market Quant Agent (CN market)
#
# Tests the current CN main chain:
#   1. DSL validation (CN DSL with constraints)
#   2. DSL canonicalization + validation
#   3. RAG knowledge API (/api/knowledge)
#   4. CN backtest API (/api/cn/backtest/report)
#   5. LLM inference (if vLLM is running)
#   6. CN training data readiness
#
# The legacy crypto chain (Binance/Freqtrade paper trading) is covered by the
# offline unit tests only (tests/test_*.py, 282 passed).
#
# Usage:
#   bash scripts/verify_e2e.sh

set -euo pipefail

PYTHON="${VENV_PYTHON:-python3}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "============================================"
echo "  Track 2: End-to-End Verification (CN market)"
echo "============================================"
echo ""

PASS=0
FAIL=0
SKIP=0

pass() { echo "  ✅ $1"; PASS=$((PASS+1)); }
fail() { echo "  ❌ $1"; FAIL=$((FAIL+1)); }
skip() { echo "  ⏭️  $1"; SKIP=$((SKIP+1)); }

CN_DSL='{"strategy":{"name":"CN_EMA","market":{"exchange":"cn_stock","instrument":"510300.SH","timeframe":"1d"},"indicators":[{"name":"ema_fast","type":"EMA","params":{"period":20,"field":"close"}},{"name":"ema_slow","type":"EMA","params":{"period":50,"field":"close"}}],"entry":{"long":"ema_fast > ema_slow","short":null},"exit":{"long":"ema_fast < ema_slow","short":null},"constraints":{"t_plus_one":true,"price_limit":0.1,"allow_short":false,"lot_size":100},"risk":{"stop_loss":-0.05,"max_position_pct":0.3,"max_drawdown":-0.15}}}'

# --- Test 1: DSL Validator (CN shape) ---
echo "[1/6] DSL Schema Validation (CN market)..."
${PYTHON} -c "
from src.dsl.validator import validate_dsl
import json, sys

dsl = json.loads('''$CN_DSL''')
is_valid, errors = validate_dsl(dsl)
assert is_valid, f'CN DSL should pass validation: {errors}'
print('  CN DSL (constraints included) passed validation')

invalid = json.loads('''$CN_DSL''')
invalid['strategy']['risk']['stop_loss'] = 0.03
is_valid, errors = validate_dsl(invalid)
assert not is_valid, 'Positive stop_loss should be rejected'
print('  Invalid strategy (positive stop_loss) correctly rejected')
" && pass "DSL validation (CN)" || fail "DSL validation (CN)"

# --- Test 2: DSL Canonicalization ---
echo ""
echo "[2/6] DSL Canonicalization + Validation..."
${PYTHON} -c "
from src.dsl.canonicalizer import canonicalize_dsl
from src.dsl.validator import validate_dsl
import json, copy

dsl = json.loads('''$CN_DSL''')
canon = copy.deepcopy(dsl)
canon, repairs, errors = canonicalize_dsl(canon)
assert not errors, f'Canonicalization errors: {errors}'
valid, verrors = validate_dsl(canon)
assert valid, f'Canonicalized DSL failed validation: {verrors}'
print(f'  Canonicalized OK, repairs: {len(repairs)}, validation: PASS')
" && pass "DSL canonicalization" || fail "DSL canonicalization"

# --- Test 3: RAG Knowledge API ---
echo ""
echo "[3/6] RAG Knowledge API (/api/knowledge)..."
if curl -s http://localhost:8080/health >/dev/null 2>&1; then
    RESPONSE=$(curl -s "http://localhost:8080/api/knowledge?query=T%2B1%20%E6%B6%A8%E8%B7%8C%E5%81%9C" || true)
    SUCCESS=$(echo "$RESPONSE" | ${PYTHON} -c "import sys,json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "False")
    if [ "$SUCCESS" = "True" ]; then
        pass "RAG knowledge API returned context"
    else
        fail "RAG knowledge API error: $RESPONSE"
    fi
else
    skip "API not running (start with: uvicorn src.api:app --host 0.0.0.0 --port 8080)"
fi

# --- Test 4: CN Backtest API ---
echo ""
echo "[4/6] CN Backtest API (/api/cn/backtest/report)..."
if curl -s http://localhost:8080/health >/dev/null 2>&1; then
    RESPONSE=$(curl -s -X POST http://localhost:8080/api/cn/backtest/report \
        -H "Content-Type: application/json" -d "$CN_DSL" || true)
    HAS_VERDICT=$(echo "$RESPONSE" | grep -c "风控结论" || true)
    if [ "$HAS_VERDICT" -gt 0 ]; then
        VERDICT=$(echo "$RESPONSE" | grep -A2 "风控结论" | tail -1 | tr -d ' ')
        pass "CN backtest report generated (verdict: ${VERDICT})"
    else
        fail "CN backtest API returned unexpected response: $(echo "$RESPONSE" | head -c 300)"
    fi
else
    skip "API not running (start with: uvicorn src.api:app --host 0.0.0.0 --port 8080)"
fi

# --- Test 5: LLM Inference (vLLM) ---
echo ""
echo "[5/6] LLM Inference (vLLM)..."
if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:8000/v1/models | ${PYTHON} -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('data',[])))" 2>/dev/null || echo "0")
    if [ "$MODELS" -gt "0" ]; then
        MODEL_ID=$(curl -s http://localhost:8000/v1/models | ${PYTHON} -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "models/qwen-trader-merged")
        RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$MODEL_ID\", \"messages\": [{\"role\": \"system\", \"content\": \"You are a CN market quant assistant. Output ONLY valid YAML.\"}, {\"role\": \"user\", \"content\": \"Create a simple EMA crossover strategy for 510300.SH with EMA 20 and 50, stop loss 3%.\"}], \"max_tokens\": 256}")
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

# --- Test 6: CN Training Data ---
echo ""
echo "[6/6] CN Training Data..."
DATA_FILE="${PROJECT_ROOT}/training/data/processed/cn_market_train.jsonl"
if [ -f "$DATA_FILE" ]; then
    COUNT=$(wc -l < "$DATA_FILE")
    if [ "$COUNT" -gt "100" ]; then
        pass "CN training data ready ($COUNT samples)"
    else
        fail "CN training data too small ($COUNT samples)"
    fi
else
    fail "CN training data missing (run: python training/data/generate_cn_market_pairs.py)"
fi

# --- Summary ---
echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================"

if [ "$FAIL" -gt "0" ]; then
    exit 1
fi
