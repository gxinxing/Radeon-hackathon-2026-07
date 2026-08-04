#!/bin/bash
# Track 2 Demo — Agentic AI Quantitative Trading Agent
# AMD AI DevMaster Hackathon 2026

cd "/Users/simon/Documents/01_AI and Code Development/Radeon-hackathon-2026-07/track2-agentic-ai"

echo "================================================================"
echo "  AMD AI DevMaster Hackathon 2026 — Track 2: Agentic AI"
echo "  Quantitative Trading Agent on AMD Radeon GPU"
echo "  Model: Qwen2.5-7B (LoRA fine-tuned on ROCm)"
echo "================================================================"
echo ""

# Step 1: Run unit tests
echo ">>> Step 1: Unit Tests (282 tests, offline)"
echo "----------------------------------------------------------------"
python3 -m pytest tests/ --ignore=tests/test_e2e.py -q 2>&1 | tail -5
echo ""

# Step 2: Demo - General Assistant (intent routing + RAG + tools)
echo ">>> Step 2: Agent Demo — Intent Routing + RAG + Multi-Tool Pipeline"
echo "----------------------------------------------------------------"
EXTERNAL_TOOLS_MODE=mock python3 scripts/demo_general_assistant.py --all --slow 2>&1
echo ""

# Step 3: DSL Pipeline
echo ">>> Step 3: E2E Pipeline — NL → DSL → Canonicalize → Validate → Transpile"
echo "----------------------------------------------------------------"
python3 -c "
import sys
sys.path.insert(0, '.')
from src.dsl.canonicalizer import canonicalize_dsl
from src.dsl.validator import validate_dsl
from src.dsl.transpiler import transpile_to_freqtrade
from src.dsl.transpiler_backtrader import transpile_to_backtrader
import copy, yaml

dsl = yaml.safe_load('''
strategy:
  name: BTC_EMA_Crossover
  market:
    exchange: binance
    pair: BTC/USDT
    timeframe: 1h
  indicators:
    - {name: ema_fast, type: EMA, params: {period: 20, field: close}}
    - {name: ema_slow, type: EMA, params: {period: 50, field: close}}
  entry:
    long: ema_fast > ema_slow
    short: null
  exit:
    long: ema_fast < ema_slow
    short: null
  risk:
    stop_loss: -0.03
    max_position_pct: 0.3
''')

canon = copy.deepcopy(dsl)
canon, repairs, errors = canonicalize_dsl(canon)
valid, verrors = validate_dsl(canon)

print(f'  Strategy: {canon[\"strategy\"][\"name\"]}')
print(f'  Canonicalization repairs: {len(repairs)}')
for r in repairs[:5]:
    print(f'    {r.field}: {r.raw} -> {r.normalized} ({r.repair_type})')
print(f'  Validation: {\"PASS\" if valid else \"FAIL\"}')
print(f'  Freqtrade transpile: {\"OK\" if transpile_to_freqtrade(canon) else \"FAIL\"}')
print(f'  Backtrader transpile: {\"OK\" if transpile_to_backtrader(canon) else \"FAIL\"}')
"
echo ""

# Step 4: Risk Agent — Veto Power
echo ">>> Step 4: Risk Agent — Veto Power (LLM proposes, deterministic code disposes)"
echo "----------------------------------------------------------------"
python3 -c "
import sys
sys.path.insert(0, '.')
from src.agent.risk_agent import run_risk_agent
from src.agent.protocol import AgentMessage

# Case 1: Valid trade — should pass
msg1 = AgentMessage(payload={'view': 'long', 'confidence': 0.72, 'position_ratio': 0.25, 'stop_loss': -0.05, 'reason': 'EMA crossover confirmed by volume'})
r1 = run_risk_agent(msg1)
p1 = r1.payload
print(f'  Case 1 (valid trade):     allow={p1[\"allow_execute\"]}  position={p1[\"final_position_ratio\"]}')

# Case 2: Over position limit — should be vetoed
msg2 = AgentMessage(payload={'view': 'long', 'confidence': 0.85, 'position_ratio': 0.45, 'stop_loss': -0.05, 'reason': 'Strong breakout'})
r2 = run_risk_agent(msg2)
p2 = r2.payload
failed2 = [c['name'] for c in p2['check_details'] if not c['passed']]
print(f'  Case 2 (over limit 45%):   allow={p2[\"allow_execute\"]}  vetoed_by={failed2}')

# Case 3: No stop loss — should be vetoed
msg3 = AgentMessage(payload={'view': 'long', 'confidence': 0.80, 'position_ratio': 0.20, 'stop_loss': 0, 'reason': 'Momentum signal'})
r3 = run_risk_agent(msg3)
p3 = r3.payload
failed3 = [c['name'] for c in p3['check_details'] if not c['passed']]
print(f'  Case 3 (no stop loss):     allow={p3[\"allow_execute\"]}  vetoed_by={failed3}')

# Case 4: Neutral view — passes with 0 position
msg4 = AgentMessage(payload={'view': 'neutral', 'confidence': 0.50, 'position_ratio': 0.0, 'stop_loss': -0.05, 'reason': 'Uncertain market'})
r4 = run_risk_agent(msg4)
p4 = r4.payload
print(f'  Case 4 (neutral view):     allow={p4[\"allow_execute\"]}  position={p4[\"final_position_ratio\"]}')
"
echo ""

# Step 5: RL Reward Function
echo ">>> Step 5: RL Reward — 8-Dimensional Strategy Scoring"
echo "----------------------------------------------------------------"
python3 -c "
import sys
sys.path.insert(0, '.')
from src.agent.reward import compute_reward

# Good strategy
good = {
    'total_return': 0.15, 'alpha': 0.08, 'sharpe_ratio': 1.5,
    'sortino_ratio': 2.0, 'max_drawdown': -0.08, 'consecutive_losses': 3,
}
r1 = compute_reward(good)
print(f'  Good strategy:  reward={r1.total:.3f}  grade={r1.grade}')
print(f'    feedback: {r1.feedback}')

# Bad strategy
bad = {
    'total_return': -0.10, 'alpha': -0.05, 'sharpe_ratio': -0.5,
    'sortino_ratio': -0.3, 'max_drawdown': -0.25, 'consecutive_losses': 8,
}
r2 = compute_reward(bad)
print(f'  Bad strategy:   reward={r2.total:.3f}  grade={r2.grade}')
print(f'    feedback: {r2.feedback}')
"
echo ""

echo "================================================================"
echo "  Demo Complete — AMD Radeon GPU | Qwen2.5-7B | ROCm 7.2"
echo "  282 Tests | ReAct Agent | Risk Veto | RL Reward | DSL Pipeline"
echo "================================================================"
