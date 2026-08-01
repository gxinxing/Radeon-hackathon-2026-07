# Video Demo Script: Domestic Market Quantitative Agent

**Duration**: ~5 minutes | **Format**: Screen recording with English voiceover

---

## Part 1: Opening (0:00 – 0:20)

**Screen**: Project title page

**Narration**:
> This is a domestic market quantitative strategy agent powered by AMD ROCm GPU.
> Users describe trading strategies in Chinese natural language, and the system
> automatically generates a strategy DSL, executes a simulated backtest, and
> produces a risk report — all running on AMD hardware.

**On-screen text**:
```
AMD ROCm 7.2.1 | gfx1100 | Qwen2.5-7B | LoRA | vLLM
Track 2: Agentic AI
```

---

## Part 2: AMD GPU Evidence (0:20 – 1:00)

**Screen 1**: Terminal — `rocminfo`

**Action**: Run `rocminfo | grep -E "Name:|Marketing Name:|Device Type:"`

**Narration**:
> The system runs on an AMD Radeon Graphics GPU (gfx1100) with ROCm 7.2.1,
> powered by an AMD EPYC 9334 32-core processor.

**Screen 2**: Terminal — vLLM service check

**Action**: Run `curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool`

**Narration**:
> vLLM version 0.16.1 is serving the fine-tuned model named
> models/qwen-trader-merged — a Qwen2.5-7B model with LoRA weights
> fine-tuned for the Chinese domestic market.

**On-screen text**:
```
GPU: AMD Radeon Graphics (gfx1100)
ROCm: 7.2.1
vLLM: 0.16.1
Model: models/qwen-trader-merged (Qwen2.5-7B + CN LoRA)
```

---

## Part 3: LoRA Training Results (1:00 – 1:45)

**Screen**: Terminal — training log

**Action**: Run `tail -20 /persistent/track2/logs/cn_qlora_train.log`

**Narration**:
> The model was fine-tuned using 400 domestic market strategy samples on this
> AMD GPU. Three epochs, 39 training steps, final loss of 0.2848,
> token accuracy of 98.1%, and peak GPU memory of 16.21 GB.

**On-screen text**:
```
Training: 39 steps, loss=0.2848, token_accuracy=98.1%
Peak GPU Memory: 16.21 GB
LoRA: r=64, alpha=128, FP16
Training time: 615 seconds (~10 min)
```

---

## Part 4: Evaluation Quality (1:45 – 2:30)

**Screen**: Terminal — run evaluation

**Action**: Run the evaluation script

```bash
cd /persistent/radeon-repo/track2-agentic-ai
/opt/venv/bin/python scripts/eval_cn_market_v2.py \
  --vllm-url http://127.0.0.1:8000/v1 \
  --model models/qwen-trader-merged \
  --output /persistent/track2/eval/cn_market_eval_final.json
```

**Narration**:
> 24 evaluation cases cover 4 ETF instruments and 6 strategy types.
> Before enhancement, the pass rate was only 45.83%, mainly due to JSON
> format errors and non-compliant lot sizes. After enhancing the JSON
> extractor and CN market canonicalizer, the pass rate reached 100%.

**On-screen text**:
```
Before: 11/24 passed (45.83%)
After:  24/24 passed (100%)
Checks: json_valid, instrument_match, timeframe_match,
        short_disabled, constraints_valid, numeric_types_valid
```

---

## Part 5: Dify Workflow Demo (2:30 – 3:45)

**Screen 1**: Dify workflow editor

**Action**: Show the 6-node workflow:
1. User Input → 2. RAG Knowledge → 3. LLM → 4. Code Execution → 5. Backtest → 6. Answer

**Narration**:
> The Dify workflow implements the complete agent pipeline. The RAG node
> retrieves domestic market rules. The LLM node uses the AMD GPU model to
> generate a strategy DSL. The code node canonicalizes and validates it.
> The backtest node executes a simulated trade, and the final node returns
> a PASS or REJECT risk report.

**Screen 2**: Terminal — test 3 strategy types

**Action**: Run 3 curl requests testing the backtest API

**Test 1: EMA Trend Strategy (510300.SH)**
```bash
curl -s -X POST http://127.0.0.1:8080/api/cn/backtest/report \
  -H "Content-Type: application/json" \
  -d '{"strategy":{"name":"CN_EMA_Trend",...}}'
```

**Narration**:
> CSI 300 ETF daily EMA20/50 crossover strategy — the system returns PASS,
> with a simulated return of -2.61% and maximum drawdown of -3.28%.

**Test 2: RSI Mean Reversion (510500.SH)**
```
Instrument: 510500.SH | 30m | RSI(14)
Result: -0.49% | Max drawdown -1.27% | REVIEW
```

**Test 3: ADX+EMA Strategy (159915.SZ)**
```
Instrument: 159915.SZ | 1d | EMA20/50 + ADX(14)>25
Result: +6.07% | Max drawdown -2.93% | PASS
```

**On-screen text**:
```
All backtests use deterministic synthetic historical data
(for system demonstration only, not investment advice)
Market constraints: T+1, 100-share lot size, no short selling,
10% price limits, commission, stamp duty
```

---

## Part 6: Architecture Summary (3:45 – 4:30)

**Screen**: Architecture diagram

```
┌─────────────────────────────────────────────────────────┐
│                    AMD GPU (gfx1100 / ROCm 7.2.1)        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  vLLM (port 8000)                                  │  │
│  │  Model: qwen-trader-cn-merged (Qwen2.5-7B + LoRA) │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          │                               │
│  ┌───────────────────────▼───────────────────────────┐  │
│  │  FastAPI (port 8080)                               │  │
│  │  /api/knowledge  → RAG domestic market rules       │  │
│  │  /api/cn/backtest/report → simulated backtest      │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          │                               │
│  ┌───────────────────────▼───────────────────────────┐  │
│  │  Dify Workflow                                     │  │
│  │  Input → RAG → LLM → Code → Backtest → Answer      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Narration**:
> The entire system runs on AMD GPU: vLLM inference, LoRA training, RAG
> retrieval, and the backtest engine all run on the same AMD server.
> The Dify workflow orchestrates all components into a complete agent
> pipeline — from Chinese natural language input to risk decision.

---

## Part 7: Disclaimers (4:30 – 5:00)

**Screen**: Disclaimer text

**Narration**:
> All backtests use deterministic synthetic historical data for system
> demonstration only. No real trading or live market operations were performed.
> The system does not involve any cryptocurrency content. Results do not
> constitute investment advice.

**On-screen text**:
```
IMPORTANT DISCLAIMERS
- Market data: Deterministic synthetic historical data (demo only)
- Trading mode: Simulated / Paper Trading
- Investment advice: Results do not constitute investment advice
- Market scope: Chinese domestic securities market only (A-share ETFs)
- Hardware: AMD GPU (gfx1100) + ROCm 7.2.1
```
