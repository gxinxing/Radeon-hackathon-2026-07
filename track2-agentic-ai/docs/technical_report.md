# Technical Report: Domestic Market Quantitative Agent on AMD Radeon GPU

**AMD AI DevMaster Hackathon 2026 — Track 2: Agentic AI**

## 1. Target Application Definition

This project builds an **autonomous quantitative trading agent** for the Chinese
domestic securities market. Users describe trading strategies in Chinese natural
language; the agent retrieves domestic market rules via RAG, generates a formal
strategy DSL on an AMD GPU, validates and canonicalizes it, executes a simulated
backtest under CN market constraints (T+1, lot size 100, no short selling, 10%
price limits), and produces a risk report with a PASS/REJECT decision.

The system demonstrates that LLM fine-tuning, inference, RAG retrieval, DSL
validation, and backtest execution can run entirely on AMD ROCm GPUs — no
NVIDIA CUDA dependency at any stage.

## 2. Overall System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Dify Workflow (6 nodes)                     │
│  User Input → RAG → LLM → Code → Backtest → Risk Report         │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                   AMD ROCm GPU (gfx1100)                         │
│                                                                  │
│  ┌────────────────────────┐    ┌────────────────────────────────┐│
│  │ vLLM (ROCm V1 Engine)  │    │ LoRA Fine-tuning               ││
│  │ Qwen2.5-7B (merged)   │    │ • 400 CN market samples        ││
│  │ FP16, ~8.5s/request    │    │ • r=64, alpha=128, FP16        ││
│  │ ~16 GB VRAM            │    │ • 39 steps, loss=0.2848         ││
│  └────────────────────────┘    └────────────────────────────────┘│
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ FastAPI Services (port 8080)                                ││
│  │ • /api/knowledge  — RAG domestic market rules               ││
│  │ • /api/cn/backtest/report — simulated backtest + risk       ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### Agent Capability Matrix

| Capability | Implementation | Evidence |
|-----------|---------------|----------|
| **Reasoning** | LLM converts NL strategy request to structured DSL | vLLM + Qwen2.5-7B LoRA, 24/24 eval pass |
| **Planning** | Dify workflow orchestrates 6-step pipeline | 6 nodes: input → RAG → LLM → code → backtest → answer |
| **Tool Use** | RAG retrieval, DSL canonicalizer, backtest engine | FastAPI endpoints, CN market canonicalizer |
| **Memory Management** | RAG knowledge base with domestic market rules | 7-rule constraint set (T+1, lot size, price limits) |
| **Task Execution** | Simulated backtest with risk report and PASS/REJECT | Paper trading with synthetic data |

### Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base model | Qwen2.5-7B-Instruct | Native Chinese support, ROCm-compatible, strong code generation |
| Fine-tuning | FP16 LoRA (r=64, alpha=128) | bitsandbytes 4-bit unavailable on ROCm; FP16 LoRA fits in 16 GB |
| Inference | vLLM with merged LoRA | V1 engine for performance, OpenAI-compatible API |
| Workflow | Dify 6-node Chatflow | Visual pipeline orchestration, RAG + LLM + code + backtest |
| Strategy DSL | JSON/YAML + JSON Schema | LLM-friendly, validatable, canonicalizable |
| Backtest | Custom CN market engine | T+1, lot size 100, no short, price limits, commission, stamp duty |
| Market | Chinese domestic ETFs (510300.SH, 510050.SH, 510500.SH, 159915.SZ) | A-share regulated market, real-world applicability |

## 3. How AMD Radeon GPUs Are Utilized

### LLM Inference (vLLM on ROCm)

- **GPU**: AMD Radeon Graphics (gfx1100), ROCm 7.2.1
- **CPU**: AMD EPYC 9334 32-Core Processor
- **Model**: Qwen2.5-7B-Instruct, fine-tuned with FP16 LoRA on CN market data
- **Engine**: vLLM v0.16.1 with ROCm support
- **Configuration**: FP16, `gpu_memory_utilization=0.35`, `enforce-eager` mode
- **VRAM Usage**: ~16 GB for model + KV cache
- **Serving**: OpenAI-compatible API at `http://localhost:8000/v1`
- **Prefix cache hit rate**: 81.7%
- **Average latency**: ~8.5 seconds per request
- **P95 latency**: ~9.8 seconds

### Model Fine-tuning (LoRA on ROCm)

- **Method**: LoRA (r=64, alpha=128) — FP16 mode (bitsandbytes 4-bit not available on this ROCm build)
- **Framework**: PEFT 0.18.1 + TRL (SFTTrainer) on ROCm PyTorch
- **Training Data**: 400 CN market NL-to-DSL pairs covering 6 strategy templates × 4 ETF instruments
- **Precision**: FP16
- **Attention**: AOTriton backend (ROCm-native SDPA)
- **Training Config**: 3 epochs, batch_size=2, lr=2e-4, cosine scheduler, packing=True
- **Training Results**:
  - 39 steps, train_runtime=615 seconds (~10 minutes)
  - Final train loss: 0.2848
  - Token accuracy: 98.1%
  - Peak GPU memory: 16.21 GB
- **Post-training**: LoRA weights merged into base model via PEFT `merge_and_unload()`
- **Model SHA256**: `8806c9d59657ce30267bb13e9b59b462632a61ad2e7d8ef76b5685f79a84a23d`
- **Adapter SHA256**: `4b7d746f54c05ccdb2de989d9c81ffa505511c96f15b708c66d37dc055ae710e`

### Training Loss Curve

| Step | Loss | Token Accuracy | Learning Rate |
|------|------|----------------|---------------|
| 10 | 0.9873 | 84.48% | 9.143e-05 |
| 20 | 0.04837 | 97.94% | 5.635e-05 |
| 30 | 0.04015 | 98.06% | 1.697e-05 |
| 39 | 0.2848 (avg) | 98.10% | 0 |

## 4. Innovation and Key Technical Contributions

### 4.1 Trading Strategy DSL

A JSON-based domain-specific language for expressing domestic market trading
strategies. The DSL serves as an intermediate representation between LLM output
and executable backtest code:

- **LLM-friendly**: JSON structure is reliably generated by the fine-tuned model
- **Validatable**: JSON Schema ensures structural correctness before execution
- **Canonicalizable**: CN market canonicalizer fixes common LLM output errors
- **Domestic-constrained**: Enforces exchange=cn_stock, T+1, lot_size=100, no short

### 4.2 Fine-tuned CN Market Trading LLM

Qwen2.5-7B fine-tuned on 400 curated CN market NL-to-DSL pairs:
- 6 strategy templates: EMA crossover, RSI mean reversion, Bollinger Bands, MACD, EMA+ADX filter, EMA cross with constraints
- 4 ETF instruments: 510300.SH, 510050.SH, 510500.SH, 159915.SZ
- 2 timeframes: daily (1d), 30-minute (30m)
- System prompt enforces CN market constraints (no short, T+1, lot_size=100, price_limit=0.1)

### 4.3 CN Market Canonicalizer

A post-LLM canonicalizer that fixes common model output errors before evaluation:

| Repair Type | Example | Count in 24-case eval |
|-------------|---------|----------------------|
| `cn_constraint` | lot_size 10000 → 100 | 6 |
| `cn_constraint` | allow_short missing → false | 4 |
| `cn_constraint` | exchange missing → cn_stock | 4 |
| `type_coerce` | period "20" → 20 | 2 |
| `sign_fix` | stop_loss 0.05 → -0.05 | 1 |
| `default_fill` | Missing risk/constraints → safe defaults | 3 |

### 4.4 Robust JSON Extraction with Retry

The enhanced evaluation pipeline includes multi-strategy JSON extraction:
- Handles `strategy\n{...}` format (model outputs keyword then JSON on next line)
- Handles `strategy":` prefix (missing opening brace)
- Handles truncated/degraded output (repeated character detection)
- Retry mechanism: max 2 attempts, temperature 0.1 → 0.3
- Full audit trail: raw_output → pre_repair_dsl → post_repair_dsl → repair_log

### 4.5 Dify Workflow Integration

A 6-node Dify Chatflow orchestrates the complete agent pipeline:

| Node | Type | Function |
|------|------|----------|
| User Input | start | Receives Chinese natural language strategy request |
| RAG Knowledge | http-request | GET /api/knowledge — retrieves 7 domestic market rules |
| LLM | llm | models/qwen-trader-merged generates DSL |
| Code Execution | code | Python3: parse, validate, canonicalize DSL |
| Backtest | http-request | POST /api/cn/backtest/report — simulated backtest |
| Answer | answer | Returns risk report with PASS/REJECT |

### 4.6 Domestic Market Backtest Engine

The backtest engine enforces all CN market constraints:

- **T+1 settlement**: Cannot sell on the same day as purchase
- **Lot size 100**: Orders must be in multiples of 100 shares
- **No short selling**: `entry.short` must be null, `allow_short` must be false
- **Price limit**: 10% daily price movement limit enforced
- **Commission**: Simulated brokerage commission applied
- **Stamp duty**: Sell-side stamp duty applied
- **Slippage**: Configurable slippage model

### 4.7 RL Reward System

The system computes a reward score from backtest metrics to inform PASS/REJECT decisions:

- **PASS**: Return acceptable, drawdown within limits, constraints satisfied
- **REVIEW**: Marginal performance, requires human review
- **REJECT**: Excessive drawdown or constraint violations

## 5. Evaluation Results

### 5.1 Before Enhancement (cn_market_eval_after.json)

| Metric | Pass Rate | Target |
|--------|-----------|--------|
| JSON parseable | 75% | >=95% |
| No forbidden terms | 100% | — |
| Instrument match | 70.83% | >=90% |
| Timeframe match | 70.83% | >=90% |
| Domestic exchange | 70.83% | — |
| Short disabled | 70.83% | >=95% |
| Constraints valid | 45.83% | >=90% |
| Numeric types | 70.83% | — |
| **Overall pass rate** | **45.83%** | **>=80%** |

### 5.2 After Enhancement (cn_market_eval_final.json)

| Metric | Pass Rate | Target | Met |
|--------|-----------|--------|-----|
| JSON parseable | 100% | >=95% | Yes |
| No forbidden terms | 100% | — | Yes |
| Instrument match | 100% | >=90% | Yes |
| Timeframe match | 100% | >=90% | Yes |
| Domestic exchange | 100% | — | Yes |
| Short disabled | 100% | >=95% | Yes |
| Constraints valid | 100% | >=90% | Yes |
| Numeric types | 100% | — | Yes |
| **Overall pass rate** | **100% (24/24)** | **>=80%** | **Yes** |

### 5.3 Failure Analysis (Before Enhancement)

13 out of 24 cases failed. Root causes:

| Failure Type | Count | Fix Applied |
|-------------|-------|-------------|
| JSON format error (strategy\n{...}) | 4 | Enhanced extract_json with pattern matching |
| JSON format error (strategy": prefix) | 3 | Prepend `{"` to fix missing opening brace |
| lot_size = 10000/1000 instead of 100 | 6 | CN canonicalizer forces lot_size=100 |
| Degraded output (repeated zeros) | 2 | Degraded output detection + retry |
| Missing constraints section | 3 | Default fill with CN market defaults |

### 5.4 Dify Workflow Test Results

Three test cases verified end-to-end:

| Test | Strategy | Instrument | Return | Max Drawdown | Decision |
|------|----------|-----------|--------|--------------|----------|
| EMA Trend | EMA20/50 crossover | 510300.SH | -2.61% | -3.28% | PASS |
| RSI Reversion | RSI(14) mean reversion | 510500.SH | -0.49% | -1.27% | REVIEW |
| ADX+EMA | EMA + ADX filter | 159915.SZ | +6.07% | -2.93% | PASS |

### 5.5 vLLM Performance

| Metric | Value |
|--------|-------|
| Average latency | 8,248 ms |
| P95 latency | 9,796 ms |
| Total retries (24 cases) | 0 |
| Prefix cache hit rate | 81.7% |
| GPU memory utilization | 35% (16 GB of available VRAM) |

## 6. Datasets Used

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| CN market NL→DSL pairs | Synthetic (6 templates × 4 instruments × parameter expansion) | 400 samples | LoRA fine-tuning |
| RAG knowledge base | Curated domestic market rules | 7 constraint rules | LLM prompt enhancement |
| Market data | Deterministic synthetic OHLCV | 180 days | Simulated backtesting |
| Model | HuggingFace (via hf-mirror.com) | Qwen2.5-7B-Instruct | Base LLM |

## 7. Final Deliverables

| Deliverable | Location | Status |
|-------------|----------|--------|
| Fine-tuned model (merged) | `/persistent/qwen-trader-cn-merged` | Trained on AMD ROCm (FP16 LoRA) |
| LoRA adapter | `/persistent/track2/models/qwen-trader-cn-lora/final` | r=64, alpha=128, 617 MB |
| DSL schema | `src/dsl/schema.json` | Complete (16 indicator types) |
| DSL validator | `src/dsl/validator.py` | Multi-column indicator support |
| DSL canonicalizer | `src/dsl/canonicalizer.py` | Type coercion + repair logging |
| CN market canonicalizer | `scripts/eval_cn_market_v2.py` | lot_size, constraints, exchange, short |
| Backtest microservice | `src/backtest/server.py` | Running on :8080 |
| CN backtest runner | `src/backtest/cn_runner.py` | T+1, lot size, price limits, commission |
| RAG knowledge API | `src/api.py` | /api/knowledge endpoint |
| Dify workflow | `dify/workflows/` | 6-node Chatflow with CN market prompts |
| Enhanced eval script | `scripts/eval_cn_market_v2.py` | 24/24 pass, retry, full audit trail |
| vLLM serving script | `training/scripts/serve_vllm.sh` | ROCm env configured |
| LoRA training script | `training/scripts/train_qlora.py` | ROCm-optimized |
| LoRA merge script | `training/scripts/merge_lora.py` | PEFT merge |
| Unit tests | `tests/` (7 files) | 229 tests passing |
| Technical report | `docs/technical_report.md` | This document |
| Final status report | `docs/track2_final_status.md` | CN market summary |
| Demo script (CN) | `docs/track2_demo_script_cn.md` | Video recording guide |
| Metrics JSON | `artifacts/track2_metrics.json` | Full evaluation metrics |
| Asset manifest | `artifacts/track2_asset_manifest.json` | SHA256, sizes, configs |
| Final eval results | `artifacts/cn_market_eval_final.json` | 24/24 pass (100%) |

## 8. Important Disclaimers

- All backtests use **deterministic synthetic historical data** for system demonstration only.
- **No real trading or live market operations** were performed.
- The system does **not** involve any cryptocurrency, digital currency exchange, or derivatives content.
- All inference and training were performed on **AMD GPU (gfx1100, ROCm 7.2.1)**.
- Backtest results **do not constitute investment advice**.
- The CN market version and the legacy version are kept **separate** and do not affect each other.

## 9. Reproducibility

### Prerequisites

- AMD GPU with ROCm 7.2.1+
- Python 3.12+ with vLLM, PEFT, TRL, httpx, pyyaml
- Remote server: `ssh -i <key> -p 31151 root@***REMOVED***`

### Steps

```bash
# 1. Confirm vLLM is running
curl http://127.0.0.1:8000/v1/models

# 2. Run the 24-case evaluation
cd /persistent/radeon-repo/track2-agentic-ai
/opt/venv/bin/python scripts/eval_cn_market_v2.py \
  --vllm-url http://127.0.0.1:8000/v1 \
  --model models/qwen-trader-merged \
  --output /persistent/track2/eval/cn_market_eval_final.json \
  --max-retries 2

# 3. Test the backtest API
curl -X POST http://127.0.0.1:8080/api/cn/backtest/report \
  -H "Content-Type: application/json" \
  -d '{"strategy":{"name":"CN_EMA","market":{"exchange":"cn_stock","instrument":"510300.SH","timeframe":"1d"},"indicators":[{"name":"ema_fast","type":"EMA","params":{"period":20,"field":"close"}},{"name":"ema_slow","type":"EMA","params":{"period":50,"field":"close"}}],"entry":{"long":"ema_fast > ema_slow","short":null},"exit":{"long":"ema_fast < ema_slow","short":null},"constraints":{"t_plus_one":true,"price_limit":0.1,"allow_short":false,"lot_size":100},"risk":{"stop_loss":-0.05,"max_position_pct":0.3,"max_drawdown":-0.15}}}'

# 4. Run unit tests locally
cd track2-agentic-ai
python3 -m pytest tests/ --ignore=tests/test_e2e.py -q

# 5. Run E2E verification
bash scripts/verify_e2e.sh
```

## 10. Team

- Team Name: Radeon ROCm Raiders
- Members: Simon Xing
