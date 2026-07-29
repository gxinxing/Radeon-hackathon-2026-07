# Technical Report: Crypto Trading Agent on AMD Radeon GPU

**AMD AI DevMaster Hackathon 2026 — Track 2: Agentic AI**

## 1. Target Application Definition

This project builds an **autonomous crypto trading agent** that reasons like an
experienced human trader. Users describe trading strategies in natural language;
the agent generates a formal strategy DSL, backtests it on historical market data,
evaluates risk, and produces a natural language analysis report.

The system addresses a real-world problem: retail traders often have trading ideas
but lack the technical skills to formalize them into testable strategies. By
bridging natural language and executable trading logic, this agent democratizes
quantitative trading.

## 2. Overall System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Gradio Chat UI (port 7860)                   │
│  User natural language input → Agent response with analysis       │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                   Pipeline Orchestrator                            │
│                                                                    │
│  ① LLM: NL→DSL  →  ② Schema Validate  →  ③ HTTP: Backtest        │
│     (vLLM/ROCm)      (JSON Schema)        (FastAPI + CCXT)        │
│                                                                    │
│  ④ LLM: Report Gen  ←  ⑤ Parse Metrics  ←  ⑥ Risk Assessment    │
│     (vLLM/ROCm)         (Code)               (LLM + Rules)       │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                AMD ROCm GPU (51 GB VRAM)                           │
│                                                                    │
│  ┌────────────────────────┐    ┌────────────────────────────────┐│
│  │ vLLM (ROCm V1 Engine)  │    │ QLoRA Fine-tuning              ││
│  │ Qwen2.5-7B (merged)    │    │ • 7,000 training samples       ││
│  │ FP16, ~31s/step        │    │ • 3 epochs, bf16               ││
│  │ ~20 GB VRAM            │    │ • PEFT merge → vLLM serve      ││
│  └────────────────────────┘    └────────────────────────────────┘│
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ Backtest Engine (FastAPI + TA-Lib + CCXT)                    ││
│  │ • Synthetic OHLCV (GBM model) • DSL→IStrategy transpiler    ││
│  │ • Technical indicators        • Trade simulation             ││
│  └──────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base model | Qwen2.5-7B-Instruct | Native Chinese support, ROCm-compatible, strong code generation |
| Fine-tuning | QLoRA 4-bit (NF4) + bf16 | Fits 51GB VRAM, bitsandbytes ROCm support, bf16 avoids GradScaler issues |
| Inference | vLLM with merged LoRA | V1 engine for performance, avoids V0 LoRA fallback on ROCm |
| Chat UI | Gradio | Lightweight, no Docker Hub dependency, Python-native |
| Strategy DSL | YAML + JSON Schema | LLM-friendly, human-readable, validatable |
| Backtest | Custom engine + CCXT/TA-Lib | Crypto-native, synthetic data fallback for network-restricted environments |
| Market | Crypto (BTC/USDT etc.) | 24/7 trading, open API standard, universal appeal |

## 3. How AMD Radeon GPUs Are Utilized

### LLM Inference (vLLM on ROCm)

- **Model**: Qwen2.5-7B-Instruct, fine-tuned with QLoRA
- **Engine**: vLLM v0.16.1 with ROCm 7.2 support
- **Configuration**: FP16, `gpu_memory_utilization=0.50`, `enforce-eager` mode
- **Environment**: `ROCBLAS_USE_HIPBLASLT=1`, `VLLM_USE_TRITON_FLASH_ATTN=0`
- **VRAM Usage**: ~20GB for model + KV cache, leaving 31GB for Track 3 (Physical AI)
- **Serving**: OpenAI-compatible API at `http://localhost:8000/v1`

### Model Fine-tuning (LoRA on ROCm)

- **Method**: LoRA (r=64, alpha=128) — FP16 mode (bitsandbytes 4-bit not available on this ROCm build; MI210 has 64GB VRAM so FP16 LoRA fits comfortably)
- **Framework**: PEFT + TRL (SFTTrainer) on ROCm PyTorch 2.9.1
- **Training Data**: 2,000 NL→DSL pairs with Chain-of-Thought reasoning traces (6 strategy templates × parameter expansion)
- **Precision**: bf16 (fp16 causes GradScaler errors on ROCm)
- **Attention**: AOTriton backend (ROCm-native SDPA)
- **Training Config**: 3 epochs, batch_size=4, grad_accum=4, lr=2e-4, cosine scheduler, packing=True
- **Training Results**: 81 steps, final loss=0.1625, token accuracy=98.71%, peak GPU memory=16.14 GB, runtime≈82 min
- **Post-training**: LoRA weights merged into base model via PEFT `merge_and_unload()`
- **Model Path**: `/workspace/persistent/qwen-trader-merged/` (4 safetensors, ~15GB)
- **Adapter Path**: `/workspace/persistent/qwen-trader-lora/final/`

## 4. Innovation and Key Technical Contributions

### 4.1 Trading Strategy DSL

A novel YAML-based domain-specific language for expressing crypto trading
strategies. The DSL serves as an intermediate representation between LLM
output and executable code:

- **LLM-friendly**: YAML structure is reliably generated by fine-tuned LLMs
- **Validatable**: JSON Schema ensures structural correctness before execution
- **Transpilent**: Automatically converts to Freqtrade IStrategy Python classes
- **Human-readable**: Traders can review and modify the DSL before execution

### 4.2 Fine-tuned Trading LLM

Qwen2.5-7B fine-tuned on a curated dataset of:
- NL→DSL instruction pairs (2,000 samples) with Chain-of-Thought reasoning traces — strategy generation
- 6 strategy templates: EMA crossover, RSI mean reversion, Bollinger Bands, volume breakout, MACD, multi-indicator confluence
- System prompts enhanced with RAG knowledge base (20 entries covering indicators, strategies, risk management, market characteristics)

### 4.3 Full-Chain Autonomous Pipeline

The system implements a complete pipeline from idea to execution:
1. **Market Context Fetch** — Current price, 24h change, volume fetched and injected into LLM prompt
2. **NL Input** — User describes a strategy in natural language
3. **DSL Generation** — LLM generates structured YAML strategy spec using Chain-of-Thought reasoning
4. **Schema Validation** — JSON Schema validates structure and semantics
5. **Backtest** — Strategy is backtested on 90-180 days of historical data with multi-position management and slippage model
6. **Risk Assessment** — Metrics computed: Sharpe, Sortino, Calmar, max drawdown, max consecutive losses, alpha vs benchmark
7. **Report** — Natural language analysis with benchmark comparison and specific recommendations
8. **Paper Trade** — Optional deployment to Binance Testnet (API available)

### 4.4 Realistic Market Simulation

The synthetic data generator replaces naive geometric Brownian motion with:
- **GARCH(1,1) volatility clustering** — High volatility periods cluster, mimicking real market dynamics
- **Student-t distribution** — Fat tails produce realistic extreme events (flash crashes, pumps)
- **Markov regime switching** — Bull/bear/sideways market states with transition probabilities
- **Volume-price correlation** — Volume spikes on large price moves, higher in volatile regimes

### 4.5 Walk-Forward Analysis (Overfitting Detection)

The system includes a walk-forward analysis endpoint (`/api/walkforward`) that splits
historical data into in-sample (70%) and out-of-sample (30%) segments, runs the strategy
on both, and compares performance:

- **Overfitting Score**: In-sample return minus out-of-sample return. High positive values
  indicate the strategy fits historical noise rather than capturing a persistent edge.
- **Robustness Assessment**: A strategy is flagged as robust only if out-of-sample Sharpe > 0,
  return > 0, and max drawdown < 30%.
- **Visual Output**: The Gradio UI displays a comparison table with IS/OOS metrics side by side.

### 4.6 Professional-Grade Backtest Metrics

The backtest engine computes a comprehensive metric suite comparable to professional quantitative platforms:

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return (annualized, timeframe-aware) |
| Sortino Ratio | Downside-risk-adjusted return |
| Calmar Ratio | Annualized return / max drawdown |
| Max Consecutive Losses | Psychological sustainability indicator |
| Alpha vs Buy&Hold | Strategy excess return over passive benchmark |
| Volatility (Annual) | Annualized standard deviation of returns |
| Avg Trade Duration | Average holding period in candles |
| Profit Factor | Gross profit / gross loss |
| Win Rate | Percentage of profitable trades |

### 4.7 NL→DSL Generation Quality Evaluation

The fine-tuned model was evaluated against 10 natural language prompts covering
8 strategy categories (trend following, mean reversion, momentum, breakout,
confluence, short, risk-based, filtered).

**Pipeline**: LLM output → YAML extraction → canonicalization → schema validation →
semantic validation → Freqtrade/Backtrader transpilation → indicator matching

**Schema-guided canonicalization** (`src/dsl/canonicalizer.py`) normalizes common
LLM output errors before validation:
- String→int/float coercion (`period: "50"` → `50`)
- Stop-loss sign correction (`3.0` → `-0.03`, interpreted as 3% loss)
- Illegal field stripping (e.g. `exit.buy` removed, only `long`/`short` allowed)
- Safe defaults for missing `risk` section

**LLM retry**: When canonicalization encounters unrecoverable errors (expression-based
stop_loss, missing `indicators`), the system sends a retry prompt with error feedback
to the LLM, requesting corrected output.

**Results** (10 test prompts, AMD MI210 GPU, vLLM inference):

| Metric | Rate |
|--------|------|
| YAML extraction | 10/10 (100%) |
| Schema validation | 9/10 (90%) |
| Semantic validation | 9/10 (90%) |
| Freqtrade transpilation | 9/10 (90%) |
| Backtrader transpilation | 9/10 (90%) |
| Indicator matching | 9/10 (90%) |
| **Overall pass rate** | **9/10 (90%)** |

**By category**: breakout 1/1, confluence 1/1, filtered 1/1, mean_reversion 2/2,
momentum 1/1, risk_based 1/1, short 1/1, trend_following 1/2

**Failed case** (`supertrend_simple`): Model placed `stop_loss` at the `strategy`
root level instead of inside the `risk` field. The system safely rejected this
output — it did not enter the backtest or trading execution pipeline.

**Safety boundary**: The system allows 9 strategies to proceed to backtesting
while safely rejecting 1 structurally non-compliant strategy. The canonicalizer
applies 2-4 repairs per test case on average (type coercion, default fills, illegal
field stripping). All repairs are logged for audit transparency.

## 5. Datasets Used

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| NL→DSL pairs | Synthetic (6 templates × parameter expansion) | 2,000 samples | NL → strategy DSL generation with CoT |
| Market data | CCXT (synthetic GARCH+Student-t fallback) | 90-180 days OHLCV | Backtesting |
| RAG knowledge base | Curated (20 entries) | Indicators, strategies, risk, market | LLM prompt enhancement |
| Model | HuggingFace (via hf-mirror.com) | Qwen2.5-7B-Instruct | Base LLM |

## 6. Final Deliverables

| Deliverable | Location | Status |
|-------------|----------|--------|
| Fine-tuned model | `models/qwen-trader-lora/` | ✅ Trained on AMD ROCm (FP16 LoRA) |
| DSL schema | `src/dsl/schema.json` | ✅ Complete (16 indicator types) |
| DSL validator | `src/dsl/validator.py` | ✅ Multi-column indicator support |
| DSL expr parser | `src/dsl/expr_parser.py` | ✅ AST-based safe evaluation |
| DSL Freqtrade transpiler | `src/dsl/transpiler.py` | ✅ All 16 indicators |
| DSL Backtrader transpiler | `src/dsl/transpiler_backtrader.py` | ✅ All 16 indicators (stock bt API) |
| DSL specification | `docs/dsl_specification.md` | ✅ Formal spec document |
| Backtest microservice | `src/backtest/server.py` | ✅ Running on :8080 |
| Backtest runner | `src/backtest/runner.py` | ✅ Multi-position, short, slippage, walk-forward |
| Market data tools | `src/tools/market_data.py` | ✅ With synthetic fallback |
| Indicator calculator | `src/tools/indicators.py` | ✅ TA-Lib integration (16 types) |
| Paper trading | `src/tools/paper_trade.py` | ✅ Binance Testnet |
| LLM prompts | `src/llm/prompts.py` | ✅ CoT + few-shot + market context |
| Chat UI | `src/chat_app.py` | ✅ Gradio + equity curve chart + walk-forward |
| QLoRA training script | `training/scripts/train_qlora.py` | ✅ Loads YAML config, ROCm-optimized |
| LoRA merge script | `training/scripts/merge_lora.py` | ✅ PEFT merge |
| vLLM serving script | `training/scripts/serve_vllm.sh` | ✅ ROCm env configured |
| Deploy script | `training/scripts/deploy.sh` | ✅ One-command deploy |
| Setup script | `scripts/setup.sh` | ✅ Complete |
| E2E verification | `scripts/verify_e2e.sh` | ✅ 7 checks |
| NL→DSL evaluation | `scripts/eval_nl_to_dsl.py` | ✅ 10 test prompts, online/offline modes |
| RAG knowledge base | `src/knowledge_base/` | ✅ 20 entries, keyword retrieval |
| Unit tests | `tests/` (6 files) | ✅ 83 tests passing |
| Technical report | `docs/technical_report.md` | ✅ This document |

## 7. Team

- Team Name: [TODO: fill before submission]
- Members: [TODO: fill before submission]
