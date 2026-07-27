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

### Model Fine-tuning (QLoRA on ROCm)

- **Method**: 4-bit NF4 quantization + LoRA (r=64, alpha=128)
- **Framework**: PEFT + TRL (SFTTrainer) on ROCm PyTorch 2.9.1
- **Training Data**: 7,000 samples (2,000 NL→DSL pairs + 5,000 financial QA)
- **Precision**: bf16 (fp16 causes GradScaler errors with bitsandbytes on ROCm)
- **Attention**: AOTriton backend (ROCm-native SDPA)
- **Training Time**: ~2 hours for 3 epochs (234 steps × 31s/step)
- **Peak VRAM**: ~20GB during training
- **Post-training**: LoRA weights merged into base model via PEFT `merge_and_unload()`

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
- NL→DSL instruction pairs (2,000 synthetic examples) — strategy generation
- Financial QA (5,000 synthetic pairs) — trading knowledge and reasoning
- System prompts for DSL generation, report writing, and risk assessment

### 4.3 Full-Chain Autonomous Pipeline

The system implements a complete pipeline from idea to execution:
1. **NL Input** — User describes a strategy in natural language
2. **DSL Generation** — LLM generates structured YAML strategy spec
3. **Schema Validation** — JSON Schema validates structure and semantics
4. **Backtest** — Strategy is backtested on 90-180 days of historical data
5. **Risk Assessment** — Metrics computed: drawdown, Sharpe, win rate, profit factor
6. **Report** — Natural language analysis with recommendations
7. **Paper Trade** — Optional deployment to Binance Testnet (API available)

### 4.4 Synthetic Data Fallback

When exchange APIs are unreachable (network-restricted cloud instances),
the system generates realistic synthetic OHLCV data using geometric Brownian
motion with pair-specific parameters (BTC base price $65K, 70% annual volatility).
This ensures the full pipeline remains testable in any environment.

## 5. Datasets Used

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| NL→DSL pairs | Synthetic (template-based) | 2,000 samples | NL → strategy DSL generation |
| Financial QA | Synthetic (12 template types) | 5,000 samples | Trading knowledge, indicator explanations |
| Market data | CCXT (synthetic GBM fallback) | 90-180 days OHLCV | Backtesting |
| Model | HuggingFace (via hf-mirror.com) | Qwen2.5-7B-Instruct | Base LLM |

## 6. Final Deliverables

| Deliverable | Location | Status |
|-------------|----------|--------|
| Fine-tuned model | `models/qwen-trader-merged/` | ✅ Trained on AMD ROCm |
| DSL schema | `src/dsl/schema.json` | ✅ Complete |
| DSL validator | `src/dsl/validator.py` | ✅ 10 tests passing |
| DSL transpiler | `src/dsl/transpiler.py` | ✅ 10 tests passing |
| Backtest microservice | `src/backtest/server.py` | ✅ Running on :8080 |
| Backtest runner | `src/backtest/runner.py` | ✅ Full trade simulation |
| Market data tools | `src/tools/market_data.py` | ✅ With synthetic fallback |
| Indicator calculator | `src/tools/indicators.py` | ✅ TA-Lib integration |
| Paper trading | `src/tools/paper_trade.py` | ✅ Binance Testnet |
| LLM prompts | `src/llm/prompts.py` | ✅ 3 specialized prompts |
| Chat UI | `src/chat_app.py` | ✅ Gradio on :7860 |
| QLoRA training script | `training/scripts/train_qlora.py` | ✅ ROCm-optimized |
| LoRA merge script | `training/scripts/merge_lora.py` | ✅ PEFT merge |
| vLLM serving script | `training/scripts/serve_vllm.sh` | ✅ ROCm env configured |
| Deploy script | `training/scripts/deploy.sh` | ✅ One-command deploy |
| Setup script | `scripts/setup.sh` | ✅ Complete |
| E2E verification | `scripts/verify_e2e.sh` | ✅ 6 checks |
| Unit tests | `tests/` (3 files) | ✅ 22 tests passing |
| Technical report | `docs/technical_report.md` | ✅ This document |

## 7. Team

- Team Name: [TODO: fill before submission]
- Members: [TODO: fill before submission]
