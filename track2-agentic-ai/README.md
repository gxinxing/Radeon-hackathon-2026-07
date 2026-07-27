# 🤖 Crypto Trading Agent on AMD Radeon GPU

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%207.2-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![vLLM](https://img.shields.io/badge/vLLM-ROCm-blue)](https://docs.vllm.ai/)
[![Qwen2.5](https://img.shields.io/badge/Qwen2.5-7B-6E49C8?logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Freqtrade](https://img.shields.io/badge/Freqtrade-Crypto%20Trading-green)](https://www.freqtrade.io/)
[![Gradio](https://img.shields.io/badge/Gradio-Chat%20UI-FF7300)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Hackathon-lightgrey)](#license)

> Fine-tune Qwen2.5-7B on AMD Radeon GPU to act as an experienced crypto trader.
> Natural language → strategy DSL → backtest → paper trading — a full-chain AI trading agent
> powered entirely by AMD ROCm.

**AMD AI DevMaster Hackathon 2026 — Track 2: Agentic AI**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Pipeline: NL → DSL → Backtest → Trade](#-pipeline-nl--dsl--backtest--trade)
- [Key Components](#-key-components)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Fine-tuning Pipeline](#-fine-tuning-pipeline)
- [DSL Specification](#-dsl-specification)
- [Chat UI](#-chat-ui)
- [Judging Criteria Alignment](#-judging-criteria-alignment)
- [License](#-license)

---

## 🌟 Overview

This project builds a **crypto trading agent** that reasons like an experienced human trader.
Users describe strategies in natural language; the agent generates a formal strategy DSL,
backtests it on historical crypto data, evaluates risk, and optionally executes paper trades.

**Key differentiators:**
- **LLM fine-tuned on trading knowledge** — Qwen2.5-7B with QLoRA on AMD ROCm GPU, trained on
  FNSPID financial news + FinGPT instructions + custom NL→DSL pairs
- **Full-chain automation** — NL input → DSL generation → schema validation → backtest →
  risk assessment → paper trading → natural language report
- **100% AMD GPU** — LLM inference via vLLM on ROCm, no NVIDIA dependency
- **Dify-powered agent** — Workflow with LLM nodes, code validation, and tool calls

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Dify Chat UI                              │
│                                                                   │
│  "BTC放量突破前高，帮我做一个突破策略，止损3%"                     │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                   Dify Workflow Engine                              │
│                                                                    │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ LLM:    │──▶│ Code:    │──▶│ HTTP:    │──▶│ LLM:            │  │
│  │ NL→DSL  │   │ Schema   │   │ Backtest │   │ Report Gen      │  │
│  │ (vLLM)  │   │ Validate │   │ API      │   │ (vLLM)          │  │
│  └─────────┘   └──────────┘   └──────────┘   └────────────────┘  │
│       │                                               │           │
│       │    ┌─────────────────────────────────────────┘           │
│       ▼    ▼                                                      │
│  ┌──────────────┐   ┌─────────────────────────────────────────┐ │
│  │ Knowledge    │   │ Tools:                                  │ │
│  │ Base (RAG)    │   │ • get_market_data (CCXT/Binance)        │ │
│  │ 交易经验规则   │   │ • calculate_indicators (TA-Lib)         │ │
│  │ 技术分析知识   │   │ • paper_trade (Binance Testnet)         │ │
│  └──────────────┘   └─────────────────────────────────────────┘ │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                AMD ROCm GPU (51 GB VRAM)                           │
│                                                                    │
│  ┌────────────────────────┐    ┌────────────────────────────────┐ │
│  │ vLLM (ROCm V1 Engine)  │    │ QLoRA Fine-tuning              │ │
│  │ Qwen2.5-7B (merged)    │    │ • FNSPID + FinGPT + NL→DSL    │ │
│  │ ~14 GB VRAM            │    │ • PEFT merge → vLLM serve      │ │
│  │ ~56 t/s (FP16)         │    │ • ~20 GB VRAM (training)       │ │
│  └────────────────────────┘    └────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Freqtrade Backtest Engine                                    │ │
│  │ • CCXT data (Binance OHLCV) • DSL→IStrategy transpiler      │ │
│  │ • Hyperopt optimization     • FreqAI ML signals (optional)  │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Pipeline: NL → DSL → Backtest → Trade

```
① NL Input          "BTC突破前高，放量确认，做个突破策略"
    │
    ▼
② DSL Generation    LLM generates YAML strategy spec (via vLLM on ROCm)
    │
    ▼
③ Schema Validation JSON Schema validates DSL structure & semantics
    │
    ▼
④ Backtest          DSL → Freqtrade IStrategy → historical backtest
    │
    ▼
⑤ Risk Assessment   LLM evaluates: max drawdown, Sharpe, win rate, risk
    │
    ▼
⑥ Report            LLM generates natural language analysis + recommendation
    │
    ▼
⑦ Paper Trade       (Optional) Deploy to Binance Testnet for live simulation
```

---

## 🧩 Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Base model | Qwen2.5-7B-Instruct | Chinese-capable LLM, ROCm-friendly |
| Fine-tuning | QLoRA (4-bit) via PEFT | Inject trading knowledge without full retrain |
| Inference | vLLM (ROCm, V1 engine) | High-throughput local LLM serving |
| Agent framework | Dify (Docker Compose) | Workflow orchestration, chat UI, tool management |
| Strategy DSL | YAML + JSON Schema | LLM-friendly intermediate representation |
| Backtest engine | Freqtrade + CCXT | Crypto-native, event-driven backtesting |
| Technical indicators | TA-Lib | EMA, RSI, MACD, ATR, Bollinger Bands |
| Paper trading | Binance Testnet API | Risk-free live simulation |
| Data sources | FNSPID, FinGPT, CCXT | Training data + market data |

---

## 🔧 Prerequisites

### Hardware

- AMD Radeon GPU (e.g., RX 7900 XTX, MI250) with ROCm 6.2+
- Minimum 24 GB VRAM recommended (7B model + vLLM + Freqtrade)

### Cloud Environment

This project runs on the same **Anrui Cloud** (安睿云) AMD GPU instance as Track 3:

- JupyterLab terminal access
- VNC via noVNC on port 6080
- Python virtual environment at `/opt/venv/`
- ROCm 6.2 + PyTorch 2.9.1 already installed

---

## 🚀 Quick Start

### One-command setup

```bash
cd track2-agentic-ai
bash scripts/setup.sh
```

This script:
1. Installs vLLM (ROCm), Freqtrade, Dify dependencies
2. Downloads Qwen2.5-7B-Instruct model
3. Prepares fine-tuning datasets
4. Starts vLLM inference server on port 8000
5. Starts backtest microservice on port 8080
6. Starts Dify on port 3000

### End-to-end verification

```bash
bash scripts/verify_e2e.sh
```

Runs a full pipeline test: NL input → DSL → backtest → report.

---

## 📁 Project Structure

```
track2-agentic-ai/
├── src/
│   ├── dsl/
│   │   ├── schema.json              # JSON Schema for strategy DSL
│   │   ├── validator.py             # DSL schema validator
│   │   └── transpiler.py            # DSL → Freqtrade IStrategy transpiler
│   ├── backtest/
│   │   ├── server.py                # FastAPI backtest microservice
│   │   ├── runner.py                # Freqtrade backtest runner
│   │   └── data_fetcher.py          # CCXT historical data fetcher
│   ├── llm/
│   │   ├── inference.py             # vLLM client wrapper
│   │   └── prompts.py               # System prompts for NL→DSL, report gen
│   └── tools/
│       ├── market_data.py           # Market data tool (CCXT wrapper)
│       ├── indicators.py            # Technical indicator calculator
│       └── paper_trade.py           # Binance Testnet paper trading
├── training/
│   ├── data/
│   │   ├── prepare_fnspid.py       # FNSPID dataset processor
│   │   ├── prepare_fingpt.py       # FinGPT dataset processor
│   │   ├── prepare_dsl_pairs.py    # NL→DSL instruction pair generator
│   │   └── merge_datasets.py       # Merge all datasets for training
│   ├── scripts/
│   │   ├── train_qlora.py           # QLoRA training script (ROCm)
│   │   ├── merge_lora.py            # Merge LoRA weights into base model
│   │   └── serve_vllm.sh            # Start vLLM serving script
│   └── configs/
│       ├── qlora_config.yaml        # QLoRA training hyperparameters
│       └── dataset_config.yaml      # Dataset mixing configuration
├── dify/
│   ├── workflows/
│   │   └── trading_agent.yml        # Dify workflow definition (export format)
│   └── tools/
│       ├── market_data_openapi.yml  # Swagger spec for market data tool
│       └── backtest_openapi.yml     # Swagger spec for backtest API
├── docker/
│   ├── Dockerfile.vllm              # vLLM ROCm Docker image
│   ├── Dockerfile.backtest          # Backtest microservice image
│   └── docker-compose.yml           # Full stack orchestration
├── docs/
│   └── technical_report.md          # Technical report for submission
├── tests/
│   ├── test_dsl_validator.py        # DSL schema validation tests
│   ├── test_transpiler.py           # DSL→Freqtrade transpilation tests
│   └── test_e2e.py                  # End-to-end pipeline tests
├── scripts/
│   ├── setup.sh                     # One-command environment setup
│   └── verify_e2e.sh                # End-to-end verification
├── models/                           # Fine-tuned model checkpoints
├── requirements.txt
└── README.md
```

---

## 🎓 Fine-tuning Pipeline

### Data Sources

| Dataset | Size | Purpose |
|---------|------|---------|
| FNSPID | ~100K samples | Financial news → price analysis capability |
| FinGPT | ~50K samples | Financial QA, sentiment analysis |
| NL→DSL pairs | ~2K samples | Natural language → strategy DSL generation |
| Trading rules | ~500 rules | System prompt / RAG knowledge base |

### QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-7B-Instruct |
| Quantization | 4-bit (NF4) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size | 4 (with gradient accumulation 4) |
| Scheduler | cosine |
| Warmup ratio | 0.03 |

### Training Flow

```
1. Download datasets (FNSPID, FinGPT)
2. Process into unified instruction format
3. Generate NL→DSL pairs from templates
4. QLoRA fine-tune on AMD ROCm GPU
5. Merge LoRA weights into base model
6. Serve merged model via vLLM
```

---

## 📐 DSL Specification

The strategy DSL is a YAML-based intermediate representation:

```yaml
strategy:
  name: "BTC_EMA_Breakout_Volume"
  market:
    exchange: binance
    pair: BTC/USDT
    timeframe: 1h
  indicators:
    - { name: ema_fast, type: EMA, params: { period: 20, field: close } }
    - { name: ema_slow, type: EMA, params: { period: 50, field: close } }
    - { name: vol_ma, type: SMA, params: { period: 20, field: volume } }
    - { name: rsi, type: RSI, params: { period: 14 } }
  entry:
    long: "ema_fast > ema_slow AND volume > vol_ma * 1.5 AND rsi < 70"
    short: null
  exit:
    long: "ema_fast < ema_slow"
    short: null
  risk:
    stop_loss: -0.03
    trailing_stop: true
    trailing_stop_positive: 0.02
    max_open_trades: 3
    stake_amount: 0.1
```

The DSL is validated against a JSON Schema and then transpiled into a
Freqtrade `IStrategy` Python class for backtesting.

---

## 🔄 Dify Workflow

The Dify workflow chains multiple nodes:

1. **LLM Node (NL→DSL)**: User NL input + system prompt → strategy DSL (YAML)
2. **Code Node (Validate)**: JSON Schema validation of DSL
3. **HTTP Request Node (Backtest)**: POST to backtest microservice
4. **Code Node (Parse)**: Extract metrics from backtest response
5. **LLM Node (Report)**: Generate NL analysis of backtest results
6. **End**: Return report + metrics + recommendation

---

## 📊 Judging Criteria Alignment

| Criterion | Points | How We Address It |
|-----------|--------|-------------------|
| Functional completeness & application value | 60 | Full-chain: NL→DSL→Backtest→Paper Trade; real crypto market data; working chat UI |
| AMD Radeon GPU / ROCm optimization | 40 | vLLM inference on ROCm; QLoRA fine-tuning on ROCm; benchmark vs CPU inference |

---

## 📄 License

This project is submitted for the AMD AI DevMaster Hackathon.
