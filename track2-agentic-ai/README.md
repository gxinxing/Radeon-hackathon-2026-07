# 🤖 Crypto Trading Agent on AMD Radeon GPU

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%207.2-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![vLLM](https://img.shields.io/badge/vLLM-ROCm-blue)](https://docs.vllm.ai/)
[![Qwen2.5](https://img.shields.io/badge/Qwen2.5-7B-6E49C8?logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
<<<<<<< HEAD
[![Freqtrade](https://img.shields.io/badge/Freqtrade-Crypto%20Trading-green)](https://www.freqtrade.io/)
[![Gradio](https://img.shields.io/badge/Gradio-Chat%20UI-FF7300)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Hackathon-lightgrey)](#license)

> Fine-tune Qwen2.5-7B on AMD Radeon GPU to act as an experienced crypto trader.
> Natural language → strategy DSL → backtest → paper trading — a full-chain AI trading agent
> powered entirely by AMD ROCm.
=======
[![Gradio](https://img.shields.io/badge/Gradio-Chat%20UI-FF7300)](https://gradio.app/)
[![Tests](https://img.shields.io/badge/Tests-224%20passed-brightgreen)](#-tests)
[![License](https://img.shields.io/badge/License-Hackathon-lightgrey)](#license)

[English](./README.md) | [中文](./README_zh.md)

> A full-chain agentic AI trading system on AMD ROCm: ReAct agent loop, multi-agent
> pipeline with risk veto, three-tier memory, RL reward optimization, multi-path RAG,
> and Gradio chat (Dify SETUP_GUIDE available) — all powered by Qwen2.5-7B fine-tuned on AMD Radeon GPU.
>>>>>>> track3-honest

**AMD AI DevMaster Hackathon 2026 — Track 2: Agentic AI**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
<<<<<<< HEAD
- [Pipeline: NL → DSL → Backtest → Trade](#-pipeline-nl--dsl--backtest--trade)
=======
- [Pipeline](#-pipeline)
- [Agent Capabilities](#-agent-capabilities)
>>>>>>> track3-honest
- [Key Components](#-key-components)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Fine-tuning Pipeline](#-fine-tuning-pipeline)
- [DSL Specification](#-dsl-specification)
<<<<<<< HEAD
- [Chat UI](#-chat-ui)
- [Judging Criteria Alignment](#-judging-criteria-alignment)
- [License](#-license)
=======
- [Tests](#-tests)
- [Judging Criteria Alignment](#-judging-criteria-alignment)
- [License](#license)
>>>>>>> track3-honest

---

## 🌟 Overview

<<<<<<< HEAD
This project builds a **crypto trading agent** that reasons like an experienced human trader.
Users describe strategies in natural language; the agent generates a formal strategy DSL,
backtests it on historical crypto data, evaluates risk, and optionally executes paper trades.

**Key differentiators:**
- **LLM fine-tuned on trading knowledge** — Qwen2.5-7B with QLoRA on AMD ROCm GPU, trained on
  FNSPID financial news + FinGPT instructions + custom NL→DSL pairs
- **Full-chain automation** — NL input → DSL generation → schema validation → backtest →
  risk assessment → paper trading → natural language report
- **100% AMD GPU** — LLM inference via vLLM on ROCm, no NVIDIA dependency
- **Gradio-powered agent** — Chat UI with LLM nodes, code validation, and tool calls
=======
This project builds an **autonomous crypto trading agent** with five core AI agent
capabilities: reasoning, planning, tool use, memory management, and task execution.

Users describe trading strategies in natural language. The agent autonomously reasons
about market conditions, generates a strategy DSL, backtests it on historical data,
evaluates risk through a dedicated Risk Agent with veto power, and optionally executes
paper trades — all while learning from each backtest via an RL reward system.

**Key differentiators:**
- **ReAct Agent Loop** — LLM autonomously reasons (Thought), selects tools (Action),
  observes results, and iterates until the goal is met
- **Multi-Agent Pipeline** — Retrieval Agent → Reasoning Agent → Risk Agent (with
  veto power); LLM only produces trading intent, Risk Agent decides execution
- **Three-Tier Memory** — Working (RAM) + Episodic (session JSON) + Semantic
  (cross-session JSON) memory for persistent learning
- **RL Reward System** — 8-dimensional reward function [-1, +1] with three-layer
  feedback: immediate prompt injection, experience rules, DPO training pairs
- **Multi-Path RAG** — Keyword + BM25 + dense retrieval + CrossEncoder reranking
  with confidence gating (score < 0.45 → forced neutral, prevents hallucination)
- **Intent Routing + Personality** — Trading queries → full agent pipeline;
  general conversation → personality-driven response as "小R"
- **Gradio Chat (Dify SETUP_GUIDE available)** — Single HTTP call (`/api/agent/run`) runs the full pipeline;
  3-node Dify Chatflow instead of 12-node manual pipeline
- **100% AMD GPU** — LLM inference via vLLM on ROCm, LoRA + DPO fine-tuning on ROCm
>>>>>>> track3-honest

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
<<<<<<< HEAD
│                         Gradio Chat UI (port 7860)                   │
=======
│                      Gradio Chat UI (port 7860)                   │
>>>>>>> track3-honest
│                                                                   │
│  "BTC放量突破前高，帮我做一个突破策略，止损3%"                     │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
<<<<<<< HEAD
│                   Pipeline Orchestrator (Gradio)                     │
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
=======
│                   Intent Router (personality.py)                   │
│                                                                    │
│  Trading intent? ──Yes──▶ ReAct Agent Loop (core.py)              │
│       │                    ├─ Thought → Action → Observe ↻        │
│       │                    ├─ 8 tools, 3-tier memory, RL reward    │
│       │                    └─ Final Answer + metrics + trace       │
│       │                                                           │
│       └───No───▶ Personality Response (小R)                       │
│                         └─ Natural conversation with memory        │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│              Multi-Agent Pipeline (orchestrator.py)                │
│              (via /api/agent/run or MULTI_AGENT_MODE=true)         │
│                                                                    │
│  ┌─ Retrieval Agent ──────────────────────────────────────┐       │
│  │  Multi-path RAG: keyword + BM25 + dense + reranking    │       │
│  │  Confidence gating: score < 0.45 → has_valid_docs=false │       │
│  │  → If no valid docs: short-circuit to neutral           │       │
│  └────────────────────────┬───────────────────────────────┘       │
│                           │                                        │
│  ┌─ Reasoning Agent ──────▼───────────────────────────────┐       │
│  │  LoRA + RAG context → trading intent JSON               │       │
│  │  {view, confidence, position_ratio, stop_loss, reason}   │       │
│  │  Forced neutral when has_valid_docs=false               │       │
│  └────────────────────────┬───────────────────────────────┘       │
│                           │                                        │
│  ┌─ Risk Agent ────────────▼───────────────────────────────┐      │
│  │  Hard rule validation (code-level, NOT model-influenced) │      │
│  │  ✓ Position limits (≤30% total, ≤10% per asset)         │      │
│  │  ✓ Stop-loss distance (0.5%–15%)                        │      │
│  │  ✓ Confidence threshold (≥0.30)                          │      │
│  │  ✓ Reason completeness                                   │      │
│  │  ✗ VETO: allow_execute=false → no trade                  │      │
│  └──────────────────────────────────────────────────────────┘      │
>>>>>>> track3-honest
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                AMD ROCm GPU (51 GB VRAM)                           │
│                                                                    │
│  ┌────────────────────────┐    ┌────────────────────────────────┐ │
<<<<<<< HEAD
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
=======
│  │ vLLM (ROCm V1 Engine)  │    │ Fine-tuning                    │ │
│  │ Qwen2.5-7B (merged)    │    │ • LoRA: 2,000 NL→DSL pairs     │ │
│  │ FP16, ~32 t/s          │    │ • DPO: reward-ranked pairs     │ │
│  │ ~20 GB VRAM            │    │ • PEFT merge → vLLM serve      │ │
│  └────────────────────────┘    └────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Tool Services (FastAPI + CCXT + TA-Lib)                     │ │
│  │ • Market data • Backtest (multi-position, slippage)        │ │
│  │ • Walk-forward analysis • Paper trading (Binance Testnet)  │ │
│  │ • Knowledge RAG (keyword + BM25 + reranking)               │ │
│  │ • RL reward computation                                    │ │
>>>>>>> track3-honest
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

<<<<<<< HEAD
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

=======
## 🔗 Pipeline

### Mode 1: ReAct Agent Loop (default, `AGENT_MODE=true`)

```
① Intent Routing    Trading? → ReAct loop | General? → Personality response (小R)
    │
    ▼ (trading)
② Thought            LLM reasons about next step (visible to user in UI)
③ Action             LLM selects one of 8 tools to call
④ Observe            Tool result stored in 3-tier memory
    │  ↻ Repeat until Final Answer or max 8 iterations
    ▼
⑤ RL Reward          Backtest metrics → 8-dim reward [-1,+1], grade A+ to F
    │                Reward injected into next iteration's prompt
    ▼
⑥ Final Answer       Report + metrics table + DSL YAML + reasoning trace
```

### Mode 2: Multi-Agent Pipeline (`MULTI_AGENT_MODE=true` or `POST /api/agent/run`)

```
① Retrieval Agent    Multi-path RAG → confidence gate
    │                (has_valid_docs=false → short-circuit to neutral)
    ▼
② Reasoning Agent    LoRA + RAG → trading intent JSON
    │                (NOT an order — just an intent)
    ▼
③ Risk Agent         Hard rule validation → allow/reject (VETO POWER)
    │
    ▼
④ Final Decision     Only if Risk Agent approves
```

### Mode 3: Dify Chatflow (3 nodes)

```
Start → Tool(runMultiAgent) → End
```

Single HTTP call runs the full multi-agent pipeline.

---

## 🧠 Agent Capabilities

| Capability | Implementation | Code |
|-----------|---------------|------|
| **Reasoning** | ReAct loop — LLM outputs Thought before each Action | `src/agent/core.py` |
| **Planning** | LLM decides tool sequence dynamically (market→knowledge→generate→validate→backtest→walk-forward→answer) | `src/agent/prompts.py` |
| **Tool Use** | 8 registered tools with structured JSON dispatch | `src/agent/tools.py` |
| **Memory Management** | Three-tier: Working (RAM) + Episodic (session JSON) + Semantic (cross-session JSON) | `src/agent/memory.py` |
| **Task Execution** | Real backtests, walk-forward analysis, paper trading via FastAPI | `src/backtest/runner.py` |
| **RL Reward** | 8-dim reward [-1,+1], grades A+ to F, 3-layer feedback (prompt/rules/DPO) | `src/agent/reward.py` |
| **Risk Control** | Hard rule validation with veto power — LLM never has final say | `src/agent/risk_agent.py` |
| **Personality** | Intent routing (trading vs general) + 小R character with memory | `src/agent/personality.py` |

>>>>>>> track3-honest
---

## 🧩 Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
<<<<<<< HEAD
| Base model | Qwen2.5-7B-Instruct | Chinese-capable LLM, ROCm-friendly |
| Fine-tuning | QLoRA (4-bit) via PEFT | Inject trading knowledge without full retrain |
| Inference | vLLM (ROCm, V1 engine) | High-throughput local LLM serving |
| Agent framework | Gradio + httpx | Chat UI, LLM orchestration, tool calls |
| Strategy DSL | YAML + JSON Schema | LLM-friendly intermediate representation |
| Backtest engine | Freqtrade + CCXT | Crypto-native, event-driven backtesting |
| Technical indicators | TA-Lib | EMA, RSI, MACD, ATR, Bollinger Bands |
| Paper trading | Binance Testnet API | Risk-free live simulation |
| Data sources | FNSPID, FinGPT, CCXT | Training data + market data |
=======
| Agent architecture | ReAct loop + 3-agent pipeline | Reasoning, planning, tool use, memory, execution |
| Base model | Qwen2.5-7B-Instruct | Chinese-capable LLM, ROCm-friendly |
| Fine-tuning | LoRA (r=64) + DPO via PEFT/TRL | Trading knowledge + reward-based preference |
| Inference | vLLM (ROCm, V1 engine) | High-throughput local LLM serving |
| Agent framework | Gradio + httpx + FastAPI | Chat UI, LLM orchestration, tool calls, HTTP API |
| Three-tier memory | WorkingMemory + EpisodicMemory + SemanticMemory | RAM → session JSON → cross-session JSON |
| RL reward | 8-dim weighted function | Return, alpha, Sharpe, Sortino, Calmar, drawdown, losses, robustness |
| RAG | Multi-path: keyword + BM25 + dense + CrossEncoder | Confidence-gated retrieval (threshold 0.45) |
| Strategy DSL | YAML + JSON Schema | LLM-friendly intermediate representation |
| Backtest engine | Custom + CCXT + TA-Lib | Multi-position, slippage, walk-forward, 15+ metrics |
| Paper trading | Binance Testnet API | Risk-free live simulation with safety limits |
| Gradio (Dify SETUP_GUIDE available) | FastAPI + OpenAPI spec | `/api/agent/run` + `/api/agent/reward` endpoints |
>>>>>>> track3-honest

---

## 🔧 Prerequisites

### Hardware

<<<<<<< HEAD
- AMD Radeon GPU (e.g., RX 7900 XTX, MI250) with ROCm 6.2+
- Minimum 24 GB VRAM recommended (7B model + vLLM + Freqtrade)

### Cloud Environment

This project runs on the same **Anrui Cloud** (安睿云) AMD GPU instance as Track 3:

- JupyterLab terminal access
- VNC via noVNC on port 6080
- Python virtual environment at `/opt/venv/`
- ROCm 6.2 + PyTorch 2.9.1 already installed
=======
- AMD Radeon GPU (e.g., RX 7900 XTX, MI210) with ROCm 6.2+
- Minimum 24 GB VRAM recommended

### Cloud Environment

- **Anrui Cloud** (安睿云) AMD GPU instance
- JupyterLab terminal access
- VNC via noVNC on port 6080
- Python virtual environment at `/opt/venv/`
- ROCm 6.2 + PyTorch 2.9.1
>>>>>>> track3-honest

---

## 🚀 Quick Start

### One-command setup

```bash
cd track2-agentic-ai
bash scripts/setup.sh
```

<<<<<<< HEAD
This script:
1. Installs vLLM (ROCm), Freqtrade, Gradio dependencies
2. Downloads Qwen2.5-7B-Instruct model
3. Prepares fine-tuning datasets
4. Starts vLLM inference server on port 8000
5. Starts backtest microservice on port 8080
6. Starts Gradio chat UI on port 7860
=======
### Environment modes

```bash
# Default: ReAct agent loop
export AGENT_MODE=true
python src/chat_app.py

# Multi-agent pipeline
export MULTI_AGENT_MODE=true
python src/chat_app.py

# Legacy linear pipeline (fallback)
export AGENT_MODE=false
python src/chat_app.py
```

### Gradio Chat Pipeline (Dify-compatible)

```bash
# Start the API server
uvicorn src.api:app --host 0.0.0.0 --port 8080

# Call the multi-agent pipeline
curl -X POST http://localhost:8080/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "BTC EMA crossover strategy, stop loss 3%"}'

# Compute RL reward
curl -X POST http://localhost:8080/api/agent/reward \
  -H "Content-Type: application/json" \
  -d '{"metrics": {"total_return": 0.15, "sharpe_ratio": 1.8, ...}}'
```
>>>>>>> track3-honest

### End-to-end verification

```bash
bash scripts/verify_e2e.sh
```

<<<<<<< HEAD
Runs a full pipeline test: NL input → DSL → backtest → report.

=======
>>>>>>> track3-honest
---

## 📁 Project Structure

```
track2-agentic-ai/
├── src/
<<<<<<< HEAD
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
│   │   └── trading_agent.yml        # Dify workflow (optional, see SETUP_GUIDE.md)
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
=======
│   ├── agent/                        # Agent system (13 modules)
│   │   ├── core.py                  # ReAct loop + intent routing
│   │   ├── memory.py                # Three-tier memory (Working/Episodic/Semantic)
│   │   ├── personality.py           # Intent classifier + 小R prompt
│   │   ├── prompts.py               # ReAct system prompt + DSL generation
│   │   ├── reward.py                # 8-dim RL reward function
│   │   ├── rl_feedback.py           # 3-layer RL feedback (prompt/rules/DPO)
│   │   ├── tools.py                 # 8-tool registry + JSON action parser
│   │   ├── protocol.py              # AgentMessage communication protocol
│   │   ├── retrieval_agent.py       # Multi-path RAG agent
│   │   ├── reasoning_agent.py       # LoRA reasoning agent
│   │   ├── risk_agent.py            # Risk agent with veto power
│   │   └── orchestrator.py          # Multi-agent pipeline coordinator
│   ├── knowledge_base/
│   │   ├── multi_retriever.py       # Keyword + BM25 + reranking + confidence gate
│   │   ├── chunker.py               # Quant document chunker (512t, table-aware)
│   │   ├── retriever.py             # Keyword retriever (base)
│   │   ├── knowledge_entries.py     # 31 knowledge entries
│   │   └── semantic.py             # Optional semantic reranker
│   ├── dsl/                         # Strategy DSL (schema, validator, transpiler)
│   ├── backtest/                    # Backtest engine (runner, server, data_fetcher)
│   ├── tools/                       # Market data, indicators, paper trade
│   ├── llm/                         # LLM prompts
│   ├── api.py                       # Unified FastAPI server (incl. /api/agent/run)
│   └── chat_app.py                  # Gradio chat UI (3 modes)
├── training/
│   ├── data/
│   │   ├── prepare_quant_lora_dataset.py  # 4-category LoRA dataset (2000 samples)
│   │   └── prepare_dsl_pairs.py           # Original NL→DSL pairs
│   ├── scripts/
│   │   ├── train_qlora.py           # LoRA training (ROCm)
│   │   ├── train_dpo.py             # DPO training (TRL, ROCm)
│   │   └── prepare_dpo_data.py      # DPO pair generator from rewards
│   └── configs/
├── dify/
│   ├── tools/trading_api_openapi.yml  # 7 OpenAPI operations
│   └── workflows/SETUP_GUIDE.md      # 3-node Dify Chatflow guide
├── docs/
│   ├── technical_report.md          # Full technical report (16 innovations)
│   ├── lora_training_spec.md        # LoRA training spec (4 categories, anti-hallucination)
│   └── dsl_specification.md         # DSL formal spec
├── tests/
│   ├── test_agent.py                # 78 tests (memory, tools, parser, intent, personality)
│   ├── test_multi_agent.py          # 35 tests (protocol, agents, risk veto, retrieval)
│   ├── test_reward.py               # 24 tests (reward, feedback, DPO, memory)
│   └── ...                          # 87 existing tests
├── scripts/
│   ├── setup.sh
│   └── verify_e2e.sh
>>>>>>> track3-honest
├── requirements.txt
└── README.md
```

---

## 🎓 Fine-tuning Pipeline

<<<<<<< HEAD
### Data Sources

| Dataset | Size | Purpose |
|---------|------|---------|
| FNSPID | ~100K samples | Financial news → price analysis capability |
| FinGPT | ~50K samples | Financial QA, sentiment analysis |
| NL→DSL pairs | ~2K samples | Natural language → strategy DSL generation |
| Trading rules | ~500 rules | System prompt / RAG knowledge base |

### QLoRA Configuration
=======
### LoRA Training Data (4 categories, anti-hallucination)

| Category | Ratio | Purpose | Examples |
|----------|-------|---------|---------|
| A: Structured output | 60% | Force JSON format, eliminate free text | Indicator conditions → trading intent JSON |
| B: Reasoning chains | 25% | Chain-of-thought for trading decisions | Multi-step indicator analysis → intent |
| C: Tool calling | 10% | Correct tool selection format | "Get market data" → `{"tool": "get_market_data"}` |
| D: Boundary rejection | 5% | Anti-hallucination — refuse when info insufficient | Missing params → `{"view": "neutral", "confidence": 0}` |

> Key principle: LoRA learns **how to think**, not **what to know** (facts come from RAG).

### DPO Training (reward-based)

Reward-ranked strategy pairs from the RL feedback loop are used for DPO fine-tuning:

```bash
# Generate DPO pairs from accumulated rewards
python training/scripts/prepare_dpo_data.py

# Train with DPO on ROCm
python training/scripts/train_dpo.py --model /workspace/persistent/qwen-trader-merged
```

### LoRA Configuration
>>>>>>> track3-honest

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-7B-Instruct |
<<<<<<< HEAD
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
=======
| LoRA rank | 64 (SFT) / 16 (DPO) |
| LoRA alpha | 128 (SFT) / 32 (DPO) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 2e-4 (SFT) / 5e-6 (DPO) |
| Precision | bf16 |
| Epochs | 3 (SFT) / 1 (DPO) |
>>>>>>> track3-honest

---

## 📐 DSL Specification

<<<<<<< HEAD
The strategy DSL is a YAML-based intermediate representation:

=======
>>>>>>> track3-honest
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
<<<<<<< HEAD
    - { name: vol_ma, type: SMA, params: { period: 20, field: volume } }
    - { name: rsi, type: RSI, params: { period: 14 } }
  entry:
    long: "ema_fast > ema_slow AND volume > vol_ma * 1.5 AND rsi < 70"
=======
    - { name: rsi, type: RSI, params: { period: 14 } }
  entry:
    long: "ema_fast > ema_slow AND rsi < 70"
>>>>>>> track3-honest
    short: null
  exit:
    long: "ema_fast < ema_slow"
    short: null
  risk:
    stop_loss: -0.03
<<<<<<< HEAD
    trailing_stop: true
    trailing_stop_positive: 0.02
=======
>>>>>>> track3-honest
    max_open_trades: 3
    stake_amount: 0.1
```

<<<<<<< HEAD
The DSL is validated against a JSON Schema and then transpiled into a
Freqtrade `IStrategy` Python class for backtesting.

---

## 🔄 Gradio Chat Pipeline

The Gradio chat interface (`src/chat_app.py`) chains multiple steps:

1. **LLM Node (NL→DSL)**: User NL input + system prompt → strategy DSL (YAML)
2. **Code Node (Validate)**: JSON Schema validation of DSL
3. **HTTP Request Node (Backtest)**: POST to backtest microservice
4. **Code Node (Parse)**: Extract metrics from backtest response
5. **LLM Node (Report)**: Generate NL analysis of backtest results
6. **End**: Return report + metrics + recommendation
=======
---

## 🧪 Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_agent.py` | 78 | Memory (3 tiers), tools, action parser, intent, personality |
| `test_multi_agent.py` | 35 | Protocol, retrieval agent, reasoning, risk veto, BM25, chunker |
| `test_reward.py` | 24 | Reward computation, RL feedback, DPO pairs, memory consolidation |
| `test_dsl_validator.py` | 10 | DSL schema validation |
| `test_transpiler.py` | 10 | DSL → Freqtrade transpilation |
| `test_expr_parser.py` | 12 | AST-based safe expression evaluation |
| `test_dsl_advanced.py` | 17 | Advanced DSL features (short, new indicators, nested expressions) |
| `test_nl_to_dsl_quality.py` | 26 | NL→DSL extraction, validation, transpilation quality |
| `test_rag_retrieval.py` | 10 | RAG keyword/alias matching, semantic safety |
| `test_cn_market.py` | 2 | CN market backtest determinism |
| `test_e2e.py` | 3+2 | End-to-end pipeline (2 async tests need pytest-asyncio) |
| **Total** | **224 passed** | |

```bash
# Run all tests
python -m pytest tests/ -v

# Run agent tests only
python -m pytest tests/test_agent.py tests/test_multi_agent.py tests/test_reward.py -v
```
>>>>>>> track3-honest

---

## 📊 Judging Criteria Alignment

| Criterion | Points | How We Address It |
|-----------|--------|-------------------|
<<<<<<< HEAD
| Functional completeness & application value | 60 | Full-chain: NL→DSL→Backtest→Paper Trade; real crypto market data; working chat UI |
| AMD Radeon GPU / ROCm optimization | 40 | vLLM inference on ROCm; QLoRA fine-tuning on ROCm; benchmark vs CPU inference |
=======
| Functional completeness & application value | 60 | ReAct agent (8 tools); 3-agent pipeline (Retrieval→Reasoning→Risk); 3-tier memory; RL reward system; multi-path RAG with confidence gating; Dify integration; intent routing + personality; DPO training; LoRA training spec (4 categories) |
| AMD Radeon GPU / ROCm optimization | 40 | vLLM inference on ROCm; LoRA fine-tuning on ROCm; DPO training on ROCm; 6.2× batch scaling (32→202 tokens/s); RL reward computed from backtest on AMD GPU; all inference local (no cloud API) |
>>>>>>> track3-honest

---

## 📄 License

This project is submitted for the AMD AI DevMaster Hackathon.
