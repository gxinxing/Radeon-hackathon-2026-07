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

The agent uses a **ReAct (Reasoning + Acting) loop** — the LLM autonomously
decides which tools to call, observes the results, and iterates until the
user's goal is met. This replaces a fixed linear pipeline with an adaptive,
LLM-driven workflow that demonstrates all five required agent capabilities:
reasoning, planning, tool use, memory management, and task execution.

```
┌──────────────────────────────────────────────────────────────────┐
│                      Gradio Chat UI (port 7860)                   │
│  User NL input → Agent reasoning trace → Final analysis report    │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                   ReAct Agent Loop (src/agent/)                    │
│                                                                    │
│  ┌─ Thought: LLM reasons about next step ──────────────┐         │
│  │  Action: LLM selects a tool to call                  │         │
│  │  Observe: Tool result stored in ConversationMemory   │         │
│  └─ Repeat until Final Answer or max iterations ───────┘         │
│                                                                    │
│  Memory: dialogue history + tool call log + strategy/backtest     │
│  Tools: 8 registered tools (market, strategy, validate, backtest,  │
│         walk-forward, paper-trade, knowledge, final-answer)        │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                AMD ROCm GPU (51 GB VRAM)                           │
│                                                                    │
│  ┌────────────────────────┐    ┌────────────────────────────────┐│
│  │ vLLM (ROCm V1 Engine)  │    │ LoRA Fine-tuning               ││
│  │ Qwen2.5-7B (merged)    │    │ • 2,000 NL→DSL training pairs  ││
│  │ FP16, ~32 t/s          │    │ • LoRA r=64, bf16, 3 epochs    ││
│  │ ~20 GB VRAM            │    │ • PEFT merge → vLLM serve      ││
│  └────────────────────────┘    └────────────────────────────────┘│
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ Tool Services (FastAPI + CCXT + TA-Lib)                     ││
│  │ • Market data • Backtest engine • Walk-forward analysis     ││
│  │ • Paper trading (Binance Testnet) • Knowledge RAG           ││
│  └──────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
```

### Agent Capability Matrix

| Capability | Implementation | Evidence |
|-----------|---------------|----------|
| **Reasoning** | ReAct loop — LLM outputs Thought before each Action | `src/agent/core.py` — `_build_agent_prompt()`, `REACT_SYSTEM_PROMPT` |
| **Planning** | LLM decides tool sequence dynamically; typical flow: market→knowledge→generate→validate→backtest→walk-forward→answer | `src/agent/prompts.py` — decision guidelines in system prompt |
| **Tool Use** | 8 registered tools with structured JSON dispatch | `src/agent/tools.py` — `TOOL_REGISTRY`, `execute_tool()` |
| **Memory Management** | `ConversationMemory` — dialogue history, tool call log, strategy/backtest tracking | `src/agent/memory.py` — messages, tool_calls, strategies, backtest_results |
| **Task Execution** | Real backtests, walk-forward analysis, paper trading via FastAPI | `src/backtest/runner.py`, `src/tools/paper_trade.py` |

### Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base model | Qwen2.5-7B-Instruct | Native Chinese support, ROCm-compatible, strong code generation |
| Fine-tuning | FP16 LoRA (r=64) + bf16 | Fits 64GB VRAM (MI210), full-precision LoRA on ROCm, bf16 avoids GradScaler issues |
| Inference | vLLM with merged LoRA | V1 engine for performance, avoids V0 LoRA fallback on ROCm |
| Chat UI | Gradio | Lightweight, no Docker Hub dependency, Python-native |
| Strategy DSL | YAML + JSON Schema | LLM-friendly, human-readable, validatable |
| Backtest | Custom engine + CCXT/TA-Lib | Crypto-native, synthetic data fallback for network-restricted environments |
| Market | Crypto (BTC/USDT etc.) | 24/7 trading, open API standard, universal appeal |

## 3. How AMD Radeon GPUs Are Utilized

### LLM Inference (vLLM on ROCm)

- **Model**: Qwen2.5-7B-Instruct, fine-tuned with FP16 LoRA
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

The fine-tuned model was evaluated against natural language prompts covering
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

#### Standard Evaluation (10 prompts)

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

#### Large-Scale Generalization Evaluation (100 prompts)

| Metric | Rate |
|--------|------|
| YAML extraction | 96/100 (96%) |
| Schema validation | 90/100 (90%) |
| Freqtrade transpilation | 90/100 (90%) |
| Backtrader transpilation | 90/100 (90%) |
| Indicator matching | 88/100 (88%) |
| Retried | 9/100 |
| **Overall pass rate** | **88/100 (88.0%)** |

**By category**: breakout 15/15 (100%), confluence 8/10 (80%),
mean_reversion 20/20 (100%), momentum 15/15 (100%), short 8/10 (80%)

**Latency**: avg=5,663ms, p95=7,776ms per request (AMD MI210, FP16, eager mode)

**Safety boundary**: The system allows 88 strategies to proceed to backtesting
while safely rejecting 12 structurally non-compliant strategies. The canonicalizer
applies 2-4 repairs per test case on average. All repairs are logged for audit
transparency.

### 4.8 vLLM AMD ROCm Performance Benchmark

vLLM serving the fine-tuned Qwen2.5-7B model on AMD Instinct MI210 GPU:

| Mode | tokens/s | Avg Latency (ms) | P95 Latency (ms) |
|------|---------|-----------------|------------------|
| Sequential (batch=1) | 32.4 | 6,849 | 15,717 |
| Concurrent batch=2 | 59.3 | 6,901 | 9,621 |
| Concurrent batch=4 | 104.4 | 6,556 | 9,192 |
| Concurrent batch=8 | 148.1 | 6,751 | 9,469 |
| Concurrent batch=16 | 201.7 | 6,946 | 10,177 |

**Key finding**: 6.2× throughput scaling from batch=1 to batch=16 (32.4 → 201.7 tokens/s).
Average latency remains stable (~6.5-6.9s) across batch sizes, indicating efficient
GPU utilization. VRAM usage ~16GB for model + KV cache on 64GB MI210.

**Note**: These are batch throughput optimizations, not low-latency interactive
optimizations. The avg latency of ~6.8s reflects the 7B model's compute time for
~500 token outputs on AMD ROCm with eager execution mode.

### 4.9 ReAct Agent with Three-Tier Memory

The agent uses a **ReAct (Reasoning + Acting) loop** where the LLM autonomously
decides which tools to call, observes results, and iterates until the goal is met.
This replaces a fixed linear pipeline with an adaptive, LLM-driven workflow.

**Three-tier memory architecture:**

| Tier | Name | Scope | Storage | Content |
|------|------|-------|---------|---------|
| 1 | WorkingMemory | Per-session | RAM | Messages, tool calls, current strategy/backtest |
| 2 | EpisodicMemory | Per-session | JSON file | All strategies, backtests, thoughts, user requests |
| 3 | SemanticMemory | Cross-session | JSON file | User preferences, strategy stats, experience rules |

The semantic memory is loaded at startup and persists across sessions, allowing
the agent to reference past strategy performance and user preferences.

### 4.10 Multi-Agent Pipeline with Risk Veto

A three-agent pipeline replaces the single-agent loop for production use:

1. **Retrieval Agent** — Multi-path RAG (keyword + BM25 + dense + reranking) with
   confidence gating. If no document passes the 0.45 threshold,
   `has_valid_docs=false` and the pipeline short-circuits to neutral.
2. **Reasoning Agent** — LoRA + RAG context → structured trading intent JSON
   (view, confidence, position ratio, stop loss). Forced neutral when RAG is insufficient.
3. **Risk Agent** — Hard rule validation (code-level, not model-influenced) with
   **unique veto power**. Checks: position limits, stop-loss distance, confidence
   threshold, reason completeness. `allow_execute=false` blocks all trades.

> **The LLM only produces trading intent. The Risk Agent decides execution.**

### 4.11 RL Reward System for Strategy Self-Optimization

An 8-dimensional reward function computes a normalized score [-1, +1] from
backtest metrics, feeding a three-layer RL feedback loop:

| Layer | Mechanism | Scope |
|-------|-----------|-------|
| L1 (Immediate) | Reward injected into current session prompt | Per-session |
| L2 (Experience) | High/low reward patterns → experience rules → semantic memory | Cross-session |
| L3 (DPO Data) | Reward-ranked strategy pairs → DPO training data → LoRA update | Weight update |

**Reward components:** Return (20%) + Alpha (15%) + Sharpe (15%) + Sortino (10%)
+ Calmar (5%) + Drawdown penalty (12%) + Consecutive losses (8%)
+ Walk-forward robustness (15%).

After each backtest, the agent computes reward and displays:
`🎯 RL Reward: +0.42 (Grade: A) — 收益率15%表现良好`

### 4.12 Multi-Path RAG with Confidence Gating

The retrieval system combines three search paths with a reranking + gating pipeline:

```
Query → ① Keyword (top 6) + ② BM25 (top 6) + ③ Dense vector (top 6)
      → Merge & deduplicate (≤10 candidates)
      → CrossEncoder reranking (keep top 4)
      → Confidence gate (score < 0.45 → clear results)
      → LLM context
```

The confidence gate is a hard gate: if no document passes the threshold,
the reasoning agent is forced to output `neutral` — preventing hallucination.

### 4.13 Intent Routing + Personality

The agent classifies each message as trading intent or general conversation:

- **Trading** (70+ keywords: 策略/回测/BTC/RSI/...) → ReAct agent loop with 8 tools
- **General** (你好/讲个笑话/...) → Personality-driven direct response as "小R"

The personality prompt gives the agent a name, humor, opinions, and memory —
making it feel "alive" rather than a cold tool. It remembers user preferences
across sessions via semantic memory.

### 4.14 Dify Integration via HTTP Endpoints

Two new FastAPI endpoints enable Dify Chatflow integration with just 3 nodes:

| Endpoint | Dify Tool | Purpose |
|----------|-----------|---------|
| `POST /api/agent/run` | `runMultiAgent` | Full pipeline in one call |
| `POST /api/agent/reward` | `computeReward` | RL reward computation |

Dify Chatflow: Start → Tool(runMultiAgent) → End — no 12-node manual pipeline needed.

### 4.15 LoRA Training Dataset Spec

A formal spec for constructing anti-hallucination training data with 4 categories:

| Category | Ratio | Purpose |
|----------|-------|---------|
| A: Structured output | 60% | Force JSON format, eliminate free text |
| B: Reasoning chains | 25% | Chain-of-thought for trading decisions |
| C: Tool calling | 10% | Correct tool selection format |
| D: Boundary rejection | 5% | Anti-hallucination — refuse when info insufficient |

Key principle: LoRA learns **how to think**, not **what to know** (facts come from RAG).

### 4.16 Original: Corrected Training Data Generation

Using the model + canonicalizer pipeline, 51 diverse NL prompts were processed
to generate high-quality training data for future v2 LoRA fine-tuning:

- **Input**: 51 diverse NL prompts (Chinese + English, 7 strategy categories)
- **Output**: 43/51 valid samples (84% yield rate)
- **Average repairs per sample**: 1.9
- **Saved to**: `/workspace/persistent/corrected_train.jsonl`

Only samples that passed canonicalization + schema validation + transpilation
were retained. This ensures the v2 training data contains only structurally
correct DSL with proper types, negative stop_loss, and valid indicator references.

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
| LoRA training script | `training/scripts/train_qlora.py` | ✅ Loads YAML config, ROCm-optimized |
| LoRA merge script | `training/scripts/merge_lora.py` | ✅ PEFT merge |
| vLLM serving script | `training/scripts/serve_vllm.sh` | ✅ ROCm env configured |
| Deploy script | `training/scripts/deploy.sh` | ✅ One-command deploy |
| Setup script | `scripts/setup.sh` | ✅ Complete |
| E2E verification | `scripts/verify_e2e.sh` | ✅ 7 checks |
| NL→DSL evaluation | `scripts/eval_nl_to_dsl.py` | ✅ 10 test prompts, online/offline modes |
| RAG knowledge base | `src/knowledge_base/` | ✅ 40+ entries, keyword + semantic retrieval |
| DSL canonicalizer | `src/dsl/canonicalizer.py` | ✅ Type coercion + repair logging |
| Unit tests | `tests/` (7 files) | ✅ 93 tests passing, 2 deselected (async) |
| Batch eval (100 prompts) | `scripts/gen_eval_dataset.py` | ✅ 88/100 (88%) |
| vLLM benchmark | `scripts/vllm_benchmark.py` | ✅ 6.2× scaling, 201.7 tokens/s |
| Corrected training data | `scripts/gen_corrected_dataset.py` | ✅ 43 valid samples |
| ReAct Agent core | `src/agent/core.py` | ✅ ReAct loop + intent routing + personality |
| Three-tier memory | `src/agent/memory.py` | ✅ Working + Episodic + Semantic (file-backed) |
| Agent prompts | `src/agent/prompts.py` | ✅ ReAct + DSL generation + reward feedback |
| Agent tools | `src/agent/tools.py` | ✅ 8-tool registry + multi-path RAG retrieval |
| Intent classifier | `src/agent/personality.py` | ✅ Trading vs general conversation routing |
| RL reward function | `src/agent/reward.py` | ✅ 8-dim reward [-1,+1], grades A+ to F |
| RL feedback loop | `src/agent/rl_feedback.py` | ✅ L1 prompt + L2 experience + L3 DPO pairs |
| Multi-agent protocol | `src/agent/protocol.py` | ✅ AgentMessage JSON structure |
| Retrieval Agent | `src/agent/retrieval_agent.py` | ✅ Multi-path RAG + confidence gating |
| Reasoning Agent | `src/agent/reasoning_agent.py` | ✅ LoRA + RAG → trading intent JSON |
| Risk Agent | `src/agent/risk_agent.py` | ✅ Hard rules + veto power (5 checks) |
| Multi-agent orchestrator | `src/agent/orchestrator.py` | ✅ Retrieval→Reasoning→Risk pipeline |
| Multi-path retrieval | `src/knowledge_base/multi_retriever.py` | ✅ Keyword+BM25+reranking+confidence gate |
| Quant chunker | `src/knowledge_base/chunker.py` | ✅ 512 tokens, table-aware, metadata tags |
| Dify HTTP endpoints | `src/api.py` | ✅ /api/agent/run + /api/agent/reward |
| OpenAPI spec (Dify) | `dify/tools/trading_api_openapi.yml` | ✅ 7 operations including runMultiAgent |
| Dify Chatflow guide | `dify/workflows/SETUP_GUIDE.md` | ✅ 3-node Chatflow with runMultiAgent |
| LoRA training spec | `docs/lora_training_spec.md` | ✅ 4 categories (60/25/10/5), anti-hallucination |
| LoRA dataset generator | `training/data/prepare_quant_lora_dataset.py` | ✅ 2000 samples (4 categories) |
| DPO data generator | `training/scripts/prepare_dpo_data.py` | ✅ Reward-ranked preference pairs |
| DPO trainer | `training/scripts/train_dpo.py` | ✅ TRL DPOTrainer, ROCm-optimized |
| Agent tests | `tests/test_agent.py` | ✅ 78 tests (memory, tools, parser, intent, personality) |
| Multi-agent tests | `tests/test_multi_agent.py` | ✅ 35 tests (protocol, agents, risk veto, retrieval) |
| RL reward tests | `tests/test_reward.py` | ✅ 24 tests (reward, feedback, DPO, memory) |
| Technical report | `docs/technical_report.md` | ✅ This document |

## 7. Team

- Team Name: Radeon ROCm Raiders
- Members: Simon Xing
