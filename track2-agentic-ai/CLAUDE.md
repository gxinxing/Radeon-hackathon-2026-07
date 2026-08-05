# Track 2: Agentic AI — Domestic Market Quantitative Agent

## Competition Context

- **Track**: Track 2 — Agentic AI
- **Goal**: Build an AI agent that reasons, plans, uses tools, and executes tasks
- **Key Requirement**: Local inference on AMD Radeon GPU via ROCm
- **Judging**: Functional completeness (60pts) + AMD ROCm optimization (40pts)
- **Timeline**: Submit by August 6, 2026

## Project: Domestic Market Quantitative Agent

Fine-tune Qwen2.5-7B on AMD ROCm GPU to act as a Chinese A-share quant strategist.
Chinese natural language → RAG (domestic market rules) → LLM DSL → canonicalize →
CN backtest (T+1, 100-share lot, no short, 10% price limits) → risk report
(PASS/REVIEW/REJECT) — full-chain agentic trading system.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | Qwen2.5-7B-Instruct | Chinese-capable, ROCm-friendly, code generation |
| Fine-tuning | FP16 LoRA (r=64, alpha=128) | 400 CN-market NL→DSL pairs; bitsandbytes unavailable on ROCm |
| Inference | vLLM (ROCm, merged LoRA, port 8000) | ~8.2s/request, ~16GB VRAM, eager mode |
| Agent orchestration | graph_engine (:8083) | OpenAI-compatible local layer: intent router → strategy/compute/general nodes; strips upstream tool_call so Open WebUI never sees "Tool not found" |
| Chat UI | Open WebUI (→ graph_engine /v1) | Standard OpenAI-compatible chat frontend |
| Workflow | Dify 6-node Chatflow | Input → RAG → LLM → code → backtest → answer |
| Backtest engine | `src/backtest/cn_runner.py` | Deterministic synthetic CN market data, T+1 / lot 100 / no short / price limits |
| Strategy DSL | YAML + JSON Schema | LLM-friendly, human-readable, validatable; CN shape includes `constraints` |
| Market | Domestic (A-share / ETF) | CN stock exchanges, .SH/.SZ instruments; crypto is legacy only |

## Development Constraints

1. **Same AMD GPU instance as Track 3** — no new instances, share VRAM
2. **All inference on ROCm** — vLLM on the host (gfx1100, ROCm 7.2.1)
3. **Dify runs in Docker** — co-located with vLLM on same machine
4. **No live trading** — backtests run on deterministic synthetic data, demonstration only
5. **All submission materials in English** (Chinese allowed in user-facing docs)

## Key File Locations

- DSL: `src/dsl/` — schema, validator, canonicalizer, transpiler
- CN Backtest: `src/backtest/cn_runner.py` + `src/api.py` (`/api/cn/backtest/report`)
- RAG: `src/api.py` `/api/knowledge` + `src/knowledge_base/`
- Orchestration: `graph_engine.py` (OpenAI-compatible, :8083)
- Chat UI: `src/chat_app.py` (legacy Gradio UI, not the main entry)
- Agent system: `src/agent/` — retrieval/reasoning/risk agents + multi-agent orchestrator
- Training: `training/` — CN market data prep, FP16 LoRA scripts, configs
- Dify: `dify/` — workflow SQL patches + SETUP_GUIDE, tool OpenAPI spec
- Docker: `docker/` — compose for API + vLLM (ROCm host service)
- Landing: `landing/index.html` — single-file marketing page

## Common Tasks

1. **DSL iteration** — Update `schema.json`, validator, canonicalizer together; CN DSL carries `constraints` (t_plus_one / lot_size / allow_short / price_limit)
2. **CN backtest** — `src/backtest/cn_runner.py`; keep deterministic + synthetic
3. **Evaluation** — `scripts/eval_cn_market_v2.py` (24-case CN market eval)
4. **Testing** — `python3 -m pytest tests/ --ignore=tests/test_e2e.py -q` → 282 passed; E2E via `scripts/verify_e2e.sh` (CN chain) and `scripts/run_demo.sh`
5. **Serve** — vLLM :8000 → graph_engine :8083 → Open WebUI; FastAPI :8080
