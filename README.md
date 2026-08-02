# AMD ROCm Local Quant Agent

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%207.2.1-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![Qwen2.5](https://img.shields.io/badge/Qwen2.5--7B-local%20model-6E49C8)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![vLLM](https://img.shields.io/badge/vLLM-ROCm-blue)](https://docs.vllm.ai/)
[![Tests](https://img.shields.io/badge/tests-232%20passed-brightgreen)](./docs/track2_final_status.md)

**AMD AI DevMaster Hackathon 2026 — Track 2: Development & Local Deployment of Private AI Agents**

An auditable, locally deployed investment and quantitative assistant running inference and fine-tuning on an AMD Radeon GPU. It combines a general conversational assistant with a grounded quantitative workflow for the domestic securities market.

The system is designed to answer ordinary investment questions, retrieve knowledge, perform local calculations, generate a constrained strategy DSL, backtest it, and return a risk-aware report. It does not place real orders.

## Highlights

- ReAct agent loop: reasoning, planning, tool selection, observation, and final answer.
- Multi-agent architecture: Retrieval Agent → Reasoning Agent → independent Risk Agent with veto power.
- Three-layer memory: working, episodic, and semantic memory with preference extraction.
- Multi-path RAG: keyword/BM25 retrieval, optional reranking, confidence gating, and source-aware answers.
- AMD-local model serving: Qwen2.5-7B with FP16 LoRA adaptation and vLLM on ROCm.
- Structured DSL pipeline: natural language → JSON/YAML DSL → canonicalization → schema/semantic validation → backtest → report.
- Dify integration: six-node workflow with RAG, local LLM, code validation, backtest, and risk report.
- General-assistant fallback: non-quantitative questions receive a natural response instead of being forced through the strategy pipeline.

## Architecture

```text
User request
    │
    ├── General question ───────────────► grounded assistant response
    │
    └── Quantitative request
          │
          ▼
     Intent router
          │
          ▼
  Retrieval Agent ──► Reasoning Agent ──► Risk Agent (veto)
          │                  │                  │
          └────── RAG ───────┴──── local tools ┘
                             │
                             ▼
       DSL → validation → backtest → paper report
                             │
                             ▼
                 AMD Radeon GPU / ROCm
                 Qwen2.5-7B + vLLM + LoRA
```

## Verified results

| Area | Result |
|---|---:|
| AMD GPU | gfx1100, ROCm 7.2.1 |
| LoRA adaptation | 400 domestic-market samples, 39 steps, 615 seconds |
| Training quality | loss 0.2848, token accuracy 98.1%, peak VRAM 16.21 GB |
| vLLM serving | FP16, local OpenAI-compatible endpoint, average latency ~8.2 s |
| CN-market evaluation | 24/24 after canonicalization and validation |
| Dify workflow | 6 nodes, three deterministic demo cases |
| Test suite | 232 passed; 2 known pre-existing async failures |

The evaluation uses deterministic synthetic historical data for reproducibility. Results are demonstrations of system behavior, not investment advice.

## Quick start

```bash
# Run on an AMD ROCm environment
bash scripts/setup.sh

# Start the API
python -m uvicorn src.api:app --host 0.0.0.0 --port 8080

# Start the Gradio assistant in another terminal
python src/chat_app.py
```

Useful endpoints:

```text
http://localhost:8080/docs       FastAPI documentation
http://localhost:7860            Gradio UI
http://localhost:8000/v1         vLLM-compatible local endpoint
```

Run the reproducible checks:

```bash
bash scripts/verify_e2e.sh
python -m pytest tests/ -v
```

## Dify workflow

The Dify setup guide is at [`dify/workflows/SETUP_GUIDE.md`](./dify/workflows/SETUP_GUIDE.md). The workflow is:

```text
User Input → RAG Retrieval → Local LLM → DSL Validation
           → Backtest API → Risk Report
```

The local model is configured as an OpenAI-compatible endpoint. Dify is an orchestration layer; model inference remains on the AMD ROCm host.

## Model assets

Large model weights are intentionally excluded from GitHub. The repository contains training scripts, configuration, checksums, and reproducibility instructions. Place the merged model under `models/` or configure the vLLM model path through the environment before starting the service.

## Repository layout

```text
src/agent/          ReAct loop, memory, routing, and multi-agent orchestration
src/knowledge_base/ RAG, chunking, retrieval, and confidence gating
src/dsl/            Schema, canonicalizer, semantic validator, transpilers
src/backtest/       Deterministic backtest and risk metrics
src/tools/          Market, indicator, paper-report, and external-tool adapters
training/           LoRA/DPO data preparation, training, merge, and serving
dify/               OpenAPI tools and workflow setup guide
docs/               Technical report, DSL specification, demos, and handoff notes
scripts/            Setup, evaluation, benchmark, and end-to-end verification
tests/              Agent, RAG, DSL, memory, reward, and integration tests
```

## Scope and safety

- No real trading or real order execution is performed.
- Demo market data is deterministic synthetic data unless an explicitly configured public data adapter is used.
- External results carry source, timestamp, mode, confidence, and limitations where available.
- Risk rules are implemented in code and can veto model output.
- The project is for hackathon demonstration and research only.

## Documentation

- [Technical report](./docs/technical_report.md)
- [Track 2 final status](./docs/track2_final_status.md)
- [DSL specification](./docs/dsl_specification.md)
- [LoRA training specification](./docs/lora_training_spec.md)
- [Dify workflow setup](./dify/workflows/SETUP_GUIDE.md)
- [中文说明](./README_zh.md)
