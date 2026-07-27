# Demo Script: Crypto Trading Agent on AMD Radeon GPU

## Demo Flow (3-5 minutes)

### 1. Introduction (30s)
- "This is Track 2: Agentic AI — a crypto trading agent powered by AMD ROCm GPU"
- Show the Gradio chat UI at http://localhost:7860

### 2. AMD ROCm GPU Showcase (30s)
- Show `rocm-smi` output: AMD Radeon Graphics, 51GB VRAM
- Show vLLM running on port 8000: `curl http://localhost:8000/v1/models`
- Explain: Qwen2.5-7B fine-tuned with QLoRA, served via vLLM on ROCm

### 3. Live Demo: NL → DSL → Backtest → Report (2 min)
Type in the chat:
> "BTC放量突破前高，帮我做一个EMA突破策略，止损3%"

Watch the pipeline:
1. **DSL Generation**: LLM generates YAML strategy DSL (powered by vLLM on AMD GPU)
2. **Schema Validation**: JSON Schema validates the DSL structure
3. **Backtest**: Strategy is backtested on 90 days of BTC/USDT data
4. **Report**: LLM generates a Chinese-language analysis report

Show the output:
- Strategy DSL (YAML format)
- Backtest metrics table (win rate, return, drawdown, Sharpe ratio)
- AI-generated analysis report

### 4. Technical Highlights (1 min)
- Show the technical architecture diagram
- Highlight: QLoRA fine-tuning on ROCm (bf16, 4-bit NF4)
- Highlight: Full-chain pipeline: NL → DSL → Validate → Backtest → Report
- Highlight: 22 unit tests passing (DSL validation + transpilation)

### 5. Closing (30s)
- "This project demonstrates that AMD Radeon GPUs can power production-grade
   AI agents for financial trading, entirely without NVIDIA hardware."
- Mention Track 3 synergy: same GPU instance runs both robot RL training and LLM inference
