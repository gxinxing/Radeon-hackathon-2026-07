# Demo Recording Guide — Crypto Trading Agent

Technical setup guide for recording the 3-5 minute demo video.

## Prerequisites

### GPU Instance (AMD Cloud)
- vLLM running at `http://localhost:8000/v1` (model: `models/qwen-trader-merged`)
- Trading API at `http://localhost:8080`
- Training log at `/workspace/persistent/qlora_train.log`
- Benchmark at `/workspace/persistent/vllm_benchmark.json`

### Local Machine
- Screen recording software (OBS Studio recommended, free)
- Browser with Dify access (or Gradio as fallback)
- Terminal SSH or VNC to GPU instance

## Recording Setup

### OBS Studio Configuration
```
Resolution: 1920×1080
FPS: 30
Format: MP4 (H.264)
Audio: Microphone (narration)
Sources: Display Capture (main monitor)
```

### Display Layout (1920×1080)

```
┌────────────────────────────────────────────┐
│  Browser (Dify or Gradio)                 │
│  ┌──────────────────────────────────┐     │
│  │                                  │     │
│  │  Chat Interface                  │     │
│  │  (left 60% of screen)             │     │
│  │                                  │     │
│  └──────────────────────────────────┘     │
│                                            │
│  ┌──────────────────────────────────┐     │
│  │  Terminal (right 40%)             │     │
│  │  rocm-smi / training log          │     │
│  │  vLLM benchmark                   │     │
│  └──────────────────────────────────┘     │
└────────────────────────────────────────────┘
```

## Pre-Recording Checklist

1. **vLLM Health Check**
   ```bash
   curl -s http://localhost:8000/v1/models | python3 -m json.tool
   # Should return model: "models/qwen-trader-merged"
   ```

2. **API Health Check**
   ```bash
   curl -s http://localhost:8080/health
   # Should return: {"status": "ok"}
   ```

3. **Pre-warm vLLM** (first inference is slower due to KV cache init)
   ```bash
   curl -s http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"models/qwen-trader-merged","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
   ```

4. **Have terminal commands ready**:
   ```bash
   # AMD evidence (Part 2)
   rocm-smi --showproductname
   tail -5 /workspace/persistent/qlora_train.log
   cat /workspace/persistent/vllm_benchmark.json | python3 -m json.tool
   ```

5. **Have Dify prompts ready** (copy-paste):
   ```
   BTC放量突破前高，使用EMA20/EMA50，止损3%，帮我回测并分析风险
   ```
   ```
   用ATR动态止损的EMA策略，止损设为 ema_fast - atr * 3
   ```

6. **Test error recovery prompt** beforehand — ensure it triggers the retry path

## Fallback: Gradio UI

If Dify has issues, use Gradio (`src/chat_app.py`) as fallback:

```bash
cd /workspace/persistent/radeon-repo/track2-agentic-ai
export VLLM_BASE_URL=http://localhost:8000/v1
export BACKTEST_API_URL=http://localhost:8080
export MODEL_NAME=models/qwen-trader-merged
/opt/venv/bin/python src/chat_app.py
```

Access at `http://localhost:7860`

Gradio shows the same pipeline: NL → DSL → backtest → report with equity curve chart.

## Recording Timeline

| Time | Section | Screen | Duration |
|------|---------|--------|----------|
| 0:00 | Opening | Title slide / README | 15s |
| 0:15 | AMD Evidence | Terminal: rocm-smi, training log, benchmark | 30s |
| 0:45 | Dify Workflow | Dify editor: 12-node flow | 30s |
| 1:15 | Live Interaction | Dify chat: EMA crossover prompt | 90s |
| 2:45 | Error Recovery | Dify chat: ATR expression stop_loss → retry | 30s |
| 3:15 | Closing | Summary stats slide | 20s |
| **Total** | | | **~3:55** |

## Post-Recording

1. Trim dead air at start/end
2. Add subtitles for Chinese text (for English-speaking judges)
3. Add on-screen text for key metrics (201.7 tokens/s, 88% pass rate, 6.2×)
4. Export as MP4, 1080p, < 500MB
5. Upload to YouTube (unlisted) or include in submission

## Key Metrics to Display

```
QLoRA: 81 steps, loss=0.1625, 98.71% token accuracy, 16GB VRAM
vLLM: 201.7 tokens/s (batch=16), 6.2× scaling, 32.4 tokens/s (batch=1)
NL→DSL: 9/10 (90%) standard, 88/100 (88%) large-scale
Schema: 90% validation rate
Corrected data: 43 high-quality samples
```
