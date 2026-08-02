# Video Demo Script — Crypto Trading Agent on AMD Radeon GPU

**Duration**: ~4 minutes | **Format**: Screen recording with voiceover

---

## Part 1: Opening (0:00 – 0:15)

**Screen**: Project title slide / README top section

**Narration**:
> A private crypto trading agent powered by AMD ROCm — converting natural language
> into validated and backtested trading strategies, entirely on AMD hardware.

**Action**: Show project title, badges (ROCm, vLLM, Qwen2.5), and one-line description.

---

## Part 2: AMD Evidence (0:15 – 0:45)

**Screen 1**: Terminal — `rocm-smi` output

**Action**: Run `rocm-smi --showproductname` in terminal.

**Narration**:
> Running on AMD Instinct MI210 GPU with ROCm 7.2. The model was fine-tuned
> using QLoRA on this GPU — 81 training steps, final loss 0.16, peak memory 16 GB.

**Screen 2**: Terminal — training log + vLLM benchmark

**Action**: Show `tail -5 /tmp/qlora_train.log` then `cat /workspace/persistent/vllm_benchmark.json`

**Narration**:
> vLLM serves the merged Qwen2.5-7B model on this AMD GPU. Batch throughput
> scales 6.2 times — from 32 tokens per second at batch size 1, to 201 tokens
> per second at batch 16.

**On-screen text**:
```
Training: 81 steps, loss=0.1625, GPU=16GB
vLLM: 201.7 tokens/s (6.2× scaling)
```

---

## Part 3: Dify Workflow (0:45 – 1:15)

**Screen**: Dify workflow editor (browser)

**Action**: Show the 12-node Chatflow in Dify's visual editor. Pan across nodes.

**Narration**:
> The agent is orchestrated as a Dify workflow. Natural language input goes to
> the fine-tuned Qwen model to generate a strategy DSL. A canonicalizer node
> fixes common LLM output errors. If validation fails, a retry branch sends
> error feedback back to the model. Validated strategies proceed to backtest,
> and the model generates a risk analysis report.

**On-screen text** (node labels highlighted):
```
Start → LLM(DSL) → Canonicalizer → IF/ELSE → Backtest → Report → End
                     ↘ Retry ← (on failure)
```

---

## Part 4: Live Interaction (1:15 – 2:45)

**Screen**: Dify chat interface

**Action**: Type the following prompt:

```
BTC放量突破前高，使用EMA20/EMA50，止损3%，帮我回测并分析风险
```

**Narration**:
> Let's test with a real trading idea in Chinese: "BTC volume breakout, EMA 20
> crosses 50, 3% stop loss, backtest and analyze risk."

**Wait**: ~6-8 seconds for vLLM inference on AMD GPU.

**Show as output appears**:

1. **DSL Output** — YAML strategy specification
   - Indicators: EMA 20 (fast), EMA 50 (slow)
   - Entry: `ema_fast > ema_slow`
   - Exit: `ema_fast < ema_slow`
   - Risk: `stop_loss: -0.03`

2. **Validation** — "Schema validation passed" (canonicalizer repairs shown)

3. **Backtest Results** — metrics table:
   - Total trades: N
   - Win rate: X%
   - Total return: X%
   - Buy & Hold return: X% (benchmark)
   - Alpha: +X% (vs B&H)
   - Max drawdown: X%
   - Sharpe ratio: X
   - Sortino ratio: X

4. **Risk Report** — Chinese language analysis:
   - 策略概述 (Strategy summary)
   - 回测表现 (Backtest performance vs benchmark)
   - 风险分析 (Risk analysis)
   - 建议 (Recommendation: APPROVE/MODIFY/REJECT)

**Narration**:
> The agent generated a valid strategy DSL, ran a 180-day backtest with
> slippage modeling, and produced a risk report comparing the strategy
> against a buy-and-hold benchmark — all powered by the AMD GPU.

---

## Part 5: Error Recovery (2:45 – 3:15)

**Screen**: Dify chat interface (new conversation)

**Action**: Type a prompt that triggers an error:

```
用ATR动态止损的EMA策略，止损设为 ema_fast - atr * 3
```

**Narration**:
> Now let's test error recovery. This prompt asks for an expression-based
> stop loss, which our schema doesn't support.

**Show**:
1. LLM generates DSL with `stop_loss: "ema_fast - atr * 3"`
2. Canonicalizer detects: "cannot parse as number"
3. Retry branch activates — sends error feedback to LLM
4. LLM regenerates with `stop_loss: -0.04` (numeric)
5. Validation passes
6. Backtest runs successfully

**On-screen text**:
```
Attempt 1: stop_loss = "ema_fast - atr * 3" → REJECTED
Retry: "Fix: stop_loss must be negative number"
Attempt 2: stop_loss = -0.04 → VALIDATED → Backtest succeeded
```

**Narration**:
> The system isn't just a chatbot — it has constraints, validation, and
> automatic recovery. Invalid outputs never reach the backtest engine.

---

## Part 6: Closing (3:15 – 3:35)

**Screen**: Summary slide with key metrics

**On-screen text**:
```
NL→DSL Generation Quality
├── 10-prompt evaluation:    9/10 (90%)
├── 100-prompt evaluation:  88/100 (88%)
└── Schema validation:      90%

AMD ROCm Performance
├── QLoRA training:  81 steps, loss=0.1625, 16GB VRAM
├── vLLM throughput: 201.7 tokens/s (batch=16)
└── Batch scaling:   6.2× (batch 1→16)

Pipeline
├── Canonicalizer: type coercion + repair logging
├── LLM retry: error feedback on unrecoverable failures
├── Backtest: multi-position, slippage, walk-forward
└── Paper trading: explicit user confirmation only
```

**Narration**:
> 88 percent pass rate on 100 diverse prompts. 201 tokens per second throughput
> on AMD hardware. A complete pipeline from natural language to validated,
> backtested trading strategies — with safety boundaries that prevent invalid
> outputs from reaching execution.

---

## Recording Notes

- Record at 1920×1080, 30fps
- Use Dify web UI in full screen (dark theme if available)
- Terminal font size: 14-16pt for readability
- Pre-warm vLLM before recording (first inference is slower)
- Have a backup Gradio UI ready if Dify has issues
- Keep training log and benchmark JSON visible in a second tab
- Test the error recovery prompt beforehand to ensure it triggers retry
