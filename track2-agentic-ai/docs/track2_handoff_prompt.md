# Track 2 — Handoff Prompt for Next Agent

> 给 Simon 的说明：下面这段英文 prompt 是**自包含的**，可直接粘贴给任意一个 coding agent（WorkBuddy 子代理、OpenClaw、Claude、Codex 均可）。它不依赖本对话的上下文，所有事实都已在内部写明。把 `<PROJECT_ROOT>` 替换成实际绝对路径再粘贴即可。

---

You are a senior AI engineer helping finish a hackathon submission. Below is a fully self-contained brief — do not ask for context, just execute and report.

## 1. Mission
Project: **AMD Radeon Hackathon — Track 2: Agentic AI (Crypto Trading Agent)**.
Goal: a natural-language → strategy DSL → backtest → analysis report pipeline that runs entirely on **AMD ROCm GPUs**.
Submission deadline: **2026-08-06**. All submission artifacts (report, README, demo) must be in **English**.
Hard constraint: **This is an AMD ROCm project — CUDA/NVIDIA is NOT available.** bitsandbytes 4-bit quantization does NOT work on ROCm; do not rely on it.

## 2. Project location
`<PROJECT_ROOT>` = the `track2-agentic-ai/` directory (subfolder of the Radeon hackathon repo).
Key paths:
- `src/dsl/` — schema.json, validator.py, transpiler.py (DSL → Freqtrade IStrategy)
- `src/backtest/` — server.py (FastAPI), runner.py, data_fetcher.py
- `src/tools/` — indicators.py, paper_trade.py (Binance Testnet)
- `src/llm/prompts.py` — 3 system prompts
- `src/chat_app.py` — Gradio NL→DSL→backtest→report
- `src/api.py` — unified FastAPI service
- `training/scripts/train_qlora.py` — QLoRA fine-tune Qwen2.5-7B
- `training/data/` — data generators (prepare_dsl_pairs.py, prepare_fingpt.py, prepare_fnspid.py, merge_datasets.py)
- `tests/` — test_dsl_validator.py, test_transpiler.py, test_e2e.py
- `docs/technical_report.md`, `README.md`, `requirements.txt`, `scripts/setup.sh`, `scripts/verify_e2e.sh`

## 3. Verified current state (as of 2026-07-28, confirmed by file inspection)

**Present & likely working:**
- DSL schema, validator, transpiler, backtest runner/server, indicators, paper_trade, prompts, chat_app, api — all exist.
- 25 unit/integration tests: `test_transpiler.py` (10), `test_dsl_validator.py` (10), `test_e2e.py` (5).

**MISSING / BROKEN (must fix):**
1. **`training/data/processed/merged_train.jsonl` does NOT exist.** Only generator scripts are present; the processed dataset was never produced. The training command references this path.
2. **`models/` is EMPTY** — no LoRA checkpoint, no base model cache. Model download/training only happens on the GPU cloud instance.
3. **`docker/Dockerfile.vllm` MISSING** — README lists it but only `Dockerfile.backtest` and `docker-compose.yml` exist.
4. **`dify/workflows/trading_agent.yml` MISSING** — only `SETUP_GUIDE.md` exists; the actual Dify workflow export was never created.
5. **`docs/technical_report.md` and `README.md` contain `[TODO: fill before submission]` placeholders** (team info, etc.).
6. **No demo video** has been produced.

## 4. Critical bugs to fix (P0 — block a working submission)

### Bug A — `training/scripts/train_qlora.py` is broken on ROCm (confirmed by reading full file)
- The `try/except` at lines 87–100 wraps ONLY `BitsAndBytesConfig(...)` construction + `import bitsandbytes`. Importing bnb *succeeds* on ROCm, so the exception never fires. The real failure happens later at `AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)` (≈line 149) because **bitsandbytes 4-bit CUDA extensions are not built for ROCm** → crash, and the intended FP16 fallback is dead code.
- `torch_dtype=torch.float16` (line 141) conflicts with `bf16=True` in `SFTConfig` (line 167) and `bnb_4bit_compute_dtype=torch.bfloat16` (line 91).
- **Fix:** Remove the 4-bit bitsandbytes path. Use full **bf16 LoRA**: set `torch_dtype=torch.bfloat16` (or omit quantization_config entirely), keep `bf16=True`, drop the dead try/except fallback. This runs reliably on AMD Radeon (e.g., 7900XTX 24GB / MI-series). Note VRAM in a comment. Keep `HF_ENDPOINT` mirror and `ROCBLAS_USE_HIPBLASLT=1`.

### Bug B — `src/chat_app.py` YAML parsing (verify & fix)
- The NL→DSL flow feeds LLM output into a YAML loader. Confirm `chat_app.py` **strips ```yaml / ``` fences** before `yaml.safe_load`. If it doesn't, any LLM reply wrapped in a code fence crashes the whole pipeline. Add robust fence-stripping + error message to the user (not a traceback).

### Bug C — `src/dsl/transpiler.py` Freqtrade output correctness (verify & fix)
- Generate the Freqtrade `IStrategy` class and confirm it actually compiles and runs a minimal backtest. Check that `populate_indicators`, `populate_entry_trades`, `populate_exit_trades` are correctly wired, indicator names match `src/tools/indicators.py`, and the timeframe/parameters from the DSL are passed through. Add a test in `tests/test_transpiler.py` that writes the generated strategy to a temp file and imports/compiles it.

### Bug D — Secrets hygiene (verify)
- `src/tools/paper_trade.py` and `src/api.py` must read API keys / Testnet secrets from **environment variables** (or a `.env` not committed), never hardcoded. Grep for any literal key/token and remove it; add a `.env.example` with placeholders. Confirm `.gitignore` covers `.env`.

### Bug E — Backtest engine is single-position & metrics are wrong (VERIFIED by code inspection)
- `_simulate_trades()` keeps only ONE `position: float` (runner.py L176) and the entry gate is `if position == 0 and row.get("enter_long")` (L222) → at most one open trade ever. But `max_open_trades` is read from the DSL (L97) and passed in (L170) yet **never enforced**. A DSL declaring `max_open_trades: 3` silently runs single-position — any judge setting >1 will spot it instantly. **Fix:** hold a list/array of open positions capped at `max_open_trades`, size each by `risk.per_trade_pct` of current equity.
- Sharpe annualization hardcoded `np.sqrt(252 * 24)` (L299) assumes 1h candles. Derive periods/year from the strategy timeframe: `15m`→`252*24*4`, `1h`→`252*24`, `4h`→`252*6`, `1d`→`252`.
- No buy-and-hold benchmark. Add a `buy_hold_return` field (first→last price) and report strategy vs benchmark in the analysis.
- `df.eval(py_expr)` (L159) executes LLM-generated expressions → injection risk. Replace with a restricted evaluator (whitelist of columns + arithmetic ops, or `numexpr` on sanitized input).
- No slippage (only 0.1% fee). Add a configurable slippage (default 0.05%) applied on fill.

**DEFER these — do NOT implement unless ahead of schedule (risk missing 2026-08-06):**
- GARCH / Student-t / regime-switching synthetic data in `data_fetcher.py`: judges won't stress synthetic-data realism; DEFER.
- Sortino / Calmar / max-consecutive-loss / drawdown-duration metrics: add ONLY if P0+P1 are done; low marginal score.
- Walk-forward / parameter-sensitivity analysis: DEFER (3h+, over-engineering risk vs deadline).
- Chain-of-Thought training data + few-shot + live market-context injection in prompts: HIGH leverage BUT requires **retraining the LoRA on the GPU instance** (longest pole, regression risk). DEFER unless the current model emits invalid DSL — if it does, that is a P0 (fix transpiler/prompt first, don't retrain blindly).
- Swap FNSPID (stock news) → crypto news data: low priority; the model is fine-tuned for DSL *generation*, not sentiment; DEFER.

## 5. Gaps to fill (P1)

1. **Generate training data:** run `python training/data/prepare_dsl_pairs.py` (then `merge_datasets.py` if needed) to produce `training/data/processed/merged_train.jsonl`. Verify the JSONL has the `instruction`/`input`/`output`/`source` fields the trainer expects (see `load_training_data` in `train_qlora.py`).
2. **`docker/Dockerfile.vllm`:** create a vLLM serving image (base ROCm PyTorch, install vLLM, `vllm serve`merged LoRA). Match what `serve_vllm.sh` expects.
3. **`dify/workflows/trading_agent.yml`:** export (or hand-author) the Dify workflow YAML that calls the unified API (`src/api.py` / `dify/tools/trading_api_openapi.yml`).
4. **Fill `[TODO]` placeholders** in `technical_report.md` and `README.md` (team name, members, affiliation, hackathon track). Keep English.
5. **Demo video:** script + record a 3–5 min walkthrough (NL query → DSL → backtest → report; show `rocm-smi` in a corner to evidence AMD). Save to `demos/`.

## 6. Acceptance criteria (definition of done)
- [ ] `pytest tests/` — all 25 tests pass.
- [ ] A backtest with `max_open_trades: 3` actually opens **up to 3 concurrent positions** (single-position bug fixed); a regression test in `tests/` asserts this.
- [ ] Sharpe ratio matches the strategy timeframe (15m/1h/4h/1d give different annualization), and the report shows a buy-and-hold benchmark line.
- [ ] `bash scripts/verify_e2e.sh` runs green using **synthetic data only** (no network needed) — this proves the NL→DSL→backtest→report chain works offline.
- [ ] `train_qlora.py` at least **constructs and launches** on ROCm without crashing (actual multi-epoch training is done on the GPU cloud instance; locally just prove it imports and the args/config are valid, e.g. `python train_qlora.py --help` + a dry import).
- [ ] `docker/Dockerfile.vllm` builds (`docker build -f docker/Dockerfile.vllm .` if docker available; else at least `hadolint`/syntax-clean).
- [ ] No hardcoded secrets; `.env.example` present; `.gitignore` covers `.env`.
- [ ] `technical_report.md` and `README.md` have zero `[TODO]` placeholders.
- [ ] Demo video exists in `demos/`.

## 7. Working rules
- Make **one logical change per commit**; commit with a clear message; **push** after each milestone (remote is `origin/main`).
- Do NOT refactor for the sake of it. This is a hackathon demo, not production — prefer minimal, correct fixes over rewrites.
- Do NOT add new dependencies unless strictly required; if you do, update `requirements.txt` and note it.
- If you cannot verify something on this machine (e.g., real GPU training, Dify UI export), implement the code/asset, document exactly what must be done on the cloud instance, and flag it clearly in your final report.
- At the end, report in **under 200 words**: what you changed, what passes, and what still needs the GPU cloud instance or manual steps.
