#!/bin/bash
set -euo pipefail

PYTHON=/opt/venv/bin/python
PROJECT=/workspace/radeon-repo/track2-agentic-ai
cd "$PROJECT"

export HF_ENDPOINT=https://hf-mirror.com
export ROCBLAS_USE_HIPBLASLT=1
export HIP_VISIBLE_DEVICES=0
export VLLM_USE_TRITON_FLASH_ATTN=0

log() { echo "[$(date +%H:%M:%S)] $1"; }

# ===== PHASE 0: Wait for training to complete =====
log "=== PHASE 0: Waiting for QLoRA training to complete ==="
while pgrep -f train_qlora.py > /dev/null 2>&1; do
    STEP=$(grep -oP '\d+/\d+' /tmp/qlora_train.log 2>/dev/null | tail -1 || echo "?")
    log "Training in progress: step $STEP ..."
    sleep 60
done
log "Training process finished."

# Check if adapter was saved
ADAPTER_DIR="models/qwen-trader-lora/final"
if [ ! -d "$ADAPTER_DIR" ]; then
    log "ERROR: LoRA adapter not found at $ADAPTER_DIR"
    log "Last 20 lines of training log:"
    tail -20 /tmp/qlora_train.log
    exit 1
fi
log "LoRA adapter found at $ADAPTER_DIR"

# ===== PHASE 1: Merge LoRA weights =====
log "=== PHASE 1: Merging LoRA weights ==="
BASE_MODEL=$(ls -d models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$BASE_MODEL" ]; then
    BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
fi
log "Base model: $BASE_MODEL"

$PYTHON training/scripts/merge_lora.py \
    --base-model "$BASE_MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --output-path models/qwen-trader-merged 2>&1

if [ ! -f "models/qwen-trader-merged/config.json" ]; then
    log "ERROR: Merged model not saved properly"
    exit 1
fi
log "Merged model saved to models/qwen-trader-merged"

# ===== PHASE 2: Start vLLM =====
log "=== PHASE 2: Starting vLLM server ==="
fuser -k 8000/tcp 2>/dev/null || true
sleep 2

nohup $PYTHON -m vllm.entrypoints.openai.api_server \
    --model models/qwen-trader-merged \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.50 \
    --dtype float16 \
    --trust-remote-code \
    --enforce-eager > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
log "vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready (up to 5 minutes)
log "Waiting for vLLM to start..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models | grep -q "data"; then
        log "vLLM is ready!"
        break
    fi
    if [ "$i" -eq 60 ]; then
        log "ERROR: vLLM failed to start within 5 minutes"
        tail -30 /tmp/vllm.log
        exit 1
    fi
    sleep 5
done

# ===== PHASE 3: Run NL->DSL evaluation =====
log "=== PHASE 3: Running NL->DSL quality evaluation ==="
# Start backtest API for full pipeline test
fuser -k 8080/tcp 2>/dev/null || true
sleep 1
nohup $PYTHON -m uvicorn src.api:app --host 0.0.0.0 --port 8080 > /tmp/api.log 2>&1 &
API_PID=$!
sleep 3

$PYTHON scripts/eval_nl_to_dsl.py --vllm-url http://localhost:8000/v1 2>&1 | tee /tmp/eval_results.txt
log "Evaluation complete. Results saved to /tmp/eval_results.txt"

# ===== PHASE 4: Summary =====
log "=== PIPELINE COMPLETE ==="
log "Training log:   /tmp/qlora_train.log"
log "vLLM log:       /tmp/vllm.log"
log "Eval results:   /tmp/eval_results.txt"
log "Merged model:   models/qwen-trader-merged/"
log "vLLM running:  http://localhost:8000/v1 (PID $VLLM_PID)"
log "API running:   http://localhost:8080 (PID $API_PID)"
