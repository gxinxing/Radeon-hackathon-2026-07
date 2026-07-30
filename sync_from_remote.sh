#!/bin/bash
# Pull remote commits and push to GitHub.
# Runs every 15 minutes via local launchd/cron.

REMOTE_HOST="radeon"
REMOTE_REPO="/workspace/radeon-repo"
LOCAL_REPO="/Users/simon/Documents/01_AI and Code Development/​Radeon-hackathon-2026-07"

LOG="$LOCAL_REPO/sync.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Local sync start ===" >> "$LOG"

# 1. Trigger remote commit
ssh "$REMOTE_HOST" "bash $REMOTE_REPO/auto_sync.sh" >> "$LOG" 2>&1

# 2. Rsync key files from remote to local (training logs, benchmark, runs)
rsync -avz --exclude='*.pt' --exclude='__pycache__' --exclude='.git' \
    "$REMOTE_HOST:$REMOTE_REPO/track3-data/" "$LOCAL_REPO/track3-data/" >> "$LOG" 2>&1

rsync -avz --exclude='*.bin' --exclude='*.safetensors' --exclude='__pycache__' \
    "$REMOTE_HOST:$REMOTE_REPO/track2-agentic-ai/data/" "$LOCAL_REPO/track2-agentic-ai/data/" >> "$LOG" 2>&1

# 3. Copy latest training log
rsync -avz "$REMOTE_HOST:$REMOTE_REPO/train_v8.log" "$LOCAL_REPO/train_v8.log" >> "$LOG" 2>&1

# 4. Commit and push from local
cd "$LOCAL_REPO"
git add -A >> "$LOG" 2>&1
if git diff --cached --quiet; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No changes to push." >> "$LOG"
else
    git commit -m "sync: remote training data [$(date '+%m-%d %H:%M')]" >> "$LOG" 2>&1
    git push origin main >> "$LOG" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pushed to GitHub." >> "$LOG"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Sync done ===" >> "$LOG"
