#!/bin/bash
# Run 3v3 match: Team A (RL) vs Team B (rule-based)
cd /workspace/radeon-repo
PORT=9881
MODEL=/workspace/radeon-repo/runs/hierarchical_soccer_chase_hl/model_499.pt
LOGDIR=/workspace/radeon-repo/match_logs

echo "=== Starting 3v3 Match ==="
echo "Model: $MODEL"
echo "Port: $PORT"

# Kill any stale processes
pkill -f match_worker 2>/dev/null
pkill -f match_coordinator 2>/dev/null
sleep 2

# Start coordinator
/opt/venv/bin/python /workspace/radeon-repo/match_coordinator.py \
    --port $PORT --n-teams 2 --duration 25.0 --log-dir $LOGDIR \
    > /tmp/match_final_coord.log 2>&1 &
COORD_PID=$!
echo "Coordinator PID: $COORD_PID"
sleep 3

# Start Team A (3 RL robots)
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role A_attacker --has-ball --port $PORT \
    --model $MODEL --init-pos -3 0 0.7 > /tmp/match_final_a1.log 2>&1 &
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role A_defender --port $PORT \
    --model $MODEL --init-pos -4 -1.5 0.7 > /tmp/match_final_a2.log 2>&1 &
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role A_keeper --port $PORT \
    --model $MODEL --init-pos -6 0 0.7 > /tmp/match_final_a3.log 2>&1 &

# Start Team B (3 rule-based robots)
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role B_attacker --has-ball --port $PORT \
    --init-pos 3 0 0.7 > /tmp/match_final_b1.log 2>&1 &
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role B_defender --port $PORT \
    --init-pos 4 -1.5 0.7 > /tmp/match_final_b2.log 2>&1 &
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role B_keeper --port $PORT \
    --init-pos 6 0 0.7 > /tmp/match_final_b3.log 2>&1 &

echo "All 6 workers + coordinator started"
echo "Waiting 60s for match to complete..."
sleep 60

echo "=== COORD LOG ==="
tail -5 /tmp/match_final_coord.log
echo "=== A_attacker LOG ==="
tail -8 /tmp/match_final_a1.log
echo "=== B_attacker LOG ==="
tail -8 /tmp/match_final_b1.log
echo "=== MATCH LOGS ==="
ls -lht $LOGDIR/ | head -5
echo "=== DONE ==="
