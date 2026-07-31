#!/bin/bash
# Run 3v3 match with ONNX inference (Team A) vs rule-based (Team B)
cd /workspace/radeon-repo
PORT=9882
ONNX=/workspace/radeon-repo/models/chase_v8_policy.onnx
LOGDIR=/workspace/radeon-repo/match_logs

echo "=== 3v3 Match: ONNX (Team A) vs Rule (Team B) ==="
pkill -f match_worker 2>/dev/null; pkill -f match_coordinator 2>/dev/null; sleep 2

/opt/venv/bin/python /workspace/radeon-repo/match_coordinator.py \
    --port $PORT --n-teams 2 --duration 25.0 --log-dir $LOGDIR > /tmp/coord_onnx.log 2>&1 &
sleep 5

# Team A: 3 robots with ONNX model
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role A_attacker --has-ball --port $PORT --onnx $ONNX --init-pos -3 0 0.7 > /tmp/onnx_a1.log 2>&1 &
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role A_defender --port $PORT --onnx $ONNX --init-pos -4 -1.5 0.7 > /tmp/onnx_a2.log 2>&1 &
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role A_keeper --port $PORT --onnx $ONNX --init-pos -6 0 0.7 > /tmp/onnx_a3.log 2>&1 &

# Team B: 3 rule-based robots
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role B_attacker --has-ball --port $PORT --init-pos 3 0 0.7 > /tmp/onnx_b1.log 2>&1 &
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role B_defender --port $PORT --init-pos 4 -1.5 0.7 > /tmp/onnx_b2.log 2>&1 &
/opt/venv/bin/python /workspace/radeon-repo/match_worker.py \
    --role B_keeper --port $PORT --init-pos 6 0 0.7 > /tmp/onnx_b3.log 2>&1 &

echo "All 6 workers started. Waiting 90s..."
sleep 90

echo "=== COORD ==="
tail -3 /tmp/coord_onnx.log
echo "=== A_attacker (ONNX) ==="
tail -8 /tmp/onnx_a1.log
echo "=== B_attacker (Rule) ==="
tail -5 /tmp/onnx_b1.log
echo "=== MATCH LOGS ==="
ls -lht $LOGDIR/ | head -3
echo "=== DONE ==="
