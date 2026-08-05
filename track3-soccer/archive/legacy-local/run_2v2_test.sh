#!/bin/bash
# Minimal 2-worker test: verify coordinator<->worker protocol works
cd /workspace/radeon-repo
PORT=9892
ONNX=models/chase_v8_policy.onnx
pkill -f match_worker 2>/dev/null; pkill -f match_coordinator 2>/dev/null; sleep 2

echo '=== Starting 2-worker test ==='
/opt/venv/bin/python match_coordinator.py --port $PORT --n-teams 1 --duration 25.0 --log-dir match_logs > /tmp/t2_coord.log 2>&1 &
sleep 5

/opt/venv/bin/python match_worker.py --role A_attacker --has-ball --port $PORT --onnx $ONNX --init-pos -3 0 0.7 > /tmp/t2_a1.log 2>&1 &
/opt/venv/bin/python match_worker.py --role B_attacker --port $PORT --init-pos 3 0 0.7 > /tmp/t2_b1.log 2>&1 &

echo '2 workers launched. Waiting 180s...'
sleep 180

echo '=== COORD ==='
tail -5 /tmp/t2_coord.log
echo '=== A1 ==='
tail -8 /tmp/t2_a1.log
echo '=== B1 ==='
tail -5 /tmp/t2_b1.log
echo '=== LOGS ==='
ls -lht match_logs/ | head -3
