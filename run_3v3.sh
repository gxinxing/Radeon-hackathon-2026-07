#!/bin/bash
# Launch 3v3 match: 1 ONNX agent + 2 rule allies vs 3 rule opponents
# Usage: bash run_3v3.sh [onnx_path] [duration]
set -e

ONNX="${1:-models/chase_v8_policy.onnx}"
DURATION="${2:-25}"
PORT=9878
PYTHON="/opt/venv/bin/python"

cd /workspace/radeon-repo

echo "============================================"
echo "  3v3 Match: ONNX+2 Allies vs 3 Opponents"
echo "============================================"
echo "  ONNX: $ONNX"
echo "  Duration: ${DURATION}s"
echo "  Port: $PORT"
echo ""

mkdir -p match_logs

$PYTHON match_coordinator.py --port $PORT --duration $DURATION --n-teams 2 > /tmp/coord.log 2>&1 &
COORD_PID=$!
echo "Coordinator PID: $COORD_PID"
sleep 5

# Team 1 (Left, attacks +x) — A_attacker has ONNX + ball authority
$PYTHON match_worker.py --role A_attacker --has-ball --port $PORT --onnx "$ONNX" --init-pos 1.0 0.0 0.7 > /tmp/worker_0.log 2>&1 &
$PYTHON match_worker.py --role A_defender --port $PORT --onnx "$ONNX" --init-pos -1.0 1.5 0.7 > /tmp/worker_1.log 2>&1 &
$PYTHON match_worker.py --role A_keeper --port $PORT --onnx "$ONNX" --init-pos -5.0 0.0 0.7 > /tmp/worker_2.log 2>&1 &

# Team 2 (Right, attacks -x) — rule-based, NO --has-ball
$PYTHON match_worker.py --role B_attacker --port $PORT --init-pos -1.0 -1.0 0.7 > /tmp/worker_3.log 2>&1 &
$PYTHON match_worker.py --role B_defender --port $PORT --init-pos 3.5 -1.5 0.7 > /tmp/worker_4.log 2>&1 &
$PYTHON match_worker.py --role B_keeper --port $PORT --init-pos 5.5 0.0 0.7 > /tmp/worker_5.log 2>&1 &

echo "All 6 workers started. Waiting for match..."
wait $COORD_PID 2>/dev/null
sleep 3
kill $(jobs -p) 2>/dev/null

echo "=== SUMMARY ==="
for i in 0 1 2 3 4 5; do
    echo "--- Worker $i ---"
    grep -E 'step|Finished|Connected|END' /tmp/worker_${i}.log 2>/dev/null | tail -5
done
echo "=== COORDINATOR ==="
cat /tmp/coord.log
echo "=== LOG FILES ==="
ls -lh match_logs/ 2>/dev/null
