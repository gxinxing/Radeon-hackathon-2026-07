#!/bin/bash
<<<<<<< HEAD
# Launch 3v3 match: 1 RL agent + 2 rule allies vs 3 rule opponents
# Usage: bash run_3v3.sh [model_path] [duration]
set -e

MODEL="${1:-runs/hierarchical_soccer_chase_hl/model_1894.pt}"
=======
# Launch 3v3 match: 1 ONNX agent + 2 rule allies vs 3 rule opponents
# Usage: bash run_3v3.sh [onnx_path] [duration]
set -e

ONNX="${1:-models/chase_v8_policy.onnx}"
>>>>>>> track3-honest
DURATION="${2:-25}"
PORT=9878
PYTHON="/opt/venv/bin/python"

<<<<<<< HEAD
cd /workspace/amd-physical-ai-soccer

echo "============================================"
echo "  3v3 Match: RL+2 Allies vs 3 Opponents"
echo "============================================"
echo "  Model: $MODEL"
=======
cd /workspace/radeon-repo

echo "============================================"
echo "  3v3 Match: ONNX+2 Allies vs 3 Opponents"
echo "============================================"
echo "  ONNX: $ONNX"
>>>>>>> track3-honest
echo "  Duration: ${DURATION}s"
echo "  Port: $PORT"
echo ""

mkdir -p match_logs

<<<<<<< HEAD
# Start coordinator (n-teams=2 → 6 workers)
$PYTHON match_coordinator.py --port $PORT --duration $DURATION --n-teams 2 > /tmp/coord.log 2>&1 &
COORD_PID=$!
echo "Coordinator PID: $COORD_PID"
sleep 2

# Team 1 (Left, attacks +x)
# Worker 0: RL agent (has ball authority, attacker position)
$PYTHON match_worker.py --role agent_rl --has-ball --model "$MODEL" --port $PORT --init-pos 1.0 0.0 0.7 > /tmp/worker_0.log 2>&1 &
W0=$!
echo "Worker 0 (RL agent, has ball): PID $W0"
sleep 2

# Worker 1: Rule ally (defender)
$PYTHON match_worker.py --role ally_def --port $PORT --init-pos -1.0 1.5 0.7 > /tmp/worker_1.log 2>&1 &
W1=$!
echo "Worker 1 (rule ally, defender): PID $W1"
sleep 1

# Worker 2: Rule ally (goalkeeper)
$PYTHON match_worker.py --role ally_gk --port $PORT --init-pos -5.0 0.0 0.7 > /tmp/worker_2.log 2>&1 &
W2=$!
echo "Worker 2 (rule ally, goalkeeper): PID $W2"
sleep 1

# Team 2 (Right, attacks -x)
# Worker 3: Rule opponent (attacker)
$PYTHON match_worker.py --role opp_att --port $PORT --init-pos -1.0 -1.0 0.7 > /tmp/worker_3.log 2>&1 &
W3=$!
echo "Worker 3 (rule opp, attacker): PID $W3"
sleep 1

# Worker 4: Rule opponent (defender)
$PYTHON match_worker.py --role opp_def --port $PORT --init-pos 3.5 -1.5 0.7 > /tmp/worker_4.log 2>&1 &
W4=$!
echo "Worker 4 (rule opp, defender): PID $W4"
sleep 1

# Worker 5: Rule opponent (goalkeeper)
$PYTHON match_worker.py --role opp_gk --port $PORT --init-pos 5.5 0.0 0.7 > /tmp/worker_5.log 2>&1 &
W5=$!
echo "Worker 5 (rule opp, goalkeeper): PID $W5"

echo ""
echo "All 6 workers started. Waiting for match..."
echo ""

# Wait for coordinator to finish (it sends END to all workers)
wait $COORD_PID 2>/dev/null
echo "Coordinator finished"

# Give workers a moment to receive END and clean up
sleep 3

# Kill any remaining workers
kill $W0 $W1 $W2 $W3 $W4 $W5 2>/dev/null

echo ""
=======
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

>>>>>>> track3-honest
echo "=== SUMMARY ==="
for i in 0 1 2 3 4 5; do
    echo "--- Worker $i ---"
    grep -E 'step|Finished|Connected|END' /tmp/worker_${i}.log 2>/dev/null | tail -5
done
<<<<<<< HEAD
echo ""
echo "=== COORDINATOR ==="
cat /tmp/coord.log
echo ""
=======
echo "=== COORDINATOR ==="
cat /tmp/coord.log
>>>>>>> track3-honest
echo "=== LOG FILES ==="
ls -lh match_logs/ 2>/dev/null
