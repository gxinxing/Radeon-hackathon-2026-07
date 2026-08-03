#!/bin/bash
# Launch 1v1 match: RL agent vs rule opponent
# Usage: bash run_1v1.sh [model_path] [duration]
set -e

MODEL="${1:-runs/hierarchical_soccer_chase_hl/model_1894.pt}"
DURATION="${2:-20}"
PORT=9876

cd /workspace/amd-physical-ai-soccer

echo "============================================"
echo "  1v1 Match: RL Agent vs Rule Opponent"
echo "============================================"
echo "  Model: $MODEL"
echo "  Duration: ${Duration}s"
echo "  Port: $PORT"
echo ""

# Start coordinator in background
/opt/venv/bin/python match_coordinator.py --port $PORT --duration $DURATION &
COORD_PID=$!
echo "Coordinator PID: $COORD_PID"
sleep 1

# Start RL agent (has ball, from init position [1, 0])
/opt/venv/bin/python match_worker.py --role agent --has-ball --model "$MODEL" --port $PORT --init-pos 1.0 0.0 0.7 &
AGENT_PID=$!
echo "Agent PID: $AGENT_PID"
sleep 1

# Start rule opponent (no ball, from init position [-3, 0])
/opt/venv/bin/python match_worker.py --role opponent --port $PORT --init-pos -3.0 0.0 0.7 &
OPP_PID=$!
echo "Opponent PID: $OPP_PID"

# Wait for all to finish
wait $COORD_PID 2>/dev/null
echo "Coordinator finished"

# Kill any remaining workers
kill $AGENT_PID $OPP_PID 2>/dev/null
echo "Match complete"
