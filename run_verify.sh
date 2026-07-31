#!/bin/bash
cd /workspace/radeon-repo
PORT=9884
M=models/chase_v8_policy.onnx
pkill -f match_worker 2>/dev/null; pkill -f match_coordinator 2>/dev/null; sleep 3

/opt/venv/bin/python match_coordinator.py --port $PORT --n-teams 2 --duration 25.0 --log-dir match_logs > /tmp/vc.log 2>&1 &
sleep 5

/opt/venv/bin/python match_worker.py --role A_attacker --has-ball --port $PORT --onnx $M --init-pos -3 0 0.7 > /tmp/va1.log 2>&1 &
/opt/venv/bin/python match_worker.py --role A_defender --port $PORT --onnx $M --init-pos -4 -1.5 0.7 > /tmp/va2.log 2>&1 &
/opt/venv/bin/python match_worker.py --role A_keeper --port $PORT --onnx $M --init-pos -6 0 0.7 > /tmp/va3.log 2>&1 &
/opt/venv/bin/python match_worker.py --role B_attacker --has-ball --port $PORT --init-pos 3 0 0.7 > /tmp/vb1.log 2>&1 &
/opt/venv/bin/python match_worker.py --role B_defender --port $PORT --init-pos 4 -1.5 0.7 > /tmp/vb2.log 2>&1 &
/opt/venv/bin/python match_worker.py --role B_keeper --port $PORT --init-pos 6 0 0.7 > /tmp/vb3.log 2>&1 &
echo 'Workers started'
