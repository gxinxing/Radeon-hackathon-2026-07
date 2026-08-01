# Track 3 Demo Script: AMD ROCm Humanoid Soccer Platform

## Demo Overview
This script demonstrates the failure recovery and OOD evaluation platform for humanoid soccer on AMD ROCm GPU.

## Prerequisites
- Remote GPU access: `ssh -i ~/.ssh/id_ed25519 -p 31036 root@***REMOVED***`
- Python venv: `/opt/venv/bin/python3`
- Genesis: 1.3.1 (physics simulation)
- PyTorch: 2.9.1+gitff65f5b (HIP backend)

## Demo Steps

### Step 1: Verify GPU and Environment
```bash
ssh -i ~/.ssh/id_ed25519 -p 31036 root@***REMOVED***
rocm-smi | head -15
/opt/venv/bin/python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.hip)"
# Expected: 2.9.1+gitff65f5b True 7.2.53211-e1a6bc5663
```

### Step 2: Verify Model Assets
```bash
ls -lh /persistent/track3/models/base/t1_walk.pt
ls -lh /persistent/track3/models/onnx/chase_v8_policy.onnx
sha256sum /persistent/track3/models/base/t1_walk.pt
# Expected: ef1d61e19082b83405f4320a08f4cfc2d7d7f003ed3790dab013778ba442dec7
```

### Step 3: Import Test
```bash
cd /workspace/radeon-repo
/opt/venv/bin/python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src')
from soccer_env_hierarchical import SoccerEnvHierarchical
from match_3v3.policy import SharedRLPolicy, RulePolicy
from match_3v3.scene import PlayerState, BallState, Team, Role
print('All imports OK')
"
```

### Step 4: Single-Robot Verification
```bash
cd /workspace/radeon-repo
/opt/venv/bin/python3 -c "
import genesis as gs
gs.init(backend=gs.gpu, logging_level='warning')
print('Genesis OK on AMD GPU')
"
```

### Step 5: Run 3v3 RL vs Rule Match
```bash
cd /workspace/radeon-repo
bash run_3v3_onnx.sh
# This launches:
#   - 1 match coordinator (port 9882)
#   - 3 RL workers (Team A, ONNX chase_v8)
#   - 3 Rule workers (Team B, geometric)
# Duration: ~90s (including Genesis kernel compilation)
```

### Step 6: Analyze Match Results
```bash
python3 /tmp/analyze_match.py /persistent/track3/match_logs/<latest>.json
# Shows: goals, falls, recoveries, ball possession, final positions
```

### Step 7: Batch Matches
```bash
bash /tmp/run_batch_3v3.sh 5 rl_vs_rule models/chase_v8_policy.onnx
# Runs 5 sequential RL vs Rule matches
# Each match: ~70s (5s setup + 25s match + 40s cleanup)
```

### Step 8: View Training Evidence
```bash
# Training log (300/300 iterations complete)
tail -20 /persistent/track3/logs/train_chase_v7.log

# GPU utilization during training
head -10 /persistent/track3/benchmark/gpu_samples.csv

# Training TensorBoard
ls /persistent/track3/tensorboard/
```

## Key Metrics to Show
1. **GPU Utilization:** 93-100% during training (gpu_samples.csv)
2. **Training Complete:** 300/300 iterations, reward=23.94
3. **Match Stability:** 0 abnormal exits in 6 matches
4. **Recovery Rate:** 88.8% (RL vs Rule) vs 55.7% (rule vs rule baseline)
5. **6 Workers Connected:** All 6 Genesis workers connect and complete matches

## Demo Talking Points
- "The platform runs 6 concurrent Genesis physics simulations on AMD ROCm GPU"
- "RL policy provides locomotion and chase direction; rule layer handles tactics"
- "Robots fall and recover during matches — episodes don't terminate on falls"
- "88.8% recovery rate demonstrates the failure recovery capability"
- "The platform is designed for OOD evaluation, not just winning matches"
- "All training was done on AMD Radeon Graphics with ROCm 7.2"
