# Track 3: Final Status Report

## AMD ROCm-Driven Humanoid Soccer Failure Recovery & OOD Training Platform

**Date:** 2026-08-01  
**Track:** Radeon Hackathon 2026 — Track 3  
**GPU:** AMD Radeon Graphics (gfx1100, ROCm 7.2)  

---

## 1. Project Status: ✅ Platform Operational

### Completed Items
- [x] t1_walk.pt locomotion policy trained and restored
- [x] chase_v6/v7/v8 HL chase policies trained via PPO on AMD ROCm GPU
- [x] ONNX export of chase policies (19-dim input → 3-dim velocity command)
- [x] 3v3 multi-agent match infrastructure (coordinator + 6 workers)
- [x] Rule-based opponent with role assignment (attacker/defender/goalkeeper)
- [x] 20 rule vs rule matches completed with full statistics
- [x] 6 RL vs Rule matches completed (1 initial + 5 batch)
- [x] ModuleNotFoundError fixed (import path corrected)
- [x] All matches run without abnormal exits
- [x] Match logs saved with per-step robot/ball trajectories
- [x] Fall detection and recovery statistics implemented
- [x] Asset audit with SHA256 hashes
- [x] GPU/ROCm evidence collected (rocm-smi, training logs, GPU samples)

### Incomplete Items
- [ ] Disturbance matches (framework ready, not executed)
- [ ] RL vs Rule goal scoring (0 goals in 6 matches)
- [ ] ONNX ROCm Execution Provider (CPU-only inference)
- [ ] 10-match extended statistical set (5 completed)

---

## 2. Policy Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  3v3 Match Coordinator                   │
│         (TCP socket sync, 50Hz, goal detection)          │
├──────────────┬──────────────┬──────────────┬────────────┤
│  Worker A1   │  Worker A2   │  Worker A3   │  Workers B │
│  (RL/ONNX)   │  (RL/ONNX)   │  (RL/ONNX)   │  (Rule)   │
├──────────────┼──────────────┼──────────────┼────────────┤
│ SharedRLPolicy│ SharedRLPolicy│ SharedRLPolicy│ RulePolicy│
│  (ONNX inference)│(ONNX)    │  (ONNX)      │ (geometric)│
├──────────────┼──────────────┼──────────────┼────────────┤
│ chase_v8.onnx│ chase_v8.onnx│ chase_v8.onnx│  N/A      │
│  (19→3 MLP)  │  (19→3 MLP)  │  (19→3 MLP)  │           │
├──────────────┼──────────────┼──────────────┼────────────┤
│ t1_walk.pt   │ t1_walk.pt   │ t1_walk.pt   │ t1_walk.pt│
│ (720→joints) │ (720→joints) │ (720→joints) │(720→joints)│
├──────────────┼──────────────┼──────────────┼────────────┤
│           Genesis Physics Engine (AMD ROCm GPU)         │
└─────────────────────────────────────────────────────────┘
```

### RL vs Rule Boundary
| Component | Controlled By | Type |
|-----------|--------------|------|
| Walking/Locomotion | RL (t1_walk.pt) | PPO-trained, frozen |
| Chase Direction | RL (chase_v8 ONNX) | PPO-trained, 19→3 MLP |
| Role Assignment | Rule | Geometric (closest to ball = attacker) |
| Kicking | Rule | Distance + alignment check |
| Goalkeeper | Rule | Track ball Y, stay on goal line |
| Defender | Rule | Position between ball and own goal |

---

## 3. Match Results Summary

### 3.1 Rule vs Rule (20 matches, control group)

| Metric | Left | Right |
|--------|------|-------|
| Avg Goals | 1.05 | 1.30 |
| Win Rate | 30% | 40% |
| Draw Rate | 30% | 30% |
| Avg Falls | 3.05 | 2.60 |
| Recovery Rate | 55.7% | 48.1% |
| Avg Shots | 3.6 | 3.65 |
| Shot Accuracy | 36.1% | 24.7% |

### 3.2 RL vs Rule (6 matches)

| Metric | RL Team | Rule Team |
|--------|---------|-----------|
| Avg Goals | 0.0 | 0.0 |
| Win Rate | 0% | 0% |
| Draw Rate | 100% | 100% |
| Avg Total Falls | 34.3 | (shared) |
| Avg Recovery Rate | 88.8% | (shared) |
| Abnormal Exits | 0 | 0 |
| Avg Sim Steps | 1241 | 1241 |

### 3.3 Honest Assessment

The RL policy (chase_v8) demonstrates:
- ✅ Stable chase behavior (ball moves towards opponent half in 4/6 matches)
- ✅ High recovery rate (88.8% vs 55.7% baseline)
- ✅ Zero abnormal exits in 6 consecutive matches
- ✅ No immediate episode termination on falls
- ❌ No goal scoring capability (lacks kick integration)
- ❌ Does not outperform rule-based policy in goals

**Project Positioning:** This is a **failure recovery and OOD evaluation platform**, not a competitive soccer policy. The RL component provides locomotion stability and chase direction, while the rule layer handles tactical decisions.

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (clipped) |
| Total Steps | 7,372,800 |
| Iterations | 300 |
| Steps/sec | 3,056 |
| Mean Reward | 23.94 |
| Mean Episode Length | 209 |
| Action Std (final) | 0.08 |
| Clip Range | 0.2 (v7), 0.6 (clip06) |
| HL Obs Dim | 19 |
| HL Action Dim | 3 |
| HL dt | 0.1s (decimation=5) |
| Walk Obs Dim | 720 |
| Walk Model | t1_walk.pt (frozen) |
| GPU | AMD Radeon (gfx1100) |
| GPU Util | 93-100% |

---

## 5. Model Assets

| File | Path (remote) | Size | SHA256 |
|------|----------------|------|--------|
| t1_walk.pt | /persistent/track3/models/base/t1_walk.pt | 2.1M | ef1d61e1... |
| chase_v7.onnx | /persistent/track3/models/onnx/chase_v7_policy.onnx | 183K | 06dd0c69... |
| chase_v8.onnx | /persistent/track3/models/onnx/chase_v8_policy.onnx | 183K | 6d7ce912... |
| best.pt | /persistent/track3/models/checkpoints/best.pt | 1.1M | 76d713f8... |
| model_400.pt | /persistent/track3/models/checkpoints/model_400.pt | 1.1M | e405dc21... |
| model_450.pt | /persistent/track3/models/checkpoints/model_450.pt | 1.1M | cc3f1cae... |

---

## 6. File Locations

### Local (amd-physical-ai-soccer/)
```
reports/asset_audit.md           — Remote asset audit with SHA256
reports/rl_vs_rule_report.json   — Full RL vs Rule match data
reports/rl_vs_rule_summary.csv   — CSV summary of 6 matches
reports/recovery_ood_report.md   — Recovery & OOD analysis
results/match_000-019.json        — 20 rule vs rule results
results/summary.json              — Rule vs rule summary
scripts/analyze_match.py          — Match log analyzer
scripts/run_batch_3v3.sh          — Batch match runner
gpu_evidence_final.txt            — ROCm SMI output
benchmark_final.txt              — GPU benchmark
gpu_stress_report.txt             — GPU stress test
```

### Remote (/persistent/track3/)
```
models/base/t1_walk.pt
models/onnx/chase_v6/v7/v8_policy.onnx
models/checkpoints/best.pt, model_400.pt, model_450.pt
logs/train_chase_v7.log (300/300 iterations complete)
logs/post_train.log
match_logs/match_20260801_14*.json (5 new RL vs Rule)
benchmark/gpu_samples.csv (GPU utilization during training)
benchmark/module_f_benchmark.json
benchmark/module_e_comparison.json
tensorboard/events.out.tfevents.*
demo/hierarchical_chase_hl.mp4
```

---

## 7. Reproduction Commands

```bash
# 1. SSH to remote GPU
ssh -i ~/.ssh/id_ed25519 -p 31036 root@***REMOVED***

# 2. Single RL vs Rule 3v3 match (25s)
cd /workspace/radeon-repo
bash run_3v3_onnx.sh

# 3. Batch 5 RL vs Rule matches
bash /tmp/run_batch_3v3.sh 5 rl_vs_rule models/chase_v8_policy.onnx

# 4. Analyze match log
python3 /tmp/analyze_match.py /persistent/track3/match_logs/<latest>.json

# 5. Rule vs Rule (single-process)
/opt/venv/bin/python scripts/match_eval_3v3.py --matches 20 --steps 1000 --seed 42

# 6. Verify environment
cd /workspace/radeon-repo
/opt/venv/bin/python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src')
from soccer_env_hierarchical import SoccerEnvHierarchical
from match_3v3.policy import SharedRLPolicy, RulePolicy
print('All imports OK')
"
```
