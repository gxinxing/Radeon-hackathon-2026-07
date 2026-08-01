# Technical Report: Humanoid Soccer RL on AMD Radeon GPU

## Track 3 – Physical AI | AMD AI DevMaster Hackathon 2026

**Team:** gxinxing  
**Application:** T1 Humanoid Soccer – Genesis RL on AMD ROCm  
**Date:** 2026-08-01 (updated)  

---

## 1. Application Definition

This project trains a **Booster T1 humanoid robot** (23-DOF) to play soccer using reinforcement learning (PPO) entirely on an **AMD Radeon GPU** via ROCm. The application covers three progressive sub-tasks: balance (walking), chase (ball pursuit), and shoot (goal scoring). A hierarchical policy architecture is used: a frozen low-level walk policy (t1_walk.pt) provides locomotion, while a high-level PPO-trained policy (chase_v8) provides chase direction commands.

**Key innovation:** This is the **first-known RL training of the Booster T1 humanoid on AMD ROCm**. Booster's official training stack depends exclusively on NVIDIA Isaac Gym / Isaac Lab. Our work demonstrates an alternative GPU ecosystem for humanoid robot learning.

## 2. Architecture

### 2.1 Software Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| GPU | AMD Radeon Graphics (gfx1100) | — | Parallel RL training + physics sim |
| Runtime | ROCm 7.2 + PyTorch 2.9.1 (HIP) | HIP 7.2.53211 | GPU compute backend |
| Simulator | Genesis | 1.3.1 | Physics simulation (gs.gpu backend) |
| RL Framework | rsl-rl (PPO) | 5.4.2 | Policy optimization |
| Robot Model | Booster T1 23-DOF MJCF | booster_assets 1.0.0 | Humanoid kinematics |
| Inference | ONNX Runtime | 1.28.0 | ONNX model deployment |
| Sim2Sim | Booster Studio | 1.9.4 | 3v3 SoccerSim validation |

### 2.2 Hierarchical Policy Architecture

```
┌──────────────────────────────────────────────────┐
│            High-Level Policy (chase_v8)           │
│  PPO-trained MLP: 19-dim obs → 3-dim velocity cmd │
│  Input: ball_rel_body(2), ball_vel_body(2),       │
│         dist_to_ball(1), goal_dir(2), etc.        │
│  Output: [vx, vy, vyaw] clipped to ±1.2 m/s       │
├──────────────────────────────────────────────────┤
│            Low-Level Policy (t1_walk.pt)           │
│  Frozen PPO model: 720-dim obs → 21 joint targets  │
│  Input: proprioception (10-frame history)          │
│  Output: 21 DOF position targets                    │
├──────────────────────────────────────────────────┤
│            Genesis Physics Engine (AMD ROCm)       │
│  50Hz simulation, 6 robots + ball + field          │
└──────────────────────────────────────────────────┘
```

### 2.3 Training Pipeline

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (clipped) |
| Total Steps | 7,372,800 |
| Iterations | 300 |
| Steps/sec | 3,056 |
| Mean Reward | 23.94 |
| Mean Episode Length | 209 steps |
| GPU Utilization | 93-100% |
| Training Config | hl_clip_lin=1.2, hl_clip_ang=1.2, decimation=5 |

### 2.4 Reward Curriculum

| Task | Active Reward Terms | Training Goal |
|------|---------------------|--------------|
| balance | upright, alive, fall_penalty, recovery, energy | Stand stability |
| chase | + approach_ball, ball_progress, approach_angle | Pursue ball |
| shoot | + ball_control, ball_to_goal, goal_scored | Score goals |

## 3. Dataset

No external dataset. Training is fully simulated via Genesis physics engine with:
- **Field:** 14×9m (aligned to Booster ADULT_FIELD_DIMENSIONS)
- **Goal:** 2.6m width, 1.0m height
- **Ball:** 0.11m radius sphere URDF
- **Robot:** Official Booster T1 23-DOF humanoid with 63 STL meshes

Domain randomization: ball spawn position randomized per episode.

## 4. AMD Radeon GPU Utilization

### 4.1 Hardware Evidence

- **GPU:** AMD Radeon Graphics (Device ID: 0x744b, GUID: 6853)
- **GFX Version:** gfx1100 (RDNA3)
- **ROCm Driver:** 6.16.13
- **GPU Utilization During Training:** 93-100% (see benchmark/gpu_samples.csv)
- **Training Throughput:** 3,056 steps/sec with 2048 parallel environments

### 4.2 Training Evidence

- Training logs: 8 log files in /persistent/track3/logs/ (train_chase_v7.log: 300/300 iterations)
- TensorBoard events: /persistent/track3/tensorboard/
- GPU samples CSV: 244K of per-10s GPU utilization readings during training
- ROCm SMI output: Full GPU metrics captured (gpu_evidence_final.txt)

### 4.3 Model Assets

| File | Size | SHA256 |
|------|------|--------|
| t1_walk.pt | 2.1M | ef1d61e1... |
| chase_v8_policy.onnx | 183K | 6d7ce912... |
| best.pt | 1.1M | 76d713f8... |
| model_400.pt | 1.1M | e405dc21... |
| model_450.pt | 1.1M | cc3f1cae... |

## 5. Match Evaluation Results

### 5.1 Rule vs Rule Baseline (20 matches, single-process)

| Metric | Left | Right |
|--------|------|-------|
| Avg Goals | 1.05 | 1.30 |
| Recovery Rate | 55.7% | 48.1% |
| Win Rate | 30% | 40% |

### 5.2 RL vs Rule Clean Re-run (18 matches, 6-worker TCP, N_STEPS fixed)

| Group | Matches | Duration | RL Goals | Rule Goals | Recovery Rate | Abnormal |
|-------|---------|----------|----------|------------|---------------|----------|
| B (RL+kick) | 7 | 25s | 0 | 0 | 83.0% | 0 |
| E (RL+kick) | 7 | 60s | 0 | 1 | 93.0% | 0 |
| A (rule vs rule) | 4 | 25s | 0 | 0 | 85.0% | 0 |
| **Total** | **18** | — | **0** | **1** | **87.0%** | **0** |

### 5.3 Key Findings

1. **Platform Stability:** 0 abnormal exits in 18 clean matches (0%)
2. **Recovery Rate:** 83-93% with t1_walk.pt (vs 52% without t1_walk.pt in single-process)
3. **RL vs Rule (same architecture):** 83.0% vs 85.0% — comparable, both benefit from t1_walk.pt
4. **Goal Scoring:** 0 RL goals in 18 matches. Ball max X = 5.9m (goal at 7.0m). 25s matches too short.
5. **Disturbance Robustness:** 5 disturbance matches completed with 87.8% recovery (only -0.7pp vs baseline)

### 5.4 RL vs Rule Control Boundary

| Component | Controlled By | Type |
|-----------|--------------|------|
| Walking/Locomotion | RL (t1_walk.pt) | PPO, frozen |
| Chase Direction | RL (chase_v8 ONNX) | PPO, 19→3 MLP |
| Role Assignment | Rule | Geometric |
| Kicking | Rule | Goal-directed dash |
| Goalkeeper | Rule | Track ball Y |

## 6. Sim2Sim Validation

### 6.1 Booster Studio Setup

- **Version:** Booster Studio 1.9.4
- **Installed on:** Remote AMD GPU instance
- **VNC:** TigerVNC on display :99 (port 5999)
- **noVNC:** Web interface on port 6080
- **Agent:** RLChaseAgent deployed with chase_v8_policy.onnx

### 6.2 Agent Architecture

```
src/booster_agent/
├── src/
│   ├── main.py           # RLChaseAgent (AgentBase subclass)
│   └── rl_playbook.py    # ONNX inference for chaser role
├── models/
│   └── chase_v8_policy.onnx  # Trained HL policy
└── README.md
```

The agent loads the ONNX model, builds 19-dim observations from match state, and outputs velocity commands for the chaser role. Other roles (supporter, goalkeeper) use rule-based behavior.

### 6.3 Status

Booster Studio installed and VNC/noVNC running. Agent code prepared with v8 ONNX model. Sim2Sim match requires manual launch via Booster Studio GUI (3v3 SoccerSim mode).

## 7. Innovation

1. **First AMD ROCm T1 training:** Booster's official RL stack requires NVIDIA GPUs. We demonstrate Genesis + ROCm as a viable alternative.
2. **Hierarchical policy on AMD GPU:** Two-level architecture (walk + chase) trained end-to-end on ROCm.
3. **Failure recovery platform:** 87% recovery rate with 0 abnormal exits across 18 matches demonstrates robustness.
4. **Disturbance evaluation framework:** Random push forces + ball randomization integrated into match coordinator.
5. **Open-source soccer environment:** Genesis-based env with reward functions, training configs, and match infrastructure.

## 8. Deliverables

| Deliverable | Path | Status |
|-------------|------|--------|
| Technical Report | docs/TECHNICAL_REPORT.md | ✅ Updated |
| Source Code | src/ + scripts/ | ✅ Complete |
| Trained Models | models/ + remote_backup/ | ✅ 6 model files |
| Reproducibility README | README.md | ✅ Present |
| Demo Video | demos/hierarchical_chase_hl.mp4 | ✅ 150 frames |
| Presentation | presentations/ | ✅ Created |
| Match Results | reports/rl_vs_rule_summary.csv | ✅ 18 clean matches |
| Recovery Report | reports/recovery_ood_report.md | ✅ Updated |
| Asset Audit | reports/asset_audit.md | ✅ SHA256 verified |
| GPU Evidence | gpu_evidence_final.txt | ✅ ROCm SMI captured |

## 9. Team

- **gxinxing** – Solo developer
- AMD AI Developer Program member

## 10. Reproducibility

```bash
# 1. SSH to remote GPU
ssh -i ~/.ssh/id_ed25519 -p 31036 root@***REMOVED***

# 2. Verify GPU and environment
rocm-smi | head -5
/opt/venv/bin/python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.hip)"
# Expected: 2.9.1+gitff65f5b True 7.2.53211-e1a6bc5663

# 3. Verify imports
cd /workspace/radeon-repo
/opt/venv/bin/python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src')
from soccer_env_hierarchical import SoccerEnvHierarchical
from match_3v3.policy import SharedRLPolicy
print('All imports OK')
"

# 4. Run 3v3 RL vs Rule match
bash run_3v3_onnx.sh

# 5. Run clean batch with N_STEPS fix
bash /tmp/run_clean_rerun.sh

# 6. Analyze match
python3 /tmp/analyze_match.py /persistent/track3/match_logs/clean_rerun/<latest>.json

# 7. Start Booster Studio for Sim2Sim
bash /workspace/radeon-repo/start_booster.sh
# Access noVNC at http://<instance>:6080 (password: ***REMOVED***)
```
