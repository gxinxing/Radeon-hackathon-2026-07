# Technical Report: Humanoid Soccer RL on AMD Radeon GPU

## Track 3 – Physical AI | AMD AI DevMaster Hackathon 2026

**Team:** gxinxing  
**Application:** T1 Humanoid Soccer – Genesis RL on AMD ROCm  
**Date:** 2026-07-19

---

## 1. Application Definition

This project trains a **Booster T1 humanoid robot** (23-DOF) to play soccer using reinforcement learning (PPO) entirely on an **AMD Radeon RX 7900 XT** GPU via ROCm. The application covers three progressive sub-tasks: balance, chase (ball pursuit), and shoot (goal scoring).

**Key innovation:** This is the **first-known RL training of the Booster T1 humanoid on AMD ROCm**. Booster's official training stack (`booster_gym`, `booster_train`) depends exclusively on NVIDIA Isaac Gym / Isaac Lab. Our work demonstrates an alternative GPU ecosystem for humanoid robot learning.

## 2. Architecture

### 2.1 Software Stack

| Layer | Technology | Purpose |
|---|---|---|
| GPU | AMD Radeon RX 7900 XT (gfx1100, 48GB VRAM) | Parallel RL training |
| Runtime | ROCm 7.2 + PyTorch 2.9.1 (HIP) | GPU compute backend |
| Simulator | Genesis 1.2.3 (`gs.amdgpu` backend) | Physics simulation |
| RL Framework | rsl-rl 5.4.2 (PPO) | Policy optimization |
| Robot Model | Booster T1 23-DOF MJCF (booster_assets) | Humanoid kinematics |
| Bridge | Custom distill pipeline | RL → behavioral rules |

### 2.2 Training Pipeline

```
soccer_agent.yaml (config)
    ↓
Genesis Scene (gs.amdgpu, 48GB VRAM)
    + T1_23dof.xml (MJCF humanoid)
    + ball.urdf + goal.urdf
    ↓
SoccerEnv (2048-4096 parallel envs)
    obs: [ang_vel, gravity, commands, dof_pos-default, dof_vel, last_action] × 10-frame history
    action: dof_targets = default + action * 0.25
    ↓
rsl-rl OnPolicyRunner (PPO)
    → model checkpoints (model_*.pt)
    → eval + distill → booster_distilled.py
```

### 2.3 Reward Curriculum

| Task | Active Reward Terms | Training Goal |
|---|---|---|
| balance | upright, alive, fall_penalty, recovery, energy | Stand stability |
| chase | + approach_ball | Pursue ball |
| shoot | + ball_control, ball_to_goal, goal_scored | Score goals |

## 3. Dataset

No external dataset. Training is fully simulated via Genesis physics engine with:
- **Field:** 14×9m (aligned to Booster ADULT_FIELD_DIMENSIONS)
- **Goal:** 2.6m width, matching real Booster pitch
- **Ball:** 0.11m radius sphere URDF
- **Robot:** Official Booster T1 23-DOF humanoid with 63 STL meshes

Domain randomization: ball spawn position randomized per episode (min 1.5m from robot).

## 4. AMD Radeon GPU Utilization

### 4.1 Hardware

- **GPU:** AMD Radeon RX 7900 XT (gfx1100, RDNA3)
- **VRAM:** 48GB (51.5GB total)
- **Driver:** ROCm 7.2

### 4.2 Throughput Performance

| Benchmark | Result |
|---|---|
| Matmul 1024² | 0.7 TFLOPS |
| Matmul 2048² | 11.2 TFLOPS |
| Matmul 4096² | 19.5 TFLOPS |
| Matmul 8192² | 18.9 TFLOPS |
| Sustained 8192² (60s) | 18.5 TFLOPS |
| Peak VRAM (stress) | 1.11 GB (matmul only) |

### 4.3 Training Throughput

| Configuration | Steps/sec | VRAM | Temp |
|---|---|---|---|
| 1× chase, 2048 envs | 44,129 | 11% | 43°C |
| 2× parallel (chase+shoot), 2048 envs each | 44k + 32k | 21% | 48°C |
| 3× parallel (balance+chase+shoot), 2048-4096 envs | ~76k combined | 60% | 58°C |
| GPU stress (8192² matmul) | 18.5 TFLOPS sustained | 63% | 54°C |

### 4.4 Multi-Task Parallelism

The RX 7900 XT's 48GB VRAM allows **3 concurrent training tasks** (balance + chase + shoot) with 6144-10240 total parallel environments. This is a key advantage for curriculum-based RL: instead of sequential training, all sub-tasks train simultaneously, reducing wall-clock time by ~3×.

## 5. Innovation

1. **First AMD ROCm T1 training:** Booster's official RL stack requires NVIDIA GPUs. We demonstrate Genesis + ROCm as a viable alternative.
2. **Booster T1 humanoid in Genesis:** Loaded official booster_assets T1_23dof MJCF with 29 DOFs, configured with real PD gains (kp=200, kd=5) matching booster_gym.
3. **Soccer-specific reward curriculum:** balance → chase → shoot progression with soccer field geometry matching real Booster pitch dimensions.
4. **Bridge distillation pipeline:** RL-learned behaviors distilled into behavioral rule constants for the Booster real-match SDK (sim3v3 framework).
5. **booster_deploy contract alignment:** Observation/action format aligned to booster_deploy's TorchScript deployment contract (10-frame history, action_scale=0.25).

## 6. Deliverables

- **Source code:** `amd-physical-ai-soccer/` (env, train, eval, benchmark, export, distill)
- **Trained models:** 5 complete training runs (1000 iterations each)
- **GPU evidence:** `gpu_evidence_final.txt`, `benchmark_final.txt`, `gpu_stress_report.txt`
- **Bridge pipeline:** `bridge/` (genesis_logger, distill, SPEC, booster_distilled)
- **Booster T1 assets:** Official 23-DOF model integrated
- **Reproducibility:** README with step-by-step cloud setup instructions

## 7. Team

- **gxinxing** – Solo developer
- AMD AI Developer Program member
- GitHub: github.com/gxinxing/Radeon-hackathon-2026-07

## 8. Reproducibility

```bash
# 1. Cloud setup (Radeon Cloud, ROCm/PyTorch template)
source /opt/venv/bin/activate
pip install genesis-world rsl-rl-lib

# 2. Verify GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.9.1+gitff65f5b True

# 3. Upload scaffold + T1 assets to /workspace/

# 4. Train
cd /workspace/amd-physical-ai-soccer
PYTHONPATH=. python scripts/train.py --task chase --num_envs 2048 --max_iterations 1000

# 5. Benchmark
python scripts/gpu_stress_test.py --duration 60

# 6. Eval + Distill
python scripts/eval.py -e t1_chase --task chase --num_envs 1 --steps 600
python bridge/distill.py --log bridge/rollout.jsonl --out bridge/booster_distilled.py
```
