# ⚽ Humanoid Robot Soccer Policy Training on AMD Radeon GPU

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%207.2-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1+ROCm-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Genesis](https://img.shields.io/badge/Genesis-1.3.1-blue)](https://genesis-embodied-ai.github.io/)
[![rsl_rl](https://img.shields.io/badge/rsl__rl-5.4.2-green)](https://github.com/leggedrobotics/rsl_rl)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2017-orange)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-Hackathon-lightgrey)](#license)

[English](./README.md) | [中文](./README_zh.md)

> Train humanoid robot soccer policies (balance, chase, shoot) using the Genesis physics
> engine and ROCm PyTorch on AMD Radeon GPUs — the **first AMD-GPU humanoid soccer training
> pipeline**, proving competitive robot policies can be trained without NVIDIA hardware.

**AMD AI DevMaster Hackathon 2026 — Track 3: Physical AI**

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Why This Project Exists](#-why-this-project-exists)
- [Architecture](#-architecture)
- [Key Results](#-key-results)
- [Prerequisites](#-prerequisites)
- [Setup](#-setup)
- [Usage](#-usage)
- [Reward Function Design](#-reward-function-design)
- [Distributed Multi-Robot Match](#-distributed-multi-robot-match-1v1--3v3)
- [Project Structure](#-project-structure)
- [Technical Stack](#-technical-stack)
- [Known Limitations](#-known-limitations)
- [Data Sources](#-data-sources)
- [Team](#-team)
- [License](#-license)

---

## 🌟 Highlights

| Metric | Value | Note |
|--------|-------|------|
| Reward improvement | **-24 → +112** | coop_hl multi-agent training (24-dim obs, 500 iter) |
| Action std (fixed) | **5.78 → 0.09** | entropy_coef 0.01→0.003 resolved noise blowup |
| Episode length | **19 → 208 steps** | From instant-fall to sustained walking |
| Total goals | **2,058** | 1,407% improvement over v7 (146 goals) |
| Ball distance (min) | **3.07m → 0.32m** | Multi-agent coop training with teammate/opponent awareness |
| ONNX inference | **0.4 ms** | 19→3 dim, real-time capable (46,467 params) |
| 1v1 match | **200 steps, ball displaced 20m** | ONNX inference verified on AMD Radeon GPU |
| Training throughput | **4,618 steps/s** (peak) | 2048 parallel envs on AMD Radeon (51 GB VRAM) |

---

## 🔍 Why This Project Exists

Booster Robotics' official RL training frameworks (Booster Gym / Booster Train) depend on
NVIDIA Isaac Gym and Isaac Lab, which require CUDA and NVIDIA GPUs. This project builds an
alternative training pipeline that runs **entirely on AMD Radeon GPUs** using:

- **Genesis** — GPU-accelerated physics simulation (AMD Radeon compatible)
- **ROCm PyTorch** — AMD's GPU compute platform (replaces CUDA)
- **rsl_rl** — PPO-based reinforcement learning runner

The result is the first AMD-GPU humanoid soccer training pipeline, proving that competitive
robot policies can be trained without NVIDIA hardware.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                      │
│                                                          │
│  ┌──────────────┐    ┌───────────────┐                  │
│  │  High-Level   │    │  Low-Level    │                  │
│  │  PPO Policy   │───▶│  Frozen Walk  │──▶ PD Control   │
│  │  (19→3 dims)  │    │  (720→21)     │    (50 Hz)       │
│  │  vx,vy,wz     │    │  t1_walk.pt   │                  │
│  └──────┬───────┘    └───────────────┘                  │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐     │
│  │  Genesis Physics Engine (AMD Radeon GPU)        │     │
│  │  Soccer field + T1 humanoid + ball               │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  Reward: approach_ball(10) + ball_control(2)             │
│          + ball_to_goal(8) + upright(0.5) - fall         │
│          + approach_angle(3) + directed_contact(5)        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                   Validation Pipeline                     │
│                                                          │
│  Trained .pt  ──▶  ONNX export  ──▶  1v1 Match           │
│  checkpoint                       (ONNX Runtime, Genesis)  │
└──────────────────────────────────────────────────────────┘
```

### Hierarchical Policy Design

The policy is split into two levels:

| Level | Observation | Action | Frequency | Model |
|-------|------------|--------|-----------|-------|
| High-level | 19-dim (ball pos/vel, goal dir, proprioception) | 3-dim (vx, vy, wz) | 10 Hz | Trainable PPO |
| Low-level | 720-dim (10-frame proprioception history) | 21-dim (joint targets) | 50 Hz | Frozen `t1_walk.pt` |

This design solves a key problem: the original flat policy (720-dim obs) had no ball
information but was rewarded for approaching the ball. The hierarchical split lets the
high-level policy directly observe ball state while the frozen walking model handles
balance and gait.

### 19-dim Observation Space

| Index | Component | Dims | Description |
|-------|-----------|------|-------------|
| 0-2 | filtered_lin_vel | 3 | Robot velocity in body frame |
| 3-5 | filtered_ang_vel | 3 | Robot angular velocity in body frame |
| 6-7 | projected_gravity | 2 | Orientation indicator (xy) |
| 8-9 | ball_rel_body | 2 | Ball position relative to robot, body frame |
| 10-11 | ball_vel_body | 2 | Ball velocity in body frame |
| 12 | dist_to_ball | 1 | Euclidean distance to ball |
| 13-14 | goal_dir | 2 | Goal direction in body frame (normalized) |
| 15 | goal_dist | 1 | Distance to goal |
| 16-18 | last_hl_actions | 3 | Last velocity command [vx, vy, wz] |

---

## 📊 Key Results

### Training Progress (P0/P1/P2 Tuned, 500 Iterations)

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| Mean reward | -22 | +24 | ▲46 |
| Episode length | 18 | 225 | ▲207 |
| Action std | 1.0 | 0.07 | ✓ Stable |
| Ball distance (min) | 4.29m | 0.25m | 93% reduction |

### Module E: Baseline vs RL Comparison

| Scenario | Baseline min_d | RL min_d | RL Advantage |
|----------|---------------|----------|--------------|
| front_close | 1.87m | **1.28m** | 32% closer |
| front_far | 4.63m | 4.81m | Comparable |
| left | 1.50m | **2.26m** | Baseline better |
| right | 3.58m | 3.58m | Same |

### Module F: Standardized Benchmark (4 scenarios × 10 runs)

| Scenario | Mean Δd | Min dist | Falls | Reward |
|----------|---------|----------|-------|--------|
| front_close | -0.01m | 1.28m | 1/10 | 76.5 |
| front_far | -0.17m | 4.48m | 1/10 | 72.0 |
| left | -0.05m | 2.02m | 0/10 | 74.6 |
| right | -0.33m | 3.24m | 2/10 | 73.4 |

**Inference**: mean 0.41ms, p95 0.40ms (4000 samples) — real-time capable.

### 5 Critical Bug Fixes

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | Floating base lock | URDF `world_joint` commented out + `merge_fixed_links=True` | Uncomment + `merge_fixed_links=False` |
| 2 | 1-step termination | `base_euler` in degrees, `term_pitch` in radians (0.52° vs 30°) | Use degree values directly |
| 3 | Obs history misalignment | `_build_low_level_obs` updated history before physics step | Read `obs_buf` only, update after step |
| 4 | Conservative local optimum | `approach_ball=1` → standing still gives +34 reward | Curriculum + `approach_ball=10` |
| 5 | Action std explosion | `entropy_coef=0.01` → std=5.78, noise dominates | `entropy_coef=0.003` → std=0.07 |

---

## 🔧 Prerequisites

### Hardware

- AMD Radeon GPU (e.g., RX 7900 XTX, MI250) with ROCm 6.2+
- Minimum 16 GB VRAM recommended for 2048 parallel environments

### Cloud Environment

This project was developed on **Anrui Cloud** (安睿云) AMD GPU instances:

- JupyterLab terminal access
- VNC via noVNC on port 6080 (password: `***REMOVED***`)
- Python virtual environment at `/opt/venv/`

---

## 🚀 Setup

### Step 1: Install ROCm PyTorch

```bash
# Use the ROCm-specific PyTorch wheel
/opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Verify AMD GPU is detected
/opt/venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'HIP available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'ROCm version: {torch.version.hip}')
"
```

### Step 2: Install Python Dependencies

```bash
/opt/venv/bin/pip install -r requirements.txt
```

### Step 3: Obtain the Pre-trained Walking Model

The frozen low-level walking model (`t1_walk.pt`) comes from Booster Robotics' deployment
framework. Clone and set up:

```bash
cd /workspace

# Clone Booster Deploy (contains t1_walk.pt and URDF models)
git clone https://github.com/BoosterRobotics/booster_deploy.git
git clone https://github.com/BoosterRobotics/booster_assets.git

# Install booster_assets (provides URDF models)
cd booster_assets
/opt/venv/bin/pip install -e .
cd ..

# Verify the walk model exists
ls -lh /workspace/booster/booster_deploy/tasks/locomotion/models/t1_walk.pt
```

### Step 4: Clone This Repository

```bash
cd /workspace
git clone https://github.com/gxinxing/radeon-hackathon-2026.git amd-physical-ai-soccer
cd amd-physical-ai-soccer
```

### Step 5: Verify Environment

```bash
# Check ROCm
rocm-smi

# Check PyTorch + Genesis + rsl_rl
/opt/venv/bin/python -c "
import torch; import genesis as gs; import rsl_rl
print(f'PyTorch {torch.__version__} | HIP: {torch.cuda.is_available()}')
print(f'Genesis {gs.__version__}')
print('rsl_rl OK')
"

# Verify t1_walk.pt can walk without falling for 30 seconds
/opt/venv/bin/python verify_t1_walk.py
```

---

## 📖 Usage

### Training

```bash
cd /workspace/amd-physical-ai-soccer

# Quick test (256 envs, 100 iterations — ~5 minutes)
/opt/venv/bin/python train_hierarchical.py \
    --num_envs 256 \
    --max_iterations 100

# Full training (2048 envs, 500 iterations — ~2-4 hours)
/opt/venv/bin/python train_hierarchical.py \
    --max_iterations 500

# Resume from checkpoint
/opt/venv/bin/python train_hierarchical.py \
    --resume runs/hierarchical_soccer_chase_hl/model_250.pt

# Custom walk model path
/opt/venv/bin/python train_hierarchical.py \
    --pretrained /path/to/custom_walk.pt
```

Models are saved to `runs/hierarchical_soccer_chase_hl/`:

```bash
ls runs/hierarchical_soccer_chase_hl/
# model_50.pt  model_100.pt  ...  model_500.pt  cfgs.pkl
```

### Rendering Demo Video

```bash
# Render 300 steps with the latest checkpoint
/opt/venv/bin/python render_hierarchical.py --steps 300

# Render with a specific model
/opt/venv/bin/python render_hierarchical.py \
    --model runs/hierarchical_soccer_chase_hl/model_500.pt \
    --steps 500
```

Output: `demos/hierarchical_chase_hl_v4.mp4`

### Exporting ONNX for Deployment

```bash
# Export via raw MLP extraction (recommended — bypasses rsl_rl tracing limit)
/opt/venv/bin/python export_onnx_mlp.py \
    --model runs/hierarchical_soccer_chase_hl/model_500.pt \
    --output models/chase_v3_policy.onnx

# Alternative: standard export
/opt/venv/bin/python export_onnx.py \
    --model runs/hierarchical_soccer_chase_hl/model_500.pt \
    --output models/soccer_policy.onnx
```

### 1v1 Match Verification (ONNX Runtime)

```bash
# Run 1v1 match: RL agent vs ball, ONNX inference
python match_1v1_onnx.py --onnx models/chase_v8_policy.onnx --steps 200
```

### 3v3 Match Evaluation

```bash
# Run match evaluation locally (no GPU needed for rule-based)
/opt/venv/bin/python scripts/match_eval_3v3.py

# With RL policy
/opt/venv/bin/python scripts/match_eval_3v3.py \
    --checkpoint runs/hierarchical_soccer_chase_hl/model_500.pt
```

### GPU Benchmark Collection

```bash
# Start benchmark collector in background
/opt/venv/bin/python benchmark_collect.py \
    --log /tmp/train_output.log \
    --output benchmark/ \
    --interval 5 &

# Run training (logs to /tmp/train_output.log)
/opt/venv/bin/python train_hierarchical.py --max_iterations 500 \
    2>&1 | tee /tmp/train_output.log

# Stop collector when training finishes
kill $(cat /tmp/benchmark_pid)
```

Output: `benchmark/gpu_samples.csv` and `benchmark/gpu_samples.json`

---

## 🎯 Reward Function Design

The reward function (`reward.py`) implements a curriculum with task-specific term sets:

| Task | Reward Terms | Purpose |
|------|-------------|---------|
| `balance` | upright, alive, tracking_vel, feet_swing, feet_slip | Maintain balance while walking |
| `chase` | balance terms + approach_ball | Approach the ball |
| `chase_hl` | upright, alive, approach_ball, ball_control, ball_progress, ball_contact, directed_contact, approach_angle, ball_to_goal, goal_scored | Hierarchical (no gait terms) |

Key reward shaping techniques:

- **Exponential kernel** for velocity tracking: `exp(-(cmd - actual)² / σ)`
- **Distance delta** for ball approach: `prev_dist - current_dist` (rewards getting closer)
- **Exponential proximity** for ball control: `exp(-(dist - radius) * 3.0)`
- **Penalties** for falling, energy usage, and jerky actions

### P0/P1/P2 Parameter Tuning

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `hl_clip` (lin/ang) | 0.05 / 0.05 | 0.8 / 1.0 | Unlock full walking speed |
| `upright` | 5.0 | 0.5 | Frozen model balances — stop reward saturation |
| `alive` | 3.0 | 0.0 | Remove passive survival bonus |
| `approach_ball` | 1.0 | 10.0 | Ball chase must dominate reward |
| `entropy_coef` | 0.01 | 0.003 | Fix action_std explosion (5.78→0.07) |
| `learning_rate` | 3e-3 | 1e-3 | Stabilize value loss |

---

## ⚔ Distributed Multi-Robot Match (1v1 / 3v3)

Since Genesis cannot handle multiple robots in one scene on ROCm, we use a
multi-process distributed architecture:

```
┌──────────────────────────────────────────────────┐
│           Match Coordinator (socket)              │
│  - 50Hz sync loop                                 │
│  - Broadcasts ball + robot positions to all       │
│  - Pairwise collision detection + push-back       │
│  - Structured JSON match log                       │
└──┬──────┬──────┬──────┬──────┬──────┬────────────┘
   │      │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼      ▼
┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐
│RL   ││Rule││Rule││Rule││Rule││Rule││Rule│
│Agent││Ally││Ally││Opp ││Opp ││Opp │
│+ball││     ││     ││     ││     ││     │
└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘
  GPU     GPU    GPU    GPU    GPU    GPU
 (shared AMD Radeon, 6 processes)
```

**Launch 1v1 match:**
```bash
bash run_1v1.sh runs/hierarchical_soccer_chase_hl/model_1894.pt 25
```

**Launch 3v3 match (distributed, 6 robots):**

> Note: 3v3 distributed match requires 6 Genesis processes to compile kernels
> simultaneously (2-3 min each). The coordinator uses a 600s accept deadline.
> 1v1 match above is the verified validation path.
```bash
bash run_3v3.sh runs/hierarchical_soccer_chase_hl/model_1894.pt 25
```

**1v1 Verification Results:**
- 200 steps at 10Hz (20s simulated)
- Ball velocity non-zero: max 1.46 m/s, avg 0.94 m/s
- Robot moved 4.05m, ball displaced 20.02m (actively pushed)
- Robot height stable at 0.89-0.92m (no falls)
- ONNX inference: 200 steps, no errors
- Match log saved to `match_logs/match_1v1.json`

---

## 📁 Project Structure

```
.
├── train_hierarchical.py          # Hierarchical training entry point
├── soccer_env_hierarchical.py     # Hierarchical env (high-level + frozen walk)
├── soccer_env_v4.py               # Base soccer env (flat policy, v4)
├── reward.py                      # Reward functions (balance/chase/shoot curriculum)
├── render_hierarchical.py         # Demo video renderer
├── verify_t1_walk.py              # Verify t1_walk.pt walks 30s without falling
├── export_onnx.py                 # Standard ONNX export
├── export_onnx_mlp.py             # ONNX export via raw MLP extraction
├── benchmark_collect.py           # ROCm GPU benchmark collector
├── match_coordinator.py           # Distributed match coordinator (socket sync)
├── match_worker.py                # Distributed match worker (1 robot per process)
├── run_1v1.sh                     # Launch 1v1 match (2 workers)
├── run_3v3.sh                     # Launch 3v3 match (6 workers)
├── match_3v3.py                   # 3v3 match simulation runner (legacy)
├── match_evaluator.py             # Match evaluation logic
├── match_scene.py                 # Match scene setup
├── soccer_env_1v1.py              # 1v1 environment (Genesis multi-entity, WIP)
├── disturbance.py                 # Disturbance injection (push, wind)
├── inject_proxy.py                # Proxy injection for agent framework
├── configs/
│   ├── hierarchical_agent.yaml    # Hierarchical training config
│   ├── curriculum_stage1.yaml     # Stage 1 curriculum config
│   ├── soccer_agent.yaml          # Flat policy training config
│   └── match_3v3.yaml            # Match simulation config
├── src/
│   ├── soccer_env/
│   │   └── soccer_scene.py        # Genesis soccer field scene builder
│   ├── match_3v3/
│   │   ├── __init__.py
│   │   ├── policy.py              # Policy interface (rule-based + RL)
│   │   ├── roles.py               # Role assignment (attacker/defender/keeper)
│   │   ├── scene.py               # Match scene and state definitions
│   │   └── result.py              # Match result tracking
│   └── soccer_env/
│       └── soccer_scene.py        # Genesis soccer field scene builder
├── scripts/
│   └── match_eval_3v3.py         # Match evaluation script
├── tests/
│   └── test_match_contract.py    # Match contract tests
├── docs/                          # Technical report and documentation
├── models/                        # Trained checkpoints and ONNX exports
├── benchmark/                     # GPU performance data + Module E/F results
├── training_logs/                 # Training logs from AMD GPU
├── match_logs/                    # 1v1/3v3 match trajectory logs (JSON)
├── demos/                         # Demo videos
├── presentations/                 # Posters and slides
├── urdf/t1/                       # T1 humanoid URDF + meshes
├── requirements.txt
└── README.md
```

---

## 🛠 Technical Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Physics simulation | Genesis 1.3.1 | GPU-accelerated, AMD Radeon compatible, Python-native |
| Deep learning | PyTorch 2.9.1 (ROCm 6.2) | AMD GPU support via HIP/ROCm |
| RL algorithm | rsl_rl 5.4.2 (PPO) | Lightweight, well-tested, compatible with Genesis |
| Robot platform | Booster T1 (humanoid, 31 DoF) | Standard platform for RoboCup soccer |
| Match validation | Genesis 1v1 (ONNX Runtime) | In-engine verification of trained policy |
| Cloud GPU | Anrui Cloud (安睿云) AMD GPU (51 GB VRAM) | AMD Radeon GPU with JupyterLab + VNC |

---

## ⚠ Known Limitations

1. **Genesis ROCm multi-entity crash**: Genesis physics engine on AMD ROCm crashes with
   `hipErrorLaunchFailure` when two or more robot URDF entities are loaded in the same scene.
   This is a platform-level bug, not a memory issue (VRAM usage only 0.9 GB / 51.5 GB).

2. **Workaround — Distributed multi-process architecture**: Each robot runs in its own
   Genesis process (proven stable with 1 robot). A socket-based coordinator syncs state
   between processes. 1v1 match verified with ONNX inference (200 steps, ball displaced 20m).
   3v3 distributed match (6 robots) is designed but not yet fully verified at runtime.

3. **Close-range ball control**: When the ball is within ~2m, the velocity-command interface
   cannot express fine motor adjustments needed for ball possession. A residual joint-level
   policy would be needed for dribbling and shooting.

---

## 📦 Data Sources

This project does **not** use external datasets. All training data is generated on-the-fly
by the Genesis physics simulation:

| Data | Source | Purpose |
|------|--------|---------|
| `t1_walk.pt` | [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repo | Frozen low-level walking policy (720→21) |
| T1 URDF model | [booster_assets](https://github.com/BoosterRobotics/booster_assets) repo | Robot physics model for Genesis |
| Soccer field | `src/soccer_env/soccer_scene.py` | 14m × 9m RoboCup 3v3 field |
| Reward function | `reward.py` | Curriculum: balance → chase → shoot |
| Training configs | `configs/hierarchical_agent.yaml` | PPO hyperparameters, reward weights |

---

## 👥 Team

- Team Name: Individual Submission
- Members: Simon

---

## 📄 License

This project is submitted for the AMD AI DevMaster Hackathon. See the competition
repository for licensing terms.
