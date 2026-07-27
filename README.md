# Humanoid Robot Soccer Policy Training on AMD Radeon GPU

Train humanoid robot soccer policies (balance, chase, shoot) using the Genesis physics
engine and ROCm PyTorch on AMD Radeon GPUs, then validate via Sim2Sim in Booster Studio's
3v3 soccer simulator.

This project is a submission for the **AMD AI DevMaster Hackathon** — Track 3: Physical AI.

## Why This Project Exists

Booster Robotics' official RL training frameworks (Booster Gym / Booster Train) depend on
NVIDIA Isaac Gym and Isaac Lab, which require CUDA and NVIDIA GPUs. This project builds an
alternative training pipeline that runs entirely on AMD Radeon GPUs using:

- **Genesis** — GPU-accelerated physics simulation (AMD Radeon compatible)
- **ROCm PyTorch** — AMD's GPU compute platform (replaces CUDA)
- **rsl_rl** — PPO-based reinforcement learning runner

The result is the first AMD-GPU humanoid soccer training pipeline, proving that competitive
robot policies can be trained without NVIDIA hardware.

## Architecture

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
│  Reward: approach_ball + ball_control + ball_to_goal     │
│          + upright + alive - fall - energy - action_rate │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                   Deployment Pipeline                     │
│                                                          │
│  Trained .pt  ──▶  ONNX export  ──▶  Booster Studio      │
│  checkpoint                       3v3 SoccerSim (Sim2Sim) │
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

## Prerequisites

### Hardware

- AMD Radeon GPU (e.g., RX 7900 XTX, MI250) with ROCm 6.2+
- Minimum 16 GB VRAM recommended for 2048 parallel environments

### Cloud Environment

This project was developed on **Anrui Cloud** (安睿云) AMD GPU instances:

- JupyterLab terminal access
- VNC via noVNC on port 6080 (password: `***REMOVED***`)
- Python virtual environment at `/opt/venv/`

## Setup

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
git clone <this-repo-url> amd-physical-ai-soccer
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

## Data Sources

This project does **not** use external datasets. All training data is generated on-the-fly
by the Genesis physics simulation:

| Data | Source | Purpose |
|------|--------|---------|
| `t1_walk.pt` | [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repo | Frozen low-level walking policy (720→21) |
| T1 URDF model | [booster_assets](https://github.com/BoosterRobotics/booster_assets) repo | Robot physics model for Genesis |
| Soccer field | `src/soccer_env/soccer_scene.py` | 14m × 9m RoboCup 3v3 field |
| Reward function | `reward.py` | Curriculum: balance → chase → shoot |
| Training configs | `configs/hierarchical_agent.yaml` | PPO hyperparameters, reward weights |

### Configuration

All training parameters are centralized in YAML config files:

```bash
configs/
├── hierarchical_agent.yaml   # Hierarchical training (high-level PPO + frozen walk)
├── soccer_agent.yaml          # Flat policy training (v4, all 720-dim)
└── match_3v3.yaml             # 3v3 match simulation config
```

Key configurable parameters in `hierarchical_agent.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 2048 | Parallel simulation environments |
| `max_iterations` | 500 | PPO training iterations |
| `high_level.decimation` | 5 | High-level control frequency = 50Hz / 5 = 10Hz |
| `reward.approach_ball` | 5.0 | Reward weight for decreasing ball distance |
| `reward.goal_scored` | 20.0 | Reward for scoring a goal |
| `train.algorithm.learning_rate` | 3e-3 | PPO learning rate (adaptive schedule) |
| `train.policy.actor_hidden_dims` | [256, 128, 64] | Actor network architecture |

## Usage

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

Output: `demo/hierarchical_chase_hl.mp4`

### Exporting ONNX for Deployment

```bash
# Export the trained high-level policy to ONNX
/opt/venv/bin/python export_onnx.py

# Custom model and output path
/opt/venv/bin/python export_onnx.py \
    --model runs/hierarchical_soccer_chase_hl/model_500.pt \
    --output models/soccer_policy.onnx
```

Output: `models/soccer_policy.onnx`

### Sim2Sim Validation in Booster Studio

1. Install Booster Studio on the cloud instance:

```bash
bash setup_booster.sh
```

2. Access Booster Studio via noVNC:

```
https://radeon-global.anruicloud.com/instances/<instance-id>/proxy/6080/vnc.html
```

3. Load the ONNX model in Booster Studio's 3v3 SoccerSim
4. Run matches against the official Booster AI

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

## Project Structure

```
.
├── train_hierarchical.py          # Hierarchical training entry point
├── soccer_env_hierarchical.py     # Hierarchical env (high-level + frozen walk)
├── soccer_env_v4.py               # Base soccer env (flat policy, v4)
├── reward.py                      # Reward functions (balance/chase/shoot curriculum)
├── render_hierarchical.py         # Demo video renderer
├── verify_t1_walk.py              # Verify t1_walk.pt walks 30s without falling
├── export_onnx.py                 # Export trained policy to ONNX
├── benchmark_collect.py           # ROCm GPU benchmark collector
├── match_coordinator.py            # Distributed match coordinator (socket sync)
├── match_worker.py                # Distributed match worker (1 robot per process)
├── run_1v1.sh                      # Launch 1v1 match (2 workers)
├── run_3v3.sh                      # Launch 3v3 match (6 workers)
├── match_3v3.py                   # 3v3 match simulation runner (legacy)
├── match_evaluator.py             # Match evaluation logic
├── match_scene.py                 # Match scene setup
├── soccer_env_1v1.py              # 1v1 environment (Genesis multi-entity, WIP)
├── disturbance.py                 # Disturbance injection (push, wind)
├── inject_proxy.py                # Proxy injection for agent framework
├── export_onnx_mlp.py             # ONNX export via raw MLP extraction
├── configs/
│   ├── hierarchical_agent.yaml    # Hierarchical training config
│   ├── soccer_agent.yaml          # Flat policy training config
│   └── match_3v3.yaml             # Match simulation config
├── src/
│   ├── soccer_env/
│   │   └── soccer_scene.py         # Genesis soccer field scene builder
│   ├── match_3v3/
│   │   ├── __init__.py
│   │   ├── policy.py               # Policy interface (rule-based + RL)
│   │   ├── roles.py                # Role assignment (attacker/defender/keeper)
│   │   ├── scene.py                # Match scene and state definitions
│   │   └── result.py               # Match result tracking
│   └── booster_agent/              # Booster Studio Sim2Sim agent
│       ├── src/main.py             # Agent entry (ONNX policy)
│       └── src/rl_playbook.py     # RL-enhanced playbook
├── scripts/
│   └── match_eval_3v3.py          # Match evaluation script
├── tests/
│   └── test_match_contract.py     # Match contract tests
├── docs/                           # Technical report and documentation
├── models/                         # Trained checkpoints and ONNX exports
├── benchmark/                      # GPU performance data + Module E/F results
├── training_logs/                  # Training logs from AMD GPU
├── match_logs/                     # 1v1/3v3 match trajectory logs (JSON)
├── demos/                          # Demo videos
├── presentations/                  # Posters and slides
├── urdf/t1/                        # T1 humanoid URDF + meshes
├── requirements.txt
└── README.md
```

## Reward Function Design

The reward function (`reward.py`) implements a curriculum with task-specific term sets:

| Task | Reward Terms | Purpose |
|------|-------------|---------|
| `balance` | upright, alive, tracking_vel, feet_swing, feet_slip | Maintain balance while walking |
| `chase` | balance terms + approach_ball | Approach the ball |
| `dribble` | chase terms + ball_control | Keep ball close while moving |
| `shoot` | dribble terms + ball_to_goal, goal_scored | Kick ball toward goal |
| `chase_hl` | upright, alive, approach_ball, ball_control, ball_to_goal, goal_scored | Hierarchical (no gait terms — frozen model handles gait) |

Key reward shaping techniques:

- **Exponential kernel** for velocity tracking: `exp(-(cmd - actual)² / σ)`
- **Distance delta** for ball approach: `prev_dist - current_dist` (rewards getting closer)
- **Exponential proximity** for ball control: `exp(-(dist - radius) * 3.0)`
- **Penalties** for falling, energy usage, and jerky actions

## Technical Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Physics simulation | Genesis (`genesis-world`) | GPU-accelerated, AMD Radeon compatible, Python-native |
| Deep learning | PyTorch (ROCm 6.2 build) | AMD GPU support via HIP/ROCm |
| RL algorithm | rsl_rl (PPO) | Lightweight, well-tested, compatible with Genesis |
| Robot platform | Booster T1 (humanoid, 31 DoF) | Standard platform for RoboCup soccer |
| Sim2Sim validation | Booster Studio 1.9.4 | Official simulator for 3v3 soccer |
| Cloud GPU | Anrui Cloud (安睿云) AMD GPU instance | AMD Radeon GPU with JupyterLab + VNC |

## Known Limitations

1. **Genesis ROCm multi-entity crash**: Genesis physics engine on AMD ROCm crashes with
   `hipErrorLaunchFailure` when two or more robot URDF entities are loaded in the same scene.
   This is a platform-level bug, not a memory issue (VRAM usage only 0.9 GB / 51.5 GB).

2. **Workaround — Distributed multi-process architecture**: Each robot runs in its own
   Genesis process (proven stable with 1 robot). A socket-based coordinator syncs state
   between processes. Verified with 6 concurrent processes (3v3 match).

### Distributed Multi-Robot Match (1v1 / 3v3)

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

**Launch 3v3 match (6 robots):**
```bash
bash run_3v3.sh runs/hierarchical_soccer_chase_hl/model_1894.pt 25
```

**Results:**
- 1v1: Agent 119 steps, Opponent 112 steps, zero GPU crash
- 3v3: All 6 workers 75-84 steps, 1240 steps logged, zero GPU crash
- Match logs saved to `match_logs/match_YYYYMMDD_HHMMSS.json`

## License

This project is submitted for the AMD AI DevMaster Hackathon. See the competition
repository for licensing terms.

## Team

- Team Name: [Your Team Name]  <!-- TODO: fill in before submission -->
- Members: [Team member details]  <!-- TODO: fill in before submission -->
