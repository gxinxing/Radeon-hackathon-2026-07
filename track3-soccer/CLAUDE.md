# AMD AI DevMaster Hackathon — Track 3: Physical AI

## Competition Context

This project is a submission for the **AMD AI DevMaster Hackathon** (Track 3: Physical AI).
The goal is to demonstrate robotics/embodied AI solutions powered by **AMD Radeon GPUs and ROCm**.

- **Track**: Track 3 — Physical AI Challenge: Robotics Simulation and Application Design based on AMD Radeon GPUs and ROCm
- **Timeline**: Submission by August 6, 2026 (UTC+8)
- **Submission**: Fork `AMD-DEV-CONTEST/Radeon-hackathon-2026-07`, open a PR titled `Track 3, <Team Name>, <Application Name>`
- **All submission materials must be in English.**

## Project: Humanoid Robot Soccer Policy Training on AMD Radeon GPU

Train humanoid robot soccer policies (balance, chase, shoot) using the Genesis physics
engine + ROCm PyTorch on AMD Radeon GPUs, then validate via Sim2Sim in Booster Studio's
3v3 soccer simulator.

### Why this matters

Booster Robotics' official RL training frameworks (Isaac Gym / Isaac Lab) only support
NVIDIA GPUs. This project builds an alternative training pipeline on AMD Radeon GPU using
Genesis + ROCm PyTorch + rsl_rl, proving that competitive humanoid robot policies can be
trained without NVIDIA hardware.

## Judging Criteria (100 points)

| Dimension | Points | What to show |
|-----------|--------|--------------|
| Robot capability performance | 30 | Trained policies successfully perform balance/chase/shoot |
| AMD Radeon GPU / ROCm adoption | 20 | Genesis simulation + PyTorch training explicitly on ROCm |
| Innovation and originality | 20 | First AMD-GPU humanoid soccer training pipeline (Isaac Gym alternative) |
| Real-world application value | 20 | Sim2Sim validation in Booster Studio; path to real robot deployment |
| Upstream open-source contributions | 10 | Reusable Genesis soccer environment, training configs, reward functions |

## Submission Deliverables

### 1. Technical Report (`docs/technical_report.md`)
Must include:
- Target application definition (humanoid robot soccer)
- Overall system architecture and solution design
- Datasets used for training and evaluation
- How AMD Radeon GPUs are utilized (training, inference, simulation)
- Innovations and key technical contributions
- Final deliverables and output forms
- Team member introductions

### 2. Project Source Code (`src/`)
- Genesis soccer environment (`src/soccer_env/`)
- RL training pipeline (`src/training/`)
- Booster Studio agent for Sim2Sim validation (`src/booster_agent/`)
- Trained model checkpoints (`models/`)
- Preferably a Docker image (`docker/`)

### 3. Reproducibility README (`README.md`)
- Environment setup instructions (ROCm, Genesis, dependencies)
- Execution and usage instructions (train, render, evaluate)
- Dependency specifications (exact versions)
- Step-by-step reproduction procedures
- Evaluators must be able to reproduce results by following instructions

### 4. Demo Video (`demos/`, 3-5 minutes)
- Complete workflow: command-line training → simulation rendering → Sim2Sim validation
- Show actual AMD Radeon GPU execution (ROCm visible in logs)
- Show trained policies in action (balance/chase/shoot)

### 5. Supplementary Materials (`presentations/`)
- Poster or PPT highlighting key scenarios and value

## Technical Stack

| Component | Technology | Note |
|-----------|-----------|------|
| Physics simulation | Genesis (`genesis-world`) | GPU-accelerated, AMD Radeon compatible |
| Deep learning | PyTorch (ROCm build) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| RL algorithm | rsl_rl (OnPolicyRunner) | PPO-based |
| Robot platform | Booster K1/T1 (humanoid) | 22-31 DoF humanoids |
| Sim2Sim validation | Booster Studio 1.9.4 | 3v3 SoccerSim |
| Cloud GPU | 安睿云 (Anrui Cloud) AMD GPU instance | JupyterLab + VNC |
| Agent framework | booster_agent_framework (Python) | Rule-based baseline + RL-enhanced |

## Project Structure

```
src/
├── soccer_env/              # Genesis soccer environment
│   ├── envs/soccer_env.py   # Main env: balance, chase, shoot tasks
│   └── configs/             # Training configs (YAML)
├── training/                # RL training pipeline
│   ├── train.py             # Training entry (ROCm PyTorch + rsl_rl)
│   ├── render_all.py        # Demo video rendering
│   └── evaluate.py          # Policy evaluation
└── booster_agent/           # Booster Studio 3v3 agent (Sim2Sim)
    ├── main.py              # Strategy: Phase state machine + role dispatch
    ├── player.py            # Player control: kick/walk/guard/dribble
    ├── param.py             # All tunable parameters
    ├── safety_adaptation.py # Online safety adaptation
    ├── framework/           # Platform pipeline (DO NOT MODIFY)
    └── utils/               # Geometry, path planning, obstacles
```

## Development Constraints

1. **AMD GPU only** — No NVIDIA CUDA dependencies. Isaac Gym/Isaac Lab are NOT available.
2. **Genesis is the simulation engine** — Not Isaac Gym, not MuJoCo (though MuJoCo is acceptable for Sim2Sim).
3. **ROCm PyTorch** — Must use ROCm-specific PyTorch wheels.
4. **booster_agent/framework/ is read-only** — Platform pipeline, modifications break deployment.
5. **Strategy changes go in `main.py`** — Phase state machine, role dispatch, behavior.
6. **Player actions go in `player.py`** — Kick, walk, guard, dribble.
7. **Parameter tuning goes in `param.py`** — All tunable values centralized.
8. **All output materials in English** — Technical report, README, PR description.

## Key Files Reference

- `BOOSTER_SDK_DEV_GUIDE.md` — Booster SDK, hardware specs, AMD GPU strategy
- `docs/Booster Agent Framework Python API.md` — Agent framework API
- `docs/BoosterOS 开发者接口文档 - V1.0.md` — BoosterOS developer API
- `scripts/setup_rocm.sh` — ROCm environment setup
- `scripts/train_all.sh` — One-command training for all tasks

## Cloud Environment

- **Instance**: 安睿云 AMD GPU (JupyterLab accessible)
- **Access**: VNC via noVNC on port 6080 (password: ***REMOVED***)
- **Booster Studio**: Running inside VNC, used for Sim2Sim validation
- **Training**: JupyterLab Terminal, `/workspace/amd-physical-ai-soccer/`

## RL Training Pipeline

Three sub-tasks trained independently:
1. **balance** — Stand and maintain balance on the soccer field
2. **chase** — Chase and approach the ball
3. **shoot** — Kick the ball toward the goal

Training uses Genesis physics engine with GPU acceleration on AMD Radeon.
Models saved as PyTorch checkpoints (`.pt`), exported to ONNX for deployment.

## Sim2Sim Validation

Trained policies are validated in Booster Studio's 3v3 SoccerSim:
- Load model → run in Genesis simulation (AMD GPU)
- Export to ONNX → load in Booster agent
- Deploy agent in Booster Studio simulator
- Run matches against official Booster AI

## Common Tasks for AI Agents

When working on this project, tasks typically fall into:

1. **Training pipeline** — Modify environments, reward functions, training configs
2. **Agent strategy** — Improve soccer tactics in `src/booster_agent/main.py`
3. **Documentation** — Write/update technical report, README, architecture docs
4. **Reproducibility** — Ensure Docker/scripts can reproduce results
5. **Demo preparation** — Render videos, prepare presentation materials

Always consider the judging criteria. If a change doesn't improve one of the 5 dimensions,
it likely isn't worth doing.
