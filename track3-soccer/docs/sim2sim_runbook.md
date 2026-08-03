# Booster Studio 3v3 SoccerSim — Sim2Sim Validation Runbook

This document provides step-by-step instructions for deploying the trained ONNX policy
into Booster Studio's 3v3 SoccerSim for Sim2Sim validation.

## Prerequisites

- Cloud instance access (Anrui Cloud AMD GPU with VNC)
- Booster Studio 1.9.4+ installed
- Booster agent framework: `src/booster_agent/`
- A **valid** ONNX policy re-exported from the 2048-envs checkpoint (see Step 0)
  - **MANDATORY size gate**: file MUST be **> 50 KB**. A ~2 KB file is a stub/empty
    export (no trained weights) and will make the robot stand still in Sim2Sim.

> ⚠️ **Do NOT reuse the committed `models/chase_v3_policy.onnx` / `chase_v4_policy.onnx` /
> `chase_v5_policy.onnx`** — all three are 1973-byte stubs (verified). Re-export from the
> real checkpoint on the cloud instance before doing anything else.

## Step 0: Re-export & validate the ONNX on the cloud instance (CRITICAL)

The trained checkpoints live only on the cloud instance, and the committed ONNX files are
empty stubs. Re-export from the 2048-envs checkpoint, then gate on file size + weight shape.

```bash
cd /workspace/amd-physical-ai-soccer
git pull   # make sure export_onnx.py + latest config are present

# 0a. Locate the 2048-envs checkpoint (pick the highest-iteration model_*.pt)
ls -t runs/hierarchical_soccer_*/model_*.pt | head -5
#    → note the path of the 2048-envs run's latest checkpoint, e.g.
#      runs/hierarchical_soccer_chase_v5_2048/model_300.pt

# 0b. Re-export with explicit checkpoint + output name
python export_onnx.py \
    --model runs/hierarchical_soccer_chase_v5_2048/model_300.pt \
    --output models/chase_v6_2048_policy.onnx

# 0c. SIZE GATE — must be >> 50 KB, else the export silently failed
ls -la models/chase_v6_2048_policy.onnx
#    ✅ good: several hundred KB
#    ❌ bad:  ~2 KB  → re-run 0a/0b, check for errors in export output

# 0d. Weight-shape check (onnxruntime is installed in /opt/venv)
/opt/venv/bin/python -c "
import onnx
m = onnx.load('models/chase_v6_2048_policy.onnx')
print('ops:', m.opset_import[0].version, '| nodes:', len(m.graph.node))
for i in m.graph.initializer:
    print(f'  {i.name}: {list(i.dims)}  ({i.dims and i.dims[0]*i.dims[-1] if i.dims else 0} params)')
total = sum(i.dims[0]*i.dims[-1] for i in m.graph.initializer if i.dims)
print(f'TOTAL params in initializers: {total}  (a real policy has >10k)')
"
#    ✅ good: TOTAL params > 10000 and several MatMul/Gemm nodes
#    ❌ bad:  TOTAL params == 0  → stub, do NOT proceed to VNC
```

Only continue to Step 1 once `chase_v6_2048_policy.onnx` passes both the size gate and the
weight-shape check.

## Step 1: Access the Cloud Instance via VNC

```bash
# From your local machine, open the noVNC URL in a browser:
https://radeon-global.anruicloud.com/instances/<instance-id>/proxy/6080/vnc.html

# Password: ***REMOVED***
```

## Step 2: Verify Booster Studio is Running

```bash
# In JupyterLab terminal, check if Booster Studio processes are running:
ps aux | grep -i booster

# If not running, start Booster Studio:
bash install_booster_studio.sh

# Or manually:
cd /opt/booster_studio
./booster_studio &
```

You should see the Booster Studio GUI in the VNC window.

## Step 3: Prepare the ONNX Policy

```bash
# Use the re-exported 2048-envs policy from Step 0
cd /workspace/amd-physical-ai-soccer
ls -la models/chase_v6_2048_policy.onnx

# Quick validation (MUST pass the size + param checks below):
/opt/venv/bin/python -c "
import onnx
model = onnx.load('models/chase_v6_2048_policy.onnx')
print(f'Input:  {model.graph.input[0].name} {[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]}')
print(f'Output: {model.graph.output[0].name} {[d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]}')
print(f'Opset:  {model.opset_import[0].version}')
print(f'Nodes:  {len(model.graph.node)}')
total = sum(i.dims[0]*i.dims[-1] for i in model.graph.initializer if i.dims)
print(f'TOTAL params in initializers: {total}')
"
```

Expected output (a REAL policy):
```
Input:  obs [1, <obs_dim>]
Output: action [1, 3]
Opset:  17
Nodes:  >= 7
TOTAL params in initializers: > 10000
```
> If `TOTAL params` is 0 or the file is ~2 KB, the export failed — go back to Step 0.
> Note: `<obs_dim>` must match the dimension the Booster agent feeds (see Step 4); a
> mismatch causes inference errors or a frozen robot.

## Step 4: Configure the Booster Agent

The agent framework is in `src/booster_agent/`. It loads the ONNX model and provides
the observation interface for Booster Studio.

```bash
# Verify agent code is in place
ls -la src/booster_agent/src/main.py
ls -la src/booster_agent/src/rl_playbook.py

# The agent reads the ONNX model from:
# src/booster_agent/models/chase_v6_2048_policy.onnx
# Copy your re-exported (validated) ONNX there:
cp models/chase_v6_2048_policy.onnx src/booster_agent/models/
ls -la src/booster_agent/models/chase_v6_2048_policy.onnx
```

### Agent Configuration

The agent (`src/booster_agent/src/main.py`) does the following:
1. Receives robot + ball state from Booster Studio SoccerSim
2. Constructs the 19-dim observation vector (ball pos/vel, goal dir, proprioception)
3. Runs ONNX inference → 3-dim velocity command (vx, vy, wz)
4. Sends velocity command to the robot via booster_agent_framework

Key parameters:
- **Observation dimension**: 19 (must match training)
- **Action dimension**: 3 (vx, vy, wz)
- **Control frequency**: 10 Hz (high-level), 50 Hz (low-level walk model)
- **Action clip**: [-0.8, 0.8] for linear, [-1.0, 1.0] for angular

## Step 5: Launch 3v3 SoccerSim

### 5.1 Open Booster Studio

In the VNC window:
1. Open Booster Studio
2. Navigate to **3v3 SoccerSim** mode
3. Select the **T1 humanoid** robot for all 6 slots (3 per team)

### 5.2 Load the RL Agent

1. Click on **Agent Configuration** for Team A, Position 1 (Attacker)
2. Select **Custom ONNX Agent**
3. Browse to: `/workspace/amd-physical-ai-soccer/src/booster_agent/src/main.py`
4. The agent will load `chase_v6_2048_policy.onnx` automatically

### 5.3 Configure Opponents

For the remaining 5 robot slots:
- Team A positions 2-3: **Rule-based** (Booster default AI)
- Team B positions 1-3: **Rule-based** (Booster default AI)

### 5.4 Start the Match

1. Click **Start Match**
2. The match runs for a configurable duration (default: 5 minutes simulated)
3. Observe the RL agent's behavior:
   - Does it approach the ball?
   - Does it maintain balance?
   - Does it attempt to kick the ball toward the goal?
   - Does it fall frequently?

## Step 6: Record Demo Video

1. In VNC, use the screen recording tool (or `ffmpeg`):
   ```bash
   # Record VNC display (adjust display number as needed):
   ffmpeg -f x11grab -video_size 1920x1080 -i :1.0 -framerate 30 \
     -t 180 demos/sim2sim_3v3.mp4
   ```

2. Capture at least 3 minutes of gameplay showing:
   - RL agent chasing the ball
   - RL agent in proximity to the ball (demonstrating balance)
   - Any successful kicks or ball interactions
   - Match context (other robots playing)

## Step 7: Collect Results

After the match, record the following metrics:

| Metric | Value | Notes |
|--------|-------|-------|
| Match duration | ___ minutes | Total simulated time |
| RL agent falls | ___ count | Number of times the robot fell |
| Ball touches | ___ count | RL agent made contact with ball |
| Ball progress | ___ meters | Ball moved toward goal by RL agent |
| Goals scored | ___ count | By RL agent's team |
| Goals against | ___ count | By opponent team |

Save results to: `benchmark/sim2sim_results.json`

## Troubleshooting

### Agent fails to load
- Verify ONNX model path is correct (absolute path recommended)
- Check Python version compatibility (3.10+)
- Ensure `onnxruntime` is installed: `/opt/venv/bin/pip install onnxruntime`

### Robot falls immediately
- This may indicate observation mismatch between training and deployment
- Check that the 19-dim observation vector is constructed correctly in `main.py`
- Verify action clip values match training config

### Robot doesn't move
- Check that the agent is receiving game state updates
- Verify the velocity command is being sent to the correct robot
- Look for errors in the Booster Studio console

### VNC connection issues
- Ensure port 6080 is accessible via the instance proxy URL
- Try refreshing the noVNC page
- Verify the instance is still running

## Evidence for Submission

After completing Sim2Sim validation, collect the following as submission evidence:

1. **Demo video**: `demos/sim2sim_3v3.mp4` (3+ minutes)
2. **Results JSON**: `benchmark/sim2sim_results.json`
3. **Screenshot**: A screenshot of Booster Studio running the match
4. **This runbook**: Serves as documentation of the Sim2Sim process

These items directly address the "Real-world application value" judging criterion (20 points).
