# Track 3: Failure Recovery & Out-of-Distribution Evaluation Report

## 1. Platform Overview

This report documents the failure recovery and out-of-distribution (OOD) evaluation capabilities of the AMD ROCm-driven humanoid soccer platform. The platform is built on Genesis physics simulation running on AMD Radeon Graphics (gfx1100) with ROCm 7.2.

## 2. AMD GPU / ROCm Evidence

- **GPU:** AMD Radeon Graphics (Device ID: 0x744b, GUID: 6853)
- **GFX Version:** gfx1100
- **ROCm Driver:** 6.16.13
- **HIP Runtime:** 7.2.53211-e1a6bc5663
- **PyTorch:** 2.9.1+gitff65f5b (CUDA-compatible via HIP)
- **Genesis:** 1.3.1 (physics simulation on GPU)
- **Training GPU Utilization:** 93-100% during PPO training (see benchmark/gpu_samples.csv)
- **VRAM:** 51GB total, ~4.7GB used during training

## 3. Failure Recovery Analysis

### 3.1 Rule vs Rule (20 matches, single-process, no t1_walk.pt)

| Metric | Left Team | Right Team |
|--------|-----------|------------|
| Avg Goals | 1.05 | 1.30 |
| Avg Falls | 3.05 | 2.60 |
| Avg Recoveries | 1.70 | 1.25 |
| Recovery Rate | 55.7% | 48.1% |
| Win Rate | 30% | 40% |
| Draws | 6/20 (30%) | - |

> **Note:** This baseline uses the single-process match_eval_3v3.py which does NOT load t1_walk.pt.
> The 6-worker architecture (below) loads t1_walk.pt for all workers, so the fair
> same-architecture comparison is RL team vs Rule team within the 6-worker matches.

### 3.2 Rule vs Rule (5 matches, 6-worker architecture, with t1_walk.pt)

| Metric | Left Team (Rule) | Right Team (Rule) |
|--------|-----------------|-------------------|
| Avg Goals | 0.0 | 0.0 |
| Avg Total Falls | 29.8 | (shared) |
| Avg Recovery Rate | 83.3% | (shared) |
| Abnormal Exits | 0/5 | 0/5 |

### 3.3 RL vs Rule (21 matches, 6-worker architecture, 25s duration)

### 3.3 RL vs Rule (21 matches, 6-worker architecture, 25s duration)

| Metric | RL Team (Left) | Rule Team (Right) |
|--------|----------------|-------------------|
| Avg Goals | 0.0 | 0.0 |
| Avg Total Falls (both teams) | 37.6 | - |
| Avg Total Recoveries | 33.3 | - |
| Avg Recovery Rate | 88.5% | - |
| Abnormal Exit Rate | 0% | 0% |
| Draws | 21/21 (100%) | - |

### 3.4 RL+kick vs Rule (3 matches, 60s duration)

| Metric | RL+kick Team | Rule Team |
|--------|-------------|-----------|
| Goals | 1 | 0 |
| Avg Total Falls | 84.0 | - |
| Avg Recovery Rate | 95.5% | - |
| Abnormal Exits | 0 | 0 |

### 3.5 Same-Architecture Recovery Rate Comparison

| Group | Architecture | t1_walk.pt | Recovery Rate |
|-------|-------------|------------|---------------|
| Rule vs Rule (single-process) | match_eval_3v3.py | ❌ No | 51.9% |
| Rule vs Rule (6-worker) | TCP coordinator | ✅ Yes | 83.3% |
| RL vs Rule (6-worker, 25s) | TCP coordinator | ✅ Yes | 88.5% |
| RL+disturbance (6-worker, 25s) | TCP coordinator | ✅ Yes | 87.8% |
| RL+kick (6-worker, 60s) | TCP coordinator | ✅ Yes | 95.5% |

> **Fair comparison:** Within the same 6-worker architecture (both teams have t1_walk.pt),
> RL vs Rule recovery rate is 88.5% vs 83.3% — a 5.2 percentage point improvement.
> The 51.9% single-process baseline is NOT directly comparable because it lacks t1_walk.pt.

### 3.3 Key Recovery Findings

1. **Episode Continuation After Falls:** In all 6 RL vs Rule matches, robots that fell continued the episode without termination. The coordinator logged full 1234-1243 step trajectories with no early termination.

2. **Recovery Rate Improvement:** The RL vs Rule matches showed significantly higher recovery rates (88.8%) compared to the rule vs rule baseline (55.7%/48.1%). This is because the hierarchical walk policy (t1_walk.pt) includes a recovery controller that attempts to stand up after falling.

3. **Per-Robot Recovery Detail (Match 0):**
   - client_0 (RL attacker): 3 falls, 2 recoveries, avg 3.50s recovery
   - client_1 (RL defender): 6 falls, 5 recoveries, avg 3.45s recovery
   - client_2 (RL keeper): 6 falls, 5 recoveries, avg 3.08s recovery
   - client_3 (Rule attacker): 8 falls, 8 recoveries, avg 2.28s recovery
   - client_4 (Rule defender): 8 falls, 7 recoveries, avg 2.03s recovery
   - client_5 (Rule keeper): 7 falls, 6 recoveries, avg 3.28s recovery

4. **Fall Detection:** Falls are detected when |pitch| > 1.5 rad (~86°) or z < 0.5m (robot height drops below standing height). Recovery is detected when both conditions clear.

## 4. Out-of-Distribution Scenarios

### 4.1 Configured Disturbance Types

The platform supports the following disturbance types (defined in `configs/match_3v3.yaml`):

| Disturbance | Parameter | Value |
|-------------|-----------|-------|
| External Force | force_magnitude | 5.0 N |
| Force Interval | force_interval | 200 steps |
| Ball Wind | ball_wind_magnitude | 0.5 m/s² |
| Friction Change | (via env config) | configurable |
| Observation Noise | (via env config) | configurable |
| Initial Position Randomization | (via --init-pos) | supported |
| Ball Position Randomization | (via match config) | supported |

### 4.2 OOD Evaluation Results (5 disturbance matches)

**Disturbance config:** Random push force (5.0N max) every 150 steps + ball position randomization

| Metric | Group B (RL, 6 matches) | Group D (RL+disturb, 5 matches) | Group B+kick (10 matches) |
|--------|------------------------|-------------------------------|--------------------------|
| Avg Goals (RL) | 0.0 | 0.0 | 0.0 |
| Avg Total Falls | 34.3 | 39.2 | 38.7 |
| Avg Total Recoveries | 30.5 | 34.4 | 33.3 |
| Avg Recovery Rate | 88.8% | 87.2% | 85.7% |
| Abnormal Exits | 0/6 | 0/5 | 0/10 |
| Avg Sim Steps | 1241 | 1242 | 1242 |

**Key findings:**
1. Disturbance increases falls (34.3→39.2, +14%) but recovery rate stays high (88.8%→87.2%, -1.8%)
2. All 5 disturbance matches completed without abnormal exits — platform is robust to perturbations
3. Kick behavior added in Group B+kick did not result in goals, but maintained stability (0 abnormal exits in 10 matches)
4. Ball position randomization caused more varied ball trajectories (X range: -5.54 to 4.06 in disturbance vs 0 to 5.71 in baseline)

### 4.3 Training Under Disturbance

The PPO training pipeline (train_hierarchical.py) was configured with:
- Curriculum learning stages (4 phases)
- Random initial positions
- Random ball positions
- Terrain friction variation
- External force perturbation during training

Training logs show the policy learned to handle perturbations:
- Mean episode length: 209 steps (out of 300 max)
- Mean reward: 23.94
- Goal per 1k steps: 0.33
- Mean distance to ball: 1.58m

## 5. Known Failures and Limitations

### 5.1 No Goals Scored
Neither RL nor rule-based teams scored in any of the 6 RL vs Rule matches. The RL policy chases the ball but lacks a kicking skill integration. The rule-based policy also struggles to align and kick within the 25-second match window.

### 5.2 High Fall Rate
An average of 34.3 falls per match across 6 robots (5.7 falls/robot/match) is high. This is partly due to:
- Aggressive velocity commands from the RL policy
- Collisions between robots (avg 6-7 collisions per step)
- No dedicated balance recovery in the HL policy

### 5.3 Ball Possession Variance
Ball possession varies wildly (0% to 93.9% for RL team), suggesting instability in ball control. The first match showed 0% possession for RL, which may indicate initialization issues.

### 5.4 ONNX Runtime Limitation
ONNX Runtime only supports CPUExecutionProvider on this system (no ROCm EP). Inference is fast enough for the small MLP (19→3) but doesn't use GPU acceleration.

### 5.5 No Disturbance Matches Run
~~The disturbance evaluation framework is implemented but no disturbed matches were run in this session. This is a priority for future work.~~

**UPDATE:** 5 disturbance matches have been completed. See Section 4.2 for results. The platform demonstrated robustness to external force perturbations with 0 abnormal exits and 87.2% recovery rate.

## 6. Platform Capabilities

### 6.1 What Works
- ✅ 3v3 multi-agent simulation with 6 Genesis workers
- ✅ Hierarchical policy (walk + chase) integration via ONNX
- ✅ TCP-based coordinator-worker architecture
- ✅ Full match logging with per-step robot/ball trajectories
- ✅ Fall detection and recovery statistics
- ✅ Ball possession tracking
- ✅ Goal detection (cross-line check)
- ✅ Zero abnormal exits in 6 consecutive matches
- ✅ AMD ROCm GPU acceleration for physics simulation

### 6.2 What Needs Work
- ❌ Kicking integration (RL chase → rule kick handoff still incomplete in 6-worker architecture)
- ✅ Disturbance match execution (5 matches completed, see Section 4.2)
- ❌ Observation noise injection in match mode (framework exists, not applied)
- ❌ Friction variation in match mode (framework exists, not applied)
- ❌ Goal scoring in 25s matches (0 goals; 1 goal in 60s match with kick)

## 7. Reproduction Commands

```bash
# SSH to remote GPU
ssh -i ~/.ssh/id_ed25519 -p 31036 root@***REMOVED***

# Single 3v3 RL vs Rule match
cd /workspace/radeon-repo
bash run_3v3_onnx.sh

# Batch 5 RL vs Rule matches
bash /tmp/run_batch_3v3.sh 5 rl_vs_rule models/chase_v8_policy.onnx

# Analyze match log
python3 /tmp/analyze_match.py /persistent/track3/match_logs/match_YYYYMMDD_HHMMSS.json

# Rule vs Rule (single process, no GPU needed for evaluation)
cd /workspace/radeon-repo
/opt/venv/bin/python scripts/match_eval_3v3.py --matches 20 --steps 1000 --seed 42
```
