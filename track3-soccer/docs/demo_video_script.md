# Demo Video Script (3-5 minutes)

## Overview

This script guides the creation of a 3-5 minute demonstration video showing the complete
workflow: training → simulation → 3v3 match → ROCm performance.

## Video Structure

### Part 1: Introduction (30s)

**Narration/Text overlay:**
> "Humanoid Robot Soccer Policy Training on AMD Radeon GPU — Track 3: Physical AI"
> "Genesis Physics Engine + ROCm PyTorch + rsl_rl PPO"
> "First AMD-GPU humanoid soccer training pipeline"

**Visual:** Title card with project name, AMD ROCm badge, Genesis badge

---

### Part 2: Training Command (30s)

**Visual:** Terminal showing training command and startup:

```bash
$ python train_hierarchical.py --max_iterations 500 --num_envs 2048

[Genesis] Running on [AMD Radeon Graphics] with backend gs.amdgpu
[hierarchical] Frozen walk model loaded from t1_walk.pt
[hierarchical] HL obs dim=19, HL action dim=3
[hierarchical] HL clip: lin=1.2 m/s, ang=1.2 rad/s

Learning iteration 0/500
  Total steps: 49152 | Steps per second: 2229
  Mean reward: -24.44 | Mean episode length: 76.02
```

**Narration:**
> "Training runs on AMD Radeon GPU with 2048 parallel environments.
> The hierarchical architecture combines a frozen walking model with a
> trainable high-level PPO policy that sees ball position and goal direction."

**Fast-forward through training iterations, showing reward climbing from -24 to +105**

---

### Part 3: Training Results (45s)

**Visual:** Training metrics table (from train_v8.log):

```
Training Complete — 500 iterations
─────────────────────────────────
Peak reward:        +105.61
Final reward:       +93.07
Total goals:        1,358
Steps per second:   4,618
Total steps:        24.6M
Episode length:     220 steps
Action std:         0.08
Falls:              0
Training time:      1h 35min
```

**Narration:**
> "After 500 iterations and 24.6 million steps, the policy achieved
> 1,358 goals with zero falls. Peak throughput was 4,618 steps per second
> on the AMD Radeon GPU."

**Show reward curve if available (TensorBoard screenshot or text-based ASCII chart)**

---

### Part 4: Single-Robot Demo (60s)

**Visual:** `demos/hierarchical_chase_hl_v12.mp4` (150 frames, 300 steps)

**Text overlay:**
> "Trained policy: robot chases ball, maintains balance (height 0.92m, 0 falls)"
> "Ball distance: 0.13m minimum — robot reaches and contacts the ball"

**Narration:**
> "The trained policy successfully chases the ball, reaching a minimum distance
> of 0.13 meters. The robot maintains balance throughout, with height stable
> at 0.92 meters and zero falls over 300 steps."

---

### Part 5: Reward Function Innovation (45s)

**Visual:** Side-by-side comparison table:

```
v7 (old reward)              v8 (new reward)
─────────────────            ─────────────────
approach_ball: clamp(≥0)    approach_ball: tanh(Δ)
ball_to_goal: 3.0            ball_to_goal: 8.0
—                            approach_angle: 3.0 (NEW)
—                            directed_contact: 5.0 (NEW)
clip_lin: 0.8 m/s            clip_lin: 1.2 m/s

146 goals                    1,358 goals (▲830%)
```

**Narration:**
> "The key innovation was the reward function. The original hard clamp on
> approach_ball killed the gradient when the robot touched the ball,
> so it learned to camp at 0.25 meters. By switching to tanh soft clamp
> and adding approach_angle and directed_contact rewards, the robot
> learned to approach from the correct direction and make purposeful
> contact — 830% more goals."

---

### Part 6: ROCm GPU Performance (30s)

**Visual:** GPU benchmark charts/data:

```
AMD Radeon GPU Performance (612 samples, 102 min)
─────────────────────────────────────────────────
GPU utilization:  avg 86%, max 100%
VRAM usage:       avg 19.9 GB / 51.5 GB (39%)
Temperature:      avg 40°C, max 52°C
Power:            avg 98 W, max 191 W
Training FPS:     4,618 steps/s (peak)

Multi-track GPU sharing:
  Track 3 (RL training):  ~5 GB VRAM
  Track 2 (vLLM serving): ~18 GB VRAM
  Total: 46% VRAM usage, both running concurrently
```

**Narration:**
> "The AMD Radeon GPU sustained 86% average utilization with 4,618 steps
> per second. The 51 GB VRAM allowed simultaneous training and vLLM
> inference, demonstrating efficient GPU sharing."

---

### Part 7: 3v3 Distributed Match (45s)

**Visual:** `demos/3v3_match_full.gif` (2D top-down animation)

**Text overlay:**
> "3v3 Match: Team A (RL, blue) vs Team B (rule-based, red)"
> "6 robots, 1240 steps, 7 collisions, 25 seconds"
> "Distributed multi-process: each robot in its own Genesis process"

**Narration:**
> "The 3v3 match validates the trained policy in a multi-agent setting.
> Six robots run in separate Genesis processes, coordinated by a socket-based
> server at 50 Hz. The ball was actively displaced 4.77 meters, showing
> real contact behavior."

---

### Part 8: ONNX Deployment (15s)

**Visual:** Terminal showing ONNX export:

```
$ python export_onnx_mlp.py --model model_499.pt --output chase_v8_policy.onnx

ONNX exported: chase_v8_policy.onnx (182.4 KB)
  Input:  obs  [batch, 19]
  Output: action [batch, 3] (vx, vy, wz)
  Opset: 17 | Parameters: 46,467
  Architecture: Linear(19,256)→ELU→Linear(256,128)→ELU→Linear(128,64)→ELU→Linear(64,3)
```

**Narration:**
> "The trained policy exports to a 182 KB ONNX model, ready for deployment."

---

### Part 9: Conclusion (15s)

**Visual:** Summary card:

```
✅ First AMD-GPU humanoid soccer training pipeline
✅ 1,358 goals (830% improvement from reward fix)
✅ 4,618 steps/s on AMD Radeon (51 GB VRAM)
✅ 3v3 distributed match — 6 robots, 0 crashes
✅ ONNX deployment model (182 KB, 46,467 params)
✅ Fully reproducible — Genesis + ROCm PyTorch
```

**Narration:**
> "This project proves that competitive humanoid robot soccer policies can be
> trained entirely on AMD Radeon GPUs, without any NVIDIA dependencies."

---

## Available Demo Files

| File | Duration | Content |
|------|----------|---------|
| `demos/hierarchical_chase_hl_v12.mp4` | ~10s | Single robot chasing ball (v8 model) |
| `demos/3v3_match_full.gif` | ~40s | 3v3 match top-down animation (6 robots) |
| `demos/3v3_match_demo.gif` | ~40s | 3v3 match (3 robots, earlier run) |
| `train_v8.log` | — | Full training log (500 iterations) |
| `track3-data/benchmark/gpu_samples.csv` | — | GPU utilization/VRAM/temp/power data |

## Recording Instructions

1. **Screen record** the terminal showing training command + output
2. **Play** `hierarchical_chase_hl_v12.mp4` for single-robot demo
3. **Play** `3v3_match_full.gif` for match visualization
4. **Overlay** metrics tables and narration text
5. **Add voice narration** or subtitles for each section
6. **Export** as MP4, 3-5 minutes, 1080p

## Total Estimated Duration

| Part | Time |
|------|------|
| Introduction | 30s |
| Training command | 30s |
| Training results | 45s |
| Single-robot demo | 60s |
| Reward innovation | 45s |
| ROCm performance | 30s |
| 3v3 match | 45s |
| ONNX deployment | 15s |
| Conclusion | 15s |
| **Total** | **~5 min 15s** |
