# Track 3 Remote Asset Audit Report

**Date:** 2026-08-01  
**Remote:** root@***REMOVED***:31036  
**Remote Path:** /persistent/track3  
**Source Code:** /workspace/radeon-repo  

## 1. Model Assets

| File | Size | SHA256 | Status |
|------|------|--------|--------|
| models/base/t1_walk.pt | 2.1M | ef1d61e19082b83405f4320a08f4cfc2d7d7f003ed3790dab013778ba442dec7 | ✅ |
| models/onnx/chase_v6_policy.onnx | 183K | — | ✅ |
| models/onnx/chase_v7_policy.onnx | 183K | 06dd0c69102275a00e8c665fc8d31d161351dbddc00d7b5d6757bec938c0e136 | ✅ |
| models/onnx/chase_v8_policy.onnx | 183K | 6d7ce9121116663eaca7d14ab7d845d6f951a9c5c55521d88483f0e6d7c3fab5 | ✅ |
| models/onnx/chase_v8_policy.onnx.data | 182K | — | ✅ |
| models/checkpoints/best.pt | 1.1M | 76d713f8ae121d2669502fd2a29289144fea1e9b92f9a1e0f80a3cfed63130da | ✅ |
| models/checkpoints/model_400.pt | 1.1M | e405dc21f50c9de3c87623b570dfe6fceb14a55ddfe8b10201425b95a9ee7060 | ✅ |
| models/checkpoints/model_450.pt | 1.1M | cc3f1caec2c11228d210fa83b3c50f171f1916c981e13a6bc23fc48f540c33cc | ✅ |
| models/checkpoints/cfgs.pkl | 2.1K | — | ✅ |

## 2. Training Logs

| Log File | Size | Last Status |
|----------|------|-------------|
| train_hl_stage1.log | 467K | HL stage1 training |
| train_chase_v4_1024.log | 460K | Chase v4 |
| train_v8.log | 540K | V8 training |
| train_chase_v5_2048.log | 320K | Chase v5 |
| train_chase_v7.log | 321K | 300/300 iters, reward=23.94, ep_len=209 |
| train_chase_clip06.log | 459K | Chase clip06 |
| train_hl_chase_v3.log | 235K | HL chase v3 |
| post_train.log | 9.1K | Post-train: demo OK, ONNX export OK, 3v3 match FAILED |

## 3. Match Logs (8 files)

| File | Size | n_clients | Notes |
|------|------|-----------|-------|
| match_20260731_002906.json | 243K | 0 | Empty match, no workers connected |
| match_20260731_003638.json | 243K | — | |
| match_20260731_003734.json | 243K | — | |
| match_20260731_003926.json | 804K | — | Has robot data |
| match_20260731_004058.json | 243K | — | |
| match_20260731_004322.json | 243K | — | |
| match_20260731_004742.json | 1.4M | — | Has robot data |
| match_20260731_032656.json | 1.4M | — | Has robot data |

## 4. Benchmark Files

| File | Size |
|------|------|
| module_f_benchmark.json | 9.0K |
| module_e_comparison.json | 2.2K |
| gpu_samples.csv | 244K |

## 5. Other Assets

- `tensorboard/events.out.tfevents.1785451714.<REDACTED>.36267.0`
- `demo/hierarchical_chase_hl.mp4` (150 frames, total_rew=142.7)

## 6. GPU / ROCm Environment

- **GPU:** AMD ROCm device (Device ID 0x744b, GUID 6853)
- **Temperature:** 33.0°C, Power: 57.0W/241.0W
- **VRAM:** 35% used
- **Python venv:** /opt/venv/bin/python3
- **PyTorch:** 2.9.1+gitff65f5b, CUDA available=True
- **HIP:** 7.2.53211-e1a6bc5663
- **Genesis:** 1.3.1
- **ONNX Runtime:** 1.28.0 (CPUExecutionProvider only — no ROCm EP)

## 7. Source Code Structure

```
/workspace/radeon-repo/
├── match_worker.py          # Per-robot worker (Genesis env + ONNX inference)
├── match_coordinator.py     # TCP coordinator for multi-process matches
├── soccer_env_hierarchical.py  # Hierarchical env (walk + HL)
├── configs/
│   ├── hierarchical_agent.yaml
│   ├── match_3v3.yaml
│   └── ...
├── src/
│   ├── match_3v3/
│   │   ├── __init__.py      # Exports Scene3v3, SceneConfig, etc.
│   │   ├── policy.py         # SharedRLPolicy, RulePolicy
│   │   ├── scene.py          # PlayerState, BallState, Team, Role
│   │   ├── roles.py          # RoleAssigner
│   │   ├── strategy.py
│   │   ├── result.py         # MatchResult, MatchSummary
│   │   └── multiagent_obs.py
│   ├── booster_agent/
│   └── soccer_env/
└── scripts/
    └── match_eval_3v3.py     # Single-process match evaluator
```

## 8. Policy Architecture

### Hierarchical Design:
1. **t1_walk.pt** — Low-level locomotion policy (720-dim obs → joint targets)
   - Frozen, provides walking capability
   - Located at: /workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt (source)
   - Backup at: /persistent/track3/models/base/t1_walk.pt

2. **chase_v7/v8 ONNX** — High-level chase policy (19-dim obs → 3-dim velocity cmd)
   - MLP: 19→256→128→64→3 (actor), 19→256→128→64→1 (critic)
   - Input: [robot_pos(3), robot_quat(4), robot_vel(3), ball_pos(3), ball_vel(3), ...] = 19
   - Output: [vx, vy, vyaw] velocity command
   - obs_normalizer: Identity (no normalization needed)

3. **RulePolicy** — Geometric rule-based behavior
   - No GPU needed
   - Role-based: attacker/defender/goalkeeper
   - Outputs velocity commands + kick flag

### RL vs Rule Boundary:
- **RL components:** t1_walk.pt (locomotion), chase_v7/v8 ONNX (HL chase direction)
- **Rule components:** Role assignment, kicking, goalkeeper positioning, defender positioning
- **Integration:** HL policy outputs velocity → walk policy executes → joints move

## 9. Previous Failure Analysis

### Root Cause: Multi-issue failure chain
1. **Import error (FIXED):** `from envs.soccer_env_hierarchical` → fixed to `from soccer_env_hierarchical`
2. **Python environment:** Workers launched with `python3` (system) instead of `/opt/venv/bin/python3`
3. **Coordinator timeout:** 600s deadline, but workers crashed immediately on import
4. **Match log:** Shows `n_clients: 0`, all states empty for 25s

### Current State of match_worker.py:
- ✅ Import fix applied (`from soccer_env_hierarchical import SoccerEnvHierarchical`)
- ✅ sys.path includes both root and src/
- ✅ `match_3v3` package exists at `src/match_3v3/`
- ⚠️ ONNX Runtime only has CPUExecutionProvider (no ROCm EP) — should be OK for small MLP
- ❓ Need to verify workers actually start with `/opt/venv/bin/python3`
