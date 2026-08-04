# Development Guide — Pitfalls, Solutions & Lessons Learned

This document records all technical challenges encountered during development
and their solutions. It serves as a troubleshooting guide for anyone
reproducing or extending this project.

## 1. Genesis ROCm Multi-Entity GPU Crash

### Symptom
Loading two or more robot URDF entities in the same Genesis scene causes:
```
HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception.
hipErrorLaunchFailure: unspecified launch failure
```

### Root Cause
Genesis physics engine on AMD ROCm cannot handle multiple articulated robot
entities (25+ links each) in a single batched scene. VRAM is not the issue
(only 0.9 GB used out of 51.5 GB). This is a platform-level kernel bug.

### Solution: Distributed Multi-Process Architecture
Each robot runs in its own independent Genesis process (proven stable with
1 robot per scene). A socket-based coordinator synchronizes state between
processes at 50Hz.

- `match_coordinator.py`: Central server, manages 2-6 worker connections
- `match_worker.py`: One robot per process, connects to coordinator
- Collision: approximated via coordinator (pairwise distance check, push-back force)

**Verified**: 6 concurrent processes (3v3), 75-84 steps each, zero GPU crash.

## 2. Floating Base Lock (Robot Doesn't Fall Under Gravity)

### Symptom
Robot base position never changes — `h=0.600` stays constant even with no
PD control and only gravity acting. The robot appears frozen in mid-air.

### Root Cause (3 issues)
1. **URDF**: The `world_joint type="floating"` was commented out in `t1.urdf`
2. **Genesis**: `merge_fixed_links=True` (default) merged the `world` link,
   eliminating the floating joint
3. **`_read_state`**: `robot.get_pos()` returned the `world` link position
   (always [0,0,0.6]), not the `Trunk` link position

### Solution
1. Uncomment `world_joint type="floating"` in URDF
2. Set `fixed=False, merge_fixed_links=False` in `gs.morphs.URDF()`
3. Change `_read_state()` to read `robot.links[1].get_pos()` (Trunk link)
4. Set `init_qpos[2] = 0.7` for correct floating base spawn height

## 3. Termination Threshold Unit Mismatch

### Symptom
Robot falls (episode terminates) after 1 step despite being perfectly upright.

### Root Cause
- `base_euler` computed with `degrees=True` → values in **degrees** (e.g., 3.5°)
- `term_pitch = math.radians(30) = 0.5236` → value in **radians**
- Comparison: `abs(3.5) > 0.5236` → True → immediate termination

### Solution
Change termination threshold to degrees:
```python
self.term_pitch = env_cfg.get("termination_pitch_deg", 30)  # was math.radians(30)
self.term_roll = env_cfg.get("termination_roll_deg", 30)
```

## 4. Observation History Timing Error

### Symptom
The frozen `t1_walk.pt` model produces unstable actions in the hierarchical
environment, even though it works perfectly in `archive/verify_t1_walk.py`.

### Root Cause
`_build_low_level_obs()` updated `obs_history` **before** physics simulation,
creating a temporal mismatch. The walk model saw the observation from before
the step, not after.

### Solution
1. `_build_low_level_obs()` now just reads `self.obs_buf` (no history update)
2. `super()._update_observation()` is called **after** `_low_level_step()`
   (after physics), matching the parent env's timing convention

## 5. Conservative Local Optimum (Policy Doesn't Chase Ball)

### Symptom
After training with `approach_ball=1.0`, the RL policy outputs fixed small
velocity commands and stays in place. Ball distance never decreases.

### Root Cause
Survival rewards (upright=5, alive=3) dominate chase rewards (approach_ball=1).
The policy learns "standing still = highest reward" and never risks movement.

### Solution: Curriculum Learning
1. **Stage 1** (clip=0.05): Only balance reward, learn to stand → ep_len=241
2. **Stage 2a** (clip=0.1): Gradually allow larger commands → ep_len=241
3. **Stage 2b** (clip=0.2): Continue expanding → ep_len=241
4. **Final** (clip=0.3/0.4 + chase reward): approach_ball=30, upright=5, alive=3
5. **Chase v3** (approach_ball=30): Ball distance monotonically decreasing,
   0 falls, ep_len 200+

### Key Reward Weights (Final)
| Reward | Weight | Purpose |
|--------|--------|---------|
| upright | 5.0 | Maintain balance (high priority) |
| alive | 3.0 | Survival bonus |
| approach_ball | 30.0 | Chase ball (strong enough to override conservatism) |
| ball_control | 0.5 | Stay close to ball |
| ball_to_goal | 1.0 | Move ball toward goal |
| goal_scored | 10.0 | Score bonus |
| fall_penalty | -5.0 | Penalize falling (not dt-scaled) |
| action_rate | -1.0 | Penalize jerky commands |

## 6. ONNX Export with rsl_rl

### Symptom
`torch.onnx.export()` fails with `IndexError: too many indices for tensor of dimension 2`

### Root Cause
rsl_rl's `MLPModel.forward()` uses `obs[obs_group]` indexing that `torch.export`
cannot trace.

### Solution
Extract the raw `nn.Sequential` MLP directly and export that:
```python
mlp = policy.mlp  # nn.Sequential: Linear(19,256) → ELU → ... → Linear(64,3)
torch.onnx.export(mlp, dummy_input, output_path, opset_version=17)
```

Result: `models/chase_v3_policy.onnx` (19→3 dim, opset 17, 7 nodes)

## 7. BrokenPipe in Multi-Process Match

### Symptom
Workers crash with `BrokenPipeError` when coordinator closes the socket.

### Solution
1. `signal.signal(signal.SIGPIPE, signal.SIG_IGN)` — ignore SIGPIPE globally
2. Wrap all `socket.sendall()` in `try/except (BrokenPipeError, ...)`
3. Workers read exactly one message set per step (not drain-all)
4. 30-second recv timeout (workers wait for coordinator to start broadcasting)
