# Training vs. Inference Pipeline Diff Report

**Generated:** 2026-08-01  
**Training code:** `/workspace/radeon-repo/train_hierarchical.py` + `soccer_env_hierarchical.py`  
**Rendering code:** `/workspace/radeon-repo/render_hierarchical.py`  
**Match worker:** `/workspace/radeon-repo/match_worker.py`  
**ONNX policy:** `/workspace/radeon-repo/src/match_3v3/policy.py` (SharedRLPolicy)  
**Config:** `/workspace/radeon-repo/configs/hierarchical_agent.yaml`

---

## Summary

| # | Item | Training | Inference (render_hierarchical) | Inference (match_worker/ONNX) | Match? |
|---|------|----------|-------------------------------|-------------------------------|--------|
| 1 | Env class | `SoccerEnvHierarchical` | `SoccerEnvHierarchical` | `SoccerEnvHierarchical` | ✅ |
| 2 | Robot URDF | `urdf/t1/t1.urdf` | `urdf/t1/t1.urdf` (from config) | `urdf/t1/t1.urdf` (from config) | ✅ |
| 3 | Obs order | See training layout below | Same env class | **Different** (see §3) | ❌ |
| 4 | Obs scales | From config obs_scales | From config obs_scales | Hardcoded in _preprocess_obs | ⚠️ |
| 5 | Action scale | 0.25 (low-level) | 0.25 (from config) | Not applied (raw ONNX output) | ⚠️ |
| 6 | Action clipping | hl_clip_lin/ang from config | Same env | Manual clip in _postprocess | ⚠️ |
| 7 | Joint order | Auto-discovered from URDF | Same env | N/A (HL policy) | ✅ |
| 8 | Control freq | HL: 10Hz (decimation=5) | Same env | Same env | ✅ |
| 9 | PD params | kp=200, kd=5 (from config) | From config | From config | ✅ |
| 10 | Init posture | base_init_pos=[0,0,0.6] | From config | **Overridden** by --init-pos arg | ⚠️ |
| 11 | Ball/field coords | field=[14,9], goal_width=2.6 | From config | From config | ✅ |
| 12 | Reset logic | env.reset() | env.reset() | env.reset() + manual pos override | ⚠️ |
| 13 | Episode termination | pitch>30°, roll>30°, fall<0.8m | Same env | Same env | ✅ |

---

## 1. Environment Class

**Training:** `SoccerEnvHierarchical` (inherits from `SoccerEnv`)  
**Rendering:** `SoccerEnvHierarchical` (try/except import fallback)  
**Match worker:** `SoccerEnvHierarchical` — **BUT IMPORT FAILS**  

### Critical Bug: ModuleNotFoundError

**File:** `match_worker.py` line 81  
**Code:** `from envs.soccer_env_hierarchical import SoccerEnvHierarchical`  
**Problem:** The file `soccer_env_hierarchical.py` is at `/workspace/radeon-repo/` (top level), NOT inside an `envs/` subdirectory. The `envs/` directory only exists in the `amd-physical-ai-soccer/` subdirectory and only contains `soccer_env.py` (the base env, not the hierarchical one).  

**render_hierarchical.py** handles this correctly with:
```python
try:
    from envs.soccer_env_hierarchical import SoccerEnvHierarchical
except ImportError:
    from soccer_env_hierarchical import SoccerEnvHierarchical
```

**match_worker.py** does NOT have this fallback, causing ALL 6 match workers to crash immediately. The match coordinator then runs for 25 seconds with 0 clients connected, producing empty match logs.

**Fix:** Add the same try/except import fallback to `match_worker.py`, or add `sys.path.insert(0, PROJECT_ROOT)` before the import.

---

## 2. Robot Model

Both training and rendering use `urdf/t1/t1.urdf` from the config. The T1 humanoid has 21 motor DOFs (23 total minus 2 fixed). The frozen walk model (`t1_walk.pt`) expects 720-dim input (proprioception with 10-frame history stacking) and outputs 21-dim joint targets.

---

## 3. Observation Order — CRITICAL MISMATCH

### Training Env (`soccer_env_hierarchical.py` `_update_observation`):
```
[0:3]   filtered_lin_vel       — robot velocity in body frame (filtered)
[3:6]   filtered_ang_vel       — robot angular velocity in body frame (filtered)
[6:8]   projected_gravity_xy   — transform_by_quat(global_gravity, inv_base_quat)[:2]
[8:10]  ball_rel_body_xy       — (ball_pos - base_pos) transformed to body frame
[10:12] ball_vel_body_xy        — ball_vel transformed to body frame
[12:13] dist_to_ball            — Euclidean distance to ball
[13:15] goal_dir_body_xy        — goal direction in body frame (normalized)
[15:16] goal_dist               — distance to goal
[16:19] last_hl_actions         — last velocity command [vx, vy, wz]
```

### ONNX Policy (`src/match_3v3/policy.py` `_preprocess_obs`):
```
[0:3]   lin_vel_body            — manual rotation (cos/sin yaw) of player.vel
[3:6]   ang_vel_body            — ⚠️ ZEROS(3) — NOT the actual angular velocity!
[6:8]   grav_xy                 — manual quaternion formula: [2*(x*z-w*y), 2*(y*z+w*x)]
[8:10]  ball_rel_body           — manual rotation of ball_rel
[10:12] ball_vel_body           — manual rotation of ball_vel
[12:13] dist_to_ball             — Euclidean norm
[13:15] goal_dir                 — manual rotation of goal_rel
[15:16] goal_dist               — Euclidean norm
[16:19] last_actions             — last velocity command
```

### Differences:

| Dim | Training | ONNX Policy | Impact |
|-----|----------|-------------|--------|
| [3:6] ang_vel | `filtered_ang_vel` (actual angular velocity) | `np.zeros(3)` | **HIGH** — policy receives zero angular velocity during deployment, but was trained with non-zero values. The policy cannot sense its own rotation, causing incorrect velocity commands. |
| [6:8] gravity | `transform_by_quat([0,0,-1], inv_base_quat)[:2]` | Manual formula `[2*(x*z-w*y), 2*(y*z+w*x)]` | **MEDIUM** — The manual formula computes the xy components of the gravity vector in body frame. If the quaternion convention matches (w,x,y,z), these should be equivalent, but sign errors are possible. |
| [0:3] lin_vel | `filtered_lin_vel` (low-pass filtered) | Raw velocity from match coordinator | **MEDIUM** — No filtering means noisier velocity estimates, but the policy was trained on filtered values. |

---

## 4. Observation Scales

**Training:** Applied in `SoccerEnv._update_observation()` (parent class) when building the 720-dim low-level observation. The high-level 19-dim observation does NOT apply obs_scales directly — the scales are used by the low-level walk model's normalizer.

**ONNX Policy:** Does not apply any obs_scales. The `_preprocess_obs` method uses raw values without normalization.

**Impact:** Low for the high-level policy (it was trained on raw body-frame values), but the low-level walk model expects normalized 720-dim input with the configured scales.

---

## 5. Action Scale

**Training:** `action_scale: 0.25` in config. This is used by the low-level walk model: `target_dof_pos = exec_actions * action_scale + default_dof_pos`. The high-level policy outputs raw velocity commands that are NOT scaled by action_scale.

**ONNX Policy:** The `_postprocess` method clips the raw output to `[-clip_lin, clip_lin]` / `[-clip_ang, clip_ang]` but does NOT apply action_scale. This is correct for the high-level policy.

**render_hierarchical.py:** Uses the same env class, so action_scale is applied correctly in the low-level loop.

---

## 6. Action Clipping

**Training env:** Clips high-level actions to `[-hl_clip_lin, hl_clip_lin]` and `[-hl_clip_ang, hl_clip_ang]` (1.2, 1.2 from config).

**ONNX Policy `_postprocess`:** Clips to the same bounds, plus applies a 0.05 deadzone (`cmd[np.abs(cmd) < 0.05] = 0.0`). The deadzone is NOT present in training.

**Impact:** Small — the deadzone only affects near-zero commands. But if the policy outputs small values, the deadzone could cause the robot to stop in the ONNX path while continuing to move in the env path.

---

## 7. Joint Order

Joint order is auto-discovered from the URDF at runtime. Both training and rendering use the same URDF, so joint order is consistent. The high-level policy does not directly control joints.

---

## 8. Control Frequency

- Physics dt: 0.002s (500 Hz)
- Low-level control dt: 0.02s (50 Hz) — substeps=10
- High-level control dt: 0.1s (10 Hz) — decimation=5

All consistent between training and rendering.

---

## 9. PD Parameters

kp=200.0, kd=5.0 — set in config and applied to robot joints via `robot.set_dofs_kp/kv`. Consistent.

---

## 10. Initial Posture

**Training:** `base_init_pos: [0.0, 0.0, 0.6]` from config.

**render_hierarchical.py:** Uses the same config value.

**match_worker.py:** Overrides `env.init_base_pos` and `env.init_qpos` with the `--init-pos` argument (e.g., `-3 0 0.7`, `3 0 0.7`). The z-height is set to 0.7 instead of 0.6.

**Impact:** Medium — different initial heights (0.6 vs 0.7) could affect the first few steps of the policy, but should not cause persistent failure.

---

## 11. Ball and Field Coordinates

Field: 14.0 × 9.0, goal_width: 2.6, ball_radius: 0.11. Consistent across all scripts.

---

## 12. Reset Logic

**Training:** `env.reset()` sets robot to `init_qpos`, samples random ball position, zeros all velocities.

**render_hierarchical.py:** Calls `env.reset()` then `env.scene.reset()`. The scene reset may reinitialize the visualizer but should not affect physics.

**match_worker.py:** Calls `env.reset()` then manually overrides `init_base_pos` and `init_qpos[0, 0:3]`. This partial override does NOT update the full init_qpos (only the position part), which could leave the robot in an inconsistent state.

---

## 13. Episode Termination

Termination conditions (from SoccerEnv.step):
- `episode_length > max_episode_length` (timeout)
- `|pitch| > 30°`
- `|roll| > 30°`
- Scene rigid solver error

Consistent across training and rendering.

---

## Additional Issues Found

### A. Model Checkpoint Path Mismatch

**run_3v3_final.sh** references `runs/hierarchical_soccer_chase_hl/model_499.pt`, but this run directory only contains `model_0.pt` and `model_49.pt`. The training log shows 500 iterations completed, but only 2 checkpoints were saved (possibly due to disk space or a changed save_interval).

**Available complete runs:**
- `hierarchical_soccer_coop_hl` — model_0 to model_499 (500 iterations)
- `curriculum_p4` — model_700 to model_996 (latest, most trained)

### B. ONNX Export Failure

Post-training log shows: `ERROR: No model found at runs/hierarchical_soccer_chase_hl/model_500.pt`. The export script looks for `model_500.pt` but training only runs to iteration 499 (0-indexed). The rsl_rl library saves the final model as `model_499.pt`, not `model_500.pt`.

### C. render_training.py Wrong Task

`render_training.py` hardcodes `env_cfg["task"] = "balance"` and `log_dir = "runs/booster_soccer_balance"`. This would load a non-existent model and render with the wrong task (balance instead of chase_hl).

### D. render_all.py Wrong Paths

`render_all.py` hardcodes `sys.path.insert(0, "/workspace/amd-physical-ai-soccer")` and `os.chdir("/workspace/amd-physical-ai-soccer")`. The actual code is at `/workspace/radeon-repo/`, not `/workspace/amd-physical-ai-soccer/`.

### E. match_evaluator.py Is a Stub

`match_evaluator.py` uses `np.random.poisson` to generate fake match statistics. It does NOT run any actual Genesis simulation. Match results from this script are synthetic and do not reflect actual model performance.

### F. Observation Filtering Mismatch

The training env uses `filtered_lin_vel` and `filtered_ang_vel` (low-pass filtered), but the ONNX policy uses raw velocity from the match coordinator (which is itself a noisy position-difference estimate). This means the policy receives noisier, unfiltered input during deployment.

---

## Required Fixes

1. **match_worker.py**: Add import fallback for `soccer_env_hierarchical` (try `envs.` prefix first, then top-level)
2. **match_worker.py**: Fix model path to use actual available checkpoint (curriculum_p4/model_996.pt or coop_hl/model_499.pt)
3. **ONNX policy _preprocess_obs**: Set `ang_vel_body` to actual angular velocity instead of zeros
4. **ONNX policy _preprocess_obs**: Verify projected_gravity computation matches `transform_by_quat` convention
5. **render_training.py**: Change task from "balance" to "chase_hl" and fix log_dir
6. **render_all.py**: Fix hardcoded paths to `/workspace/radeon-repo`
7. **match_worker.py**: Fix init_qpos override to update full state, not just position
8. **ONNX export script**: Fix model_500.pt reference to model_499.pt
