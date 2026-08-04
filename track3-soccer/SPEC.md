# Track 3 Soccer Robot — Spec-Driven Development (SDD)

> **Spec ID:** T3-SDD-2026-08-05
> **Author:** Simon Xing
> **Date:** 2026-08-05
> **Status:** Active
> **Deadline:** 2026-08-06

---

## 0. Meta

| Field | Value |
|-------|-------|
| Track | Track 3 — Physical AI (人形机器人足球) |
| Hardware | AMD Radeon RX 7900 XT (gfx1100), 48GB VRAM |
| Software | ROCm 7.2.1, PyTorch 2.9.1 (HIP), Genesis 1.3.1, rsl_rl |
| Instance | anruicloud <REDACTED> (重建, GPU 干净) |
| Local tests | 151 passed |
| GitHub (private) | `gxinxing/Radeon-hackathon-2026-07-track3` |
| GitHub (fallback, private) | `gxinxing/Radeon-hackathon-2026-07-track3-2` |

---

## 1. Goal

在 2026-08-06 截止前，交付一条完整的、可验证的证据链：

```
项目变更 → AMD GPU 训练 → 收敛证据 → checkpoint 对比 → 多机器人角色 demo → ROCm 性能
```

满足官方 `COMPETITION_ACCEPTANCE.md` 的 5 大维度 + Go/No-Go 清单。

---

## 2. Official Requirements (Acceptance Criteria)

### 2.1 Scoring Dimensions (100 pts)

| # | Dimension | Pts | Requirement |
|---|-----------|-----|-------------|
| 1 | 训练收敛 | 30 | ① 奖励曲线 + best checkpoint ② 奖励分量（approach_ball / ball_control / ball_progress / goal_scored / fall_penalty）③ 任务指标（进球率/到球距离等） |
| 2 | 基线对比 | (含在 1+5) | 至少两种策略相同条件下对比（规则 vs RL），含跌倒率 + 足球指标。保留行走验证 |
| 3 | 可复现性 | (含在 1+4) | 环境/奖励变更文档、算法/网络/超参/训练命令/评估命令/配置/模型 SHA-256 |
| 4 | AMD ROCm 性能 | 20 | GPU/ROCm 版本、并行环境数、训练吞吐 steps/s、推理延迟/FPS、峰值 VRAM |
| 5 | 多机器人 | 20 | 多机器人加载、独立控制、角色分配、完整比赛生命周期 |

### 2.2 Go/No-Go Checklist

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | 本地测试通过 | ✅ | 151 passed |
| 2 | 校验和/commit ID | ✅ | SHA-256 文件, commit `37b661d` |
| 3 | 低层行走门禁 | ✅ | t1_walk.pt: stance 60步/0跌倒, gait 150步/0跌倒/6.4m |
| 4 | 单 Agent 足球评估 | ✅ | ONNX 100步, 球位移 1.04m, 0跌倒 |
| 5 | 多机器人生命周期 | ✅ | 6 机器人, 10s 干净退出 |
| 6 | **完整演示日志+视频** | ❌ | **关键缺口: 无有效比赛视频** |
| 7 | 奖励曲线和任务指标 | ✅ | training_curve.csv, 500轮 |
| 8 | ROCm 性能测量 | ✅ | 4618 steps/s, 23.7GB VRAM, 0.4ms 推理 |
| 9 | README 命令匹配 | ✅ | 已修正 |
| 10 | GitHub 无密钥/大文件 | ✅ | 已清理 |

### 2.3 Gaps (Must Fix)

| Gap | Root Cause | Fix |
|-----|-----------|-----|
| **无有效比赛视频** | T04 补丁未 apply; 碰撞参数太弱; 踢球逻辑错误; DOF 索引 bug | Apply T04 补丁 → 跑 eval → 录视频 |
| **奖励分量未输出** | 代码支持但 eval 未实际输出 | 跑 eval_hierarchical_short.py 输出 JSON |

---

## 3. Architecture

### 3.1 Control Pipeline (Hierarchical)

```
High-Level PPO Policy (19-dim obs → 3-dim action: vx, vy, wz @ 10Hz)
    │  velocity commands
    ▼
Frozen t1_walk.pt (720-dim obs → 21-dim joint actions @ 50Hz)
    │  joint targets: target = action * 0.25 + policy_default_pos
    ▼
Genesis Physics (AMD Radeon GPU, gfx1100, ROCm 7.2.1)
```

### 3.2 Observation (19-dim, body frame)

| Dims | Content |
|------|---------|
| 0-2 | ball position (body frame) |
| 3-5 | ball velocity (body frame) |
| 6-8 | goal direction (body frame) |
| 9 | distance to ball |
| 10 | distance to goal |
| 11-13 | base angular velocity |
| 14-16 | projected gravity |
| 17-18 | last velocity command (vx, wz) |

### 3.3 Reward Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| approach_ball | 10 | tanh(prev_dist - current_dist), soft-clamped |
| ball_progress | 10 | Potential-based: ball→goal distance reduction |
| ball_to_goal | 8 | Ball velocity toward goal |
| goal_scored | 30 | Binary, episode ends on goal |
| directed_contact | 5 | Foot near ball + ball moving toward goal |
| approach_angle | 3 | Approach from goal-opposite side |
| ball_control | 2 | Foot within 0.15m of ball |
| upright | 0.5 | Torso up projection clamp |
| fall_penalty | -5 | Binary, base height < fall_height |

### 3.4 Key Files

| File | Role |
|------|------|
| `soccer_env_hierarchical.py` | Hierarchical env: frozen walk + trainable HL |
| `soccer_env_v4.py` | Base env: scene, robot, ball, obs (720-dim) |
| `reward.py` | 8-dimensional reward function |
| `configs/hierarchical_agent.yaml` | PPO config, reward scales, env params |
| `scripts/eval_hierarchical_short.py` | Single-robot eval harness (ONNX/Rule) |
| `match_1v1_onnx.py` | 1v1 match: ONNX agent vs rule opponent |
| `scripts/soccer_env_3v3.py` | 3v3 shared-physics env (T04 target) |
| `models/pretrained/t1_walk.pt` | Frozen walking model (720→21) |
| `models/chase_v8_policy.onnx` | Exported high-level policy (19→3) |

---

## 4. Bug Analysis & Fixes

### 4.1 T04 Patch (Ready, Not Yet Applied)

**File:** `.graph_engine/patches/instance_soccer_env_3v3.patch` (3742B, 6 hunks, py_compile OK)

| Bug | Old Value | New Value | Impact |
|-----|-----------|-----------|--------|
| Collision pairs | 256 | 4096 | 6-robot contact detection precision |
| Collision tolerance | 1e-5 | 1e-4 | Stability under multi-robot contact |
| Solver iterations | (none) | 100 | Constraint solver convergence |
| Kick distance | 0.3m | 0.5m | Robot can actually reach ball |
| Kick cooldown | 1.0s | 0.5s | More frequent kicks |
| Chase logic | Abandon ball <0.3m, turn to goal | Always chase, accelerate <0.5m | Ball actually gets kicked |

### 4.2 DOF Index Bug (Diagnosed, Fix Applied, Not Verified)

- **Root cause:** `dof_start` (solver global offset) used as `dof_idx_local` (entity-local index) → pollutes floating base qpos
- **Fix:** Changed to use `joint.dof_idx_local` in `scripts/soccer_env_3v3.py`
- **Status:** Applied locally + remote, but never verified on clean GPU

### 4.3 Newton Solver Memory (Diagnosed)

- **Root cause:** gfx1100 local memory limit 65,536 bytes; 6-robot scene needs 66,560 bytes (exceeds by 1KB)
- **Mitigation:** Use CG sparse solver (`sparse_solve=True`), slower but fits
- **Fallback:** Reduce to 1v1 (2 robots) if 3v3 still crashes

### 4.4 Camera Rendering (Not Yet Solved)

- **Root cause:** `soccer_env_v4.py` uses `gs.renderers.Rasterizer()`. Camera `render()` returns empty frames.
- **Hypothesis:** Need to call `scene.render_all_cameras()` before `cam.render()`, or the Rasterizer needs different camera API.
- **Fallback:** Monkey-patch `gs.Scene` to use default renderer before env init.

---

## 5. Execution Plan

### Phase 1: Deploy T04 Patch (ETA: 10 min)

```
Objective: Apply collision + kick fixes to remote instance

Steps:
1. Upload instance_soccer_env_3v3.patch to /workspace/
2. cd /workspace/amd-physical-ai-soccer
3. git apply --check ../instance_soccer_env_3v3.patch  (dry-run)
4. git apply ../instance_soccer_env_3v3.patch
5. python3 -m py_compile scripts/soccer_env_3v3.py  (verify)
6. grep -n 'max_collision_pairs\|KICK_DISTANCE\|KICK_COOLDOWN' scripts/soccer_env_3v3.py

Acceptance:
- max_collision_pairs=4096, tolerance=1e-4, iterations=100
- KICK_DISTANCE=0.5, KICK_COOLDOWN=0.5
- py_compile passes
```

### Phase 2: Single-Robot Eval + Video (ETA: 30 min)

```
Objective: Verify walk model + ONNX policy on clean GPU, record video, output reward components

Steps:
1. Run eval_hierarchical_short.py with --controller onnx --steps 100
   - Confirm: 0 falls, ball displacement > 0.5m
   - Output: JSON with reward_components (approach_ball, ball_control, etc.)
2. Fix camera rendering (try render_all_cameras() or monkey-patch renderer)
3. Run eval again with camera, output MP4 video (1280x720, 30fps)

Acceptance:
- status=passed, falls=0, ball_displacement > 0.5m
- reward_components JSON has all 5 components with non-null values
- Video file exists, >100 frames, not all-black
```

### Phase 3: 1v1 Match Demo (ETA: 30-60 min)

```
Objective: Two robots in same scene, ONNX vs Rule, record match video

Steps:
1. Run match_1v1_onnx.py with patched env
   - ONNX agent (chase_v8_policy.onnx) vs RulePolicy opponent
   - 200 steps, log ball trajectory + kick events
2. Record camera video during match
3. Output: match_log JSON + MP4 video

Acceptance:
- Both robots stay upright (fall_count < 2)
- Ball displacement > 5m (demonstrates active play)
- At least 1 kick event
- Video file exists, >150 frames
```

### Phase 4: 3v3 Attempt (Optional, ETA: 60 min)

```
Objective: 6 robots, 3 roles each team, full match

Prerequisite: Phase 3 passed, GPU VRAM stable

Steps:
1. Apply DOF index fix (dof_idx_local) if not already
2. Set sparse_solve=True in RigidOptions
3. Run run_3v3.sh with patched env
4. Record video

Acceptance:
- At least 4/6 robots stay upright for 50+ steps
- Ball moves > 2m
- At least 1 kick event
- Video file exists

Fallback: If Newton solver OOM, use 1v1 as multi-robot evidence (2 robots = multi-robot)
```

### Phase 5: Package & Push (ETA: 20 min)

```
Objective: Update GitHub with new evidence

Steps:
1. Download new videos + JSON results to local
2. Update README validation status
3. Update acceptance/ with new eval results
4. Commit and push to gxinxing/Radeon-hackathon-2026-07-track3 (private)
5. Verify Go/No-Go checklist all green

Acceptance:
- Go/No-Go #6 (video): ✅
- Go/No-Go #7 (reward components): ✅
- GitHub has latest code + evidence
```

### Phase 6: Persistent Backup (实时, 贯穿全程)

```
Objective: 确保所有产出物在实例销毁后不丢失

原则: 每一个 Phase 产出的文件，必须立即同步到持久化存储。
      不要等"全部做完"再备份——实例随时可能被销毁。

三层备份策略:

Layer 1 — 远端持久化 (/workspace/persistent/)
  路径: /workspace/persistent/track3/
  内容:
    - models/checkpoints/     ← best.pt, cfgs.pkl
    - demo/                    ← 视频和截图
    - logs/                    ← 训练日志
    - eval/                    ← 评估 JSON
    - benchmark/               ← GPU 遥测
    - match_logs/              ← 比赛日志 JSON
    - tensorboard/             ← 训练曲线
  规则: 每个 Phase 完成后立即执行
    cp /workspace/demo_artifacts/*.mp4 /workspace/persistent/track3/demo/
    cp /workspace/demo_artifacts/*.json /workspace/persistent/track3/eval/
    cp /workspace/demo_artifacts/*.png /workspace/persistent/track3/demo/

Layer 2 — 本地同步 (JupyterLab API 下载)
  路径: 本地 track3-soccer/acceptance/ 和 demos/
  规则: 每个重要产出物通过 JupyterLab contents API 下载到本地
    demo_video.mp4 → demos/
    eval_result.json → acceptance/single_agent/
    match_log.json → match_logs/

Layer 3 — GitHub 推送 (最终保险)
  仓库: gxinxing/Radeon-hackathon-2026-07-track3 (private)
  规则: Phase 5 统一推送，确保代码+证据在 GitHub 上

备份检查清单 (每个 Phase 结束时过一遍):
  [ ] 新产出的文件是否已 cp 到 /workspace/persistent/?
  [ ] 关键 JSON/MP4 是否已下载到本地?
  [ ] 远端 /workspace/ 下是否有未备份的新文件?

风险: 如果实例被意外销毁，/workspace/ 下的文件全部丢失。
      只有 /workspace/persistent/ (PVC) 会保留。
      因此: 产出即备份，不留到最后。
```

---

## 6. Constraint Matrix

| Constraint | Value | Source |
|------------|-------|--------|
| GPU VRAM | 48 GB (clean, ~28MB used after rebuild) | rocm-smi |
| Genesis version | 1.3.1 | pip install genesis-world |
| T1 DOFs | 29 | URDF |
| Walk model input | 720-dim (10 × 72) | t1_walk.pt |
| HL policy input | 19-dim | chase_v8_policy.onnx |
| HL policy output | 3-dim (vx, vy, wz) | ONNX |
| Control freq (HL) | 10 Hz | config decimation=5 |
| Control freq (LL) | 50 Hz | DECIMATION=10 |
| Training envs | 2048 | config |
| Training iterations | 500 | train_v8.log |
| Scene build time (T1) | ~96s (first), ~10s (cached) | measured |
| Scene build time (Franka) | ~10s (first) | measured |
| Newton solver limit | 65,536 bytes (gfx1100) | diagnosis |
| 6-robot Newton need | 66,560 bytes | diagnosis |

---

## 7. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Camera render returns empty | High | Medium | Try 3 approaches: render_all_cameras, monkey-patch renderer, use env camera directly |
| 3v3 Newton solver OOM | High | Low | Use 1v1 as fallback (2 robots still satisfies "multi-robot") |
| DOF index fix not working | Medium | High | Verify on single robot first, then scale up |
| GPU VRAM leak after multiple runs | Medium | Medium | Kill all GPU processes between runs; if leak, destroy+rebuild |
| ONNX model is stub (1973 bytes) | Low | Critical | chase_v8 is 186KB (verified real); v3-v5 are stubs (documented) |
| Instance kernel service dies again | Medium | High | Clean kernels after each run; monitor kernel count |

---

## 8. Success Definition

**Minimum Viable Submission (MVS):**
- ✅ All Go/No-Go items pass except #6 video
- ✅ Single-robot eval: 0 falls, ball displacement > 0.5m
- ✅ Reward components JSON with all 5 components
- ✅ 1v1 match video (both robots upright, ball moves, kick events)
- ✅ Honest documentation of 3v3 limitations

**Full Success:**
- All of MVS plus:
- ✅ 3v3 match video (6 robots, roles, at least 4 upright, ball moves, kicks)
- ✅ 3v3 match log JSON with scored events

---

## 9. Non-Goals

- ❌ Re-training the walk model (t1_walk.pt is frozen, validated)
- ❌ Re-training the high-level policy (chase_v8 is exported, validated)
- ❌ Franka pick-and-place (separate fallback repo, not soccer)
- ❌ Sim2Sim with Booster Studio (T10, future track)
- ❌ Multi-agent 24-dim training (T08, future track)
- ❌ Real robot deployment (out of scope)

---

## 10. RL-Driven Agent Workflow

> 详见 `.graph_engine/rl_workflow.md`

### 评分标准 (-100 到 +100)

**正向指标（做到才加分，做不到得0）：**

| 指标 | 范围 | 计算 |
|------|------|------|
| 机器人位移 | +0~+25 | clamp(disp/2.0,0,1)*25 |
| 球位移 | +0~+25 | clamp(disp/3.0,0,1)*25 |
| Kick次数 | +0~+15 | clamp(kicks/3,0,1)*15 |
| Walk活跃度 | +0~+10 | clamp(std/0.3,0,1)*10 |
| 步数完成 | +0~+5 | done/target*5 |

**负向指标（做不好直接扣）：**

| 指标 | 范围 | 触发 |
|------|------|------|
| 机器人不动 | -25 | disp<0.01m |
| 球不动 | -15 | ball<0.01m |
| Walk全零 | -20 | std<0.001 |
| 跌倒 | -10/次 | fall>0 |
| GPU崩溃 | -30 | HIP error |
| 超时 | -15 | 被杀 |
| 退步 | -10 | delta<-5 |

### 进退规则

| delta | 动作 |
|-------|------|
| >+10 | 继续 |
| 0~+10 | 继续+微调 |
| -5~0 | 换参数/方向/回退 |
| <-5 | 回退+换方向 |
| 总分<-30 | 必须回退 |
| 2轮负分 | 人工裁决 |

### 当前经验表

| 轮次 | 方向 | 总分 | delta |
|------|------|------|-------|
| R0 | 无修复 | -75 | 基线 |
| R1 | 修control_dofs_position | -32 | +43 |
| R2 | 修all_dof_pos初始化 | TBD | ? |

---

## 10. Change Log

| Date | Change |
|------|--------|
| 2026-08-05 | Initial spec created. Based on COMPETITION_ACCEPTANCE.md, Graph Engine task chain T04-T09, and clean GPU validation results. |
