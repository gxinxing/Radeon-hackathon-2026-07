# ⚽ 人形机器人足球 — AMD Radeon GPU（赛道三）

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%207.2.1-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![Genesis](https://img.shields.io/badge/Genesis-1.3.1-blue)](https://genesis-embodied-ai.github.io/)
[![rsl_rl](https://img.shields.io/badge/rsl__rl-5.4.2-green)](https://github.com/leggedrobotics/rsl_rl)

**AMD AI DevMaster Hackathon 2026 — 赛道三：Physical AI（人形机器人足球）**

基于 AMD Radeon (ROCm) 的分层足球训练：高层 PPO 策略学习 平衡→追球→踢球，驱动冻结的行走进模模型，在 Genesis 物理引擎中运行。全程不使用 NVIDIA 硬件。

## 项目状态（2026-08-06，10h 竞速窗口）

| 项目线 | 状态 | 证据 |
|--------|--------|----------|
| **主提交线（Path-B）：单机器人射门** | ✅ **PASSED** | chase → kick → goal (`scored=True`, ball 7.35m), 0 falls, `demos/exp/match_1v1_shoot_20260805.mp4` |
| **Task-9 10h 竞速训练（v2 参数）** | 🔄 进行中 | P1 no-opponent chase: per-episode falls 1.0→0.8, mean reward ~300, episode length ~188/200 (`training_logs/task9_p1.log`) |
| 基线（训练前，用于对比） | ❌ FAILED | `fallen=300`, `robot_disp=0.78m`, `kicks=0` — see `reports/task9_baseline/` |
| 3v3 共享物理场景 | ⚠️ Demo-level | 6-robot rule-walk demo renders (green field); neural-walk multi-robot robustness still limited |

**提交策略**：单机器人射门视频为保底交付物；10h Task-9 窗口向 3v3 稳定性训练（降倒地 + 位移 + 踢球），止损线与门禁见 `SPEC.md` §3（Task-9-v2）。

## 架构

```text
High-Level PPO policy (19-dim obs → 3-dim action: vx, vy, wz @ 10 Hz)
    │  velocity commands (clipped: lin/ang ≤ 0.6)
    ▼
Frozen walk model t1_walk.pt (720-dim obs → 21-dim joint targets @ 50 Hz)
    │  action_scale=0.16, clip_actions=1.2
    ▼
Genesis physics on AMD Radeon GPU (ROCm 7.2.1, gfx1100, 2048 parallel envs)
```

### Task-9-v2 reward / hyper-parameter snapshot (authoritative: `configs/hierarchical_agent.yaml`)

| Key | Value | Purpose |
|-----|-------|---------|
| `hl_clip_lin / hl_clip_ang` | **0.6 / 0.6** | Cap high-level speed, suppress sprint-falls |
| `action_scale` / `clip_actions` | **0.16 / 1.2** | Constrain low-level joint targets |
| `fall_penalty` | **-14.0** | Strong disincentive to fall |
| `alive` / `upright` | **0.35 / 1.3** | Reward staying upright |
| `action_rate` / `energy_penalty` | **-2.0 / -0.02** | Penalize jitter |
| `approach_ball` / `ball_progress` / `ball_to_goal` | **5.0 / 6.0 / 5.0** | Chase + advance the ball |
| `goal_scored` | **30.0** | Terminal goal reward |
| `learning_rate` | **0.00015 (1.5e-4, LOCKED)** | Stability; never raised |
| `episode_length_s` | 20.0 | Shorten episodes, less garbage data |

## 10h 竞速计划（A → B → C）

| 阶段 | 时间窗 | 任务 | 硬性规则 |
|-------|--------|------|------------|
| A (P1) | 0–4h | No-opponent chase (`task=chase_hl`, 2048 envs) | Checkpoint every 25 iters → `models/task9_p1.pt`; stop-loss if `per_ep fallen > 80` (raise `fall_penalty=-18`, cut `hl_clip=0.5`) |
| B (1v1) | 4–8h | 1v1 adversarial training | **No 3v3 / no coop training**; gate to enter: `fallen < 60` & `robot_disp ≥ 1.8m` |
| C (Eval) | 8–10h | Freeze weights, batch 3v3 evaluation only | No further weight updates; save `models/task9_1v1.pt` |

Checkpoints (local time): 2h = 04:37 · 4h = 06:37 · 8h = 10:37 · 10h = 12:37.

## 演示视频

| Video | Path | Content |
|-------|------|---------|
| Single-robot shoot (primary) | `demos/exp/match_1v1_shoot_20260805.mp4` | chase → kick → goal, 0 falls |
| Final film (4:37, 1080p) | `acceptance/final_video/track3_final_20260806.mp4` | narration, no black/blue frames |
| Single-robot chase | `demos/match_1v1_20260805.mp4` | 200 steps, 0 falls, ball 12m |
| 3v3 rule-walk | `demo_artifacts/match_rule_walk.mp4` | 6-robot scene, green field, 100 frames |

## 快速开始（AMD 实例 `/workspace`）

```bash
# P1 training (no opponent, Task-9-v2 params)
/opt/venv/bin/python3.12 run_task9_p1.py --max_iterations 240 --num_envs 2048 --phase A
# resume from latest checkpoint
/opt/venv/bin/python3.12 run_task9_p1.py --max_iterations 240 --num_envs 2048 --phase A --resume
# single-scene evaluation of the trained policy (300 steps)
/opt/venv/bin/python3.12 run_task9_eval.py --ckpt runs/task9_p1/model_latest.pt --steps 300
# single-robot shoot demo (Path-B)
/opt/venv/bin/python3.12 run_1v1_shoot.py
```

本地（无 GPU）： `python3 -m pytest track3-soccer/tests/ --ignore=tests/test_e2e.py -q`.

## 目录结构（官方规范：`SPEC.md` §1.2 / §12）

```text
track3-soccer/
├── SPEC.md                  # single source of truth (tasks, gates, stop-losses)
├── FILES_MANIFEST.md        # real-path index + dead-path blacklist
├── run_task9_p1.py          # P1 training script (rsl_rl 5.4.2, resume support)
├── run_task9_eval.py        # single-scene policy evaluation
├── configs/                 # hierarchical_agent.yaml (v2 params)
├── scripts/                 # soccer_env_v4.py, soccer_env_3v3.py, reward.py, ...
├── demos/                   # submission videos (match_1v1_*, exp/)
├── demo_artifacts/          # 3v3 match outputs (protected)
├── reports/                 # task9_baseline/, checkpoints, evidence
├── acceptance/              # final film, acceptance evidence
└── docs/                    # technical_report.md etc.
```

持久目录（实例重建不丢）： `/workspace/persistent/track3/` on the remote —
scripts, configs, checkpoints, logs, baseline and `FILES_MANIFEST.md` are mirrored there.

## 验证状态（已实测）

| Check | Result |
|-------|--------|
| Local tests | 151 passed |
| Single-robot shoot (Task-B) | 300 steps, 0 falls, **scored=True**, ball 7.35m |
| Single-robot chase | 200 steps, 0 falls, ball 12m |
| P1 training throughput | ~4,600 steps/s, 2048 envs on AMD Radeon |
| 3v3 rule-walk demo | 100-frame green-field video |
| Baseline (pre-training) | fallen=300 / disp=0.78m / kicks=0 — FAILED (backup in `reports/task9_baseline/`) |

## 已知限制

- **3v3 neural-walk robustness**: the frozen walk model was trained single-robot; shared-physics
  contact perturbation still causes falls in 6-robot scenes. Task-9 (1v1 + disturbance) targets
  this; the single-robot shoot path remains the guaranteed submission.
- **Close-range ball control** (~2m): no fine dribbling.
- **ROCm solver**: Newton solver exceeds gfx1100 local-memory limits for 6 robots; CG solver
  fallback is slower but functional.

## 官方评分维度对齐

| Official requirement | Where |
|----------------------|-------|
| Technical report | `docs/technical_report.md` |
| Source code + Docker | this repo + `Dockerfile` |
| Reproducibility | `## Quick Start` above |
| Demo video | `demos/exp/match_1v1_shoot_20260805.mp4` (+ final film) |
| Supplementary | `demos/`, `acceptance/`, `reports/task9_baseline/` |
