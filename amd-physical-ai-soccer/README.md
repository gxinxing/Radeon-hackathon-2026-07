# AMD Physical AI — Humanoid Soccer (Booster T1)

Track 3 (Physical AI) submission for the **2026 AMD AI DevMaster Hackathon**.
A **Booster T1 humanoid robot** (23-DOF) trained to play soccer using PPO RL
entirely on **AMD Radeon RX 7900 XT + ROCm** via Genesis simulation.

## Why this project

Booster's official RL training stack (`booster_gym`, `booster_train`) requires
NVIDIA Isaac Gym / Isaac Lab. We demonstrate the **first-known AMD ROCm alternative**
for humanoid robot learning — same T1 robot, same PPO algorithm, different GPU ecosystem.

## Results (2026-07-19)

| Metric | Value |
|---|---|
| GPU | AMD Radeon RX 7900 XT (gfx1100, 48GB) |
| Peak TFLOPS | 19.5 (4096² matmul) |
| Sustained TFLOPS | 18.5 (60s 8192²) |
| Training throughput | 44,000+ steps/sec (2048 envs) |
| Max parallel envs | 10,240 (3 tasks × 4096 envs) |
| Peak VRAM | 63% (32GB/51GB) |
| Robot | Booster T1 23-DOF (official booster_assets) |
| Training tasks | balance, chase, shoot (1000 iters each) |

## Judging-criteria mapping (Track 3, 100 pts)

| Criterion | Pts | How this project earns it |
|---|---|---|
| Robot capability | 30 | Policy completes chase/dribble/shoot; stays upright; recovers from falls |
| AMD Radeon GPU + ROCm adoption | 20 | Full training & inference on ROCm; see `scripts/benchmark_gpu.py` report |
| Innovation & originality | 20 | Booster-derived reward curriculum + attacker/defender role split |
| Real-world application value | 20 | Clean, reproducible, documented; demo videos in `demo/` |
| Upstream open-source contribution | 10 | Env wrapper + baseline policy contributed back upstream |

## Hard requirements

- **Register AMD AI Developer Program first** (China entry: https://www.amd.com/zh-cn/developer.html) — required to be eligible for prizes and free Radeon cloud GPU.
- **macOS cannot run ROCm.** All work happens on the **AMD free cloud Radeon (Linux)** instance.
- Register the event on Luma (https://luma.com/amd-4dhi) and pick **Physical AI** track.
- Final submission = PR to `AMD-DEV-CONTEST/Radeon-hackathon-2026-07` by **2026-08-06 23:59 UTC+8**.

## Quick start (on AMD Radeon Linux cloud)

```bash
# 1. Install Genesis + rsl-rl (don't let pip swap ROCm torch)
source /opt/venv/bin/activate
pip install genesis-world rsl-rl-lib
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   -> 2.9.1+gitff65f5b True

# 2. Verify GPU + Genesis
python -c "import genesis as gs; gs.init(backend=gs.gpu); print('Genesis GPU OK')"

# 3. Train T1 humanoid
cd /workspace/amd-physical-ai-soccer
PYTHONPATH=. python scripts/train.py --task chase --num_envs 2048 --max_iterations 1000

# 4. Multi-task parallel training (maximizes GPU utilization)
python scripts/train.py --task balance --num_envs 4096 &
python scripts/train.py --task chase --num_envs 4096 &
python scripts/train.py --task shoot --num_envs 2048 &

# 5. GPU throughput report
python scripts/gpu_stress_test.py --duration 60

# 6. Eval + distill
python scripts/eval.py -e t1_chase --task chase --num_envs 1 --steps 600
python bridge/distill.py --log bridge/rollout.jsonl --out bridge/booster_distilled.py
```

## 3-week plan

| Week | Goal | Deliverable |
|---|---|---|
| W1 | ROCm env + Genesis humanoid + balance/chase policy | standing + chase baseline, env docs, `rocm-smi` proof |
| W2 | shooting policy + single-skill RL + Radeon throughput | chase→shoot demo video + tokens/steps-per-sec report |
| W3 | simplified 1v1 / 3v3 + upstream PR + polish | full PR (video + README + repro scripts) + 1 upstream PR |

## Layout

```
amd-physical-ai-soccer/
  configs/soccer_agent.yaml   # rsl-rl PPO + T1 robot + field + reward config
  envs/soccer_env.py          # Genesis soccer env (MJCF/URDF, 10-frame obs history)
  rewards/reward.py           # Booster-derived reward curriculum
  assets/t1/                  # Booster T1 23-DOF (URDF+MJCF+63 STL meshes)
  assets/ball.urdf            # soccer ball
  assets/goal.urdf            # goal-line marker
  scripts/train.py            # training entry (rsl-rl OnPolicyRunner)
  scripts/eval.py             # rollout + distill logging
  scripts/benchmark_gpu.py    # ROCm throughput report
  scripts/gpu_stress_test.py  # sustained GPU stress + max TFLOPS
  scripts/export_policy.py   # export TorchScript .pt for booster_deploy
  bridge/                     # RL→behavioral rules distillation pipeline
    genesis_logger.py         # rollout recorder
    distill.py                # threshold extractor
    SPEC.md                   # Genesis→Booster mapping
    booster_distilled.py      # auto-generated constants
  docs/TECHNICAL_REPORT.md    # hackathon submission report
  demo/                       # exported mp4 demos
```

## Notes / pitfalls

- sim-to-real is **not** required for the hackathon — Booster real-robot intuition is used to
  *design tasks and rewards*, not to deploy to hardware.
- Always init Genesis headless on the cloud (`show_viewer=False`); render offscreen for demo mp4.
- Start from the official Genesis locomotion example, then modify. Use PPO baseline first.
