# AMD Physical AI — Humanoid Soccer (Booster-inspired)

Track 3 (Physical AI) submission for the **2026 AMD AI DevMaster Hackathon**.
A humanoid soccer policy trained entirely in simulation (Genesis) on **AMD Radeon GPU + ROCm**.
Task design is anchored on real Booster T1 RoBoLeague 3v3 gameplay: chase → dribble → shoot,
with recovery-from-fall and attacker/defender role coordination.

## Why this project

Robot soccer is the most visible embodied-AI benchmark right now (Booster T1 / RoboCup /
RoBoLeague). We decompose real soccer skills observed in Booster 3v3 matches into RL sub-tasks
and train them on a single AMD Radeon GPU, demonstrating GPU-accelerated training + inference
via the ROCm software stack.

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

## Quick start (on the AMD Radeon Linux cloud instance)

```bash
# 0. verify GPU is visible
rocm-smi

# 1. PyTorch on ROCm (match the ROCm version of the instance, e.g. rocm6.2)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#   -> expect: True  <your Radeon name>   (under ROCm, torch.cuda maps to the AMD GPU)

# 2. sim + RL deps
pip install -r requirements.txt

# 3. sanity check the sim + confirm the robot URDF path the env will load
python -c "import genesis as gs; gs.init(backend=gs.gpu); print('genesis ok')"
#    VERIFY robot asset exists; the env loads configs/soccer_agent.yaml -> env.robot_urdf
#    (default urdf/h1/urdf/h1.urdf). Find the asset dir and confirm:
python -c "import genesis,os; print(os.path.join(os.path.dirname(genesis.__file__),'assets'))"
#    If H1 is elsewhere, edit env.robot_urdf in configs/soccer_agent.yaml accordingly.

# 4. train week-1 baseline (balance + chase-ball). Config is read from configs/soccer_agent.yaml.
python scripts/train.py --task chase

# 5. GPU throughput report (earns the 20 ROCm pts)
python scripts/benchmark_gpu.py
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
  configs/soccer_agent.yaml   # rsl-rl PPO + task + robot + field hyperparams
  envs/soccer_env.py          # Genesis humanoid soccer env (headless, GPU-batched)
  rewards/reward.py           # Booster-derived reward curriculum
  assets/ball.urdf            # soccer ball (sphere, no mesh needed)
  assets/goal.urdf            # goal-line marker (visual)
  scripts/train.py            # training entry (rsl-rl OnPolicyRunner)
  scripts/eval.py             # rollout + record demo video
  scripts/benchmark_gpu.py    # ROCm throughput / utilization report
  policies/ppo_runner.py      # (legacy skrl wiring, kept for reference)
  docs/ARCHITECTURE.md        # design notes + Booster mapping
  demo/                       # exported mp4 demos
```

## Notes / pitfalls

- sim-to-real is **not** required for the hackathon — Booster real-robot intuition is used to
  *design tasks and rewards*, not to deploy to hardware.
- Always init Genesis headless on the cloud (`show_viewer=False`); render offscreen for demo mp4.
- Start from the official Genesis locomotion example, then modify. Use PPO baseline first.
