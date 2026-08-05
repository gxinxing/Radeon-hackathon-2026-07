#!/usr/bin/env python3
"""4-Phase Curriculum Training for 1v1 Soccer.

P1 (0-200):    Basic navigation — no opponent
P2 (200-450):  Weak opponent (0.1 m/s) — dribble + avoidance
P3 (450-700):  Kick timing — 4th action dim triggers kick
P4 (700-1000): Full confrontation — strong opponent (0.5 m/s)

Fixed 24-dim obs + 4-dim action throughout.
Phase transitions happen at runtime via env.set_phase().
"""
import argparse, os, sys, pickle, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml, torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from soccer_env_curriculum import SoccerEnvCurriculum

# Phase schedule: (start_iter, end_iter, phase_id, opponent_speed)
PHASES = [
    (0,    200,  0, 0.0),   # P1: no opponent, basic navigation
    (200,  450,  1, 0.1),   # P2: weak opponent, dribble + avoidance
    (450,  700,  2, 0.3),   # P3: kick timing learning
    (700,  1000, 3, 0.5),   # P4: full confrontation
]

def get_phase(iteration):
    for start, end, phase_id, opp_speed in PHASES:
        if start <= iteration < end:
            return phase_id, opp_speed, start, end
    return 3, 0.5, 700, 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", default="soccer_curriculum")
    parser.add_argument("--pretrained",
                        default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open("configs/hierarchical_agent.yaml") as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"
    hl_cfg = cfg.get("high_level", {})

    # Actor: 24→256→128→64→4 (fixed throughout all phases)
    train_cfg = cfg["train"]
    train_cfg["run_name"] = args.exp_name
    train_cfg["max_iterations"] = args.max_iterations
    train_cfg["actor"]["hidden_dims"] = [256, 128, 64]
    train_cfg["critic"]["hidden_dims"] = [256, 128, 64]

    log_dir = f"runs/{args.exp_name}"
    if args.resume is None:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, cfg["obs"], cfg["reward"], cfg["command"], train_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    env = SoccerEnvCurriculum(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        command_cfg=cfg["command"],
        walk_model_path=hl_cfg.get("walk_model_path"),
        high_level_decimation=hl_cfg.get("decimation", 5),
        show_viewer=False,
        phase=0,
        opponent_speed=0.0,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.resume:
        print(f"=== Resuming from {args.resume} ===")
        runner.load(args.resume)

    print(f"\n{'='*60}")
    print(f"  4-Phase Curriculum Training")
    print(f"{'='*60}")
    print(f"  P1 (0-200):    Basic navigation, no opponent")
    print(f"  P2 (200-450):  Weak opponent (0.1 m/s), dribble+avoid")
    print(f"  P3 (450-700):  Kick timing (4th action dim)")
    print(f"  P4 (700-1000): Full confrontation (0.5 m/s)")
    print(f"  Obs: 24 dim | Action: 4 dim | Envs: {args.num_envs}")
    print(f"  Log dir: {log_dir}")
    print(f"{'='*60}\n")

    current_phase = -1
    runner.learn(num_learning_iterations=args.max_iterations,
                 init_at_random_ep_len=True)
