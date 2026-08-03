#!/usr/bin/env python3
"""Train 1v1 soccer policy: RL agent vs virtual opponent.

Uses SoccerEnv1v1Virtual (kinematic opponent, no second Genesis entity).
21-dim obs (19 base + 2 opponent relative) → 3-dim action.

Usage:
    python train_1v1.py --max_iterations 500 --num_envs 2048
    python train_1v1.py --max_iterations 100 --num_envs 256  # quick test
"""
import argparse, os, sys, pickle, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml, torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from soccer_env_1v1_virtual import SoccerEnv1v1Virtual


def main():
    parser = argparse.ArgumentParser(description="1v1 Soccer Training")
    parser.add_argument("-e", "--exp_name", default="soccer_1v1")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opponent_speed", type=float, default=0.4)
    parser.add_argument("--pretrained",
                        default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--config", default="configs/hierarchical_agent.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"
    obs_cfg = cfg["obs"]
    reward_cfg = cfg["reward"]
    command_cfg = cfg["command"]
    train_cfg = cfg["train"]
    train_cfg["run_name"] = args.exp_name
    train_cfg["max_iterations"] = args.max_iterations

    hl_cfg = cfg.get("high_level", {})
    decimation = hl_cfg.get("decimation", 5)

    log_dir = f"runs/{args.exp_name}"
    if args.resume is None:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    env = SoccerEnv1v1Virtual(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        walk_model_path=args.pretrained,
        high_level_decimation=decimation,
        show_viewer=False,
        opponent_speed=args.opponent_speed,
        opponent_init_pos=(-3.0, 0.0),
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.resume:
        print(f"\n=== Resuming from {args.resume} ===")
        runner.load(args.resume)

    print(f"\n{'='*60}")
    print(f"  1v1 Soccer Training (Virtual Opponent)")
    print(f"{'='*60}")
    print(f"  Obs dim:    {env.hl_obs_dim} (19 base + 2 opponent)")
    print(f"  Action dim: {env.num_actions} (vx, vy, wz)")
    print(f"  Opponent:   virtual kinematic, speed={args.opponent_speed} m/s")
    print(f"  Envs:       {args.num_envs}")
    print(f"  Max iters:  {args.max_iterations}")
    print(f"  Log dir:    {log_dir}")
    print(f"{'='*60}\n")

    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )

    print(f"\n=== Training complete ===")
    print(f"  Models saved in: {log_dir}")
    print(f"  Best model: {log_dir}/model_{args.max_iterations}.pt")


if __name__ == "__main__":
    main()
