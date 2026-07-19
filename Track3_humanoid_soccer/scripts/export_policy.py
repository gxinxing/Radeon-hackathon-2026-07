#!/usr/bin/env python
"""Export trained actor network to TorchScript .pt for booster_deploy.

booster_deploy expects a torch.jit.ScriptModule that takes the observation
tensor and returns action tensor. This script loads the latest checkpoint,
extracts the actor, scripts it, and saves as .pt.

Usage:
    python scripts/export_policy.py -e t1_chase
    python scripts/export_policy.py -e t1_shoot -o models/t1_soccer.pt
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--config", default="configs/soccer_agent.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    log_dir = f"runs/{args.exp_name}"

    # find latest checkpoint
    model_files = sorted(glob.glob(f"{log_dir}/model_*.pt"), key=os.path.getmtime)
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt found in {log_dir}")
    latest = model_files[-1]
    print(f"Loading checkpoint: {latest}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=42)

    # We only need the policy network, not the full env
    # Reconstruct runner with minimal env to load checkpoint
    from envs.soccer_env import SoccerEnv

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = cfg.get("task", "chase")
    env = SoccerEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        command_cfg=cfg["command"],
        show_viewer=False,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(latest)
    policy = runner.get_inference_policy(device=gs.device)

    # Get the actual actor network
    actor = runner.alg.actor_critic.actor
    actor.eval()

    # Export to TorchScript
    scripted = torch.jit.script(actor)

    out_path = args.output or f"models/{args.exp_name}_policy.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted.save(out_path)
    print(f"Exported TorchScript policy: {out_path}")
    print(f"  obs_dim: {env.obs_buf.shape[1]}")
    print(f"  action_dim: {env.num_actions}")
    print(f"  history_length: {env.obs_history_length}")

    # Verify: load and test
    loaded = torch.jit.load(out_path, map_location="cpu")
    loaded.eval()
    test_obs = torch.zeros(1, env.obs_buf.shape[1])
    test_action = loaded(test_obs)
    print(f"  test inference: obs{test_obs.shape} -> action{test_action.shape}")
    print("Done. Use this .pt with booster_deploy.")


if __name__ == "__main__":
    main()
