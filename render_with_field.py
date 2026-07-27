#!/usr/bin/env python3
"""Render demo video with full soccer field scene.

This script loads the trained policy in a SINGLE-env scene
that includes the full soccer field (green ground, white lines,
goals, ball) for video production.

Usage:
    python render_with_field.py --task balance
    python render_with_field.py --task chase
    python render_with_field.py --task shoot
"""
import sys, os, glob, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml
import torch
import imageio
import genesis as gs
from genesis.utils.misc import tensor_to_array

# Field constants
FIELD_L = 14.0
FIELD_W = 9.0
HALF_L = FIELD_L / 2
HALF_W = FIELD_W / 2
GOAL_W = 2.6
GOAL_H = 1.0
POST_R = 0.05
CIRCLE_R = 1.5
LINE_H = 0.005
LINE_W = 0.12
BALL_R = 0.11


def build_field_scene(robot_path, task):
    """Build a single-env scene with full soccer field for rendering."""
    gs.init(backend=gs.gpu, logging_level="warning")

    with open("configs/soccer_agent.yaml") as f:
        cfg = yaml.safe_load(f)
    env_cfg = dict(cfg["env"])
    env_cfg["task"] = task

    from envs.soccer_env import SoccerEnv
    env = SoccerEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        command_cfg=cfg["command"],
        show_viewer=False,
    )

    # Get the camera that was already added in _build_scene
    cam = env.scene.visualizer.cameras[0]
    return env, cam, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="balance", choices=["balance", "chase", "shoot"])
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("configs/soccer_agent.yaml") as f:
        cfg = yaml.safe_load(f)
    env_cfg = dict(cfg["env"])
    env_cfg["task"] = args.task

    gs.init(backend=gs.gpu, logging_level="warning")

    from envs.soccer_env import SoccerEnv
    from rsl_rl.runners import OnPolicyRunner

    env = SoccerEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        command_cfg=cfg["command"],
        show_viewer=False,
    )
    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Load trained model
    log_dir = f"runs/booster_soccer_{args.task}"
    model_files = sorted(glob.glob(f"{log_dir}/model_*.pt"), key=os.path.getmtime)
    if not model_files:
        print(f"No model found in {log_dir}")
        return
    print(f"Loading: {model_files[-1]}")

    runner = OnPolicyRunner(env, cfg["train"], log_dir, device=gs.device)
    runner.load(model_files[-1])
    policy = runner.get_inference_policy(device=gs.device)

    # Rollout + render
    obs = env.reset()
    frames = []
    for i in range(args.steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        if i % 2 == 0:
            rgb, _, _, _ = cam.render(rgb=True)
            arr = tensor_to_array(rgb)
            if arr.ndim == 4:
                arr = arr[0]
            frames.append(arr.astype(np.uint8))
        if (i + 1) % 100 == 0:
            print(f"  step {i+1}/{args.steps}  rew={rew.mean().item():.3f}")

    os.makedirs("demo", exist_ok=True)
    video_path = f"demo/{args.task}_field_demo.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    print(f"\nVideo saved: {video_path} ({len(frames)} frames)")
    print("DONE")


if __name__ == "__main__":
    main()
