#!/usr/bin/env python3
"""Render demo video for hierarchical soccer policy.

Loads the trained high-level PPO model + frozen t1_walk.pt,
runs rollout in a single-env scene with soccer field, and saves MP4.

Usage (on cloud):
    cd /workspace/amd-physical-ai-soccer
    python render_hierarchical.py --steps 300
    python render_hierarchical.py --model runs/hierarchical_soccer_chase_hl/model_100.pt
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml
import torch
import imageio
import genesis as gs
from genesis.utils.misc import tensor_to_array

try:
    from envs.soccer_env_hierarchical import SoccerEnvHierarchical
except ImportError:
    from soccer_env_hierarchical import SoccerEnvHierarchical

from rsl_rl.runners import OnPolicyRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Path to model checkpoint")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--task", default="chase_hl")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("configs/hierarchical_agent.yaml") as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = args.task
    hl_cfg = cfg.get("high_level", {})

    gs.init(backend=gs.gpu, logging_level="warning")

    env = SoccerEnvHierarchical(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        command_cfg=cfg["command"],
        walk_model_path=hl_cfg.get(
            "walk_model_path",
            "/workspace/booster/booster_deploy/tasks/locomotion/models/t1_walk.pt",
        ),
        high_level_decimation=hl_cfg.get("decimation", 5),
        show_viewer=False,
    )

    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Apply robot surface material for better visual quality
    try:
        robot_surface = gs.surfaces.Rough(color=(0.3, 0.5, 0.9), roughness=0.6)
        for link in env.robot.links:
            if hasattr(link, 'set_surface'):
                link.set_surface(robot_surface)
    except Exception:
        pass  # surface API varies across Genesis versions

    # Find model checkpoint
    log_dir = f"runs/hierarchical_soccer_{args.task}"
    if args.model:
        model_path = args.model
    else:
        model_files = sorted(glob.glob(f"{log_dir}/model_*.pt"), key=os.path.getmtime)
        if not model_files:
            print(f"No model found in {log_dir}")
            return
        model_path = model_files[-1]

    print(f"Loading: {model_path}")
    runner = OnPolicyRunner(env, cfg["train"], log_dir, device=gs.device)
    runner.load(model_path)
    policy = runner.get_inference_policy(device=gs.device)

    # Rollout + render
    obs = env.reset()
    frames = []
    total_rew = 0

    for i in range(args.steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        total_rew += rew.mean().item()

        if i % 2 == 0:
            try:
                rgb, _, _, _ = cam.render(rgb=True)
                arr = tensor_to_array(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
                frames.append(arr.astype(np.uint8))
            except Exception as e:
                if i == 0:
                    print(f"  Camera render failed: {e}")

        if (i + 1) % 50 == 0:
            print(f"  step {i+1}/{args.steps}  rew={rew.mean().item():.3f}  "
                  f"height={env.base_pos[0,2].item():.3f}  "
                  f"ball_dist={torch.norm(env.base_pos[0,:2]-env.ball_pos[0,:2]).item():.2f}")

    os.makedirs("demo", exist_ok=True)
    video_path = f"demo/hierarchical_{args.task}.mp4"

    if frames:
        imageio.mimsave(video_path, frames, fps=30)
        print(f"\nVideo saved: {video_path} ({len(frames)} frames, total_rew={total_rew:.1f})")
    else:
        print(f"\nNo frames rendered (camera issue), total_rew={total_rew:.1f}")

    print("DONE")


if __name__ == "__main__":
    main()
