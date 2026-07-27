#!/usr/bin/env python3
"""Render a goal-scoring showreel from a trained hierarchical soccer policy.

A single short rollout often scores no goal (unlucky ball start / angle).
This script runs one long rollout, watches for any goal event, and trims
the frames around the FIRST goal into a short clip. If no goal happens in
the whole rollout it falls back to the closest-approach window and warns.

Usage (on cloud, after training):
    cd /workspace/amd-physical-ai-soccer
    python render_goal_video.py --model runs/<exp>/model_300.pt --max_steps 1500

Outputs:
    demos/goal_showreel.mp4   (clip centered on first goal; the 30-pt demo)
    demos/chase_full.mp4      (full rollout, fallback if no goal)
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
    ap.add_argument("--model", default=None, help="Checkpoint path (default: latest in runs/hierarchical_soccer_chase_hl)")
    ap.add_argument("--task", default="chase_hl")
    ap.add_argument("--max_steps", type=int, default=1500, help="Total sim steps in the rollout")
    ap.add_argument("--pre", type=int, default=60, help="Frames kept before the goal")
    ap.add_argument("--post", type=int, default=120, help="Frames kept after the goal")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("configs/hierarchical_agent.yaml") as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"]); env_cfg["task"] = args.task
    hl_cfg = cfg.get("high_level", {})

    gs.init(backend=gs.gpu, logging_level="warning")
    env = SoccerEnvHierarchical(
        num_envs=1, env_cfg=env_cfg, obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"], command_cfg=cfg["command"],
        walk_model_path=hl_cfg.get("walk_model_path", "/workspace/booster/booster_deploy/tasks/locomotion/models/t1_walk.pt"),
        high_level_decimation=hl_cfg.get("decimation", 5), show_viewer=False,
    )
    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    log_dir = f"runs/hierarchical_soccer_{args.task}"
    if args.model:
        model_path = args.model
    else:
        files = sorted(glob.glob(f"{log_dir}/model_*.pt"), key=os.path.getmtime)
        if not files:
            print("No model found"); return
        model_path = files[-1]

    print(f"Loading: {model_path}")
    runner = OnPolicyRunner(env, cfg["train"], log_dir, device=gs.device)
    runner.load(model_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    frames, goal_flags = [], []
    min_dist = 1e9; min_dist_frame = 0

    for i in range(args.max_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)

        state = env._soccer_state()
        goal = bool(state["scored"][0].item())
        d = float(state["dist_to_ball"][0].item())
        if d < min_dist:
            min_dist, min_dist_frame = d, len(frames)

        try:
            rgb, _, _, _ = cam.render(rgb=True)
            arr = tensor_to_array(rgb)
            if arr.ndim == 4:
                arr = arr[0]
            frames.append(arr.astype(np.uint8))
            goal_flags.append(goal)
        except Exception:
            pass

        if goal:
            print(f"  GOAL at step {i}  (min dist so far {min_dist:.2f})")
        if (i + 1) % 200 == 0:
            print(f"  step {i+1}/{args.max_steps}  ball_dist={d:.2f}")

    os.makedirs("demos", exist_ok=True)
    if any(goal_flags):
        g = goal_flags.index(True)
        lo = max(0, g - args.pre); hi = min(len(frames), g + args.post)
        clip = frames[lo:hi]
        out = "demos/goal_showreel.mp4"
        imageio.mimsave(out, clip, fps=30)
        print(f"\nGoal showreel saved: {out}  ({len(clip)} frames, goal at local step {g-lo})")
    else:
        # Fallback: window around the closest approach to the ball
        lo = max(0, min_dist_frame - args.pre); hi = min(len(frames), min_dist_frame + args.post)
        clip = frames[lo:hi]
        out = "demos/chase_full.mp4"
        imageio.mimsave(out, clip, fps=30)
        print(f"\nWARNING: no goal scored. Saved closest-approach clip: {out} "
              f"(min dist {min_dist:.2f}, {len(clip)} frames). Retrain / tune before submitting.")

    print("DONE")


if __name__ == "__main__":
    main()
