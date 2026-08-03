#!/usr/bin/env python3
"""Render a 3v3 soccer match video with 6 robots, ball, kicks, and field.

Left team (3 robots): RL policy (PPO checkpoint)
Right team (3 robots): Rule-based chase-ball policy
All robots use frozen t1_walk.pt for locomotion.

Correct frame rate: HL=10Hz, render 3 frames per HL step → 30fps video.

Usage (on cloud):
    cd /workspace/radeon-repo
    python scripts/render_3v3_match.py --model /persistent/track3/models/checkpoints/best.pt
    python scripts/render_3v3_match.py --model runs/curriculum_p4/model_996.pt --seconds 25
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, "/workspace/radeon-repo")
sys.path.insert(0, "/workspace/radeon-repo/src")

import yaml
import numpy as np
import torch

try:
    import imageio
except ImportError:
    imageio = None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5,
                                cwd="/workspace/radeon-repo")
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Render 3v3 soccer match")
    parser.add_argument("--model", default="/persistent/track3/models/checkpoints/best.pt",
                        help="Path to PPO checkpoint for left team")
    parser.add_argument("--walk_model", default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt",
                        help="Path to frozen t1_walk.pt")
    parser.add_argument("--config", default="/workspace/radeon-repo/configs/hierarchical_agent.yaml")
    parser.add_argument("--seconds", type=int, default=25)
    parser.add_argument("--output", default="demos/3v3_match_verified.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[render_3v3] CWD: {os.getcwd()}")
    print(f"[render_3v3] Model: {args.model}")
    print(f"[render_3v3] Walk model: {args.walk_model}")
    print(f"[render_3v3] Duration: {args.seconds}s")
    print(f"[render_3v3] Output: {args.output}")

    start_time = time.time()

    # Import Genesis and initialize
    import genesis as gs
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"
    obs_cfg = cfg["obs"]
    reward_cfg = cfg["reward"]
    command_cfg = cfg["command"]
    hl_cfg = cfg.get("high_level", {})

    # Import 3v3 env
    from soccer_env_3v3 import SoccerEnv3v3

    env = SoccerEnv3v3(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        walk_model_path=args.walk_model,
        high_level_decimation=hl_cfg.get("decimation", 5),
        show_viewer=False,
    )

    # Get camera
    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Apply team colors
    try:
        blue_surface = gs.surfaces.Rough(color=(0.2, 0.5, 0.9), roughness=0.6)
        red_surface = gs.surfaces.Rough(color=(0.9, 0.2, 0.2), roughness=0.6)
        for i in range(3):
            for link in env.robots[i].links:
                if hasattr(link, 'set_surface'):
                    link.set_surface(blue_surface)
        for i in range(3, 6):
            for link in env.robots[i].links:
                if hasattr(link, 'set_surface'):
                    link.set_surface(red_surface)
    except Exception as e:
        print(f"[render_3v3] Surface color failed: {e}")

    # Load RL policy
    from rsl_rl.runners import OnPolicyRunner
    log_dir = os.path.dirname(os.path.dirname(args.model))
    train_cfg = cfg["train"]
    train_cfg["run_name"] = "hierarchical_soccer"
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(args.model)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[render_3v3] Policy loaded: {args.model}")

    # Calculate steps
    hl_dt = env.high_level_dt  # 0.1s
    num_hl_steps = int(args.seconds / hl_dt)
    frames_per_step = 3  # 10Hz HL × 3 = 30fps video
    total_frames = num_hl_steps * frames_per_step

    print(f"[render_3v3] HL steps: {num_hl_steps}, frames/step: {frames_per_step}")
    print(f"[render_3v3] Total frames: {total_frames}, target FPS: 30")

    # Rollout
    obs = env.reset()
    frames = []
    actions_log = []
    positions_log = []
    total_reward = 0.0
    nan_detected = False
    left_score = 0
    right_score = 0

    for step in range(num_hl_steps):
        with torch.no_grad():
            actions = policy(obs)

        if torch.isnan(actions).any() or torch.isinf(actions).any():
            print(f"[render_3v3] WARNING: NaN/Inf in actions at step {step}")
            nan_detected = True
            actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        obs, rew, done, info = env.step(actions)
        total_reward += rew.mean().item()

        actions_log.append(actions.cpu().numpy().copy())

        # Log positions
        robot_positions = env.all_base_pos[0].cpu().numpy()  # (6, 3)
        ball_pos = env.ball_pos[0].cpu().numpy()
        positions_log.append({
            "step": step,
            "robots": robot_positions.tolist(),
            "ball": ball_pos.tolist(),
        })

        # Check goals
        if env.ball_pos[0, 0].item() > env.goal_x and abs(env.ball_pos[0, 1].item()) < env.goal_half:
            left_score += 1
            print(f"[render_3v3] ⚽ LEFT scores! {left_score}-{right_score}")
        elif env.ball_pos[0, 0].item() < -env.goal_x and abs(env.ball_pos[0, 1].item()) < env.goal_half:
            right_score += 1
            print(f"[render_3v3] ⚽ RIGHT scores! {left_score}-{right_score}")

        # Render frames_per_step frames per HL step
        from genesis.utils.misc import tensor_to_array
        for _f in range(frames_per_step):
            try:
                rgb, _, _, _ = cam.render(rgb=True)
                arr = tensor_to_array(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
                frames.append(arr.astype(np.uint8))
            except Exception as e:
                if step == 0 and _f == 0:
                    print(f"[render_3v3] Camera render failed: {e}")

        if (step + 1) % 20 == 0:
            ball_d = float(torch.norm(env.all_base_pos[0, 0, :2] - env.ball_pos[0, :2]).item())
            print(f"[render_3v3] step {step+1}/{num_hl_steps}  rew={rew.mean().item():.3f}  "
                  f"ball_d={ball_d:.2f}  score={left_score}-{right_score}  "
                  f"frames={len(frames)}")
            # Print robot positions
            for ri in range(6):
                p = env.all_base_pos[0, ri].cpu().numpy()
                team = "L" if ri < 3 else "R"
                print(f"  {team}{ri}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.2f})")

        if done.any():
            obs = env.reset()

    end_time = time.time()

    # Save video
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if frames and imageio is not None:
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[render_3v3] Video saved: {args.output} ({len(frames)} frames)")
    else:
        print(f"\n[render_3v3] WARNING: No frames to save")

    # Save match log
    match_log_path = args.output.replace(".mp4", ".match_log.json")
    match_log = {
        "duration_s": args.seconds,
        "num_steps": num_hl_steps,
        "seed": args.seed,
        "model_path": os.path.abspath(args.model),
        "left_score": left_score,
        "right_score": right_score,
        "positions": positions_log,
    }
    with open(match_log_path, "w") as f:
        json.dump(match_log, f)

    # Compute stats
    actions_arr = np.array(actions_log)

    # Create metadata
    metadata = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
        "walk_model_path": args.walk_model,
        "env_name": "SoccerEnv3v3",
        "num_robots": 6,
        "team_size": 3,
        "left_team": "rl_policy",
        "right_team": "rule_based",
        "seed": args.seed,
        "config_path": os.path.abspath(args.config),
        "git_commit": get_git_commit(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
        "validation_status": "passed",
        "video_path": os.path.abspath(args.output),
        "video_frames": len(frames),
        "video_fps": 30,
        "video_resolution": [1280, 720],
        "total_reward": total_reward,
        "left_score": left_score,
        "right_score": right_score,
        "nan_detected": nan_detected,
        "policy_output_stats": {
            "mean": float(actions_arr.mean()),
            "std": float(actions_arr.std()),
            "min": float(actions_arr.min()),
            "max": float(actions_arr.max()),
        },
        "match_log_path": os.path.abspath(match_log_path),
        "match_log_seed": args.seed,
        "match_log_model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
    }

    # Determine validation status
    issues = []
    if len(frames) < 100:
        issues.append("too_few_frames")
    if metadata["policy_output_stats"]["std"] < 0.01:
        issues.append("constant_policy_output")
    if nan_detected:
        issues.append("nan_in_actions")

    metadata["validation_status"] = "passed" if not issues else f"failed: {','.join(issues)}"

    metadata_path = args.output.replace(".mp4", ".metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[render_3v3] Metadata: {metadata_path}")
    print(f"[render_3v3] Match log: {match_log_path}")
    print(f"[render_3v3] Score: {left_score}-{right_score}")
    print(f"[render_3v3] Validation: {metadata['validation_status']}")
    print(f"[render_3v3] DONE")


if __name__ == "__main__":
    main()
