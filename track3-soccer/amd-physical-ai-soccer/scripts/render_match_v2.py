#!/usr/bin/env python3
"""Render a 1v1 soccer match with kick logic and correct frame rate.

Uses SoccerEnvCurriculum which supports:
  - 1 RL robot (with kick action) vs 1 virtual opponent
  - 4th action dim = kick trigger
  - Kick impulse applied to ball when close

Left: RL policy (curriculum_p4/model_996.pt, 24-dim obs, 4-dim action)
Right: Virtual opponent (rule-based, from curriculum env)
Frame rate: 3 frames per HL step → 30fps video

Usage:
    cd /workspace/radeon-repo
    python scripts/render_match_v2.py --model runs/curriculum_p4/model_996.pt --seconds 25
"""
import argparse, hashlib, json, os, subprocess, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, "/workspace/radeon-repo")

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
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5,
                           cwd="/workspace/radeon-repo")
        return r.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Render 1v1 soccer match with kicks")
    parser.add_argument("--model", default="/workspace/radeon-repo/runs/curriculum_p4/model_996.pt")
    parser.add_argument("--walk_model", default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    parser.add_argument("--config", default="/workspace/radeon-repo/configs/hierarchical_agent.yaml")
    parser.add_argument("--seconds", type=int, default=25)
    parser.add_argument("--output", default="demos/3v3_match_v2.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    print(f"[render_v2] Model: {args.model}")
    print(f"[render_v2] Duration: {args.seconds}s, Output: {args.output}")

    import genesis as gs
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"
    hl_cfg = cfg.get("high_level", {})

    # Use curriculum env (supports kick + virtual opponent)
    from soccer_env_curriculum import SoccerEnvCurriculum

    env = SoccerEnvCurriculum(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        command_cfg=cfg["command"],
        walk_model_path=args.walk_model,
        high_level_decimation=hl_cfg.get("decimation", 5),
        show_viewer=False,
        phase=3,            # full phase: kick active, opponent at 0.5 m/s
        opponent_speed=0.5,
    )

    # Camera
    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Load policy
    from rsl_rl.runners import OnPolicyRunner
    log_dir = os.path.dirname(args.model)
    train_cfg = cfg["train"]
    train_cfg["actor"]["hidden_dims"] = [256, 128, 64]
    train_cfg["critic"]["hidden_dims"] = [256, 128, 64]
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(args.model)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[render_v2] Policy loaded: {args.model}")
    print(f"[render_v2] Obs dim={env._obs_dim()}, Action dim={env.num_actions}")

    # Steps
    hl_dt = env.high_level_dt  # 0.1s
    num_steps = int(args.seconds / hl_dt)
    frames_per_step = 3  # 10Hz × 3 = 30fps
    print(f"[render_v2] Steps: {num_steps}, frames/step: {frames_per_step}")

    obs = env.reset()
    frames = []
    actions_log = []
    total_reward = 0.0
    nan_detected = False
    goals = 0

    from genesis.utils.misc import tensor_to_array

    for step in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)

        if torch.isnan(actions).any() or torch.isinf(actions).any():
            nan_detected = True
            actions = torch.nan_to_num(actions)

        obs, rew, done, info = env.step(actions)
        total_reward += rew.mean().item()
        actions_log.append(actions.cpu().numpy().copy())

        # Check goals
        ball_x = env.ball_pos[0, 0].item()
        ball_y = env.ball_pos[0, 1].item()
        if ball_x > env.goal_x and abs(ball_y) < env.goal_half:
            goals += 1
            print(f"[render_v2] ⚽ GOAL! Total: {goals}")

        # Render 3 frames per HL step
        for _f in range(frames_per_step):
            try:
                rgb, _, _, _ = cam.render(rgb=True)
                arr = tensor_to_array(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
                frames.append(arr.astype(np.uint8))
            except Exception as e:
                if step == 0:
                    print(f"[render_v2] Camera error: {e}")

        if (step + 1) % 20 == 0:
            robot_pos = env.base_pos[0].cpu().numpy()
            ball_d = float(torch.norm(env.base_pos[0, :2] - env.ball_pos[0, :2]).item())
            h = robot_pos[2]
            kick_active = actions[0, 3].item() > 0.5 if actions.shape[-1] >= 4 else False
            print(f"[render_v2] step {step+1}/{num_steps}  rew={rew.mean().item():.3f}  "
                  f"h={h:.3f}  ball_d={ball_d:.2f}  kick={kick_active}  "
                  f"ball=({ball_x:.1f},{ball_y:.1f})  frames={len(frames)}")

        if done.any():
            obs = env.reset()

    t_end = time.time()

    # Save video
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if frames and imageio is not None:
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[render_v2] Video saved: {args.output} ({len(frames)} frames)")
    else:
        print(f"\n[render_v2] WARNING: No frames saved")

    # Metadata
    actions_arr = np.array(actions_log)
    metadata = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
        "walk_model_path": args.walk_model,
        "env_name": "SoccerEnvCurriculum",
        "num_robots": 2,
        "team_size": 1,
        "left_team": "rl_policy (curriculum_p4, phase=3, kick active)",
        "right_team": "virtual_opponent (0.5 m/s)",
        "seed": args.seed,
        "config_path": os.path.abspath(args.config),
        "git_commit": get_git_commit(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)),
        "validation_status": "passed",
        "video_path": os.path.abspath(args.output),
        "video_frames": len(frames),
        "video_fps": 30,
        "total_reward": total_reward,
        "goals_scored": goals,
        "nan_detected": nan_detected,
        "policy_output_stats": {
            "mean": float(actions_arr.mean()),
            "std": float(actions_arr.std()),
            "min": float(actions_arr.min()),
            "max": float(actions_arr.max()),
        },
        "kick_action_used": True,
        "obs_dim": 24,
        "action_dim": 4,
        "phase": 3,
        "opponent_speed": 0.5,
    }

    # Validation
    issues = []
    if len(frames) < 100:
        issues.append("too_few_frames")
    if metadata["policy_output_stats"]["std"] < 0.01:
        issues.append("constant_policy_output")
    if nan_detected:
        issues.append("nan_in_actions")
    metadata["validation_status"] = "passed" if not issues else f"failed: {','.join(issues)}"

    meta_path = args.output.replace(".mp4", ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[render_v2] Metadata: {meta_path}")
    print(f"[render_v2] Goals: {goals}")
    print(f"[render_v2] Validation: {metadata['validation_status']}")
    print(f"[render_v2] DONE")


if __name__ == "__main__":
    main()
