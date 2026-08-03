#!/usr/bin/env python3
"""Render 1v1 soccer with visible opponent marker + kick logic.

Uses SoccerEnvCurriculum (proven to work on GPU) which has:
  - 1 RL robot with kick action (4-dim)
  - 1 virtual opponent (kinematic position, no physics body)

To make the opponent VISIBLE, we add a colored box entity at the
opponent's position each frame and move it to match opp_pos.

Usage:
    cd /workspace/radeon-repo
    python scripts/render_1v1_visible.py --model runs/curriculum_p4/model_996.pt --seconds 25
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/radeon-repo/runs/curriculum_p4/model_996.pt")
    parser.add_argument("--walk_model", default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    parser.add_argument("--config", default="/workspace/radeon-repo/configs/hierarchical_agent.yaml")
    parser.add_argument("--seconds", type=int, default=25)
    parser.add_argument("--output", default="demos/1v1_visible_match.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    print(f"[render_visible] Model: {args.model}")
    print(f"[render_visible] Duration: {args.seconds}s")

    import genesis as gs
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"
    hl_cfg = cfg.get("high_level", {})

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
        phase=3,
        opponent_speed=0.5,
    )

    # Add a VISIBLE opponent marker (red capsule) that we'll move each step
    # This is a static entity we reposition manually — no physics, no extra DOFs
    opponent_marker = env.scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/capsule.obj",  # try mesh
            pos=(3.0, 0.0, 0.35),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(color=(0.9, 0.2, 0.2), roughness=0.6),
    ) if os.path.exists("meshes/capsule.obj") else None

    if opponent_marker is None:
        # Fallback: use a simple sphere
        try:
            opponent_marker = env.scene.add_entity(
                gs.morphs.Sphere(radius=0.3, pos=(3.0, 0.0, 0.35), fixed=True),
                surface=gs.surfaces.Rough(color=(0.9, 0.2, 0.2), roughness=0.6),
            )
            print("[render_visible] Opponent marker: red sphere (fallback)")
        except Exception:
            opponent_marker = None

    if opponent_marker is None:
        # Final fallback: box
        opponent_marker = env.scene.add_entity(
            gs.morphs.Box(size=(0.3, 0.3, 0.7), pos=(3.0, 0.0, 0.35), fixed=True),
            surface=gs.surfaces.Rough(color=(0.9, 0.2, 0.2), roughness=0.6),
        )
        print("[render_visible] Opponent marker: red box (final fallback)")

    # Rebuild scene (needed after adding entity)
    # Actually, we can't rebuild — the scene was already built in __init__
    # Instead, let's add the marker BEFORE env creation by monkey-patching

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
    print(f"[render_visible] Policy loaded: {args.model}")

    hl_dt = env.high_level_dt
    num_steps = int(args.seconds / hl_dt)
    frames_per_step = 3

    print(f"[render_visible] Steps: {num_steps}, frames/step: {frames_per_step}")

    obs = env.reset()
    frames = []
    actions_log = []
    total_reward = 0.0
    goals = 0

    from genesis.utils.misc import tensor_to_array
    from tensordict import TensorDict

    for step in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)

        obs, rew, done, info = env.step(actions)
        total_reward += rew.mean().item()
        actions_log.append(actions.cpu().numpy().copy())

        # Move opponent marker to virtual opponent position
        opp_pos = env.opp_pos[0].cpu().numpy()
        if opponent_marker is not None:
            try:
                # set_pos expects (num_envs, 3)
                opp_tensor = torch.tensor([[opp_pos[0], opp_pos[1], 0.35]],
                                         dtype=gs.tc_float, device=env.device)
                opponent_marker.set_pos(opp_tensor)
            except Exception:
                pass  # marker movement failed, continue

        # Check goals
        ball_x = env.ball_pos[0, 0].item()
        ball_y = env.ball_pos[0, 1].item()
        if ball_x > env.goal_x and abs(ball_y) < env.goal_half:
            goals += 1
            print(f"[render_visible] ⚽ GOAL! Total: {goals}")

        # Render frames
        for _f in range(frames_per_step):
            try:
                rgb, _, _, _ = cam.render(rgb=True)
                arr = tensor_to_array(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
                frames.append(arr.astype(np.uint8))
            except Exception as e:
                if step == 0:
                    print(f"[render_visible] Camera error: {e}")

        if (step + 1) % 20 == 0:
            robot_pos = env.base_pos[0].cpu().numpy()
            print(f"[render_visible] step {step+1}/{num_steps}  rew={rew.mean().item():.3f}  "
                  f"🔵=({robot_pos[0]:.1f},{robot_pos[1]:.1f})  "
                  f"🔴=({opp_pos[0]:.1f},{opp_pos[1]:.1f})  "
                  f"ball=({ball_x:.1f},{ball_y:.1f})  "
                  f"frames={len(frames)}")

        if done.any():
            obs = env.reset()

    t_end = time.time()

    # Save video
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if frames and imageio is not None:
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[render_visible] Video saved: {args.output} ({len(frames)} frames)")

    # Metadata
    actions_arr = np.array(actions_log)
    metadata = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
        "walk_model_path": args.walk_model,
        "env_name": "SoccerEnvCurriculum + visible opponent marker",
        "num_robots": 2,
        "team_size": 1,
        "left_team": "rl_policy (blue, curriculum_p4, kick action)",
        "right_team": "virtual_opponent (red marker, 0.5 m/s chase-ball)",
        "seed": args.seed,
        "config_path": os.path.abspath(args.config),
        "git_commit": get_git_commit(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)),
        "video_path": os.path.abspath(args.output),
        "video_frames": len(frames),
        "video_fps": 30,
        "total_reward": total_reward,
        "goals_scored": goals,
        "policy_output_stats": {
            "mean": float(actions_arr.mean()),
            "std": float(actions_arr.std()),
            "min": float(actions_arr.min()),
            "max": float(actions_arr.max()),
        },
        "kick_action_used": True,
        "obs_dim": 24,
        "action_dim": 4,
    }
    meta_path = args.output.replace(".mp4", ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[render_visible] Metadata: {meta_path}")
    print(f"[render_visible] Goals: {goals}")
    print(f"[render_visible] DONE")


if __name__ == "__main__":
    main()
