#!/usr/bin/env python3
"""Render 1v1 soccer with TWO real robots using CPU physics backend.

Genesis GPU kernel has 64KB local memory limit — 2 robots (46 DOFs) exceeds it.
CPU backend has no such limit. Slower but produces correct 2-robot video.

Usage:
    cd /workspace/radeon-repo
    python scripts/render_cpu_1v1.py --seconds 25
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
    parser.add_argument("--model", default="/workspace/radeon-repo/runs/soccer_1v1/model_499.pt")
    parser.add_argument("--walk_model", default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    parser.add_argument("--config", default="/workspace/radeon-repo/configs/hierarchical_agent.yaml")
    parser.add_argument("--seconds", type=int, default=25)
    parser.add_argument("--output", default="demos/cpu_1v1_match.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    print(f"[render_cpu] Model: {args.model}")
    print(f"[render_cpu] Backend: CPU (GPU kernel limit bypassed)")
    print(f"[render_cpu] Duration: {args.seconds}s (CPU will be slower)")

    import genesis as gs
    # Use CPU backend — no GPU local memory limit
    gs.init(backend=gs.cpu, logging_level="warning", seed=args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"
    hl_cfg = cfg.get("high_level", {})

    from soccer_env_1v1 import SoccerEnv1v1

    env = SoccerEnv1v1(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        command_cfg=cfg["command"],
        walk_model_path=args.walk_model,
        high_level_decimation=hl_cfg.get("decimation", 5),
        show_viewer=False,
        opponent_init_pos=(3.0, 0.0, 0.7),
    )

    print(f"[render_cpu] ✅ Scene built with 2 robots on CPU!")
    print(f"[render_cpu] Obs dim: {env.hl_obs_dim}")

    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Team colors
    try:
        blue = gs.surfaces.Rough(color=(0.2, 0.5, 0.9), roughness=0.6)
        red = gs.surfaces.Rough(color=(0.9, 0.2, 0.2), roughness=0.6)
        for link in env.robot.links:
            if hasattr(link, 'set_surface'):
                link.set_surface(blue)
        for link in env.opponent.links:
            if hasattr(link, 'set_surface'):
                link.set_surface(red)
        print("[render_cpu] Team colors applied")
    except Exception as e:
        print(f"[render_cpu] Colors: {e}")

    # Load policy
    from rsl_rl.runners import OnPolicyRunner
    log_dir = os.path.dirname(args.model)
    train_cfg = cfg["train"]
    train_cfg["actor"]["hidden_dims"] = [256, 128, 64]
    train_cfg["critic"]["hidden_dims"] = [256, 128, 64]
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(args.model)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[render_cpu] Policy loaded")

    hl_dt = env.high_level_dt
    num_steps = int(args.seconds / hl_dt)
    fps = 3  # 3 frames per HL step = 30fps

    obs = env.reset()
    frames = []
    total_reward = 0.0
    goals = 0
    kick_cd_a = 0.0
    kick_cd_o = 0.0

    from genesis.utils.misc import tensor_to_array

    for step in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        total_reward += rew.mean().item()

        # Kick: agent
        d_a = float(torch.norm(env.base_pos[0, :2] - env.ball_pos[0, :2]).item())
        if d_a < 0.35 and kick_cd_a < 0.01:
            g = torch.tensor([env.goal_x, 0., 0.], device=env.device) - env.ball_pos[0]
            g = g / (torch.norm(g) + 1e-6) * 3.0
            bq = env.ball.get_dofs_velocity().clone()
            bq[0, :3] = g
            env.ball.set_dofs_velocity(bq)
            kick_cd_a = 1.0
            print(f"[render_cpu] 🔵 kick! d={d_a:.2f}")

        # Kick: opponent
        d_o = float(torch.norm(env.opp_base_pos[0, :2] - env.ball_pos[0, :2]).item())
        if d_o < 0.35 and kick_cd_o < 0.01:
            g = torch.tensor([-env.goal_x, 0., 0.], device=env.device) - env.ball_pos[0]
            g = g / (torch.norm(g) + 1e-6) * 2.5
            bq = env.ball.get_dofs_velocity().clone()
            bq[0, :3] = g
            env.ball.set_dofs_velocity(bq)
            kick_cd_o = 1.0
            print(f"[render_cpu] 🔴 kick! d={d_o:.2f}")

        kick_cd_a = max(0, kick_cd_a - hl_dt)
        kick_cd_o = max(0, kick_cd_o - hl_dt)

        bx = env.ball_pos[0, 0].item()
        by = env.ball_pos[0, 1].item()
        if bx > env.goal_x and abs(by) < env.goal_half:
            goals += 1
            print(f"[render_cpu] ⚽ GOAL! {goals}")

        for _ in range(fps):
            try:
                rgb, _, _, _ = cam.render(rgb=True)
                arr = tensor_to_array(rgb)
                if arr.ndim == 4: arr = arr[0]
                frames.append(arr.astype(np.uint8))
            except Exception as e:
                if step == 0: print(f"[render_cpu] cam err: {e}")

        if (step + 1) % 10 == 0:
            ap = env.base_pos[0].cpu().numpy()
            op = env.opp_base_pos[0].cpu().numpy()
            print(f"[render_cpu] step {step+1}/{num_steps}  "
                  f"🔵=({ap[0]:.1f},{ap[1]:.1f},h={ap[2]:.2f})  "
                  f"🔴=({op[0]:.1f},{op[1]:.1f},h={op[2]:.2f})  "
                  f"ball=({bx:.1f},{by:.1f})  f={len(frames)}  "
                  f"t={time.time()-t_start:.0f}s")

        if done.any():
            obs = env.reset()

    t_end = time.time()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if frames and imageio:
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[render_cpu] Video: {args.output} ({len(frames)} frames, {t_end-t_start:.0f}s)")

    metadata = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
        "env_name": "SoccerEnv1v1 (CPU backend, 2 real robots)",
        "num_robots": 2,
        "left_team": "rl_policy (blue)",
        "right_team": "rule_based (red, chase-ball + kick)",
        "backend": "cpu",
        "seed": args.seed,
        "video_frames": len(frames),
        "video_fps": 30,
        "goals_scored": goals,
        "total_reward": total_reward,
        "render_time_s": round(t_end - t_start, 1),
        "git_commit": get_git_commit(),
    }
    meta_path = args.output.replace(".mp4", ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[render_cpu] Goals: {goals}")
    print(f"[render_cpu] DONE")


if __name__ == "__main__":
    main()
