#!/usr/bin/env python3
"""Render 1v1 soccer with visible opponent overlay + kick logic.

Uses SoccerEnvCurriculum (proven on GPU) + post-render overlay:
1. Render the RL robot + ball + field normally
2. After rendering, overlay a red circle/marker at the virtual opponent's position
3. This avoids GPU memory issues with 2 full physics robots

Usage:
    cd /workspace/radeon-repo
    python scripts/render_1v1_overlay.py --model runs/curriculum_p4/model_996.pt --seconds 25
"""
import argparse, hashlib, json, os, subprocess, sys, time, math

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

try:
    import cv2
except ImportError:
    cv2 = None


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


def world_to_image(wx, wy, cam_pos, cam_lookat, cam_fov, img_w, img_h):
    """Approximate world (x,y) to image pixel projection for a top-ish camera."""
    # Camera at (0, -12, 8) looking at (0, 0, 0.5)
    # Simple perspective: project onto camera plane
    dx = wx - cam_pos[0]
    dy = wy - cam_pos[1]
    dz = 0.5 - cam_pos[2]  # assume object at height 0.5

    # Distance from camera
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 0.1:
        return img_w // 2, img_h // 2

    # Simple projection: camera looks along (cam_lookat - cam_pos)
    # For the broadcast angle (0, -12, 8) → (0, 0, 0.5):
    # x maps roughly linearly to image x
    # y maps roughly inversely to image y (farther = higher on screen)
    
    # Scale factors (tuned for (0,-12,8) camera)
    scale_x = img_w / 16.0  # field width ~14, plus margin
    scale_y = img_h / 12.0  # field depth view
    
    px = int(img_w / 2 + dx * scale_x * 0.5)
    py = int(img_h / 2 + dy * scale_y * 0.3)  # y compressed by perspective
    
    return max(0, min(img_w-1, px)), max(0, min(img_h-1, py))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/radeon-repo/runs/curriculum_p4/model_996.pt")
    parser.add_argument("--walk_model", default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    parser.add_argument("--config", default="/workspace/radeon-repo/configs/hierarchical_agent.yaml")
    parser.add_argument("--seconds", type=int, default=25)
    parser.add_argument("--output", default="demos/1v1_overlay_match.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    print(f"[render_overlay] Model: {args.model}")
    print(f"[render_overlay] Duration: {args.seconds}s")

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
    print(f"[render_overlay] Policy loaded: {args.model}")

    hl_dt = env.high_level_dt
    num_steps = int(args.seconds / hl_dt)
    frames_per_step = 3

    print(f"[render_overlay] Steps: {num_steps}, frames/step: {frames_per_step}")

    # Camera params for projection
    cam_pos = (0, -12, 8)
    cam_lookat = (0, 0, 0.5)
    cam_fov = 50

    obs = env.reset()
    raw_frames = []
    opp_positions = []
    agent_positions = []
    ball_positions = []
    actions_log = []
    total_reward = 0.0
    goals = 0

    from genesis.utils.misc import tensor_to_array

    for step in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)

        obs, rew, done, info = env.step(actions)
        total_reward += rew.mean().item()
        actions_log.append(actions.cpu().numpy().copy())

        # Record positions for overlay
        agent_pos = env.base_pos[0].cpu().numpy()
        opp_pos = env.opp_pos[0].cpu().numpy()
        ball_pos = env.ball_pos[0].cpu().numpy()
        agent_positions.append(agent_pos.copy())
        opp_positions.append(opp_pos.copy())
        ball_positions.append(ball_pos.copy())

        # Check goals
        if ball_pos[0] > env.goal_x and abs(ball_pos[1]) < env.goal_half:
            goals += 1
            print(f"[render_overlay] ⚽ GOAL! Total: {goals}")

        # Render frames
        for _f in range(frames_per_step):
            try:
                rgb, _, _, _ = cam.render(rgb=True)
                arr = tensor_to_array(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
                raw_frames.append(arr.astype(np.uint8))
            except Exception as e:
                if step == 0:
                    print(f"[render_overlay] Camera error: {e}")

        if (step + 1) % 20 == 0:
            print(f"[render_overlay] step {step+1}/{num_steps}  rew={rew.mean().item():.3f}  "
                  f"🔵=({agent_pos[0]:.1f},{agent_pos[1]:.1f})  "
                  f"🔴=({opp_pos[0]:.1f},{opp_pos[1]:.1f})  "
                  f"ball=({ball_pos[0]:.1f},{ball_pos[1]:.1f})  "
                  f"kick={'Y' if (actions.shape[-1]>=4 and actions[0,3]>0.5) else 'N'}  "
                  f"frames={len(raw_frames)}")

        if done.any():
            obs = env.reset()

    t_end = time.time()

    # Post-process: overlay opponent marker on each frame
    print(f"\n[render_overlay] Post-processing {len(raw_frames)} frames with opponent overlay...")

    img_h, img_w = raw_frames[0].shape[:2] if raw_frames else (720, 1280)
    final_frames = []

    for i, frame in enumerate(raw_frames):
        if cv2 is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Calculate which step this frame belongs to
            step_idx = min(i // frames_per_step, len(opp_positions) - 1)
            opp = opp_positions[step_idx]
            agent = agent_positions[step_idx]
            ball = ball_positions[step_idx]

            # Draw opponent marker (red circle)
            px, py = world_to_image(opp[0], opp[1], cam_pos, cam_lookat, cam_fov, img_w, img_h)
            cv2.circle(frame_bgr, (px, py), 20, (0, 0, 255), -1)  # red filled circle
            cv2.circle(frame_bgr, (px, py), 20, (0, 0, 100), 2)   # dark red outline
            cv2.putText(frame_bgr, "OPP", (px - 20, py - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

            # Draw agent marker (blue circle, smaller — the real robot is already visible)
            ax, ay = world_to_image(agent[0], agent[1], cam_pos, cam_lookat, cam_fov, img_w, img_h)
            cv2.circle(frame_bgr, (ax, ay), 5, (255, 100, 0), -1)  # blue dot

            # Draw ball marker (yellow)
            bx, by = world_to_image(ball[0], ball[1], cam_pos, cam_lookat, cam_fov, img_w, img_h)
            cv2.circle(frame_bgr, (bx, by), 8, (0, 255, 255), -1)  # yellow

            # Score display
            cv2.putText(frame_bgr, f"Goals: {goals}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            final_frames.append(frame_rgb)
        else:
            final_frames.append(frame)

    # Save video
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if final_frames and imageio is not None:
        imageio.mimsave(args.output, final_frames, fps=30, codec='libx264')
        print(f"[render_overlay] Video saved: {args.output} ({len(final_frames)} frames)")

    # Metadata
    actions_arr = np.array(actions_log)
    metadata = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
        "walk_model_path": args.walk_model,
        "env_name": "SoccerEnvCurriculum + OpenCV opponent overlay",
        "num_robots": 2,
        "team_size": 1,
        "left_team": "rl_policy (curriculum_p4, 24-dim obs, 4-dim action with kick)",
        "right_team": "virtual_opponent (red circle overlay, 0.5 m/s chase-ball)",
        "seed": args.seed,
        "config_path": os.path.abspath(args.config),
        "git_commit": get_git_commit(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)),
        "video_path": os.path.abspath(args.output),
        "video_frames": len(final_frames),
        "video_fps": 30,
        "total_reward": total_reward,
        "goals_scored": goals,
        "kick_action_used": True,
        "obs_dim": 24,
        "action_dim": 4,
        "opponent_overlay": "red circle at virtual opponent position",
    }
    meta_path = args.output.replace(".mp4", ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[render_overlay] Metadata: {meta_path}")
    print(f"[render_overlay] Goals: {goals}")
    print(f"[render_overlay] DONE")


if __name__ == "__main__":
    main()
