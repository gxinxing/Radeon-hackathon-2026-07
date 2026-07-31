#!/usr/bin/env python3
"""Render 1v1 demo video using offscreen camera rendering (Genesis method 2).

Uses scene.add_camera() + cam.render() per step, then encodes to MP4.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="models/chase_v8_policy.onnx")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", default="demos/1v1_demo.mp4")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    import yaml, numpy as np, torch
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "configs/hierarchical_agent.yaml")) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"
    hl_cfg = cfg.get("high_level", {})

    from soccer_env_hierarchical import SoccerEnvHierarchical
    env = SoccerEnvHierarchical(
        num_envs=1, env_cfg=env_cfg, obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"], command_cfg=cfg["command"],
        walk_model_path=hl_cfg.get("walk_model_path"),
        high_level_decimation=hl_cfg.get("decimation", 5),
        show_viewer=False)

    # Get camera from scene (already added in _build_scene)
    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Load ONNX policy
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from match_3v3.policy import SharedRLPolicy
    from match_3v3.scene import PlayerState, BallState, Team, Role

    rl_policy = SharedRLPolicy(onnx_path=args.onnx)
    print(f"[render] ONNX loaded: {args.onnx} (mode={rl_policy.mode})")

    obs = env.reset()
    frames = []
    total_rew = 0

    for step in range(args.steps):
        robot_pos = env.base_pos[0].cpu().numpy()
        robot_quat = env.base_quat[0].cpu().numpy()
        robot_vel = env.filtered_lin_vel[0].cpu().numpy()
        ball_pos = env.ball_pos[0].cpu().numpy()
        ball_vel = env.ball_vel[0].cpu().numpy()

        player = PlayerState(
            team=Team.LEFT, robot_idx=0, role=Role.ATTACKER,
            pos=robot_pos, quat=robot_quat, vel=robot_vel,
        )
        ball = BallState(pos=ball_pos, vel=ball_vel)

        action_result = rl_policy.compute(player, ball)
        action = torch.tensor([action_result.velocity_cmd],
                               dtype=torch.float32, device=env.device)

        obs, rew, done, extras = env.step(action)
        total_rew += rew.mean().item()

        # Render frame every 2 steps (30fps at 10Hz HL = 15fps, every step = 30fps)
        try:
            rgb, _, _, _ = cam.render(rgb=True)
            arr = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.array(rgb)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            frames.append(arr)
        except Exception as e:
            if step == 0:
                print(f"[render] Camera render failed: {e}")

        if (step + 1) % 50 == 0:
            ball_d = float(np.linalg.norm(robot_pos[:2] - ball_pos[:2]))
            print(f"[render] step {step+1}/{args.steps}: h={robot_pos[2]:.3f} "
                  f"ball_d={ball_d:.2f} rew={rew.mean().item():.3f} "
                  f"ball=({ball_pos[0]:.1f},{ball_pos[1]:.1f}) "
                  f"frames={len(frames)}")

        if done.any():
            obs = env.reset()

    # Save video
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if frames:
        import imageio
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[render] Video saved: {args.output} ({len(frames)} frames, total_rew={total_rew:.1f})")
    else:
        print(f"\n[render] No frames rendered, total_rew={total_rew:.1f}")


if __name__ == "__main__":
    main()
