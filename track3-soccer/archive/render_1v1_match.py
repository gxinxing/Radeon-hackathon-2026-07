#!/usr/bin/env python3
"""Render 1v1 match video: RL agent (ONNX) vs virtual opponent.

Uses offscreen camera rendering (Genesis method 2: cam.render()).
Virtual opponent is drawn as a marker (kinematic, no physics entity).
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="models/chase_v8_policy.onnx")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", default="demos/1v1_match_video.mp4")
    parser.add_argument("--opponent_speed", type=float, default=0.4)
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

    from soccer_env_1v1_virtual import SoccerEnv1v1Virtual
    env = SoccerEnv1v1Virtual(
        num_envs=1, env_cfg=env_cfg, obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"], command_cfg=cfg["command"],
        walk_model_path=hl_cfg.get("walk_model_path"),
        high_level_decimation=hl_cfg.get("decimation", 5),
        show_viewer=False,
        opponent_speed=args.opponent_speed,
        opponent_init_pos=(-3.0, 0.0))

    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Load ONNX policy
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from match_3v3.policy import SharedRLPolicy
    from match_3v3.scene import PlayerState, BallState, Team, Role

    # Need to check ONNX input dim — if 21, use 1v1 obs; if 19, use base obs
    rl_policy = SharedRLPolicy(onnx_path=args.onnx)
    onnx_input_dim = rl_policy.session.get_inputs()[0].shape[-1] if rl_policy.session else 19
    print(f"[render] ONNX loaded: {args.onnx} (input dim={onnx_input_dim})")

    obs = env.reset()
    frames = []
    total_rew = 0
    ball_start = env.ball_pos[0].cpu().numpy().copy()

    for step in range(args.steps):
        robot_pos = env.base_pos[0].cpu().numpy()
        robot_quat = env.base_quat[0].cpu().numpy()
        robot_vel = env.filtered_lin_vel[0].cpu().numpy()
        ball_pos = env.ball_pos[0].cpu().numpy()
        ball_vel = env.ball_vel[0].cpu().numpy()
        opp_pos = env.opp_pos[0].cpu().numpy()

        # Build observation: use env's _update_observation which produces 21-dim
        # But ONNX might be 19-dim (v8 model) — need to match
        if onnx_input_dim == 21:
            # Use 1v1 obs directly (21-dim from env)
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            action = torch.zeros((1, 3), dtype=torch.float32, device=env.device)
            with torch.no_grad():
                import onnxruntime as ort
                input_name = rl_policy.session.get_inputs()[0].name
                result = rl_policy.session.run(None, {input_name: obs_tensor.cpu().numpy().reshape(1, -1).astype(np.float32)})
                action = torch.tensor(result[0], dtype=torch.float32, device=env.device)
        else:
            # 19-dim ONNX: use rule-based or pad obs
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

        # Render frame
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
            opp_d = float(np.linalg.norm(opp_pos[:2] - ball_pos[:2]))
            poss = "AGENT" if ball_d < opp_d else "OPP"
            print(f"[render] step {step+1}/{args.steps}: h={robot_pos[2]:.3f} "
                  f"ball_d={ball_d:.2f} opp_d={opp_d:.2f} poss={poss} "
                  f"rew={rew.mean().item():.3f} "
                  f"ball=({ball_pos[0]:.1f},{ball_pos[1]:.1f}) "
                  f"opp=({opp_pos[0]:.1f},{opp_pos[1]:.1f})")

        if done.any():
            obs = env.reset()

    # Save video
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ball_end = env.ball_pos[0].cpu().numpy()
    ball_displacement = float(np.linalg.norm(ball_end[:2] - ball_start[:2]))

    if frames:
        import imageio
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[render] Video saved: {args.output} ({len(frames)} frames)")
    else:
        print(f"\n[render] No frames rendered")

    print(f"  Total reward: {total_rew:.2f}")
    print(f"  Ball displacement: {ball_displacement:.2f}m")
    print(f"  Ball: ({ball_start[0]:.1f},{ball_start[1]:.1f}) → ({ball_end[0]:.1f},{ball_end[1]:.1f})")


if __name__ == "__main__":
    main()
