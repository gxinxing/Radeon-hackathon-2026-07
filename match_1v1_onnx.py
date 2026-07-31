#!/usr/bin/env python3
"""1v1 Match: RL agent vs rule-based opponent in single Genesis scene.

Simpler than 3v3 — only 2 robots, no coordinator needed.
RL agent (Team A) uses ONNX model, opponent (Team B) uses rule-based chase.

This is the "1v1 verification" step from 问题全景梳理:
"先训练 1v1 → 2v2 → 3v3 逐步迁移"
"""
import argparse, os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="1v1 Match: RL vs Rule")
    parser.add_argument("--onnx", default="models/chase_v8_policy.onnx")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", default="match_logs/match_1v1.json")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    import yaml, torch
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

    # Load ONNX policy for Team A
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from match_3v3.policy import SharedRLPolicy
    from match_3v3.scene import PlayerState, BallState, Team, Role

    rl_policy = SharedRLPolicy(onnx_path=args.onnx)
    print(f"[1v1] ONNX loaded: {args.onnx} (mode={rl_policy.mode})")

    obs = env.reset()
    log = []
    rl_reward = 0
    ball_start = env.ball_pos[0].cpu().numpy().copy()

    for step in range(args.steps):
        robot_pos = env.base_pos[0].cpu().numpy()
        robot_quat = env.base_quat[0].cpu().numpy()
        robot_vel = env.filtered_lin_vel[0].cpu().numpy()
        ball_pos = env.ball_pos[0].cpu().numpy()
        ball_vel = env.ball_vel[0].cpu().numpy()

        # RL agent action (Team A, attacks +x)
        player = PlayerState(
            team=Team.LEFT, robot_idx=0, role=Role.ATTACKER,
            pos=robot_pos, quat=robot_quat, vel=robot_vel,
        )
        ball = BallState(pos=ball_pos, vel=ball_vel)

        action_result = rl_policy.compute(player, ball)
        action = torch.tensor([action_result.velocity_cmd],
                               dtype=torch.float32, device=env.device)

        obs, rew, done, extras = env.step(action)
        rl_reward += rew.mean().item()

        # Log state
        log.append({
            "t": round(step * 0.1, 2),  # 10Hz
            "ball": {
                "x": round(float(ball_pos[0]), 3),
                "y": round(float(ball_pos[1]), 3),
                "z": round(float(ball_pos[2]), 3),
                "vx": round(float(ball_vel[0]), 3),
                "vy": round(float(ball_vel[1]), 3),
            },
            "robot": {
                "x": round(float(robot_pos[0]), 3),
                "y": round(float(robot_pos[1]), 3),
                "z": round(float(robot_pos[2]), 3),
                "pitch": round(float(env.base_euler[0, 1].item()), 3),
                "roll": round(float(env.base_euler[0, 0].item()), 3),
            },
            "reward": round(float(rew.mean().item()), 4),
        })

        if (step + 1) % 50 == 0:
            ball_d = float(np.linalg.norm(robot_pos[:2] - ball_pos[:2]))
            print(f"[1v1] step {step+1}/{args.steps}: h={robot_pos[2]:.3f} "
                  f"ball_d={ball_d:.2f} rew={rew.mean().item():.3f} "
                  f"ball_pos=({ball_pos[0]:.1f},{ball_pos[1]:.1f}) "
                  f"ball_vel=({ball_vel[0]:.1f},{ball_vel[1]:.1f})")

        if done.any():
            print(f"[1v1] Episode ended at step {step+1}")
            obs = env.reset()

    # Save match log
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ball_end = env.ball_pos[0].cpu().numpy()
    ball_displacement = float(np.linalg.norm(ball_end[:2] - ball_start[:2]))

    match_data = {
        "duration": round(args.steps * 0.1, 1),
        "steps": len(log),
        "n_clients": 1,
        "rl_reward": round(rl_reward, 2),
        "ball_start": {"x": round(float(ball_start[0]), 3), "y": round(float(ball_start[1]), 3)},
        "ball_end": {"x": round(float(ball_end[0]), 3), "y": round(float(ball_end[1]), 3)},
        "ball_displacement": round(ball_displacement, 3),
        "ball_velocity_nonzero": any(abs(l["ball"].get("vx", 0)) > 0.01 for l in log),
        "log": log,
    }

    with open(args.output, "w") as f:
        json.dump(match_data, f, indent=2)

    print(f"\n[1v1] Match complete: {args.output}")
    print(f"  Steps: {len(log)}")
    print(f"  Total reward: {rl_reward:.2f}")
    print(f"  Ball displacement: {ball_displacement:.2f}m")
    print(f"  Ball velocity non-zero: {match_data['ball_velocity_nonzero']}")
    print(f"  Ball end pos: ({ball_end[0]:.2f}, {ball_end[1]:.2f})")


if __name__ == "__main__":
    main()
