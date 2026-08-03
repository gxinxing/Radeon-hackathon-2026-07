#!/usr/bin/env python3
"""Render a 1v1 soccer match with TWO visible robots and kick logic.

Uses SoccerEnv1v1 which adds a real opponent robot entity to the scene.
Both robots use frozen t1_walk.pt for locomotion.
RL agent (best.pt, 19-dim) controls left robot.
Rule-based policy controls right robot.
Kick logic: when robot close to ball (<0.3m), apply impulse toward goal.

Usage:
    cd /workspace/radeon-repo
    python scripts/render_1v1_kick.py --model /persistent/track3/models/checkpoints/best.pt --seconds 25
"""
import argparse, hashlib, json, os, subprocess, sys, time

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
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5,
                           cwd="/workspace/radeon-repo")
        return r.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Render 1v1 match with 2 robots + kicks")
    parser.add_argument("--model", default="/persistent/track3/models/checkpoints/best.pt")
    parser.add_argument("--walk_model", default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    parser.add_argument("--config", default="/workspace/radeon-repo/configs/hierarchical_agent.yaml")
    parser.add_argument("--seconds", type=int, default=25)
    parser.add_argument("--output", default="demos/1v1_kick_match.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    print(f"[render_1v1_kick] Model: {args.model}")
    print(f"[render_1v1_kick] Duration: {args.seconds}s")

    import genesis as gs
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"
    hl_cfg = cfg.get("high_level", {})

    # Use SoccerEnv1v1 — has 2 real robot entities
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
        opponent_init_pos=(3.0, 0.0, 0.7),  # opponent on right side
    )

    # Override obs dim to 19 (best.pt was trained with 19-dim obs)
    # SoccerEnv1v1 default is 21 (19+2 opponent), but our model expects 19
    env.hl_obs_dim = 19

    # Camera
    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Apply team colors: blue for agent, red for opponent
    try:
        blue_surface = gs.surfaces.Rough(color=(0.2, 0.5, 0.9), roughness=0.6)
        red_surface = gs.surfaces.Rough(color=(0.9, 0.2, 0.2), roughness=0.6)
        for link in env.robot.links:
            if hasattr(link, 'set_surface'):
                link.set_surface(blue_surface)
        for link in env.opponent.links:
            if hasattr(link, 'set_surface'):
                link.set_surface(red_surface)
        print("[render_1v1_kick] Team colors applied: blue(agent) vs red(opponent)")
    except Exception as e:
        print(f"[render_1v1_kick] Surface color failed: {e}")

    # Load RL policy (19-dim obs, 3-dim action)
    from rsl_rl.runners import OnPolicyRunner
    log_dir = os.path.dirname(os.path.dirname(args.model))
    train_cfg = cfg["train"]
    train_cfg["actor"]["hidden_dims"] = [256, 128, 64]
    train_cfg["critic"]["hidden_dims"] = [256, 128, 64]
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(args.model)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[render_1v1_kick] Policy loaded: {args.model}")

    # Steps
    hl_dt = env.high_level_dt  # 0.1s
    num_steps = int(args.seconds / hl_dt)
    frames_per_step = 3  # 10Hz × 3 = 30fps

    print(f"[render_1v1_kick] Steps: {num_steps}, frames/step: {frames_per_step}")
    print(f"[render_1v1_kick] Agent obs: 19-dim, Opponent: rule-based chase")

    # Kick cooldown
    kick_cd_agent = 0.0
    kick_cd_opponent = 0.0

    obs = env.reset()
    frames = []
    actions_log = []
    total_reward = 0.0
    nan_detected = False
    goals = 0

    from genesis.utils.misc import tensor_to_array

    for step in range(num_steps):
        # Build 19-dim obs for RL policy (strip opponent dims if present)
        obs_policy = obs["policy"]
        if obs_policy.shape[-1] > 19:
            obs_policy = obs_policy[:, :19]  # truncate to 19 dims

        with torch.no_grad():
            actions = policy(TensorDict_wrapper(obs_policy))

        if torch.isnan(actions).any() or torch.isinf(actions).any():
            nan_detected = True
            actions = torch.nan_to_num(actions)

        obs, rew, done, info = env.step(actions)
        total_reward += rew.mean().item()
        actions_log.append(actions.cpu().numpy().copy())

        # Kick logic for BOTH robots
        # Agent kick (left team attacks +x)
        agent_ball_dist = float(torch.norm(env.base_pos[0, :2] - env.ball_pos[0, :2]).item())
        if agent_ball_dist < 0.3 and kick_cd_agent < 0.01:
            goal_dir = torch.tensor([env.goal_x, 0.0, 0.0], device=env.device) - env.ball_pos[0]
            goal_dir_norm = goal_dir / (torch.norm(goal_dir) + 1e-6)
            impulse = goal_dir_norm * 3.0
            ball_qvel = env.ball.get_dofs_velocity().clone()
            ball_qvel[0, 0] = impulse[0]
            ball_qvel[0, 1] = impulse[1]
            ball_qvel[0, 2] = 0.0
            env.ball.set_dofs_velocity(ball_qvel)
            kick_cd_agent = 1.0
            print(f"[render_1v1_kick] 🔵 Agent kicks! ball_d={agent_ball_dist:.2f}")

        # Opponent kick (right team attacks -x)
        opp_ball_dist = float(torch.norm(env.opp_base_pos[0, :2] - env.ball_pos[0, :2]).item())
        if opp_ball_dist < 0.3 and kick_cd_opponent < 0.01:
            goal_dir = torch.tensor([-env.goal_x, 0.0, 0.0], device=env.device) - env.ball_pos[0]
            goal_dir_norm = goal_dir / (torch.norm(goal_dir) + 1e-6)
            impulse = goal_dir_norm * 3.0
            ball_qvel = env.ball.get_dofs_velocity().clone()
            ball_qvel[0, 0] = impulse[0]
            ball_qvel[0, 1] = impulse[1]
            ball_qvel[0, 2] = 0.0
            env.ball.set_dofs_velocity(ball_qvel)
            kick_cd_opponent = 1.0
            print(f"[render_1v1_kick] 🔴 Opponent kicks! ball_d={opp_ball_dist:.2f}")

        kick_cd_agent = max(0, kick_cd_agent - hl_dt)
        kick_cd_opponent = max(0, kick_cd_opponent - hl_dt)

        # Check goals
        ball_x = env.ball_pos[0, 0].item()
        ball_y = env.ball_pos[0, 1].item()
        if ball_x > env.goal_x and abs(ball_y) < env.goal_half:
            goals += 1
            print(f"[render_1v1_kick] ⚽ AGENT scores! Total: {goals}")
        elif ball_x < -env.goal_x and abs(ball_y) < env.goal_half:
            print(f"[render_1v1_kick] ⚽ Opponent scores!")

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
                    print(f"[render_1v1_kick] Camera error: {e}")

        if (step + 1) % 20 == 0:
            agent_pos = env.base_pos[0].cpu().numpy()
            opp_pos = env.opp_base_pos[0].cpu().numpy()
            print(f"[render_1v1_kick] step {step+1}/{num_steps}  rew={rew.mean().item():.3f}  "
                  f"🔵h={agent_pos[2]:.3f} pos=({agent_pos[0]:.1f},{agent_pos[1]:.1f})  "
                  f"🔴h={opp_pos[2]:.3f} pos=({opp_pos[0]:.1f},{opp_pos[1]:.1f})  "
                  f"ball=({ball_x:.1f},{ball_y:.1f})  frames={len(frames)}")

        if done.any():
            obs = env.reset()

    t_end = time.time()

    # Save video
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if frames and imageio is not None:
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[render_1v1_kick] Video saved: {args.output} ({len(frames)} frames)")
    else:
        print(f"\n[render_1v1_kick] WARNING: No frames saved")

    # Metadata
    actions_arr = np.array(actions_log)
    metadata = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
        "walk_model_path": args.walk_model,
        "env_name": "SoccerEnv1v1",
        "num_robots": 2,
        "team_size": 1,
        "left_team": "rl_policy (blue, best.pt, 19-dim obs)",
        "right_team": "rule_based (red, chase-ball + kick)",
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
        "kick_logic": "manual: impulse 3.0 m/s toward goal when ball_dist < 0.3m",
    }

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

    print(f"[render_1v1_kick] Metadata: {meta_path}")
    print(f"[render_1v1_kick] Goals: {goals}")
    print(f"[render_1v1_kick] Validation: {metadata['validation_status']}")
    print(f"[render_1v1_kick] DONE")


class TensorDict_wrapper:
    """Wrap a tensor so it looks like a TensorDict with ['policy'] key."""
    def __init__(self, tensor):
        self._tensor = tensor
    def __getitem__(self, key):
        return self._tensor


if __name__ == "__main__":
    main()
