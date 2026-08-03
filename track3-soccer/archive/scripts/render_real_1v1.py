#!/usr/bin/env python3
"""Render 1v1 soccer with TWO real physical robots + kick logic.

Uses SoccerEnv1v1 which adds a real opponent robot entity.
Checkpoint: runs/soccer_1v1/model_499.pt (21-dim obs, 3-dim action)
Both robots use frozen t1_walk.pt for locomotion.
Kick: manual impulse when robot close to ball.

Usage:
    cd /workspace/radeon-repo
    python scripts/render_real_1v1.py --seconds 25
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

try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None


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
    parser.add_argument("--output", default="demos/real_1v1_match.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    print(f"[render_real_1v1] Model: {args.model}")
    print(f"[render_real_1v1] Duration: {args.seconds}s")

    # Kill stale GPU processes
    os.system("pkill -9 -f genesis 2>/dev/null; sleep 2")
    os.system("rocminfo > /dev/null 2>&1")  # reset GPU

    import genesis as gs

    # Monkey-patch to reduce collision pairs for 2-robot scene
    _orig_build_scene = None

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"
    hl_cfg = cfg.get("high_level", {})

    # Import 1v1 env
    from soccer_env_1v1 import SoccerEnv1v1

    # Monkey-patch the parent _build_scene to use lower collision settings
    import soccer_env_v4
    _orig_v4_build = soccer_env_v4.SoccerEnv._build_scene

    def _patched_build_scene(self, show_viewer):
        """Build scene with reduced collision pairs for 2-robot setup."""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=soccer_env_v4.PHYSICS_DT, substeps=1),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=64,  # minimal: just robot-ground + ball
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -10, 6), camera_lookat=(0, 0, 0.5), camera_fov=50),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=True,
                ambient_light=(0.7, 0.7, 0.7),
                shadow=True,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=show_viewer,
        )

        # Ground
        self.scene.add_entity(
            gs.morphs.URDF(file=soccer_env_v4._genesis_asset("urdf", "plane", "plane.urdf"), fixed=True))

        # Green field plane
        self.scene.add_entity(
            morph=gs.morphs.Plane(pos=(0, 0, 0.001), plane_size=(14.0, 9.0), fixed=True),
            surface=gs.surfaces.Rough(color=(0.12, 0.45, 0.15), roughness=0.9))

        # Minimal goal markers (just crossbars, no posts to save collision pairs)
        _gs_s = gs.surfaces.Rough(color=(0.95, 0.95, 0.95), roughness=0.5)
        for gx in [-7.0, 7.0]:
            self.scene.add_entity(
                morph=gs.morphs.Box(size=(0.1, 2.6, 0.1), pos=(gx, 0, 1.0), fixed=True),
                surface=_gs_s)

        # Robot (agent) — merge fixed links to reduce DOF count
        robot_path = self.cfg["robot_urdf"]
        if not os.path.isabs(robot_path):
            ga = soccer_env_v4._genesis_asset(robot_path)
            robot_path = ga if os.path.exists(ga) else os.path.abspath(robot_path)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=robot_path, pos=soccer_env_v4.INIT_POS, quat=soccer_env_v4.INIT_QUAT,
                          fixed=False, merge_fixed_links=True))

        # Ball
        ball_path = os.path.join(os.path.dirname(__file__), "..", "assets", "ball.urdf")
        if not os.path.exists(ball_path):
            ball_path = "/workspace/assets/ball.urdf"
        self.ball = self.scene.add_entity(gs.morphs.URDF(file=os.path.abspath(ball_path)))

        # Camera
        self.scene.add_camera(res=(1280, 720), pos=(0, -10, 6), lookat=(0, 0, 0.5), fov=50, GUI=False)
        self.scene.build(n_envs=self.num_envs)

    # Apply patch
    soccer_env_v4.SoccerEnv._build_scene = _patched_build_scene

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

    # Restore original
    soccer_env_v4.SoccerEnv._build_scene = _orig_v4_build

    print(f"[render_real_1v1] Scene built! 2 robots in scene.")
    print(f"[render_real_1v1] Agent robot: {type(env.robot)}")
    print(f"[render_real_1v1] Opponent robot: {type(env.opponent)}")

    # Camera
    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Apply team colors
    try:
        blue_surface = gs.surfaces.Rough(color=(0.2, 0.5, 0.9), roughness=0.6)
        red_surface = gs.surfaces.Rough(color=(0.9, 0.2, 0.2), roughness=0.6)
        for link in env.robot.links:
            if hasattr(link, 'set_surface'):
                link.set_surface(blue_surface)
        for link in env.opponent.links:
            if hasattr(link, 'set_surface'):
                link.set_surface(red_surface)
        print("[render_real_1v1] Team colors: blue(agent) vs red(opponent)")
    except Exception as e:
        print(f"[render_real_1v1] Colors failed: {e}")

    # Load policy (21-dim obs, 3-dim action)
    from rsl_rl.runners import OnPolicyRunner
    log_dir = os.path.dirname(args.model)
    train_cfg = cfg["train"]
    train_cfg["actor"]["hidden_dims"] = [256, 128, 64]
    train_cfg["critic"]["hidden_dims"] = [256, 128, 64]
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(args.model)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[render_real_1v1] Policy loaded: {args.model}")
    print(f"[render_real_1v1] Obs dim: {env.hl_obs_dim}, Action dim: {env.num_actions}")

    hl_dt = env.high_level_dt
    num_steps = int(args.seconds / hl_dt)
    frames_per_step = 3

    obs = env.reset()
    frames = []
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

        # Kick logic for agent (left attacks +x)
        agent_ball_dist = float(torch.norm(env.base_pos[0, :2] - env.ball_pos[0, :2]).item())
        if agent_ball_dist < 0.35:
            goal_dir = torch.tensor([env.goal_x, 0.0, 0.0], device=env.device) - env.ball_pos[0]
            goal_dir_norm = goal_dir / (torch.norm(goal_dir) + 1e-6)
            impulse = goal_dir_norm * 3.0
            ball_qvel = env.ball.get_dofs_velocity().clone()
            ball_qvel[0, 0] = impulse[0]
            ball_qvel[0, 1] = impulse[1]
            ball_qvel[0, 2] = 0.0
            env.ball.set_dofs_velocity(ball_qvel)
            print(f"[render_real_1v1] 🔵 Agent kicks! dist={agent_ball_dist:.2f}")

        # Kick logic for opponent (right attacks -x)
        opp_ball_dist = float(torch.norm(env.opp_base_pos[0, :2] - env.ball_pos[0, :2]).item())
        if opp_ball_dist < 0.35:
            goal_dir = torch.tensor([-env.goal_x, 0.0, 0.0], device=env.device) - env.ball_pos[0]
            goal_dir_norm = goal_dir / (torch.norm(goal_dir) + 1e-6)
            impulse = goal_dir_norm * 2.5
            ball_qvel = env.ball.get_dofs_velocity().clone()
            ball_qvel[0, 0] = impulse[0]
            ball_qvel[0, 1] = impulse[1]
            ball_qvel[0, 2] = 0.0
            env.ball.set_dofs_velocity(ball_qvel)
            print(f"[render_real_1v1] 🔴 Opponent kicks! dist={opp_ball_dist:.2f}")

        # Check goals
        ball_x = env.ball_pos[0, 0].item()
        ball_y = env.ball_pos[0, 1].item()
        if ball_x > env.goal_x and abs(ball_y) < env.goal_half:
            goals += 1
            print(f"[render_real_1v1] ⚽ AGENT scores! {goals}")
        elif ball_x < -env.goal_x and abs(ball_y) < env.goal_half:
            print(f"[render_real_1v1] ⚽ OPPONENT scores!")

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
                    print(f"[render_real_1v1] Camera error: {e}")

        if (step + 1) % 20 == 0:
            agent_pos = env.base_pos[0].cpu().numpy()
            opp_pos = env.opp_base_pos[0].cpu().numpy()
            print(f"[render_real_1v1] step {step+1}/{num_steps}  rew={rew.mean().item():.3f}  "
                  f"🔵=({agent_pos[0]:.1f},{agent_pos[1]:.1f},h={agent_pos[2]:.2f})  "
                  f"🔴=({opp_pos[0]:.1f},{opp_pos[1]:.1f},h={opp_pos[2]:.2f})  "
                  f"ball=({ball_x:.1f},{ball_y:.1f})  "
                  f"frames={len(frames)}")

        if done.any():
            obs = env.reset()

    t_end = time.time()

    # Save video
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if frames and imageio is not None:
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[render_real_1v1] Video saved: {args.output} ({len(frames)} frames)")

    # Metadata
    actions_arr = np.array(actions_log)
    metadata = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
        "walk_model_path": args.walk_model,
        "env_name": "SoccerEnv1v1 (2 real robots)",
        "num_robots": 2,
        "team_size": 1,
        "left_team": "rl_policy (blue, 21-dim obs, 3-dim action)",
        "right_team": "rule_based (red, chase-ball + kick)",
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
        "kick_logic": "manual impulse 3.0 m/s when ball_dist < 0.35m",
        "scene_config": "self_collision=False, max_collision_pairs=64, merge_fixed_links=True",
    }
    meta_path = args.output.replace(".mp4", ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[render_real_1v1] Metadata: {meta_path}")
    print(f"[render_real_1v1] Goals: {goals}")
    print(f"[render_real_1v1] DONE")


if __name__ == "__main__":
    main()
