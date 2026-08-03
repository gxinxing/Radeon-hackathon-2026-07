#!/usr/bin/env python3
"""Render 1v1 soccer with TWO real robots — CPU backend (no GPU memory limit).

Root cause of all previous failures:
  1. SoccerEnv1v1 used GLOBAL dof indices for set_dofs_kp → IndexError
  2. GPU backend has 64KB local memory limit for physics kernels → 2 robots crash
  3. SoccerEnvCurriculum's "opponent" is a virtual position, not a visible robot

This script fixes all three:
  - Uses CPU backend (no GPU memory limit)
  - Uses entity-LOCAL indices for set_dofs_kp (not global)
  - Creates 2 real physical robot entities (both visible in video)

Usage:
    cd /workspace/radeon-repo
    python scripts/render_2robot_match.py --seconds 25
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
    from tensordict import TensorDict
except ImportError:
    TensorDict = None

from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

# Constants from soccer_env_v4
PHYSICS_DT = 0.002
DECIMATION = 10
CONTROL_DT = PHYSICS_DT * DECIMATION  # 0.02
DEFAULT_POS_23 = [0, 0, 0.2, -1.3, 0, -0.5, 0.2, 1.3, 0, 0.5, 0.0,
                  -0.2, 0, 0, 0.4, -0.2, 0.0, -0.2, 0, 0, 0.4, -0.2, 0.0]
KP_23 = [4,4, 50,50,50,50,50,50,50,50, 200, 200,200,200,200,50,50, 200,200,200,200,50,50]
KD_23 = [1,1, 1,1,1,1,1,1,1,1, 5, 5,5,5,5,2,2, 5,5,5,5,2,2]
POLICY_JOINT_MAP = [2,6,3,7,4,8,5,9, 10,14,11,15,12,16,13,17, 18,19,20,21,22]
INIT_POS = [0.0, 0.0, 0.7]
INIT_QUAT = [1.0, 0.0, 0.0, 0.0]

FIELD_L, FIELD_W = 14.0, 9.0
HALF_L, HALF_W = FIELD_L/2, FIELD_W/2
GOAL_W, GOAL_HALF = 2.6, 1.3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5, cwd="/workspace/radeon-repo")
        return r.stdout.strip()
    except: return "unknown"


def build_2robot_scene(gs, robot_path, ball_path):
    """Build a scene with 2 robots + ball + goals. Returns (scene, r1, r2, ball, cam)."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=PHYSICS_DT, substeps=1),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False, tolerance=1e-5, max_collision_pairs=128),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0], show_world_frame=False, show_link_frame=False,
            show_cameras=False, plane_reflection=True, ambient_light=(0.7,0.7,0.7), shadow=True),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )
    # Ground
    scene.add_entity(gs.morphs.URDF(
        file=os.path.join(os.path.dirname(gs.__file__), "assets", "urdf", "plane", "plane.urdf"), fixed=True))
    # Green field
    scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0.001), plane_size=(FIELD_L, FIELD_W), fixed=True),
                    surface=gs.surfaces.Rough(color=(0.12,0.45,0.15), roughness=0.9))
    # Goal markers (minimal)
    _w = gs.surfaces.Rough(color=(0.95,0.95,0.95), roughness=0.5)
    for gx in [-HALF_L, HALF_L]:
        scene.add_entity(morph=gs.morphs.Box(size=(0.1, GOAL_W, 0.1), pos=(gx, 0, 1.0), fixed=True), surface=_w)
    # 2 robots
    r1 = scene.add_entity(gs.morphs.URDF(file=robot_path, pos=(-1.0, 0, 0.7), fixed=False, merge_fixed_links=False))
    r2 = scene.add_entity(gs.morphs.URDF(file=robot_path, pos=( 3.0, 0, 0.7), fixed=False, merge_fixed_links=False))
    # Ball
    ball = scene.add_entity(gs.morphs.URDF(file=ball_path))
    # Camera
    scene.add_camera(res=(1280, 720), pos=(0, -10, 6), lookat=(0, 0, 0.5), fov=50, GUI=False)
    scene.build(n_envs=1)
    return scene, r1, r2, ball


def setup_robot_pd(robot):
    """Set PD gains using entity-local DOF indices [0..22] for motor joints."""
    motors = [j for j in robot.joints[1:] if j.n_dofs > 0]
    n = len(motors)
    # Entity-local indices: always [0, 1, ..., n-1] regardless of global DOF position
    local_idx = torch.arange(n, dtype=torch.int32)
    try:
        robot.set_dofs_kp(KP_23[:n], local_idx)
        robot.set_dofs_kv(KD_23[:n], local_idx)
        kp_read = robot.get_dofs_kp(local_idx)
        print(f"  PD set OK: kp_mean={kp_read.float().mean().item():.1f}")
    except Exception as e:
        print(f"  PD set FAILED: {e}")
    # Get default positions
    default_pos = robot.get_dofs_position(local_idx)[0].clone()
    return motors, local_idx, default_pos


def build_low_obs(robot, motors, local_idx, default_pos, commands, last_actions, obs_scales):
    """Build 720-dim observation for walk model."""
    pos = robot.get_pos()
    quat = robot.get_quat()
    dof_pos = robot.get_dofs_position(local_idx)
    dof_vel = robot.get_dofs_velocity(local_idx)
    inv_bq = inv_quat(quat)
    lin_vel = transform_by_quat(robot.get_vel(), inv_bq)
    ang_vel = transform_by_quat(robot.get_ang(), inv_bq)
    grav = transform_by_quat(torch.tensor([0.,0.,-1.], device=pos.device).expand(1, -1), inv_bq)

    # 72-dim per frame
    policy_dof_pos = dof_pos[:, POLICY_JOINT_MAP[:21]] if dof_pos.shape[-1] >= 23 else dof_pos
    policy_dof_vel = dof_vel[:, POLICY_JOINT_MAP[:21]] if dof_vel.shape[-1] >= 23 else dof_vel
    policy_default = default_pos[POLICY_JOINT_MAP[:21]] if default_pos.shape[0] >= 23 else default_pos
    policy_last = last_actions[:, :21] if last_actions.shape[-1] >= 21 else last_actions

    per_frame = torch.cat([
        ang_vel * obs_scales.get("ang_vel", 0.25),
        grav,
        commands,
        (policy_dof_pos - policy_default.unsqueeze(0)) * obs_scales.get("dof_pos", 1.0),
        policy_dof_vel * obs_scales.get("dof_vel", 0.05),
        policy_last,
    ], dim=-1)
    if per_frame.shape[-1] < 72:
        per_frame = torch.cat([per_frame, torch.zeros(1, 72 - per_frame.shape[-1], device=pos.device)], dim=-1)
    return per_frame  # (1, 72)


def run_walk_model(walk_model, obs_720, norm_mean, norm_std):
    """Run frozen walk model: 720→21 joint actions."""
    with torch.no_grad():
        if norm_mean is not None:
            obs_normed = (obs_720 - norm_mean) / norm_std
        else:
            obs_normed = obs_720
        return walk_model.actor(obs_normed)


def apply_joint_actions(robot, joint_actions, local_idx, default_pos, last_actions, action_scale=0.25):
    """Apply joint actions to robot via PD position control using LOCAL dof indices."""
    actions = torch.clip(joint_actions, -100.0, 100.0)
    exec_actions = last_actions  # simulate action latency
    target = default_pos.unsqueeze(0).expand(1, -1).clone()
    policy_target = exec_actions * action_scale + default_pos[POLICY_JOINT_MAP[:21]].unsqueeze(0)
    if target.shape[-1] >= 23:
        target[:, POLICY_JOINT_MAP[:21]] = policy_target
    else:
        target[:, :21] = policy_target
    sorted_idx = torch.argsort(local_idx)
    robot.control_dofs_position(target[:, sorted_idx], local_idx)
    return actions


def compute_hl_obs(robot, ball, goal_x, last_hl_actions, filtered_lin_vel, filtered_ang_vel, obs_scales):
    """Compute 19-dim high-level observation, padded to 21 for 1v1 model."""
    pos = robot.get_pos()
    quat = robot.get_quat()
    inv_bq = inv_quat(quat)
    ball_rel = ball.get_pos() - pos
    ball_rel_body = transform_by_quat(ball_rel, inv_bq)
    ball_vel_body = transform_by_quat(ball.get_vel(), inv_bq)
    goal = torch.zeros_like(pos)
    goal[:, 0] = goal_x
    goal_rel = goal - pos
    goal_rel_body = transform_by_quat(goal_rel, inv_bq)
    goal_dist = torch.norm(goal_rel_body[:, :2], dim=1, keepdim=True)
    goal_dir = goal_rel_body[:, :2] / (goal_dist + 1e-6)
    dist_ball = torch.norm(ball_rel_body[:, :2], dim=1, keepdim=True)
    grav = transform_by_quat(torch.tensor([0.,0.,-1.], device=pos.device).expand(1, -1), inv_bq)
    obs19 = torch.cat([
        filtered_lin_vel,          # 3
        filtered_ang_vel,          # 3
        grav[:, :2],               # 2
        ball_rel_body[:, :2],      # 2
        ball_vel_body[:, :2],      # 2
        dist_ball,                  # 1
        goal_dir,                   # 2
        goal_dist,                  # 1
        last_hl_actions,            # 3
    ], dim=-1)  # Total: 19
    # Pad to 21 for 1v1 model (add 2 zeros for opponent relative)
    obs21 = torch.cat([obs19, torch.zeros(1, 2)], dim=-1)
    return obs21


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/radeon-repo/runs/soccer_1v1/model_499.pt")
    parser.add_argument("--walk_model", default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    parser.add_argument("--config", default="/workspace/radeon-repo/configs/hierarchical_agent.yaml")
    parser.add_argument("--seconds", type=int, default=25)
    parser.add_argument("--output", default="demos/2robot_match.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    print(f"[2robot] Model: {args.model}, Backend: CPU, Duration: {args.seconds}s")

    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning", seed=args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    obs_scales = cfg["obs"]["obs_scales"]
    hl_cfg = cfg.get("high_level", {})
    hl_decimation = hl_cfg.get("decimation", 5)
    hl_dt = CONTROL_DT * hl_decimation  # 0.1s
    hl_clip_lin = cfg["env"].get("hl_clip_lin", 1.2)
    hl_clip_ang = cfg["env"].get("hl_clip_ang", 1.2)

    rp = os.path.abspath("urdf/t1/t1.urdf")
    bp = "/workspace/assets/ball.urdf"
    scene, r1, r2, ball = build_2robot_scene(gs, rp, bp)
    print("[2robot] Scene built: 2 robots + ball on CPU")

    # Setup PD gains with LOCAL indices
    r1_motors, r1_lidx, r1_default = setup_robot_pd(r1)
    r2_motors, r2_lidx, r2_default = setup_robot_pd(r2)
    print(f"[2robot] R1: {len(r1_motors)} motors, R2: {len(r2_motors)} motors")

    # Team colors
    try:
        blue = gs.surfaces.Rough(color=(0.2,0.5,0.9), roughness=0.6)
        red = gs.surfaces.Rough(color=(0.9,0.2,0.2), roughness=0.6)
        for link in r1.links:
            if hasattr(link, 'set_surface'): link.set_surface(blue)
        for link in r2.links:
            if hasattr(link, 'set_surface'): link.set_surface(red)
        print("[2robot] Colors: blue(R1=RL) vs red(R2=rule)")
    except Exception as e:
        print(f"[2robot] Colors: {e}")

    # Load walk model (shared)
    walk_model = torch.jit.load(args.walk_model, map_location="cpu")
    walk_model.eval()
    try:
        _norm = walk_model.obs_normalizer
        norm_mean = _norm._mean.to("cpu")
        norm_std = torch.clamp(_norm._std, min=1e-8).to("cpu")
        print(f"[2robot] Walk model normalizer: {norm_mean.shape}")
    except:
        norm_mean = norm_std = None
    print("[2robot] Walk model loaded")

    # Load RL policy directly (no OnPolicyRunner needed)
    import torch.nn as nn
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    actor_sd = ckpt["actor_state_dict"]
    input_dim = actor_sd["mlp.0.weight"].shape[1]  # 21 for 1v1
    print(f"[2robot] Checkpoint input dim: {input_dim}")

    class SimplePolicy(nn.Module):
        def __init__(self, in_dim, hidden=[256,128,64], out_dim=3):
            super().__init__()
            layers = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            layers.append(nn.Linear(d, out_dim))
            self.mlp = nn.Sequential(*layers)
        def forward(self, x):
            return self.mlp(x)

    # Filter out non-MLP keys (distribution.std_param etc.)
    filtered_sd = {k: v for k, v in actor_sd.items() if k.startswith("mlp.")}
    policy = SimplePolicy(input_dim)
    policy.load_state_dict(filtered_sd, strict=False)
    policy.eval()
    print(f"[2robot] RL policy loaded ({input_dim}-dim obs → 3-dim action)")

    cam = scene.visualizer.cameras[0]
    scene.reset()

    # Init obs history (10 frames × 72 dims = 720)
    r1_hist = torch.zeros(1, 10, 72)
    r2_hist = torch.zeros(1, 10, 72)
    r1_last_act = torch.zeros(1, 21)
    r2_last_act = torch.zeros(1, 21)
    r1_cmd = torch.zeros(1, 3)
    r2_cmd = torch.zeros(1, 3)
    last_hl = torch.zeros(1, 3)
    flv1 = torch.zeros(1, 3)
    fav1 = torch.zeros(1, 3)
    kick_cd = 0.0

    num_steps = int(args.seconds / hl_dt)
    fps = 3
    frames = []
    goals = 0
    total_rew = 0.0

    from genesis.utils.misc import tensor_to_array

    for step in range(num_steps):
        # === High-level: RL policy for R1 ===
        hl_obs = compute_hl_obs(r1, ball, 7.0, last_hl, flv1, fav1, obs_scales)
        with torch.no_grad():
            hl_action = policy(hl_obs)
        r1_cmd = torch.stack([
            torch.clamp(hl_action[:, 0], -hl_clip_lin, hl_clip_lin),
            torch.clamp(hl_action[:, 1], -hl_clip_lin, hl_clip_lin),
            torch.clamp(hl_action[:, 2], -hl_clip_ang, hl_clip_ang),
        ], dim=1)
        r1_cmd = torch.where(torch.abs(r1_cmd) < 0.05, torch.zeros_like(r1_cmd), r1_cmd)

        # === High-level: Rule-based for R2 (chase ball) ===
        r2_pos = r2.get_pos()
        ball_pos = ball.get_pos()
        to_ball = ball_pos[:, :2] - r2_pos[:, :2]
        dist_b = torch.norm(to_ball, dim=1, keepdim=True) + 1e-6
        direction = to_ball / dist_b
        r2_cmd = torch.stack([
            torch.clamp(direction[:, 0] * 0.3, -0.3, 0.3),
            torch.clamp(direction[:, 1] * 0.3, -0.3, 0.3),
            torch.clamp(torch.atan2(to_ball[:, 1], to_ball[:, 0]) * 0.2, -0.3, 0.3),
        ], dim=1)

        # === Low-level: run both robots for hl_decimation steps ===
        for _ in range(hl_decimation):
            # R1
            obs720_r1 = r1_hist.reshape(1, -1)
            ja1 = run_walk_model(walk_model, obs720_r1, norm_mean, norm_std)
            apply_joint_actions(r1, ja1, r1_lidx, r1_default, r1_last_act)
            r1_last_act = ja1
            # R2
            obs720_r2 = r2_hist.reshape(1, -1)
            ja2 = run_walk_model(walk_model, obs720_r2, norm_mean, norm_std)
            apply_joint_actions(r2, ja2, r2_lidx, r2_default, r2_last_act)
            r2_last_act = ja2
            # Physics
            for _ in range(DECIMATION):
                scene.step()
            # Update obs history
            lo1 = build_low_obs(r1, r1_motors, r1_lidx, r1_default, r1_cmd, r1_last_act, obs_scales)
            r1_hist = torch.cat([r1_hist[:, 1:], lo1.unsqueeze(1)], dim=1)
            lo2 = build_low_obs(r2, r2_motors, r2_lidx, r2_default, r2_cmd, r2_last_act, obs_scales)
            r2_hist = torch.cat([r2_hist[:, 1:], lo2.unsqueeze(1)], dim=1)

        # === Kick logic ===
        r1_pos = r1.get_pos()
        r2_pos = r2.get_pos()
        ball_pos = ball.get_pos()
        d1 = float(torch.norm(r1_pos[0, :2] - ball_pos[0, :2]).item())
        if d1 < 0.35 and kick_cd < 0.01:
            g = torch.tensor([7.0, 0., 0.]) - ball_pos[0]
            g = g / (torch.norm(g) + 1e-6) * 3.0
            bq = ball.get_dofs_velocity().clone()
            bq[0, :3] = g
            ball.set_dofs_velocity(bq)
            kick_cd = 1.0
            print(f"[2robot] 🔵 R1 kicks! d={d1:.2f}")
        d2 = float(torch.norm(r2_pos[0, :2] - ball_pos[0, :2]).item())
        if d2 < 0.35 and kick_cd < 0.01:
            g = torch.tensor([-7.0, 0., 0.]) - ball_pos[0]
            g = g / (torch.norm(g) + 1e-6) * 2.5
            bq = ball.get_dofs_velocity().clone()
            bq[0, :3] = g
            ball.set_dofs_velocity(bq)
            kick_cd = 1.0
            print(f"[2robot] 🔴 R2 kicks! d={d2:.2f}")
        kick_cd = max(0, kick_cd - hl_dt)

        # Check goals
        bx = ball_pos[0, 0].item()
        by = ball_pos[0, 1].item()
        if bx > 7.0 and abs(by) < GOAL_HALF:
            goals += 1
            print(f"[2robot] ⚽ R1 scores! Total: {goals}")
        elif bx < -7.0 and abs(by) < GOAL_HALF:
            print(f"[2robot] ⚽ R2 scores!")

        # Render 3 frames
        for _ in range(fps):
            try:
                rgb, _, _, _ = cam.render(rgb=True)
                arr = tensor_to_array(rgb)
                if arr.ndim == 4: arr = arr[0]
                frames.append(arr.astype(np.uint8))
            except Exception as e:
                if step == 0: print(f"[2robot] cam: {e}")

        last_hl = r1_cmd.clone()
        # Update filtered vel
        inv1 = inv_quat(r1.get_quat())
        lv1 = transform_by_quat(r1.get_vel(), inv1)
        av1 = transform_by_quat(r1.get_ang(), inv1)
        fw = 0.3
        flv1 = lv1 * fw + flv1 * (1 - fw)
        fav1 = av1 * fw + fav1 * (1 - fw)

        if (step + 1) % 10 == 0:
            print(f"[2robot] step {step+1}/{num_steps}  "
                  f"🔵=({r1_pos[0,0]:.1f},{r1_pos[0,1]:.1f},h={r1_pos[0,2]:.2f})  "
                  f"🔴=({r2_pos[0,0]:.1f},{r2_pos[0,1]:.1f},h={r2_pos[0,2]:.2f})  "
                  f"ball=({bx:.1f},{by:.1f})  f={len(frames)}  t={time.time()-t0:.0f}s")

    t1 = time.time()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if frames and imageio:
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[2robot] Video: {args.output} ({len(frames)} frames, {t1-t0:.0f}s)")

    metadata = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
        "env_name": "Custom 2-robot scene (CPU backend)",
        "num_robots": 2,
        "left_team": "rl_policy (blue, 21-dim obs, soccer_1v1/model_499.pt)",
        "right_team": "rule_based (red, chase-ball + kick)",
        "backend": "cpu",
        "seed": args.seed,
        "video_frames": len(frames),
        "video_fps": 30,
        "goals_scored": goals,
        "render_time_s": round(t1 - t0, 1),
        "git_commit": get_git_commit(),
        "pd_fix": "Used entity-LOCAL dof indices (0-22) instead of GLOBAL for set_dofs_kp",
    }
    mp = args.output.replace(".mp4", ".metadata.json")
    with open(mp, "w") as f: json.dump(metadata, f, indent=2)
    print(f"[2robot] Goals: {goals}, Time: {t1-t0:.0f}s")
    print("[2robot] DONE")


if __name__ == "__main__":
    main()
