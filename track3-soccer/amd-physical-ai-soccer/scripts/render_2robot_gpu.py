#!/usr/bin/env python3
"""Render 1v1 soccer: R1 via SoccerEnvHierarchical (proven), R2 via manual walk model.

Strategy:
  - Use SoccerEnvHierarchical for R1 (already works on GPU, produces standing robot)
  - After env.step() completes R1's low-level loop, manually step R2's walk model
  - R2 is added to the same scene via monkey-patching scene.build
  - Both robots share the same physics scene and ball

This avoids the 720-dim obs reconstruction issue by letting the env handle R1.
"""
import argparse, hashlib, json, os, subprocess, sys, time, math
import numpy as np, torch, yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, "/workspace/radeon-repo")

try:
    import imageio
except: imageio = None

from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

PHYSICS_DT = 0.002
DECIMATION = 10
CONTROL_DT = PHYSICS_DT * DECIMATION

KP_23 = [4,4, 50,50,50,50,50,50,50,50, 200, 200,200,200,200,50,50, 200,200,200,200,50,50]
KD_23 = [1,1, 1,1,1,1,1,1,1,1, 5, 5,5,5,5,2,2, 5,5,5,5,2,2]
POLICY_JOINT_MAP = [2,6,3,7,4,8,5,9, 10,14,11,15,12,16,13,17, 18,19,20,21,22]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

def get_git_commit():
    try:
        r = subprocess.run(["git","rev-parse","HEAD"], capture_output=True, text=True, timeout=5, cwd="/workspace/radeon-repo")
        return r.stdout.strip()
    except: return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/radeon-repo/runs/soccer_1v1/model_499.pt")
    parser.add_argument("--walk_model", default="/workspace/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    parser.add_argument("--config", default="/workspace/radeon-repo/configs/hierarchical_agent.yaml")
    parser.add_argument("--seconds", type=int, default=25)
    parser.add_argument("--output", default="demos/2robot_gpu_match.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    print(f"[2robot_gpu] Model: {args.model}, Duration: {args.seconds}s")

    import genesis as gs

    # ── Monkey-patch SoccerEnv._build_scene to add R2 before build ──
    import soccer_env_v4
    _orig_build = soccer_env_v4.SoccerEnv._build_scene

    r2_holder = {}

    def patched_build(self, show_viewer):
        original_build = gs.Scene.build
        def intercept_build(scene_self, *a, **kw):
            if 'r2' not in r2_holder:
                rp = self.cfg["robot_urdf"]
                if not os.path.isabs(rp): rp = os.path.abspath(rp)
                r2_holder['r2'] = scene_self.add_entity(
                    gs.morphs.URDF(file=rp, pos=(3.0, 0.0, 0.7), fixed=False, merge_fixed_links=False))
                print("[2robot_gpu] R2 added to scene before build")
            return original_build(scene_self, *a, **kw)
        gs.Scene.build = intercept_build
        try:
            _orig_build(self, show_viewer)
        finally:
            gs.Scene.build = original_build

    soccer_env_v4.SoccerEnv._build_scene = patched_build

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = dict(cfg["env"]); env_cfg["task"] = "chase_hl"
    hl_cfg = cfg.get("high_level", {})

    from soccer_env_hierarchical import SoccerEnvHierarchical
    env = SoccerEnvHierarchical(
        num_envs=1, env_cfg=env_cfg, obs_cfg=cfg["obs"], reward_cfg=cfg["reward"],
        command_cfg=cfg["command"], walk_model_path=args.walk_model,
        high_level_decimation=hl_cfg.get("decimation", 5), show_viewer=False)

    # Restore original
    soccer_env_v4.SoccerEnv._build_scene = _orig_build

    r2 = r2_holder['r2']
    print(f"[2robot_gpu] Scene: R1={type(env.robot)}, R2={type(r2)}")

    # Setup R2 PD gains (local indices)
    r2_motors = [j for j in r2.joints[1:] if j.n_dofs > 0]
    r2_lidx = torch.arange(len(r2_motors), dtype=torch.int32)
    r2.set_dofs_kp(KP_23[:len(r2_motors)], r2_lidx)
    r2.set_dofs_kv(KD_23[:len(r2_motors)], r2_lidx)
    r2_default = r2.get_dofs_position(r2_lidx)[0].clone()
    r2_last_act = torch.zeros(1, 21)
    r2_cmd = torch.zeros(1, 3)

    # Team colors
    try:
        blue = gs.surfaces.Rough(color=(0.2,0.5,0.9), roughness=0.6)
        red = gs.surfaces.Rough(color=(0.9,0.2,0.2), roughness=0.6)
        for link in env.robot.links:
            if hasattr(link, 'set_surface'): link.set_surface(blue)
        for link in r2.links:
            if hasattr(link, 'set_surface'): link.set_surface(red)
        print("[2robot_gpu] Colors: blue(R1) vs red(R2)")
    except: pass

    # Load RL policy directly
    import torch.nn as nn
    ckpt = torch.load(args.model, map_location=gs.device, weights_only=False)
    actor_sd = ckpt["actor_state_dict"]
    input_dim = actor_sd["mlp.0.weight"].shape[1]
    class P(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.mlp = nn.Sequential(nn.Linear(d,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,3))
        def forward(s,x): return s.mlp(x)
    policy = P(input_dim)
    policy.load_state_dict({k:v for k,v in actor_sd.items() if k.startswith("mlp.")}, strict=False)
    policy = policy.to(gs.device)
    policy.eval()
    print(f"[2robot_gpu] Policy loaded ({input_dim}-dim)")

    cam = env.scene.visualizer.cameras[0]
    env.scene.reset()

    # Init R2 obs history
    r2_hist = torch.zeros(1, 10, 72, device=gs.device)
    obs_scales = cfg["obs"]["obs_scales"]
    hl_dt = env.high_level_dt
    num_steps = int(args.seconds / hl_dt)
    fps = 3
    frames = []
    goals = 0
    kick_cd = 0.0

    # R2 walk model (reuse env's walk model)
    walk_model = env.walk_model
    norm_mean = env._norm_mean
    norm_std = env._norm_std

    obs = env.reset()
    from genesis.utils.misc import tensor_to_array

    for step in range(num_steps):
        # ── R1: RL policy ──
        hl_obs = obs["policy"]
        if hl_obs.shape[-1] < input_dim:
            hl_obs = torch.cat([hl_obs, torch.zeros(1, input_dim - hl_obs.shape[-1], device=gs.device)], dim=-1)
        elif hl_obs.shape[-1] > input_dim:
            hl_obs = hl_obs[:, :input_dim]
        with torch.no_grad():
            hl_action = policy(hl_obs)

        # ── R2: rule-based chase ball ──
        r2_pos = r2.get_pos()
        ball_pos = env.ball_pos
        to_ball = ball_pos[0, :2] - r2_pos[0, :2]
        dist_b = float(torch.norm(to_ball).item()) + 1e-6
        direction = to_ball / dist_b
        r2_cmd = torch.tensor([[float(torch.clamp(direction[0]*0.3,-0.3,0.3)),
                                  float(torch.clamp(direction[1]*0.3,-0.3,0.3)),
                                  float(torch.clamp(torch.atan2(to_ball[1],to_ball[0])*0.2,-0.3,0.3))]],
                                device=gs.device)

        # ── Env step (handles R1's full low-level loop) ──
        obs, rew, done, info = env.step(hl_action)

        # ── R2: manual low-level steps (same decimation as env) ──
        # Build R2 720-dim obs
        r2_dof_pos = r2.get_dofs_position(r2_lidx)
        r2_dof_vel = r2.get_dofs_velocity(r2_lidx)
        r2_quat = r2.get_quat()
        inv_r2q = inv_quat(r2_quat)
        r2_grav = transform_by_quat(torch.tensor([0.,0.,-1.], device=gs.device).expand(1,-1), inv_r2q)
        r2_ang_vel = transform_by_quat(r2.get_ang(), inv_r2q)

        policy_dof_pos = r2_dof_pos[:, POLICY_JOINT_MAP[:21]] if r2_dof_pos.shape[-1] >= 23 else r2_dof_pos
        policy_dof_vel = r2_dof_vel[:, POLICY_JOINT_MAP[:21]] if r2_dof_vel.shape[-1] >= 23 else r2_dof_vel
        policy_default = r2_default[POLICY_JOINT_MAP[:21]] if r2_default.shape[0] >= 23 else r2_default

        r2_frame = torch.cat([
            r2_ang_vel * obs_scales.get("ang_vel", 0.25),
            r2_grav,
            r2_cmd,
            (policy_dof_pos - policy_default.unsqueeze(0)) * obs_scales.get("dof_pos", 1.0),
            policy_dof_vel * obs_scales.get("dof_vel", 0.05),
            r2_last_act,
        ], dim=-1)
        if r2_frame.shape[-1] < 72:
            r2_frame = torch.cat([r2_frame, torch.zeros(1, 72-r2_frame.shape[-1], device=gs.device)], dim=-1)

        r2_hist = torch.cat([r2_hist[:, 1:], r2_frame.unsqueeze(1)], dim=1)
        r2_obs720 = r2_hist.reshape(1, -1)

        with torch.no_grad():
            if norm_mean is not None:
                r2_normed = (r2_obs720 - norm_mean) / norm_std
            else:
                r2_normed = r2_obs720
            r2_ja = walk_model.actor(r2_normed)

        # Apply R2 joint actions
        r2_actions = torch.clip(r2_ja, -100.0, 100.0)
        r2_target = r2_default.unsqueeze(0).expand(1,-1).clone()
        r2_policy_target = r2_last_act * 0.25 + r2_default[POLICY_JOINT_MAP[:21]].unsqueeze(0)
        if r2_target.shape[-1] >= 23:
            r2_target[:, POLICY_JOINT_MAP[:21]] = r2_policy_target
        sorted_idx = torch.argsort(r2_lidx)
        r2.control_dofs_position(r2_target[:, sorted_idx], r2_lidx)
        r2_last_act = r2_ja

        # Kick logic for both
        r1_pos = env.base_pos
        d1 = float(torch.norm(r1_pos[0,:2] - ball_pos[0,:2]).item())
        if d1 < 0.35 and kick_cd < 0.01:
            g = torch.tensor([7.,0.,0.], device=gs.device) - ball_pos[0]
            g = g/(torch.norm(g)+1e-6)*3.0
            bq = env.ball.get_dofs_velocity().clone()
            bq[0,:3] = g
            env.ball.set_dofs_velocity(bq)
            kick_cd = 1.0
            print(f"[2robot_gpu] 🔵 R1 kicks! d={d1:.2f}")

        d2 = float(torch.norm(r2_pos[0,:2] - ball_pos[0,:2]).item())
        if d2 < 0.35 and kick_cd < 0.01:
            g = torch.tensor([-7.,0.,0.], device=gs.device) - ball_pos[0]
            g = g/(torch.norm(g)+1e-6)*2.5
            bq = env.ball.get_dofs_velocity().clone()
            bq[0,:3] = g
            env.ball.set_dofs_velocity(bq)
            kick_cd = 1.0
            print(f"[2robot_gpu] 🔴 R2 kicks! d={d2:.2f}")
        kick_cd = max(0, kick_cd - hl_dt)

        # Goals
        bx = ball_pos[0,0].item(); by = ball_pos[0,1].item()
        if bx > 7.0 and abs(by) < 1.3:
            goals += 1; print(f"[2robot_gpu] ⚽ R1 scores! {goals}")
        elif bx < -7.0 and abs(by) < 1.3:
            print(f"[2robot_gpu] ⚽ R2 scores!")

        # Render
        for _ in range(fps):
            try:
                rgb,_,_,_ = cam.render(rgb=True)
                arr = tensor_to_array(rgb)
                if arr.ndim == 4: arr = arr[0]
                frames.append(arr.astype(np.uint8))
            except Exception as e:
                if step == 0: print(f"[2robot_gpu] cam: {e}")

        if (step+1) % 10 == 0:
            print(f"[2robot_gpu] step {step+1}/{num_steps}  "
                  f"🔵=({r1_pos[0,0]:.1f},{r1_pos[0,1]:.1f},h={r1_pos[0,2]:.2f})  "
                  f"🔴=({r2_pos[0,0]:.1f},{r2_pos[0,1]:.1f},h={r2_pos[0,2]:.2f})  "
                  f"ball=({bx:.1f},{by:.1f})  f={len(frames)}  t={time.time()-t0:.0f}s")

        if done.any():
            obs = env.reset()

    t1 = time.time()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if frames and imageio:
        imageio.mimsave(args.output, frames, fps=30, codec='libx264')
        print(f"\n[2robot_gpu] Video: {args.output} ({len(frames)} frames, {t1-t0:.0f}s)")

    meta = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else "N/A",
        "env_name": "SoccerEnvHierarchical + manual R2 (GPU)",
        "num_robots": 2,
        "backend": "gpu",
        "seed": args.seed,
        "video_frames": len(frames),
        "video_fps": 30,
        "goals_scored": goals,
        "render_time_s": round(t1-t0, 1),
        "git_commit": get_git_commit(),
    }
    mp = args.output.replace(".mp4",".metadata.json")
    with open(mp,"w") as f: json.dump(meta, f, indent=2)
    print(f"[2robot_gpu] Goals: {goals}")
    print("[2robot_gpu] DONE")


if __name__ == "__main__":
    main()
