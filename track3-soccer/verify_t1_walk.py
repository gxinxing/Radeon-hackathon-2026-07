#!/usr/bin/env python3
"""Verify t1_walk.pt alignment with current Genesis env and run 30s no-fall walking test.

Run on cloud instance:
    cd /workspace/amd-physical-ai-soccer
    python verify_t1_walk.py [--no_norm] [--obs_scales ang_vel=0.25,dof_vel=0.05]

Checks:
  1. Actor input dimension (from t1_walk.pt)
  2. Current env obs dimension
  3. Actor output dimension (from t1_walk.pt)
  4. Current action dimension
  5. 21 joint names and order
  6. action_scale
  7. PD kp/kd
  8. obs normalizer stats (per-component breakdown)
  9. obs_scales (ang_vel, dof_vel) diagnosis

Then runs 30-second walking playback (1500 steps @ dt=0.02s).
Fails if robot falls (base height < 0.4m or pitch/roll > 30°) within 30s.
"""
import os, sys, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# ── Config ──
T1_WALK_PATH = "/workspace/booster/booster_deploy/tasks/locomotion/models/t1_walk.pt"
CFG_PATH = "configs/soccer_agent.yaml"
WALK_DURATION_S = 30.0
DT = 0.02  # control dt = PHYSICS_DT(0.002) * DECIMATION(10)
N_STEPS = int(WALK_DURATION_S / DT)  # 1500 steps
FALL_HEIGHT = 0.4
FALL_PITCH_DEG = 30.0
FALL_ROLL_DEG = 30.0

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"


def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_norm", action="store_true",
                        help="Skip obs normalizer (use raw obs)")
    parser.add_argument("--t1_path", type=str, default=T1_WALK_PATH,
                        help="Path to t1_walk.pt")
    args = parser.parse_args()

    results = {}
    use_norm = not args.no_norm

    # ═══════════════════════════════════════════════════════
    # 1. Load t1_walk.pt and inspect model architecture
    # ═══════════════════════════════════════════════════════
    banner("STEP 1: Load t1_walk.pt & inspect architecture")

    t1_path = args.t1_path
    if not os.path.exists(t1_path):
        print(f"{FAIL} t1_walk.pt not found at {t1_path}")
        print("  Searching for alternative locations...")
        for alt in [
            os.path.expanduser("~/booster_deploy/tasks/locomotion/models/t1_walk.pt"),
            "t1_walk.pt",
            "models/t1_walk.pt",
            os.path.expanduser("~/.cache/t1_walk.pt"),
        ]:
            if os.path.exists(alt):
                t1_path = alt
                print(f"  Found at: {t1_path}")
                break
        else:
            print(f"{FAIL} Cannot find t1_walk.pt anywhere")
            sys.exit(1)

    model = torch.jit.load(t1_path, map_location="cpu")
    model.eval()

    # Inspect model structure
    print(f"\nModel type: {type(model)}")
    top_attrs = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr, None))]
    print(f"Model top-level attributes: {top_attrs}")

    # Extract actor
    actor = model.actor
    print(f"\nActor type: {type(actor)}")

    # Get actor parameters
    actor_params = list(actor.named_parameters())
    print(f"\nActor parameters ({len(actor_params)}):")
    for name, param in actor_params:
        print(f"  {name}: {tuple(param.shape)}")

    # Determine input/output dims
    input_dim = None
    output_dim = None
    for name, param in actor_params:
        if "weight" in name and ("mlp.0" in name or name == "0.weight"):
            input_dim = param.shape[1]
        if "weight" in name:
            output_dim = param.shape[0]

    # Try forward pass to confirm output dim
    try:
        test_in = torch.zeros(1, input_dim) if input_dim else torch.zeros(1, 720)
        with torch.no_grad():
            test_out = actor(test_in)
        if hasattr(test_out, 'shape'):
            output_dim = test_out.shape[-1]
        elif hasattr(test_out, 'mean') and hasattr(test_out.mean, 'shape'):
            output_dim = test_out.mean.shape[-1]
    except Exception as e:
        print(f"  Forward pass test: {e}")

    print(f"\n  ┌────────────────────────────────────┐")
    print(f"  │ Actor INPUT  dim:  {str(input_dim):>6s}            │")
    print(f"  │ Actor OUTPUT dim:  {str(output_dim):>6s}            │")
    print(f"  └────────────────────────────────────┘")
    results["actor_input_dim"] = input_dim
    results["actor_output_dim"] = output_dim

    # Extract obs normalizer
    obs_norm = None
    try:
        obs_norm = model.obs_normalizer
        norm_mean = obs_norm._mean
        norm_var = obs_norm._var
        norm_std = obs_norm._std
        norm_count = obs_norm.count
        print(f"\n  Obs normalizer found:")
        print(f"    mean shape: {tuple(norm_mean.shape)}")
        print(f"    var  shape: {tuple(norm_var.shape)}")
        print(f"    count:      {norm_count.item()}")
    except Exception as e:
        print(f"\n  {WARN} Obs normalizer not accessible: {e}")

    # ═══════════════════════════════════════════════════════
    # 2. Create env and check dimensions
    # ═══════════════════════════════════════════════════════
    banner("STEP 2: Create Genesis env & verify dimensions")

    import yaml
    import genesis as gs

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=42)

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "balance"
    obs_cfg = cfg["obs"]
    reward_cfg = cfg["reward"]
    command_cfg = cfg["command"]

    from envs.soccer_env import SoccerEnv

    env = SoccerEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    env_obs_dim = env._obs_dim()
    env_action_dim = env.num_actions
    print(f"\n  Env OBS    dim: {env_obs_dim}")
    print(f"  Env ACTION dim: {env_action_dim}")
    results["env_obs_dim"] = env_obs_dim
    results["env_action_dim"] = env_action_dim

    # ═══════════════════════════════════════════════════════
    # 3. Verify joint names and order
    # ═══════════════════════════════════════════════════════
    banner("STEP 3: 21 joint names and order")

    # POLICY_JOINT_NAMES is a module-level constant in soccer_env.py
    from envs.soccer_env import POLICY_JOINT_NAMES as ENV_POLICY_JOINT_NAMES

    all_motor_names = [j.name for j in env.motor_joints]
    print(f"\n  All motor joints ({len(all_motor_names)}):")
    for i, name in enumerate(all_motor_names):
        is_policy = "★ policy" if name in ENV_POLICY_JOINT_NAMES else ""
        print(f"    [{i:2d}] {name:<30s} {is_policy}")

    policy_names = [all_motor_names[idx] for idx in env.policy_joint_indices.tolist()]
    print(f"\n  Policy joints ({len(policy_names)}):")
    for i, name in enumerate(policy_names):
        print(f"    [{i:2d}] {name}")

    expected = [
        "Left_Shoulder_Pitch", "Right_Shoulder_Pitch", "Waist",
        "Left_Shoulder_Roll", "Right_Shoulder_Roll",
        "Left_Hip_Pitch", "Right_Hip_Pitch",
        "Left_Elbow_Pitch", "Right_Elbow_Pitch",
        "Left_Hip_Roll", "Right_Hip_Roll",
        "Left_Elbow_Yaw", "Right_Elbow_Yaw",
        "Left_Hip_Yaw", "Right_Hip_Yaw",
        "Left_Knee_Pitch", "Right_Knee_Pitch",
        "Left_Ankle_Pitch", "Right_Ankle_Pitch",
        "Left_Ankle_Roll", "Right_Ankle_Roll",
    ]
    match = policy_names == expected
    print(f"\n  Joint order match: {PASS if match else FAIL}")
    if not match:
        print("  EXPECTED vs ACTUAL:")
        for i, n in enumerate(expected):
            actual = policy_names[i] if i < len(policy_names) else "???"
            mark = "" if n == actual else "  ← MISMATCH"
            print(f"    [{i:2d}] {n:<30s} vs {actual}{mark}")
    results["joint_count"] = len(policy_names)
    results["joint_order_match"] = match

    # ═══════════════════════════════════════════════════════
    # 4. action_scale
    # ═══════════════════════════════════════════════════════
    banner("STEP 4: action_scale")
    print(f"\n  env.action_scale = {env.action_scale}")
    print(f"  Expected:         0.25  (booster_deploy T1WalkControllerCfg)")
    ok = abs(env.action_scale - 0.25) < 1e-6
    print(f"  {PASS if ok else FAIL}")
    results["action_scale"] = env.action_scale

    # ═══════════════════════════════════════════════════════
    # 5. PD kp/kd
    # ═══════════════════════════════════════════════════════
    banner("STEP 5: PD kp/kd (23 motors)")

    try:
        actual_kp_t = env.robot.get_dofs_kp(env.motors_dof_idx)
        actual_kd_t = env.robot.get_dofs_kv(env.motors_dof_idx)
        kp_list = [float(actual_kp_t[i]) for i in range(len(all_motor_names))]
        kd_list = [float(actual_kd_t[i]) for i in range(len(all_motor_names))]
    except Exception:
        if hasattr(env, 'KP_23'):
            kp_list = list(env.KP_23)
            kd_list = list(env.KD_23)
        else:
            kp_list = cfg.get("env", {}).get("kp", [0]*23)
            kd_list = cfg.get("env", {}).get("kd", [0]*23)

    print(f"\n  {'#':>2s}  {'Joint':<30s} {'KP':>8s} {'KD':>8s}  Expected_KP  Expected_KD")
    print(f"  {'─'*70}")
    expected_kp = [4.0, 4.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                   200.0, 200.0, 200.0, 200.0, 200.0, 50.0, 50.0,
                   200.0, 200.0, 200.0, 200.0, 50.0, 50.0]
    expected_kd = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
                   5.0, 5.0, 5.0, 5.0, 2.0, 2.0]
    for i, name in enumerate(all_motor_names):
        ekp = expected_kp[i] if i < len(expected_kp) else "?"
        ekd = expected_kd[i] if i < len(expected_kd) else "?"
        kp_ok = abs(kp_list[i] - ekp) < 0.01 if isinstance(ekp, float) else True
        mark = "✓" if kp_ok else "✗"
        print(f"  {mark} [{i:2d}] {name:<30s} {kp_list[i]:>8.1f} {kd_list[i]:>8.1f}  {ekp:>10.1f}  {ekd:>10.1f}")

    kp_match = all(abs(kp_list[i] - expected_kp[i]) < 0.01 for i in range(min(len(kp_list), 23))) if len(kp_list) >= 23 else False
    kd_match = all(abs(kd_list[i] - expected_kd[i]) < 0.01 for i in range(min(len(kd_list), 23))) if len(kd_list) >= 23 else False
    print(f"\n  KP match: {PASS if kp_match else FAIL}")
    print(f"  KD match: {PASS if kd_match else FAIL}")
    results["kp_match"] = kp_match
    results["kd_match"] = kd_match

    # ═══════════════════════════════════════════════════════
    # 6. obs normalizer per-component breakdown
    # ═══════════════════════════════════════════════════════
    banner("STEP 6: obs normalizer per-component analysis")

    if obs_norm is not None:
        norm_input_dim = norm_mean.shape[-1]
        print(f"\n  Normalizer input dim: {norm_input_dim}")
        print(f"  Env obs dim:          {env_obs_dim}")

        # Per-frame breakdown: ang_vel(3) + grav(3) + cmd(3) + dof_pos(21) + dof_vel(21) + act(21) = 72
        per_frame = env.obs_dim_per_frame
        n_hist = env.obs_history_length
        print(f"  Per-frame dim:        {per_frame}")
        print(f"  History length:       {n_hist}")
        print(f"  Total obs:            {per_frame * n_hist}")

        # Analyze FIRST frame (dims 0..71) of the normalizer
        components = [
            ("ang_vel",          0,  3, "obs_scales.ang_vel (typical 0.25)"),
            ("projected_gravity", 3,  6, "obs_scales.gravity (typical 1.0)"),
            ("commands",          6,  9, "obs_scales.command (typical 1.0)"),
            ("dof_pos",           9, 30, "obs_scales.dof_pos (typical 1.0)"),
            ("dof_vel",          30, 51, "obs_scales.dof_vel (typical 0.05)"),
            ("last_action",      51, 72, "no scale (raw actions)"),
        ]

        print(f"\n  Per-component normalizer stats (first frame only):")
        print(f"  {'Component':<20s} {'Dims':>6s} {'Mean range':>22s} {'Std range':>22s}  Note")
        print(f"  {'─'*90}")
        for name, start, end, note in components:
            m_slice = norm_mean[0, start:end]
            s_slice = norm_std[0, start:end]
            m_lo, m_hi = float(m_slice.min()), float(m_slice.max())
            s_lo, s_hi = float(s_slice.min()), float(s_slice.max())
            print(f"  {name:<20s} [{start:2d}:{end:2d}]  [{m_lo:>8.4f}, {m_hi:>8.4f}]  [{s_lo:>8.4f}, {s_hi:>8.4f}]  {note}")

        # Diagnose scaling
        print(f"\n  ── Scaling Diagnosis ──")
        # ang_vel: if trained with scale=0.25, mean should be small (~0.1)
        #          if trained with scale=1.0, mean could be larger (~0.5)
        ang_vel_std = float(norm_std[0, 0:3].mean())
        dof_vel_std = float(norm_std[0, 30:51].mean())
        print(f"  ang_vel avg std: {ang_vel_std:.4f}")
        print(f"    If ~0.1-0.5 → likely scaled by 0.25 (booster_deploy default)")
        print(f"    If ~1.0-5.0 → likely unscaled (scale=1.0)")
        print(f"  dof_vel avg std: {dof_vel_std:.4f}")
        print(f"    If ~0.01-0.1 → likely scaled by 0.05 (booster_deploy default)")
        print(f"    If ~1.0-10  → likely unscaled (scale=1.0)")

        match = norm_input_dim == env_obs_dim
        print(f"\n  Normalizer dim == Env obs dim: {PASS if match else FAIL}")
        results["norm_obs_match"] = match
    else:
        print(f"\n  {WARN} Obs normalizer not loaded from model")
        results["norm_obs_match"] = False
        norm_input_dim = None

    # ═══════════════════════════════════════════════════════
    # 7. Cross-check dimensions
    # ═══════════════════════════════════════════════════════
    banner("STEP 7: Cross-check dimensions")

    checks = [
        ("Actor input == Env obs",     input_dim == env_obs_dim,     f"{input_dim} vs {env_obs_dim}"),
        ("Actor output == Env action",   output_dim == env_action_dim, f"{output_dim} vs {env_action_dim}"),
        ("Joint count == 21",            len(policy_names) == 21,     f"{len(policy_names)}"),
        ("action_scale == 0.25",          abs(env.action_scale - 0.25) < 1e-6, f"{env.action_scale}"),
        ("KP match",                      kp_match,                     ""),
        ("KD match",                      kd_match,                     ""),
    ]
    if obs_norm is not None:
        checks.append(("Norm dim == Obs dim", norm_input_dim == env_obs_dim, f"{norm_input_dim} vs {env_obs_dim}"))

    all_pass = True
    for desc, ok, detail in checks:
        status = PASS if ok else FAIL
        if not ok:
            all_pass = False
        print(f"  {status} {desc}: {detail}")

    # ═══════════════════════════════════════════════════════
    # 8. Obs scales check
    # ═══════════════════════════════════════════════════════
    banner("STEP 8: obs_scales application check")

    # Check what scales the env is actually using
    s_ang_vel = env.obs_scales.get("ang_vel", "MISSING")
    s_dof_pos = env.obs_scales.get("dof_pos", "MISSING")
    s_dof_vel = env.obs_scales.get("dof_vel", "MISSING")

    print(f"\n  env.obs_scales from config:")
    print(f"    ang_vel:  {s_ang_vel}")
    print(f"    dof_pos:  {s_dof_pos}")
    print(f"    dof_vel:  {s_dof_vel}")

    # Check if _update_observation applies them
    import inspect
    obs_src = inspect.getsource(env._update_observation)
    applies_scales = "obs_scales" in obs_src
    print(f"\n  _update_observation applies obs_scales: {PASS if applies_scales else FAIL}")
    if not applies_scales:
        print(f"  {FAIL} CRITICAL: env._update_observation() does NOT apply obs_scales!")
        print(f"  The normalizer expects SCALED obs from booster_deploy training.")
        print(f"  Without obs_scales, the normalized obs will be wrong → robot falls.")
        print(f"  Fix: apply self.obs_scales in _update_observation()")

    if obs_norm is not None:
        # Diagnose what scales the normalizer was trained with
        ang_vel_std = float(norm_std[0, 0:3].mean())
        dof_vel_std = float(norm_std[0, 30:51].mean())
        print(f"\n  Normalizer std diagnosis:")
        print(f"    ang_vel avg std: {ang_vel_std:.4f}", end="")
        if ang_vel_std < 0.5:
            print(f"  → likely scaled by 0.25 ✓")
        else:
            print(f"  → likely unscaled (scale=1.0)")
        print(f"    dof_vel avg std: {dof_vel_std:.4f}", end="")
        if dof_vel_std < 0.5:
            print(f"  → likely scaled by 0.05 ✓")
        else:
            print(f"  → likely unscaled (scale=1.0)")

    # ═══════════════════════════════════════════════════════
    # 9. 30-second walking playback
    # ═══════════════════════════════════════════════════════
    banner(f"STEP 9: {WALK_DURATION_S}s walking playback ({N_STEPS} steps)")

    # Load model onto device
    model = torch.jit.load(t1_path, map_location=gs.device)
    model.eval()

    # Reset env with zero commands (stand still)
    obs = env.reset()
    # Override commands to zero (walk forward slowly)
    env.commands[:, 0] = 0.5  # forward velocity
    env.commands[:, 1] = 0.0  # lateral
    env.commands[:, 2] = 0.0  # angular

    obs_tensor = obs["policy"].to(gs.device)

    # Prepare normalizer tensors
    has_norm = False
    if use_norm and obs_norm is not None:
        norm_mean_d = norm_mean.to(gs.device)
        norm_std_d = norm_std.to(gs.device)
        has_norm = True
        print(f"\n  Using obs normalizer (count={norm_count.item()})")
    else:
        print(f"\n  {WARN} NOT using obs normalizer (raw obs → actor)")

    print(f"\n  Using obs from env (scales applied in _update_observation)")
    print(f"\n  Running {N_STEPS} steps (dt={DT}s, total={WALK_DURATION_S}s)...")
    print(f"  Fall threshold: height<{FALL_HEIGHT}m, |pitch|>{FALL_PITCH_DEG}°, |roll|>{FALL_ROLL_DEG}°\n")

    step_fallen = -1
    min_height = 999.0
    max_pitch = 0.0
    max_roll = 0.0
    start_time = time.time()

    for step in range(N_STEPS):
        # Env already applies obs_scales in _update_observation() (fixed in v4)
        # So obs_tensor is already correctly scaled.
        # We just need to apply the normalizer on top.

        # Apply normalizer
        if has_norm:
            obs_normed = (obs_tensor - norm_mean_d) / torch.clamp(norm_std_d, min=1e-8)
        else:
            obs_normed = obs_tensor

        # Run actor
        with torch.no_grad():
            action = model.actor(obs_normed)

        # Step env
        obs, rew, done, info = env.step(action)
        obs_tensor = obs["policy"].to(gs.device)

        # Track stats
        height = float(env.base_pos[0, 2].item())
        pitch = float(env.base_euler[0, 1].item())
        roll = float(env.base_euler[0, 0].item())
        min_height = min(min_height, height)
        max_pitch = max(max_pitch, abs(pitch))
        max_roll = max(max_roll, abs(roll))

        # Check fall
        fallen = (height < FALL_HEIGHT) or (abs(pitch) > FALL_PITCH_DEG) or (abs(roll) > FALL_ROLL_DEG)

        if fallen and step_fallen < 0:
            step_fallen = step
            elapsed_s = step * DT
            print(f"  ❌ FELL at step {step} ({elapsed_s:.1f}s)")
            print(f"     height={height:.3f}m  pitch={pitch:.1f}°  roll={roll:.1f}°")
            # Print first few action values for debugging
            print(f"     action[0:5]: {action[0, :5].cpu().tolist()}")
            break

        # Progress report
        if (step + 1) % 300 == 0:
            elapsed_s = (step + 1) * DT
            print(f"  step {step+1:4d}/{N_STEPS}  ({elapsed_s:.0f}s)  "
                  f"h={height:.3f}m  pitch={pitch:.1f}°  roll={roll:.1f}°  "
                  f"rew={rew.mean().item():.3f}")

    elapsed = time.time() - start_time
    sim_time = N_STEPS * DT if step_fallen < 0 else step_fallen * DT
    walked_full = step_fallen < 0

    # ═══════════════════════════════════════════════════════
    # 10. Final report
    # ═══════════════════════════════════════════════════════
    banner("FINAL REPORT")

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │ t1_walk.pt Alignment Verification                          │
  ├─────────────────────────────────────────────────────────────┤
  │ Actor input dim:     {str(input_dim):>6s}                                 │
  │ Env obs dim:         {str(env_obs_dim):>6s}                                 │
  │ Actor output dim:    {str(output_dim):>6s}                                 │
  │ Env action dim:      {str(env_action_dim):>6s}                                 │
  │ Input match:         {str(input_dim == env_obs_dim):>6s}                                 │
  │ Output match:        {str(output_dim == env_action_dim):>6s}                                 │
  ├─────────────────────────────────────────────────────────────┤
  │ Policy joints:       {str(len(policy_names)):>6s}                                 │
  │ Joint order match:   {str(match):>6s}                                 │
  │ action_scale:        {str(env.action_scale):>6s}                                 │
  │ KP match:            {str(kp_match):>6s}                                 │
  │ KD match:            {str(kd_match):>6s}                                 │
  │ Obs normalizer:      {str(has_norm):>6s}                                 │
  │ obs_scales (cfg):   ang_vel={s_ang_vel}, dof_vel={s_dof_vel}           │
  ├─────────────────────────────────────────────────────────────┤
  │ Walking test:                                              │
  │   Target duration:   {WALK_DURATION_S:>6.1f}s                                 │
  │   Actual duration:   {sim_time:>6.1f}s                                 │
  │   Min height:        {min_height:>6.3f}m                                 │
  │   Max |pitch|:       {max_pitch:>6.1f}°                                 │
  │   Max |roll|:        {max_roll:>6.1f}°                                 │
  │   Wall time:         {elapsed:>6.1f}s                                 │
  │   Result:            {'PASS ✓' if walked_full else 'FAIL ✗':>6s}                                 │
  └─────────────────────────────────────────────────────────────┘
""")

    if walked_full:
        print(f"  {PASS} t1_walk.pt walked {WALK_DURATION_S}s without falling.")
        print(f"  Training may proceed.")
    else:
        print(f"  {FAIL} t1_walk.pt fell after {sim_time:.1f}s.")
        print(f"  DO NOT train until walking is restored.")
        print(f"\n  Possible fixes to try:")
        print(f"    1. Check obs_scales in config match booster_deploy (ang_vel=0.25, dof_vel=0.05)")
        print(f"    2. Try without normalizer:")
        print(f"       python verify_t1_walk.py --no_norm")
        print(f"    3. Check if envs/soccer_env.py is v4 (720 obs) not v3 (726 obs)")
        print(f"    4. Verify robot URDF joint order matches POLICY_JOINT_NAMES")
        print(f"    5. Check physics dt=0.002, decimation=10")
        print(f"    6. Verify DEFAULT_POS_23 matches booster_deploy standing pose")

    return walked_full


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
