#!/usr/bin/env python3
"""Render a verified 3v3 soccer match video with full metadata and validation.

This is the EXPERIMENTAL version of the rendering pipeline. It does NOT modify
the original render_hierarchical.py. Instead it:
  1. Reads configs/inference_manifest.yaml for all parameters
  2. Uses the correct env class (SoccerEnvHierarchical) with import fallback
  3. Records full metadata (model SHA256, seed, config path, git commit, etc.)
  4. Logs per-step robot positions, policy outputs, and physics state
  5. Saves video + metadata + match log with guaranteed consistency

Progressive validation modes:
  --mode single   : 1 robot, 500 steps (no rendering)
  --mode multi     : 6 robots, 500 steps (no rendering)
  --mode short     : 3v3, 10 seconds, render video
  --mode full      : 3v3, 25 seconds, render video

Usage:
    python scripts/render_match_verified.py --config configs/inference_manifest.yaml --mode single
    python scripts/render_match_verified.py --config configs/inference_manifest.yaml --mode short --output demos/verified_short.mp4
    python scripts/render_match_verified.py --config configs/inference_manifest.yaml --mode full --output demos/verified_match.mp4

Requirements: Genesis + GPU (run on remote server)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Also support running from /workspace/radeon-repo
for p in [PROJECT_ROOT, "/workspace/radeon-repo", "/workspace/radeon-repo/src"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import yaml
import numpy as np

try:
    import torch
except ImportError:
    torch = None

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
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_config(config_path):
    """Load inference manifest YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_env(config, num_envs=1, show_viewer=False, config_path=None):
    """Create the hierarchical soccer environment with import fallback."""
    import genesis as gs

    # Initialize Genesis ONCE (required before any gs.tc_float usage)
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, precision="32", logging_level="warning",
                seed=config.get("seed", 42))
        gs._initialized = True

    env_cfg = {
        "robot_urdf": config["robot_urdf"],
        "episode_length_s": config.get("match_duration_seconds", 25.0),
        "dt": config["sim_dt"],
        "substeps": config["substeps"],
        "action_scale": config["action_scale"],
        "clip_actions": config["clip_actions"],
        "simulate_action_latency": True,
        "ball_radius": config["ball_radius"],
        "field": config["field"],
        "goal_width": config["goal_width"],
        "circle_radius": config["circle_radius"],
        "fall_height": config["fall_height"],
        "termination_pitch_deg": config["termination_pitch_deg"],
        "termination_roll_deg": config["termination_roll_deg"],
        "base_init_pos": config["base_init_pos"],
        "base_init_quat": config["base_init_quat"],
        "hl_clip_lin": config["action_clip"]["lin"],
        "hl_clip_ang": config["action_clip"]["ang"],
        "multiagent_obs": False,
        "task": config["task"],
    }

    obs_cfg = {"obs_scales": config["obs_scales"]}
    # Load reward config from the actual training config (reward is computed during step)
    reward_cfg_path = os.path.join(os.path.dirname(config_path), "hierarchical_agent.yaml")
    if os.path.exists(reward_cfg_path):
        with open(reward_cfg_path) as f:
            full_cfg = yaml.safe_load(f)
        reward_cfg = full_cfg.get("reward", {})
    else:
        reward_cfg = {}  # Empty reward will cause KeyError; must provide
    command_cfg = {"goal_dir": [1.0, 0.0, 0.0]}

    hl_cfg = {
        "decimation": config["control_decimation"],
        "walk_model_path": config["walk_model_path"],
    }

    # Import with fallback
    try:
        from envs.soccer_env_hierarchical import SoccerEnvHierarchical
    except ImportError:
        try:
            from soccer_env_hierarchical import SoccerEnvHierarchical
        except ImportError:
            # Try from PROJECT_ROOT/envs/
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "envs"))
            from soccer_env_hierarchical import SoccerEnvHierarchical

    env = SoccerEnvHierarchical(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        walk_model_path=hl_cfg["walk_model_path"],
        high_level_decimation=hl_cfg["decimation"],
        show_viewer=show_viewer,
    )

    print(f"[render_verified] Env class: {type(env).__name__}")
    print(f"[render_verified] Env device: {env.device}")
    print(f"[render_verified] Obs dim: {env.hl_obs_dim}")
    print(f"[render_verified] Action dim: {env.num_actions}")
    print(f"[render_verified] Walk model: {hl_cfg['walk_model_path']}")

    return env


def load_policy(env, config, use_onnx=False):
    """Load the high-level policy from checkpoint or ONNX."""
    if use_onnx:
        import onnxruntime as ort
        onnx_path = config["onnx_path"]
        print(f"[render_verified] Loading ONNX: {onnx_path}")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        class ONNXPolicy:
            def __call__(self, obs):
                obs_np = obs["policy"].cpu().numpy() if isinstance(obs, dict) else obs.cpu().numpy()
                result = sess.run(None, {sess.get_inputs()[0].name: obs_np})
                return torch.tensor(result[0], dtype=torch.float32, device=env.device)

        return ONNXPolicy(), onnx_path
    else:
        from rsl_rl.runners import OnPolicyRunner
        model_path = config["model_path"]
        log_dir = os.path.dirname(os.path.dirname(model_path))
        print(f"[render_verified] Loading .pt: {model_path}")
        print(f"[render_verified] Log dir: {log_dir}")

        train_cfg = {
            "algorithm": {"class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01,
                          "entropy_coef": 0.003, "gamma": 0.99, "lam": 0.95,
                          "learning_rate": 0.001, "max_grad_norm": 1.0,
                          "num_learning_epochs": 5, "num_mini_batches": 4,
                          "schedule": "adaptive", "use_clipped_value_loss": True,
                          "value_loss_coef": 1.0},
            "actor": {"class_name": "MLPModel", "hidden_dims": [256, 128, 64], "activation": "elu",
                      "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}},
            "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128, 64], "activation": "elu"},
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "num_steps_per_env": 24,
            "save_interval": 50,
            "logger": "tensorboard",
            "run_name": "hierarchical_soccer",
            "max_iterations": 500,
            "experiment_name": "hierarchical_soccer",
        }

        runner = OnPolicyRunner(env, train_cfg, log_dir, device=env.device)
        runner.load(model_path)
        policy = runner.get_inference_policy(device=env.device)
        return policy, model_path


def run_mode_single(config, num_steps=500):
    """Mode 1: Single robot, 500 steps, no rendering."""
    print("\n" + "=" * 60)
    print("[render_verified] MODE: single robot, 500 steps")
    print("=" * 60)

    env = create_env(config, num_envs=1, config_path=args.config)
    policy, model_path = load_policy(env, config, use_onnx=False)

    obs = env.reset()
    actions_log = []
    positions = []
    heights = []

    for i in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)

        actions_log.append(actions.cpu().numpy().copy())
        pos = env.base_pos[0].cpu().numpy()
        positions.append(pos[:2].copy())
        heights.append(pos[2])

        if (i + 1) % 100 == 0:
            print(f"  step {i+1}/{num_steps}  rew={rew.mean().item():.3f}  "
                  f"h={pos[2]:.3f}  pos=({pos[0]:.2f}, {pos[1]:.2f})")

    # Analyze results
    actions_arr = np.array(actions_log)
    positions_arr = np.array(positions)
    height_arr = np.array(heights)

    result = {
        "mode": "single",
        "steps": num_steps,
        "mean_reward": float(rew.mean().item()),
        "final_height": float(height_arr[-1]),
        "min_height": float(height_arr.min()),
        "max_height": float(height_arr.max()),
        "position_range_x": float(positions_arr[:, 0].max() - positions_arr[:, 0].min()),
        "position_range_y": float(positions_arr[:, 1].max() - positions_arr[:, 1].min()),
        "action_mean": float(actions_arr.mean()),
        "action_std": float(actions_arr.std()),
        "action_has_nan": bool(np.any(np.isnan(actions_arr))),
        "action_has_inf": bool(np.any(np.isinf(actions_arr))),
        "robot_moved": float(np.linalg.norm(positions_arr[-1] - positions_arr[0])) > 0.1,
    }

    print(f"\n[result] Robot moved: {result['robot_moved']}")
    print(f"[result] Height range: [{result['min_height']:.3f}, {result['max_height']:.3f}]")
    print(f"[result] Position range: x={result['position_range_x']:.2f}, y={result['position_range_y']:.2f}")
    print(f"[result] Action std: {result['action_std']:.4f}")

    return result


def run_mode_multi(config, num_steps=500):
    """Mode 2: 6 robots, 500 steps, no rendering."""
    print("\n" + "=" * 60)
    print("[render_verified] MODE: 6 robots, 500 steps")
    print("=" * 60)

    env = create_env(config, num_envs=6, config_path=args.config)
    policy, model_path = load_policy(env, config, use_onnx=False)

    obs = env.reset()
    all_positions = []

    for i in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)

        positions = env.base_pos.cpu().numpy()
        all_positions.append(positions.copy())

        if (i + 1) % 100 == 0:
            print(f"  step {i+1}/{num_steps}  rew={rew.mean().item():.3f}")
            for j in range(min(6, positions.shape[0])):
                print(f"    robot {j}: h={positions[j,2]:.3f} pos=({positions[j,0]:.2f}, {positions[j,1]:.2f})")

    all_pos = np.array(all_positions)
    movement = np.linalg.norm(all_pos[-1] - all_pos[0], axis=1)

    result = {
        "mode": "multi",
        "num_robots": 6,
        "steps": num_steps,
        "mean_reward": float(rew.mean().item()),
        "robot_movements": [float(m) for m in movement],
        "all_robots_moved": bool(np.all(movement > 0.05)),
        "min_height": float(all_pos[:, :, 2].min()),
        "max_height": float(all_pos[:, :, 2].max()),
    }

    print(f"\n[result] All robots moved: {result['all_robots_moved']}")
    print(f"[result] Movements: {[f'{m:.2f}' for m in movement]}")

    return result


def run_mode_render(config, seconds=25, output_path="demos/verified_match.mp4"):
    """Mode 3/4: 3v3 match with video rendering and metadata."""
    print("\n" + "=" * 60)
    print(f"[render_verified] MODE: 3v3 match, {seconds}s video")
    print("=" * 60)

    start_time = time.time()
    seed = config["seed"]
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    # Create environment
    env = create_env(config, num_envs=1, show_viewer=False, config_path=args.config)
    policy, model_path = load_policy(env, config, use_onnx=False)

    # Get camera
    cam = None
    try:
        cam = env.scene.visualizer.cameras[0]
    except (AttributeError, IndexError) as e:
        print(f"[render_verified] WARNING: Could not get camera: {e}")

    env.scene.reset()

    # Calculate steps
    hl_dt = config["sim_dt"] * config["control_decimation"]  # 0.02 * 5 = 0.1s
    num_steps = int(seconds / hl_dt)
    render_every = max(1, int(1.0 / (hl_dt * config["render_fps"])))  # render at target fps

    print(f"[render_verified] Steps: {num_steps}, render_every: {render_every}")
    print(f"[render_verified] HL dt: {hl_dt}s, target FPS: {config['render_fps']}")

    # Rollout
    obs = env.reset()
    frames = []
    actions_log = []
    positions_log = []
    total_reward = 0.0
    nan_detected = False

    for i in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)

        # Check for NaN/Inf in actions
        if torch is not None:
            if torch.isnan(actions).any() or torch.isinf(actions).any():
                print(f"[render_verified] WARNING: NaN/Inf in actions at step {i}")
                nan_detected = True
                actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        obs, rew, done, info = env.step(actions)
        total_reward += rew.mean().item()

        actions_log.append(actions.cpu().numpy().copy())
        pos = env.base_pos[0].cpu().numpy()
        positions_log.append(pos.copy())

        # Render frame
        if cam is not None and i % render_every == 0:
            try:
                from genesis.utils.misc import tensor_to_array
                rgb, _, _, _ = cam.render(rgb=True)
                arr = tensor_to_array(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
                frames.append(arr.astype(np.uint8))
            except Exception as e:
                if i == 0:
                    print(f"[render_verified] Camera render failed: {e}")

        if (i + 1) % 50 == 0:
            print(f"  step {i+1}/{num_steps}  rew={rew.mean().item():.3f}  "
                  f"h={pos[2]:.3f}  ball_d={torch.norm(env.base_pos[0,:2]-env.ball_pos[0,:2]).item():.2f}  "
                  f"frames={len(frames)}")

    end_time = time.time()

    # Save video
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if frames and imageio is not None:
        imageio.mimsave(output_path, frames, fps=config["render_fps"])
        print(f"[render_verified] Video saved: {output_path} ({len(frames)} frames)")
    else:
        print(f"[render_verified] WARNING: No frames to save (frames={len(frames)})")

    # Compute action statistics
    actions_arr = np.array(actions_log)
    positions_arr = np.array(positions_log)

    # Save match log
    match_log_path = output_path.replace(".mp4", ".match_log.json")
    match_log = {
        "duration_s": seconds,
        "num_steps": num_steps,
        "seed": seed,
        "model_path": os.path.abspath(model_path),
        "positions": positions_arr.tolist(),
    }
    with open(match_log_path, "w") as f:
        json.dump(match_log, f)

    # Create metadata
    metadata = {
        "model_path": os.path.abspath(model_path),
        "model_sha256": sha256_file(model_path) if os.path.exists(model_path) else "N/A",
        "env_name": "SoccerEnvHierarchical",
        "num_robots": 6,
        "seed": seed,
        "config_path": os.path.abspath(args.config) if 'args' in dir() else "N/A",
        "git_commit": get_git_commit(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
        "validation_status": "pending",
        "video_path": os.path.abspath(output_path),
        "video_frames": len(frames),
        "video_fps": config["render_fps"],
        "total_reward": total_reward,
        "nan_detected": nan_detected,
        "policy_output_stats": {
            "mean": float(actions_arr.mean()),
            "std": float(actions_arr.std()),
            "min": float(actions_arr.min()),
            "max": float(actions_arr.max()),
        },
        "robot_stats": {
            "initial_pos": positions_arr[0].tolist(),
            "final_pos": positions_arr[-1].tolist(),
            "total_displacement": float(np.linalg.norm(positions_arr[-1] - positions_arr[0])),
            "min_height": float(positions_arr[:, 2].min()),
            "max_height": float(positions_arr[:, 2].max()),
        },
        "match_log_path": os.path.abspath(match_log_path),
        "match_log_seed": seed,
        "match_log_model_sha256": sha256_file(model_path) if os.path.exists(model_path) else "N/A",
    }

    # Determine validation status
    issues = []
    if len(frames) < 50:
        issues.append("too_few_frames")
    if metadata["robot_stats"]["total_displacement"] < 0.1:
        issues.append("robot_did_not_move")
    if metadata["robot_stats"]["min_height"] < 0.1:
        issues.append("robot_fell")
    if nan_detected:
        issues.append("nan_in_actions")
    if metadata["policy_output_stats"]["std"] < 0.01:
        issues.append("constant_policy_output")

    metadata["validation_status"] = "passed" if not issues else f"failed: {','.join(issues)}"

    # Save metadata
    metadata_path = output_path.replace(".mp4", ".metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[render_verified] Metadata saved: {metadata_path}")
    print(f"[render_verified] Match log saved: {match_log_path}")
    print(f"[render_verified] Validation status: {metadata['validation_status']}")

    return metadata


def main():
    global args
    parser = argparse.ArgumentParser(description="Render verified 3v3 match video")
    parser.add_argument("--config", required=True, help="Path to inference_manifest.yaml")
    parser.add_argument("--mode", choices=["single", "multi", "short", "full"],
                        default="full", help="Validation mode")
    parser.add_argument("--seconds", type=int, default=None, help="Override video duration")
    parser.add_argument("--output", default="demos/verified_match.mp4")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model")
    args = parser.parse_args()

    config = load_config(args.config)

    # Print environment info
    print(f"[render_verified] CWD: {os.getcwd()}")
    print(f"[render_verified] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[render_verified] PYTHONPATH: {':'.join(sys.path[:5])}")
    print(f"[render_verified] Config: {os.path.abspath(args.config)}")
    print(f"[render_verified] Mode: {args.mode}")

    if args.mode == "single":
        result = run_mode_single(config, num_steps=500)
        if not result["robot_moved"]:
            print("[render_verified] ❌ FAIL: Robot did not move")
            sys.exit(1)
        print("[render_verified] ✅ Single robot validation passed")

    elif args.mode == "multi":
        result = run_mode_multi(config, num_steps=500)
        if not result["all_robots_moved"]:
            print("[render_verified] ❌ FAIL: Not all robots moved")
            sys.exit(1)
        print("[render_verified] ✅ Multi-robot validation passed")

    elif args.mode == "short":
        seconds = args.seconds or 10
        metadata = run_mode_render(config, seconds=seconds, output_path=args.output)
        if "failed" in metadata["validation_status"]:
            print(f"[render_verified] ❌ FAIL: {metadata['validation_status']}")
            sys.exit(1)
        print(f"[render_verified] ✅ Short video validation passed: {args.output}")

    elif args.mode == "full":
        seconds = args.seconds or int(config.get("match_duration_seconds", 25))
        metadata = run_mode_render(config, seconds=seconds, output_path=args.output)
        if "failed" in metadata["validation_status"]:
            print(f"[render_verified] ❌ FAIL: {metadata['validation_status']}")
            sys.exit(1)
        print(f"[render_verified] ✅ Full video validation passed: {args.output}")
        print(f"[render_verified] Video: {args.output}")
        print(f"[render_verified] Metadata: {args.output.replace('.mp4', '.metadata.json')}")

    print("\n[render_verified] DONE")


if __name__ == "__main__":
    main()
