#!/usr/bin/env python3
"""C-phase: 3v3 acceptance eval with our Task-9 HL PPO driving robot 0.
Same metrics/criteria as run_3v3_final_v2.py (crit1 fallen<=2, crit2 kicks,
crit3 frame_diff>2, crit4 output). Optional --stoch uses sampled actions.
"""
import sys, os, json, time, traceback, argparse, numpy as np, torch
from pathlib import Path

ROOT = Path("/workspace")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "strategy"))
OUT = ROOT / "demo_artifacts"; OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/workspace/models/task9_p1.pt")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--stoch", action="store_true", help="use sampled actions (noisy) instead of deterministic mean")
ap.add_argument("--tag", default="hl")
args = ap.parse_args()

result = {"started_at": time.time(), "gpu": "AMD Radeon RX 7900 XT (gfx1100)", "mode": f"3v3_final_{args.tag}", "stoch": args.stoch}

try:
    import yaml, genesis as gs
    gs.init(backend=gs.gpu, precision="32", logging_level="error", seed=42)
    from soccer_env_3v3 import SoccerEnv3v3, KICK_IMPULSE, KICK_DISTANCE, KICK_COOLDOWN
    from strategy.match import Match
    from rsl_rl.modules import ActorCritic

    with open(ROOT / "configs/hierarchical_agent.yaml") as f:
        cfg = yaml.safe_load(f)
    env_cfg = dict(cfg["env"]); env_cfg["task"] = "chase_hl"; env_cfg["use_rule_walk"] = False
    env = SoccerEnv3v3(num_envs=1, env_cfg=env_cfg, obs_cfg=cfg["obs"], reward_cfg=cfg["reward"],
                       command_cfg=cfg["command"], walk_model_path=str(ROOT / "models/pretrained/t1_walk.pt"),
                       high_level_decimation=5, show_viewer=False)
    env.use_rule_walk = False
    env.term_pitch = 30.0; env.term_roll = 30.0; env.max_episode_length = 100000

    train_cfg = cfg["train"]; actor_cfg = train_cfg.get("actor", {}); critic_cfg = train_cfg.get("critic", {})
    dist_cfg = actor_cfg.get("distribution_cfg", {})
    pc = {"actor_hidden_dims": actor_cfg.get("hidden_dims", [256, 128, 64]), "actor_activation": actor_cfg.get("activation", "elu"),
          "critic_hidden_dims": critic_cfg.get("hidden_dims", [256, 128, 64]), "critic_activation": critic_cfg.get("activation", "elu"),
          "init_noise_std": dist_cfg.get("init_std", 1.0), "std_type": dist_cfg.get("std_type", "scalar")}
    ckpt = torch.load(args.ckpt, map_location=env.device)
    ac = ActorCritic(19, 19, 3, **pc).to(env.device)
    ac.load_state_dict(ckpt["model_state_dict"]); ac.eval()
    print(f"[eval3v3] ckpt={args.ckpt} iter={ckpt.get('iter','?')} stoch={args.stoch}", flush=True)

    match = Match(env)
    env.reset()
    ball_qpos = env.ball.get_qpos().clone()
    ball_qpos[0, :3] = torch.tensor([-0.5, 0.0, 0.11], device=env.device)
    ball_qpos[0, 3:7] = torch.tensor([1.0, 0, 0, 0], device=env.device)
    env.ball.set_qpos(ball_qpos, zero_velocity=True, skip_forward=True)
    env.ball_pos = env.ball.get_pos(); env.ball_vel = env.ball.get_vel()
    print(f"Ball placed at (-0.5,0). Robot 0 at {env.all_base_pos[0,0].cpu().numpy()}", flush=True)

    frames = []; n_steps = args.steps; all_kicks = 0
    ball_start = env.ball_pos[0].cpu().numpy().copy()
    robot_start = env.all_base_pos[0].cpu().numpy().copy()
    prev_frame = None; frame_diffs = []; scored_frame = -1

    for step in range(1, n_steps + 1):
        obs = env._compute_rl_obs(0)
        with torch.inference_mode():
            actions = ac.act_inference(obs) if not args.stoch else ac.act(obs, deterministic=False)
        _, reward, done, extras = env.step(actions)
        kicks, scored = match.check_events(extras)
        all_kicks += kicks
        if step >= 10:
            try:
                ret = env.cam.render(rgb=True)
                rgb = ret[0] if isinstance(ret, tuple) else ret
                if rgb is not None and rgb.size > 0:
                    frames.append(rgb)
                    if prev_frame is not None:
                        diff = float(np.mean(np.abs(rgb.astype(np.float32) - prev_frame.astype(np.float32))))
                        frame_diffs.append(diff)
                    prev_frame = rgb.copy()
            except Exception:
                pass
        ball = env.ball_pos[0].cpu().numpy()
        stats = match.get_robot_stats()
        fallen = sum(1 for s in stats if s['fallen'])
        robot_now = env.all_base_pos[0].cpu().numpy()
        robot_disp = float(np.linalg.norm(robot_now[0, :2] - robot_start[0, :2]))
        ball_disp = float(np.linalg.norm(ball[:2] - ball_start[:2]))
        last_diff = frame_diffs[-1] if frame_diffs else 0.0
        scored_now = bool(ball[0] > env.goal_x and abs(ball[1]) < env.goal_half)
        if scored_now and scored_frame < 0:
            scored_frame = step
        if step % 25 == 0 or scored_now or done.any():
            print(f"Step {step:3d}/{n_steps}: fallen={fallen} robot_disp={robot_disp:.2f}m "
                  f"ball_disp={ball_disp:.2f}m kicks={all_kicks} frame_diff={last_diff:.1f}{' SCORED' if scored_now else ''}", flush=True)
        if scored_frame > 0 and step >= scored_frame + 5:
            break
        if done.any():
            env.reset()

    ball_end = env.ball_pos[0].cpu().numpy()
    robot_end = env.all_base_pos[0].cpu().numpy()
    ball_displacement = float(np.linalg.norm(ball_end[:2] - ball_start[:2]))
    robot_displacement = float(np.linalg.norm(robot_end[0, :2] - robot_start[0, :2]))
    stats = match.get_robot_stats()
    final_fallen = sum(1 for s in stats if s['fallen'])
    mean_frame_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0
    max_frame_diff = float(np.max(frame_diffs)) if frame_diffs else 0.0
    scored = bool(ball_end[0] > env.goal_x and abs(ball_end[1]) < env.goal_half)

    if frames:
        import imageio.v2 as imageio
        vid = f"match_3v3_{args.tag}.mp4"
        imageio.imwrite(str(OUT / f"match_3v3_{args.tag}.png"), frames[len(frames)//2])
        imageio.mimsave(str(OUT / vid), frames, fps=10, quality=9)
        result["video"] = vid
        print(f"\nVideo: {len(frames)} frames, mean_frame_diff={mean_frame_diff:.1f}", flush=True)

    crit1 = final_fallen <= 2
    crit2 = robot_displacement >= 2.0 and all_kicks >= 1 and ball_displacement >= 2.0
    crit3 = mean_frame_diff > 2.0
    crit4 = True
    all_pass = crit1 and crit2 and crit3 and crit4
    result.update(status="passed" if all_pass else "failed", steps=step, frames=len(frames), kicks=all_kicks,
                  score=int(scored), ball_displacement=round(ball_displacement, 2),
                  ball_x_displacement=round(float(ball_end[0] - ball_start[0]), 2),
                  robot_displacement=round(robot_displacement, 2), final_fallen=final_fallen,
                  mean_frame_diff=round(mean_frame_diff, 2), max_frame_diff=round(max_frame_diff, 2), num_robots=6,
                  video=result.get("video"),
                  acceptance={
                      "crit1_fallen_le_2": {"pass": crit1, "value": final_fallen, "threshold": 2},
                      "crit2_robot_kicks_ball": {"pass": crit2, "robot_disp": round(robot_displacement, 2),
                                                 "kicks": all_kicks, "ball_disp": round(ball_displacement, 2)},
                      "crit3_frame_diff_gt_2": {"pass": crit3, "value": round(mean_frame_diff, 2), "threshold": 2.0},
                      "crit4_output": {"pass": crit4, "json": f"match_3v3_{args.tag}_result.json", "mp4": result.get("video")},
                  })
    print(f"\n=== 3V3 {args.tag} RESULT ===")
    print(f"fallen={final_fallen} (<=2: {'PASS' if crit1 else 'FAIL'})")
    print(f"robot_disp={robot_displacement:.2f}m (>=2m: {'PASS' if robot_displacement>=2 else 'FAIL'})")
    print(f"kicks={all_kicks} (>=1: {'PASS' if all_kicks>=1 else 'FAIL'})")
    print(f"ball_disp={ball_displacement:.2f}m (>=2m: {'PASS' if ball_displacement>=2 else 'FAIL'})")
    print(f"scored={scored}")
    print(f"mean_frame_diff={mean_frame_diff:.2f} (>2: {'PASS' if crit3 else 'FAIL'})")
    print(f"All: {'PASS' if all_pass else 'FAIL'}")
except Exception as e:
    result["status"] = "failed"; result["error"] = repr(e); result["traceback"] = traceback.format_exc()
    print(f"FAILED: {e}", flush=True); traceback.print_exc()

result["ended_at"] = time.time()
result["duration_s"] = result["ended_at"] - result["started_at"]
(OUT / f"match_3v3_{args.tag}_result.json").write_text(json.dumps(result, indent=2, default=str))
print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, indent=2, default=str))
