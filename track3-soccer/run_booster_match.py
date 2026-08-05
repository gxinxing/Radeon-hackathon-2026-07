#!/usr/bin/env python3
"""用 Booster 风格策略跑 3v3 比赛 + 录视频。

不依赖 walk model 的 obs 反馈链（绕过 all_dof_pos 问题）。
直接用策略层计算速度指令，env.step() 转给 walk model。
"""
import sys, os, json, time, traceback, numpy as np, torch
from pathlib import Path

ROOT = Path("/workspace")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "strategy"))

OUT = ROOT / "demo_artifacts"
OUT.mkdir(exist_ok=True)

result = {"started_at": time.time(), "gpu": "AMD Radeon RX 7900 XT (gfx1100)"}

try:
    import yaml
    import genesis as gs
    gs.init(backend=gs.gpu, precision="32", logging_level="info", seed=42)

    from soccer_env_3v3 import SoccerEnv3v3
    from strategy.strategy import Match

    with open(ROOT / "configs/hierarchical_agent.yaml") as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = "chase_hl"

    env = SoccerEnv3v3(
        num_envs=1, env_cfg=env_cfg, obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"], command_cfg=cfg["command"],
        walk_model_path=str(ROOT / "models/pretrained/t1_walk.pt"),
        high_level_decimation=5,
        show_viewer=False,
    )
    print(f"Env created. Robots: {env.num_robots}", flush=True)

    # 用 Booster 风格的 Match 控制器
    match = Match(env)
    print("Match controller created (Booster-style)", flush=True)

    env.reset()
    frames = []
    n_steps = 30
    all_kicks = 0
    ball_start = env.ball_pos[0].cpu().numpy().copy()

    for step in range(1, n_steps + 1):
        # 策略层决策：计算 6 个机器人的速度指令
        action_tensor = match.act()

        # 执行物理仿真
        _, reward, done, extras = env.step(action_tensor)

        # 检查事件
        kicks, scored = match.check_events(extras)
        all_kicks += kicks

        # 录像
        try:
            ret = env.cam.render(rgb=True)
            rgb = ret[0] if isinstance(ret, tuple) else ret
            if rgb is not None and rgb.size > 0:
                frames.append(rgb)
        except:
            pass

        if step % 10 == 0:
            ball = env.ball_pos[0].cpu().numpy()
            stats = match.get_robot_stats()
            fallen = sum(1 for s in stats if s['fallen'])
            print(f"Step {step}/{n_steps}: ball=({ball[0]:.1f},{ball[1]:.1f}) "
                  f"frames={len(frames)} kicks={all_kicks} fallen={fallen} "
                  f"score={match.score}", flush=True)

        if done.any():
            env.reset()

    # 保存
    ball_end = env.ball_pos[0].cpu().numpy()
    ball_displacement = float(np.linalg.norm(ball_end[:2] - ball_start[:2]))

    if frames:
        import imageio.v2 as imageio
        imageio.imwrite(str(OUT / "match_booster.png"), frames[len(frames)//2])
        imageio.mimsave(str(OUT / "match_booster.mp4"), frames, fps=10, quality=9)
        result["video"] = "match_booster.mp4"
        print(f"Video: {len(frames)} frames", flush=True)

    result.update(
        status="passed",
        steps=n_steps,
        frames=len(frames),
        kicks=all_kicks,
        score=match.score,
        ball_displacement=round(ball_displacement, 2),
        num_robots=6,
    )
    print(f"PASSED: {n_steps} steps, {len(frames)} frames, {all_kicks} kicks, "
          f"score={match.score}, ball={ball_displacement:.2f}m", flush=True)

except Exception as e:
    result["status"] = "failed"
    result["error"] = repr(e)
    result["traceback"] = traceback.format_exc()
    print(f"FAILED: {e}", flush=True)
    traceback.print_exc()

result["ended_at"] = time.time()
result["duration_s"] = result["ended_at"] - result["started_at"]
(OUT / "match_booster_result.json").write_text(json.dumps(result, indent=2, default=str))
print(json.dumps(result, indent=2, default=str))
