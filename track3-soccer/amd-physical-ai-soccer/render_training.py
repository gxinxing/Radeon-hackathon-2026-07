"""Render a demo video from trained policy."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import torch
import imageio
import genesis as gs
from genesis.utils.misc import tensor_to_array
from rsl_rl.runners import OnPolicyRunner
from envs.soccer_env import SoccerEnv

with open("configs/soccer_agent.yaml") as f:
    cfg = yaml.safe_load(f)

env_cfg = dict(cfg["env"])
env_cfg["task"] = "balance"
obs_cfg = cfg["obs"]
reward_cfg = cfg["reward"]
command_cfg = cfg["command"]
train_cfg = cfg["train"]

gs.init(backend=gs.gpu, logging_level="warning")

env = SoccerEnv(
    num_envs=1,
    env_cfg=env_cfg,
    obs_cfg=obs_cfg,
    reward_cfg=reward_cfg,
    command_cfg=command_cfg,
    show_viewer=False,
)

# 拿到 env 内部已添加的相机
cam = env.scene._visualizer._cameras[-1] if hasattr(env.scene._visualizer, "_cameras") else None
if cam is None:
    # fallback: 直接找 visualizer 的 cameras 列表
    cam = env.scene.visualizer.cameras[0]
print(f"Camera: {cam}")

env.scene.reset()

log_dir = "runs/booster_soccer_balance"
model_files = sorted(glob.glob(f"{log_dir}/model_*.pt"), key=os.path.getmtime)
print(f"Loading: {model_files[-1]}")

runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
runner.load(model_files[-1])
policy = runner.get_inference_policy(device=gs.device)

obs = env.reset()
frames = []
total_reward = 0
n_steps = 300

print(f"Rendering {n_steps} frames...")
for i in range(n_steps):
    with torch.no_grad():
        actions = policy(obs)
    obs, rew, done, info = env.step(actions)
    total_reward += rew.mean().item()

    if i % 2 == 0:
        rgb, _, _, _ = cam.render(rgb=True)
        arr = tensor_to_array(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        frames.append(arr.astype(np.uint8))

    if (i + 1) % 50 == 0:
        print(f"  step {i+1}/{n_steps}  rew={rew.mean().item():.3f}  frames={len(frames)}")

os.makedirs("demo", exist_ok=True)
video_path = "demo/training_demo.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"\nVideo saved: {video_path}")
print(f"Total reward: {total_reward:.2f}")
print(f"Frames: {len(frames)}")
print("DONE")
