"""Train with pre-trained t1_walk.pt initialization — exact weight transfer.

Loads ALL layers from the TorchScript model, including obs_normalizer.
"""
import argparse, os, sys, pickle, shutil, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from envs.soccer_env import SoccerEnv


def load_pretrained_weights(runner, pretrained_path):
    """Load t1_walk.pt (TorchScript) weights into PPO actor + obs normalizer."""
    model = torch.jit.load(pretrained_path, map_location=gs.device)
    actor = model.actor
    obs_normalizer = model.obs_normalizer

    algo = runner.alg

    # === Copy actor weights (all layers must match exactly) ===
    actor_params = list(actor.parameters())
    target_params = list(algo.actor.parameters())

    print(f"Pretrained actor: {len(actor_params)} params")
    print(f"Target actor: {len(target_params)} params")

    copied = 0
    skipped = 0
    for i, (src, dst) in enumerate(zip(actor_params, target_params)):
        if src.shape == dst.shape:
            dst.data.copy_(src.data)
            copied += 1
        else:
            skipped += 1
            print(f"  SKIP layer {i}: src={src.shape} dst={dst.shape}")

    print(f"  Copied: {copied}/{len(actor_params)} layers")
    if skipped > 0:
        print(f"  WARNING: {skipped} layers skipped (shape mismatch)")

    # === Copy obs normalizer (running mean/var) ===
    try:
        mean = obs_normalizer._mean
        var = obs_normalizer._var
        std = obs_normalizer._std
        count = obs_normalizer.count

        # rsl_rl stores normalizer in actor.obs_normalizer
        if hasattr(algo.actor, 'obs_normalizer'):
            norm = algo.actor.obs_normalizer
            if hasattr(norm, '_mean') and mean.shape == norm._mean.shape:
                norm._mean.data.copy_(mean.data)
                norm._var.data.copy_(var.data)
                norm._std.data.copy_(std.data)
                if hasattr(norm, 'count'):
                    norm.count.data.copy_(count.data)
                print(f"  Obs normalizer copied: mean={mean.shape}")
            else:
                print(f"  Obs normalizer shape mismatch: src={mean.shape}")
        else:
            print("  No obs_normalizer found in target actor")
    except Exception as e:
        print(f"  Obs normalizer not accessible: {e}")

    print("Pre-trained weights loaded!\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("-e", "--exp_name", type=str, default=None)
    parser.add_argument("-B", "--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained", type=str,
                        default="/workspace/booster/booster_deploy/tasks/locomotion/models/t1_walk.pt")
    args = parser.parse_args()

    with open("configs/soccer_agent.yaml") as f:
        cfg = yaml.safe_load(f)

    task = args.task or cfg.get("task", "chase")
    exp_name = args.exp_name or f"{cfg.get('exp_name', 'booster_soccer')}_{task}"
    num_envs = args.num_envs or cfg.get("num_envs", 2048)
    seed = args.seed or cfg.get("seed", 42)

    env_cfg = dict(cfg["env"])
    env_cfg["task"] = task
    obs_cfg = cfg["obs"]
    reward_cfg = cfg["reward"]
    command_cfg = cfg["command"]
    train_cfg = cfg["train"]
    train_cfg["run_name"] = exp_name
    if args.max_iterations:
        train_cfg["max_iterations"] = args.max_iterations

    log_dir = f"runs/{exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=seed)

    env = SoccerEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                    reward_cfg=reward_cfg, command_cfg=command_cfg,
                    show_viewer=cfg.get("show_viewer", False))

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Load pre-trained walking weights
    if os.path.exists(args.pretrained):
        print(f"\n=== Loading pre-trained weights from {args.pretrained} ===")
        load_pretrained_weights(runner, args.pretrained)
    else:
        print(f"WARNING: Pre-trained model not found at {args.pretrained}")

    runner.learn(num_learning_iterations=train_cfg["max_iterations"], init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
