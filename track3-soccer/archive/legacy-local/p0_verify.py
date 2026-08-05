"""P0 verify: Strategy A rule-walk fallback (bypasses t1_walk.pt).

Runs SoccerEnvHierarchical's production deterministic gait:
  - commands == 0  -> static standing stance (all-zero action = default pose, rock steady)
  - commands != 0  -> phase-driven leg swing scaled by forward/side speed

Outputs text only (no screenshots). Reads:
  P0_RESULT {...}   VERDICT PASS|TUNE
"""
import os, sys, json
W = '/workspace/amd-physical-ai-soccer'
os.chdir(W)
sys.path.insert(0, W)

import genesis as gs
gs.init(backend=gs.gpu, logging_level='error')
import yaml, torch
from soccer_env_hierarchical import SoccerEnvHierarchical


cfg = yaml.safe_load(open('configs/hierarchical_agent.yaml'))
env_cfg = dict(cfg['env'])
env_cfg['task'] = 'chase_hl'
env_cfg['use_rule_walk'] = True
hl = cfg.get('high_level', {})
walk_path = hl.get('walk_model_path')
env = SoccerEnvHierarchical(
    num_envs=1, env_cfg=env_cfg, obs_cfg=cfg['obs'],
    reward_cfg=cfg['reward'], command_cfg=cfg['command'],
    walk_model_path=walk_path, high_level_decimation=hl.get('decimation', 5),
    show_viewer=False,
)
env._rule_stride = 1.1
env._rule_step_amp = 0.16
env._rule_lift_amp = 0.22
env._rule_ankle_amp = 0.10
print("[p0] env built, testing rule-walk...", flush=True)

obs = env.reset()


def run(cmd, nsteps, label):
    hs = []
    falls = 0
    x0 = None
    for i in range(nsteps):
        a = torch.zeros((1, 3), device=env.device)
        a[0, 0] = cmd
        obs, rew, done, ext = env.step(a)
        h = env.base_pos[0, 2].item()
        p = abs(env.base_euler[0, 1].item())
        r = abs(env.base_euler[0, 0].item())
        if x0 is None:
            x0 = env.base_pos[0, 0].item()
        hs.append(h)
        if bool(done[0]) or h < 0.4 or p > 30.0 or r > 30.0:
            falls += 1
            break
    x1 = env.base_pos[0, 0].item()
    disp = (x1 - x0) if x0 is not None else 0.0
    mean = sum(hs) / len(hs) if hs else 0
    sd = (sum((x - mean) ** 2 for x in hs) / max(1, len(hs) - 1)) ** 0.5 if len(hs) > 1 else 0
    return dict(label=label, steps=len(hs), mean_h=round(mean, 4), std_h=round(sd, 4),
                min_h=round(min(hs), 4) if hs else 0, falls=falls, disp_x=round(disp, 3))


res = {}
res['stance'] = run(0.0, 60, 'stance_6s')
print("[p0] stance done: " + json.dumps(res['stance']), flush=True)
res['gait_fwd'] = run(0.4, 150, 'gait_fwd_15s')
print("[p0] gait done: " + json.dumps(res['gait_fwd']), flush=True)

print("P0_RESULT " + json.dumps(res))
verdict = 'PASS' if (res['stance']['falls'] == 0 and res['gait_fwd']['falls'] == 0
                     and res['gait_fwd']['disp_x'] > 0.3) else 'TUNE'
print("VERDICT " + verdict)
