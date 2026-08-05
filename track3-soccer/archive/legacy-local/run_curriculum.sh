#!/bin/bash
# 4-Phase Curriculum Training: P1→P2→P3→P4, each resumes from previous checkpoint
cd /workspace/radeon-repo
PYTHON=/opt/venv/bin/python

ln -sf /workspace/booster_assets/robots/T1/meshes /workspace/radeon-repo/urdf/t1/meshes 2>/dev/null
cp /workspace/radeon-repo/amd-physical-ai-soccer/assets/ball.urdf /workspace/assets/ball.urdf 2>/dev/null
sed -i 's|/workspace/booster/booster_deploy|/workspace/booster_deploy|g' /workspace/radeon-repo/configs/hierarchical_agent.yaml 2>/dev/null

LOG=/tmp/curriculum.log
echo "=== Curriculum Training Started: $(date) ===" | tee -a $LOG

# P1: 0-200, basic navigation, no opponent
echo "[P1] Basic navigation (0-200, no opponent)" | tee -a $LOG
$PYTHON -c "
import yaml; cfg=yaml.safe_load(open('configs/hierarchical_agent.yaml'))
cfg['env']['hl_clip_lin']=1.2; cfg['env']['hl_clip_ang']=1.2
cfg['task']='chase_hl'; cfg['num_envs']=2048
cfg['train']['run_name']='curriculum_p1'; cfg['train']['max_iterations']=200
cfg['train']['actor']['hidden_dims']=[256,128,64]; cfg['train']['critic']['hidden_dims']=[256,128,64]
yaml.dump(cfg, open('configs/curriculum_p1.yaml','w'))
print('P1 config created')
" | tee -a $LOG
$PYTHON -c "
import yaml, os, sys, pickle, shutil
sys.path.insert(0, '.')
import torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from soccer_env_curriculum import SoccerEnvCurriculum

with open('configs/curriculum_p1.yaml') as f: cfg = yaml.safe_load(f)
env_cfg = dict(cfg['env']); env_cfg['task']='chase_hl'
hl_cfg = cfg.get('high_level',{})
gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=42)
env = SoccerEnvCurriculum(num_envs=2048, env_cfg=env_cfg, obs_cfg=cfg['obs'],
    reward_cfg=cfg['reward'], command_cfg=cfg['command'],
    walk_model_path=hl_cfg.get('walk_model_path'),
    high_level_decimation=hl_cfg.get('decimation',5), show_viewer=False,
    phase=0, opponent_speed=0.0)
log_dir='runs/curriculum_p1'
os.makedirs(log_dir, exist_ok=True)
with open(f'{log_dir}/cfgs.pkl','wb') as f: pickle.dump([env_cfg,cfg['obs'],cfg['reward'],cfg['command'],cfg['train']], f)
runner = OnPolicyRunner(env, cfg['train'], log_dir, device=gs.device)
print('P1: obs_dim=24, action_dim=4, phase=0, opponent=OFF')
runner.learn(num_learning_iterations=200, init_at_random_ep_len=True)
print('P1 complete')
" 2>&1 | tee -a $LOG
echo "[P1] done: $(date)" | tee -a $LOG

# P2: 200-450, weak opponent, dribble+avoid
echo "[P2] Weak opponent + dribble (200-450, opp=0.1)" | tee -a $LOG
$PYTHON -c "
import yaml, os, sys, pickle
sys.path.insert(0, '.')
import torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from soccer_env_curriculum import SoccerEnvCurriculum

with open('configs/curriculum_p1.yaml') as f: cfg = yaml.safe_load(f)
cfg['train']['run_name']='curriculum_p2'; cfg['train']['max_iterations']=250
env_cfg = dict(cfg['env']); env_cfg['task']='chase_hl'
hl_cfg = cfg.get('high_level',{})
gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=42)
env = SoccerEnvCurriculum(num_envs=2048, env_cfg=env_cfg, obs_cfg=cfg['obs'],
    reward_cfg=cfg['reward'], command_cfg=cfg['command'],
    walk_model_path=hl_cfg.get('walk_model_path'),
    high_level_decimation=hl_cfg.get('decimation',5), show_viewer=False,
    phase=1, opponent_speed=0.1)
log_dir='runs/curriculum_p2'
os.makedirs(log_dir, exist_ok=True)
with open(f'{log_dir}/cfgs.pkl','wb') as f: pickle.dump([env_cfg,cfg['obs'],cfg['reward'],cfg['command'],cfg['train']], f)
runner = OnPolicyRunner(env, cfg['train'], log_dir, device=gs.device)
import glob; models=sorted(glob.glob('runs/curriculum_p1/model_*.pt'), key=os.path.getmtime)
if models: runner.load(models[-1]); print(f'P2: resumed from {models[-1]}')
print('P2: phase=1, opponent=0.1 m/s, dribble+avoid rewards')
runner.learn(num_learning_iterations=250, init_at_random_ep_len=True)
print('P2 complete')
" 2>&1 | tee -a $LOG
echo "[P2] done: $(date)" | tee -a $LOG

# P3: 450-700, kick timing
echo "[P3] Kick timing (450-700, opp=0.3)" | tee -a $LOG
$PYTHON -c "
import yaml, os, sys, pickle, glob
sys.path.insert(0, '.')
import torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from soccer_env_curriculum import SoccerEnvCurriculum

with open('configs/curriculum_p1.yaml') as f: cfg = yaml.safe_load(f)
cfg['train']['run_name']='curriculum_p3'; cfg['train']['max_iterations']=250
env_cfg = dict(cfg['env']); env_cfg['task']='chase_hl'
hl_cfg = cfg.get('high_level',{})
gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=42)
env = SoccerEnvCurriculum(num_envs=2048, env_cfg=env_cfg, obs_cfg=cfg['obs'],
    reward_cfg=cfg['reward'], command_cfg=cfg['command'],
    walk_model_path=hl_cfg.get('walk_model_path'),
    high_level_decimation=hl_cfg.get('decimation',5), show_viewer=False,
    phase=2, opponent_speed=0.3)
log_dir='runs/curriculum_p3'
os.makedirs(log_dir, exist_ok=True)
with open(f'{log_dir}/cfgs.pkl','wb') as f: pickle.dump([env_cfg,cfg['obs'],cfg['reward'],cfg['command'],cfg['train']], f)
runner = OnPolicyRunner(env, cfg['train'], log_dir, device=gs.device)
models=sorted(glob.glob('runs/curriculum_p2/model_*.pt'), key=os.path.getmtime)
if models: runner.load(models[-1]); print(f'P3: resumed from {models[-1]}')
print('P3: phase=2, opponent=0.3 m/s, kick action active')
runner.learn(num_learning_iterations=250, init_at_random_ep_len=True)
print('P3 complete')
" 2>&1 | tee -a $LOG
echo "[P3] done: $(date)" | tee -a $LOG

# P4: 700-1000, full confrontation
echo "[P4] Full confrontation (700-1000, opp=0.5)" | tee -a $LOG
$PYTHON -c "
import yaml, os, sys, pickle, glob
sys.path.insert(0, '.')
import torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from soccer_env_curriculum import SoccerEnvCurriculum

with open('configs/curriculum_p1.yaml') as f: cfg = yaml.safe_load(f)
cfg['train']['run_name']='curriculum_p4'; cfg['train']['max_iterations']=300
env_cfg = dict(cfg['env']); env_cfg['task']='chase_hl'
hl_cfg = cfg.get('high_level',{})
gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=42)
env = SoccerEnvCurriculum(num_envs=2048, env_cfg=env_cfg, obs_cfg=cfg['obs'],
    reward_cfg=cfg['reward'], command_cfg=cfg['command'],
    walk_model_path=hl_cfg.get('walk_model_path'),
    high_level_decimation=hl_cfg.get('decimation',5), show_viewer=False,
    phase=3, opponent_speed=0.5)
log_dir='runs/curriculum_p4'
os.makedirs(log_dir, exist_ok=True)
with open(f'{log_dir}/cfgs.pkl','wb') as f: pickle.dump([env_cfg,cfg['obs'],cfg['reward'],cfg['command'],cfg['train']], f)
runner = OnPolicyRunner(env, cfg['train'], log_dir, device=gs.device)
models=sorted(glob.glob('runs/curriculum_p3/model_*.pt'), key=os.path.getmtime)
if models: runner.load(models[-1]); print(f'P4: resumed from {models[-1]}')
print('P4: phase=3, opponent=0.5 m/s, all skills active')
runner.learn(num_learning_iterations=300, init_at_random_ep_len=True)
print('P4 complete')
" 2>&1 | tee -a $LOG
echo "[P4] done: $(date)" | tee -a $LOG

echo "=== Curriculum Complete: $(date) ===" | tee -a $LOG
