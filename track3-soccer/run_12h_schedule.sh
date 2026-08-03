#!/bin/bash
cd /workspace/radeon-repo
PYTHON=/opt/venv/bin/python
LOG=/tmp/schedule_12h.log

echo "========================================" | tee -a $LOG
echo "  12-Hour Training Schedule Started" | tee -a $LOG
echo "  $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG

ln -sf /workspace/booster_assets/robots/T1/meshes /workspace/radeon-repo/urdf/t1/meshes 2>/dev/null
cp /workspace/radeon-repo/amd-physical-ai-soccer/assets/ball.urdf /workspace/assets/ball.urdf 2>/dev/null
sed -i 's|/workspace/booster/booster_deploy|/workspace/booster_deploy|g' /workspace/radeon-repo/configs/hierarchical_agent.yaml 2>/dev/null

echo "[Schedule] Task 1: 1v1 opponent=0.4 (waiting for current run)" | tee -a $LOG
while pgrep -f 'train_1v1' > /dev/null 2>&1; do sleep 60; done
echo "[Schedule] Task 1 done: $(date)" | tee -a $LOG

echo "[Schedule] Task 2: 1v1 opponent=0.1 (weak)" | tee -a $LOG
$PYTHON train_1v1.py --max_iterations 500 --num_envs 2048 --opponent_speed 0.1 --exp_name soccer_1v1_opp01 > /workspace/radeon-repo/train_1v1_opp01.log 2>&1
echo "[Schedule] Task 2 done: $(date)" | tee -a $LOG

echo "[Schedule] Task 3: 1v1 opponent=0.8 (strong)" | tee -a $LOG
$PYTHON train_1v1.py --max_iterations 500 --num_envs 2048 --opponent_speed 0.8 --exp_name soccer_1v1_opp08 > /workspace/radeon-repo/train_1v1_opp08.log 2>&1
echo "[Schedule] Task 3 done: $(date)" | tee -a $LOG

echo "[Schedule] Task 4: 24d coop_hl (real multi-agent)" | tee -a $LOG
$PYTHON -c "import yaml; cfg=yaml.safe_load(open('configs/hierarchical_agent.yaml')); cfg['env']['multiagent_obs']=True; cfg['task']='coop_hl'; cfg['train']['max_iterations']=500; cfg['train']['run_name']='coop_hl_24d_v2'; cfg['num_envs']=2048; yaml.dump(cfg, open('configs/coop_hl_24d_v2.yaml','w')); print('Config created')" 2>&1 | tee -a $LOG
$PYTHON train_hierarchical.py --config configs/coop_hl_24d_v2.yaml --task coop_hl --num_envs 2048 --max_iterations 500 > /workspace/radeon-repo/train_coop_hl_24d_v2.log 2>&1
echo "[Schedule] Task 4 done: $(date)" | tee -a $LOG

echo "[Schedule] Task 5: Ablation no approach_angle" | tee -a $LOG
$PYTHON -c "import yaml; cfg=yaml.safe_load(open('configs/hierarchical_agent.yaml')); cfg['reward']['approach_angle']=0.0; cfg['train']['run_name']='ablation_no_angle'; cfg['train']['max_iterations']=500; yaml.dump(cfg, open('configs/ablation_no_angle.yaml','w')); print('Config created')" 2>&1 | tee -a $LOG
$PYTHON train_hierarchical.py --config configs/ablation_no_angle.yaml --max_iterations 500 --num_envs 2048 > /workspace/radeon-repo/train_ablation_no_angle.log 2>&1
echo "[Schedule] Task 5 done: $(date)" | tee -a $LOG

echo "[Schedule] Task 6: Ablation no directed_contact" | tee -a $LOG
$PYTHON -c "import yaml; cfg=yaml.safe_load(open('configs/hierarchical_agent.yaml')); cfg['reward']['directed_contact']=0.0; cfg['train']['run_name']='ablation_no_contact'; cfg['train']['max_iterations']=500; yaml.dump(cfg, open('configs/ablation_no_contact.yaml','w')); print('Config created')" 2>&1 | tee -a $LOG
$PYTHON train_hierarchical.py --config configs/ablation_no_contact.yaml --max_iterations 500 --num_envs 2048 > /workspace/radeon-repo/train_ablation_no_contact.log 2>&1
echo "[Schedule] Task 6 done: $(date)" | tee -a $LOG

echo "[Schedule] Task 7: Ablation old reward (hard clamp)" | tee -a $LOG
$PYTHON -c "import yaml; cfg=yaml.safe_load(open('configs/hierarchical_agent.yaml')); cfg['reward']['ball_to_goal']=3.0; cfg['reward']['approach_angle']=0.0; cfg['reward']['directed_contact']=0.0; cfg['env']['hl_clip_lin']=0.8; cfg['env']['hl_clip_ang']=1.0; cfg['train']['run_name']='ablation_old_reward'; cfg['train']['max_iterations']=500; yaml.dump(cfg, open('configs/ablation_old_reward.yaml','w')); print('Config created')" 2>&1 | tee -a $LOG
$PYTHON train_hierarchical.py --config configs/ablation_old_reward.yaml --max_iterations 500 --num_envs 2048 > /workspace/radeon-repo/train_ablation_old_reward.log 2>&1
echo "[Schedule] Task 7 done: $(date)" | tee -a $LOG

echo "[Schedule] Task 8: Export 1v1 ONNX + render video" | tee -a $LOG
$PYTHON -c "
import torch, torch.nn as nn, os
for run_dir in ['runs/soccer_1v1_opp04', 'runs/soccer_1v1', 'runs/soccer_1v1_opp01', 'runs/soccer_1v1_opp08']:
    if not os.path.exists(run_dir): continue
    models = sorted([f for f in os.listdir(run_dir) if f.startswith('model_') and f.endswith('.pt')], key=lambda x: int(x.split('_')[1].split('.')[0]))
    if not models: continue
    last = os.path.join(run_dir, models[-1])
    ckpt = torch.load(last, map_location='cpu', weights_only=False)
    sd = ckpt['actor_state_dict']
    in_dim = sd['mlp.0.weight'].shape[1]
    mlp = nn.Sequential(nn.Linear(in_dim,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,3))
    mapping = {'0.weight':'mlp.0.weight','0.bias':'mlp.0.bias','2.weight':'mlp.2.weight','2.bias':'mlp.2.bias','4.weight':'mlp.4.weight','4.bias':'mlp.4.bias','6.weight':'mlp.6.weight','6.bias':'mlp.6.bias'}
    mlp.load_state_dict({k:sd[v] for k,v in mapping.items()})
    mlp.eval()
    with torch.no_grad():
        torch.onnx.export(mlp, torch.randn(1,in_dim), 'models/1v1_policy.onnx', input_names=['obs'], output_names=['action'], opset_version=17, dynamo=False)
    print(f'ONNX exported from {last} (dim={in_dim}): {os.path.getsize(\"models/1v1_policy.onnx\")} bytes')
    break
" 2>&1 | tee -a $LOG
$PYTHON render_1v1_match.py --onnx models/1v1_policy.onnx --steps 300 --output demos/1v1_match_final.mp4 2>&1 | tee -a $LOG
echo "[Schedule] Task 8 done: $(date)" | tee -a $LOG

echo "========================================" | tee -a $LOG
echo "  12-Hour Schedule Complete: $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG
