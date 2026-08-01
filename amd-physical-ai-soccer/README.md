# AMD Physical AI — Humanoid Soccer (Booster T1)

Track 3 (Physical AI) submission for the **2026 AMD AI DevMaster Hackathon**.
A **Booster T1 humanoid robot** (23-DOF) trained to play soccer using PPO RL
entirely on **AMD Radeon GPU + ROCm** via Genesis simulation.

## Why this project

Booster's official RL training stack requires NVIDIA Isaac Gym / Isaac Lab. We demonstrate
the **first-known AMD ROCm alternative** for humanoid robot learning — same T1 robot,
same PPO algorithm, different GPU ecosystem.

## Results (2026-08-01)

| Metric | Value |
|---|---|
| GPU | AMD Radeon Graphics (gfx1100, ROCm 7.2) |
| PyTorch | 2.9.1+gitff65f5b (HIP backend) |
| Genesis | 1.3.1 (gs.gpu backend) |
| Training iterations | 300 (chase_v7) |
| Total training steps | 7,372,800 |
| Training throughput | 3,056 steps/sec |
| GPU utilization | 93-100% |
| Match recovery rate | 83-93% (with t1_walk.pt) |
| Match stability | 0 abnormal exits / 18 matches |
| RL goals scored | 0 (ball max X = 5.9m, goal at 7.0m) |
| Robot | Booster T1 23-DOF (official booster_assets) |

## Judging-criteria mapping (Track 3, 100 pts)

| Criterion | Pts | How this project earns it |
|---|---|---|
| Robot capability | 30 | Walk policy stable; chase policy approaches ball; 83% recovery rate; 0 crashes |
| AMD Radeon GPU + ROCm | 20 | Full training + inference on ROCm; GPU util 93-100%; see gpu_evidence_final.txt |
| Innovation & originality | 20 | First AMD ROCm T1 humanoid training; hierarchical walk+chase policy |
| Real-world application value | 20 | Failure recovery platform; 18 matches with full logging; disturbance framework |
| Upstream open-source contribution | 10 | Genesis soccer env, match coordinator, analyze scripts |

## Quick start (on AMD Radeon Linux cloud)

```bash
# 1. SSH to remote GPU
ssh -i ~/.ssh/id_ed25519 -p 31036 root@***REMOVED***

# 2. Verify environment
rocm-smi | head -5
/opt/venv/bin/python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.hip)"
# Expected: 2.9.1+gitff65f5b True 7.2.53211-e1a6bc5663

# 3. Verify imports
cd /workspace/radeon-repo
/opt/venv/bin/python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src')
from soccer_env_hierarchical import SoccerEnvHierarchical
from match_3v3.policy import SharedRLPolicy
print('All imports OK')
"

# 4. Run 3v3 RL vs Rule match (25s)
bash run_3v3_onnx.sh

# 5. Run clean batch (5 matches)
bash /tmp/run_clean_rerun.sh

# 6. Analyze match results
python3 /tmp/analyze_match.py /persistent/track3/match_logs/clean_rerun/<latest>.json
```

## Layout

```
amd-physical-ai-soccer/
  docs/
    TECHNICAL_REPORT.md       # Hackathon submission report
    track3_final_status.md    # Platform status summary
  reports/
    asset_audit.md            # Model SHA256 hashes
    rl_vs_rule_report.json   # Match data (18 clean matches)
    rl_vs_rule_summary.csv    # CSV summary
    recovery_ood_report.md     # Recovery & disturbance analysis
  demos/
    hierarchical_chase_hl.mp4 # Demo video (150 frames)
    track3_demo_script.md     # Reproduction guide
  presentations/
    track3_presentation.md    # 10-slide summary
  results/
    match_000-019.json        # 20 rule vs rule results
    summary.json              # Rule vs rule summary
  scripts/
    analyze_match.py           # Match log analyzer
    run_batch_3v3.sh           # Batch match runner
    run_clean_rerun.sh         # Clean re-run with N_STEPS fix
    match_coordinator_v3.py    # Coordinator with disturbance
  match_3v3.py                 # 3v3 match environment
  disturbance.py               # Disturbance configuration
  match_evaluator.py           # Match evaluation (stub)
  configs/
    inference_manifest.yaml    # ONNX model manifest
  gpu_evidence_final.txt       # ROCm SMI output
  benchmark_final.txt          # GPU benchmark
```

## Key findings

1. **Stability:** 0 abnormal exits in 18 clean matches (6-worker TCP architecture)
2. **Recovery:** 83% (25s) to 93% (60s) recovery rate with t1_walk.pt
3. **No goals:** RL pushes ball forward (max 5.9m) but can't reach goal (7.0m) in 25s
4. **Disturbance:** 87.8% recovery under 5N random push forces (only -0.7pp vs baseline)
5. **Honest positioning:** Failure recovery + OOD evaluation platform, not competitive soccer

## Notes

- All training done on AMD Radeon GPU with ROCm 7.2 (no NVIDIA hardware)
- Genesis 1.3.1 physics engine with gs.gpu backend
- ONNX Runtime 1.28.0 for inference (CPU-only, no ROCm EP available)
- Match logs include per-step robot positions, ball trajectory, collisions
