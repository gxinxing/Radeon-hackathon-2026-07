# AMD Radeon GPU + ROCm Adaptation Report

## Hardware
- GPU: AMD Radeon Graphics (gfx1100, RDNA3)
- VRAM: 48 GB
- Platform: Ubuntu 24.04, Radeon Cloud (anruicloud.com)

## Software Stack
- ROCm: 7.2.53211-e1a6bc5663
- PyTorch: 2.9.1+gitff65f5b (ROCm/HIP build)
- Genesis: 1.2.3 (gs.amdgpu backend)
- rsl-rl: 5.4.2 (PPO)

## GPU Performance Benchmark

### Matmul TFLOPS
| Matrix Size | TFLOPS |
|-------------|--------|
| 1024x1024   | 0.5    |
| 2048x2048   | 4.9    |
| 4096x4096   | 10.0   |
| 8192x8192   | 18.7   |

### Sustained Load (30s)
- Sustained TFLOPS: 17.9 (8192x8192 matmul)
- Peak VRAM: 1.11 GB (benchmark only)
- GPU Temperature: 44C (edge), 53C (junction)
- Clock: 2209 MHz

## RL Training Performance

### Training Configuration
- Algorithm: PPO (rsl-rl)
- Robot: Booster T1 23-DOF humanoid
- Tasks: balance, chase, shoot
- Parallel envs: 2048 + 2048 + 1024 = 5120 total
- GPU Utilization: 100pct (all three tasks concurrent)

### Training Metrics
| Task | Envs | Steps/sec | Reward (1000 iters) |
|------|------|-----------|---------------------|
| balance | 2048 | ~4000 | ~1090 |
| chase | 2048 | ~3500 | ~1080 |
| shoot | 1024 | ~5000 | ~1133 |

### VRAM Usage
- Three concurrent training tasks: 96pct VRAM utilized
- Single task (2048 envs): ~38pct VRAM
- Max parallel envs tested: 5120 (3 tasks)

## Key Achievements
1. First-known AMD ROCm training pipeline for Booster T1 humanoid robot
2. Full Genesis 1.2.3 + rsl-rl PPO on AMD GPU (no NVIDIA dependency)
3. Three concurrent training tasks maximizing GPU utilization
4. Sim-to-Sim deployment via booster_deploy (MuJoCo)
