# ⚽ 基于 AMD Radeon GPU 的人形机器人足球策略训练

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%207.2-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1+ROCm-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Genesis](https://img.shields.io/badge/Genesis-1.2.3-blue)](https://genesis-embodied-ai.github.io/)
[![rsl_rl](https://img.shields.io/badge/rsl__rl-5.4.2-green)](https://github.com/leggedrobotics/rsl_rl)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2017-orange)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-Hackathon-lightgrey)](#license)

[English](./README.md) | [中文](./README_zh.md)

> 使用 Genesis 物理引擎和 ROCm PyTorch 在 AMD Radeon GPU 上训练人形机器人足球策略
>（平衡、追球、射门）—— 这是**首个基于 AMD GPU 的人形足球训练 pipeline**，
> 证明无需 NVIDIA 硬件即可训练出有竞争力的机器人策略。

**AMD AI DevMaster 黑客马拉松 2026 — Track 3: Physical AI**

---

## 📋 目录

- [核心亮点](#-核心亮点)
- [项目背景](#-项目背景)
- [架构设计](#-架构设计)
- [关键结果](#-关键结果)
- [环境要求](#-环境要求)
- [安装步骤](#-安装步骤)
- [使用方法](#-使用方法)
- [奖励函数设计](#-奖励函数设计)
- [分布式多机器人对抗](#-分布式多机器人对抗-1v1--3v3)
- [项目结构](#-项目结构)
- [技术栈](#-技术栈)
- [已知限制](#-已知限制)
- [数据来源](#-数据来源)
- [团队信息](#-团队信息)
- [许可证](#-license)

---

## 🌟 核心亮点

| 指标 | 数值 | 说明 |
|------|------|------|
| 奖励提升 | **-22 → +24** | P0/P1/P2 调参后（500 轮迭代） |
| 动作标准差（修复后） | **5.78 → 0.07** | entropy_coef 0.01→0.003 解决噪声爆炸 |
| 回合长度 | **18 → 225 步** | 从立即摔倒到持续行走 |
| 最小球距 | **4.29m → 0.25m** | 机器人主动接近球 |
| ONNX 推理速度 | **0.4 ms** | 19→3 维，实时可用（4000 采样） |
| 3v3 对抗 | **6 机器人，0 崩溃** | 单 AMD GPU 上分布式多进程 |
| 训练吞吐量 | **~847 步/秒** | 2048 并行环境，AMD Radeon (51 GB 显存) |

---

## 🔍 项目背景

Booster Robotics 官方 RL 训练框架（Booster Gym / Booster Train）依赖 NVIDIA Isaac Gym
和 Isaac Lab，需要 CUDA 和 NVIDIA GPU。本项目构建了一个完全运行在 **AMD Radeon GPU**
上的替代训练 pipeline：

- **Genesis** — GPU 加速物理仿真（AMD Radeon 兼容）
- **ROCm PyTorch** — AMD GPU 计算平台（替代 CUDA）
- **rsl_rl** — 基于 PPO 的强化学习训练器

这是首个基于 AMD GPU 的人形足球训练 pipeline，证明无需 NVIDIA 硬件即可训练出有竞争力的
机器人策略。

---

## 🏗 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                      训练 Pipeline                        │
│                                                          │
│  ┌──────────────┐    ┌───────────────┐                  │
│  │  高层策略      │    │  低层模型      │                  │
│  │  PPO (可训练)  │───▶│  冻结行走模型  │──▶ PD 控制       │
│  │  (19→3 维)    │    │  (720→21)     │    (50 Hz)       │
│  │  vx,vy,wz     │    │  t1_walk.pt   │                  │
│  └──────┬───────┘    └───────────────┘                  │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐     │
│  │  Genesis 物理引擎 (AMD Radeon GPU)               │     │
│  │  足球场 + T1 人形机器人 + 球                      │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  奖励: approach_ball(10) + ball_control(2)               │
│        + ball_to_goal(3) + upright(0.5) - fall            │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                     部署 Pipeline                         │
│                                                          │
│  训练 .pt  ──▶  ONNX 导出  ──▶  Booster Studio           │
│  检查点                         3v3 SoccerSim (Sim2Sim)   │
└──────────────────────────────────────────────────────────┘
```

### 分层策略设计

策略分为两个层级：

| 层级 | 观测空间 | 动作空间 | 频率 | 模型 |
|------|---------|---------|------|------|
| 高层 | 19 维（球位置/速度、球门方向、本体感知） | 3 维 (vx, vy, wz) | 10 Hz | 可训练 PPO |
| 低层 | 720 维（10 帧本体感知历史） | 21 维（关节目标） | 50 Hz | 冻结 `t1_walk.pt` |

此设计解决了一个关键问题：原始扁平策略（720 维观测）没有球的信息，却因接近球而获得奖励。
分层拆分让高层策略直接观测球的状态，同时冻结的行走模型负责平衡和步态。

### 19 维观测空间

| 索引 | 分量 | 维度 | 描述 |
|------|------|------|------|
| 0-2 | filtered_lin_vel | 3 | 机器人体坐标系下的线速度 |
| 3-5 | filtered_ang_vel | 3 | 机器人体坐标系下的角速度 |
| 6-7 | projected_gravity | 2 | 姿态指示器 (xy) |
| 8-9 | ball_rel_body | 2 | 球相对机器人的位置（体坐标系） |
| 10-11 | ball_vel_body | 2 | 球速度（体坐标系） |
| 12 | dist_to_ball | 1 | 到球的欧几里得距离 |
| 13-14 | goal_dir | 2 | 球门方向（体坐标系，归一化） |
| 15 | goal_dist | 1 | 到球门的距离 |
| 16-18 | last_hl_actions | 3 | 上一次速度指令 [vx, vy, wz] |

---

## 📊 关键结果

### 训练进展（P0/P1/P2 调参后，500 轮迭代）

| 指标 | 初始 | 结束 | 变化 |
|------|------|------|------|
| 平均奖励 | -22 | +24 | ▲46 |
| 回合长度 | 18 | 225 | ▲207 |
| 动作标准差 | 1.0 | 0.07 | ✓ 稳定 |
| 最小球距 | 4.29m | 0.25m | 降低 93% |

### Module E：基线 vs RL 对比

| 场景 | 基线 min_d | RL min_d | RL 优势 |
|------|-----------|----------|---------|
| 正面近 | 1.87m | **1.28m** | 接近 32% |
| 正面远 | 4.63m | 4.81m | 相当 |
| 左侧 | 1.50m | 2.26m | 基线更好 |
| 右侧 | 3.58m | 3.58m | 相同 |

### Module F：标准化基准测试（4 场景 × 10 次）

| 场景 | 平均 Δd | 最小距离 | 摔倒次数 | 奖励 |
|------|---------|---------|---------|------|
| 正面近 | -0.01m | 1.28m | 1/10 | 76.5 |
| 正面远 | -0.17m | 4.48m | 1/10 | 72.0 |
| 左侧 | -0.05m | 2.02m | 0/10 | 74.6 |
| 右侧 | -0.33m | 3.24m | 2/10 | 73.4 |

**推理延迟**：均值 0.41ms，p95 0.40ms（4000 采样）—— 满足实时要求。

### 5 个关键 Bug 修复

| # | Bug | 根因 | 修复方案 |
|---|-----|------|---------|
| 1 | 浮动基座锁死 | URDF `world_joint` 被注释 + `merge_fixed_links=True` | 取消注释 + `merge_fixed_links=False` |
| 2 | 1 步终止 | `base_euler` 用度数，`term_pitch` 用弧度 (0.52° vs 30°) | 直接使用度数值 |
| 3 | 观测时序错位 | `_build_low_level_obs` 在物理步进前更新历史 | 只读 `obs_buf`，步进后更新 |
| 4 | 保守局部最优 | `approach_ball=1` → 原地不动得 +34 奖励 | 课程学习 + `approach_ball=10` |
| 5 | 动作标准差爆炸 | `entropy_coef=0.01` → std=5.78，噪声主导 | `entropy_coef=0.003` → std=0.07 |

---

## 🔧 环境要求

### 硬件

- AMD Radeon GPU（如 RX 7900 XTX、MI250），ROCm 6.2+
- 推荐 16 GB 以上显存（2048 并行环境）

### 云环境

本项目在**安睿云** AMD GPU 实例上开发：

- JupyterLab 终端访问
- VNC 通过 noVNC 端口 6080（密码：`***REMOVED***`）
- Python 虚拟环境位于 `/opt/venv/`

---

## 🚀 安装步骤

### 第 1 步：安装 ROCm PyTorch

```bash
# 使用 ROCm 专用 PyTorch wheel
/opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# 验证 AMD GPU 检测
/opt/venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'HIP available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'ROCm version: {torch.version.hip}')
"
```

### 第 2 步：安装 Python 依赖

```bash
/opt/venv/bin/pip install -r requirements.txt
```

### 第 3 步：获取预训练行走模型

冻结的低层行走模型（`t1_walk.pt`）来自 Booster Robotics 的部署框架：

```bash
cd /workspace

# 克隆 Booster Deploy（含 t1_walk.pt 和 URDF 模型）
git clone https://github.com/BoosterRobotics/booster_deploy.git
git clone https://github.com/BoosterRobotics/booster_assets.git

# 安装 booster_assets（提供 URDF 模型）
cd booster_assets
/opt/venv/bin/pip install -e .
cd ..

# 验证行走模型存在
ls -lh /workspace/booster/booster_deploy/tasks/locomotion/models/t1_walk.pt
```

### 第 4 步：克隆本仓库

```bash
cd /workspace
git clone https://github.com/gxinxing/radeon-hackathon-2026.git amd-physical-ai-soccer
cd amd-physical-ai-soccer
```

### 第 5 步：验证环境

```bash
# 检查 ROCm
rocm-smi

# 检查 PyTorch + Genesis + rsl_rl
/opt/venv/bin/python -c "
import torch; import genesis as gs; import rsl_rl
print(f'PyTorch {torch.__version__} | HIP: {torch.cuda.is_available()}')
print(f'Genesis {gs.__version__}')
print('rsl_rl OK')
"

# 验证 t1_walk.pt 能行走 30 秒不摔倒
/opt/venv/bin/python verify_t1_walk.py
```

---

## 📖 使用方法

### 训练

```bash
cd /workspace/amd-physical-ai-soccer

# 快速测试（256 环境，100 轮迭代 — 约 5 分钟）
/opt/venv/bin/python train_hierarchical.py \
    --num_envs 256 \
    --max_iterations 100

# 完整训练（2048 环境，500 轮迭代 — 约 2-4 小时）
/opt/venv/bin/python train_hierarchical.py \
    --max_iterations 500

# 从检查点恢复训练
/opt/venv/bin/python train_hierarchical.py \
    --resume runs/hierarchical_soccer_chase_hl/model_250.pt

# 自定义行走模型路径
/opt/venv/bin/python train_hierarchical.py \
    --pretrained /path/to/custom_walk.pt
```

模型保存在 `runs/hierarchical_soccer_chase_hl/`：

```bash
ls runs/hierarchical_soccer_chase_hl/
# model_50.pt  model_100.pt  ...  model_500.pt  cfgs.pkl
```

### 渲染演示视频

```bash
# 使用最新检查点渲染 300 步
/opt/venv/bin/python render_hierarchical.py --steps 300

# 使用指定模型渲染
/opt/venv/bin/python render_hierarchical.py \
    --model runs/hierarchical_soccer_chase_hl/model_500.pt \
    --steps 500
```

输出：`demos/hierarchical_chase_hl_v4.mp4`

### 导出 ONNX 部署模型

```bash
# 通过原始 MLP 提取导出（推荐 — 绕过 rsl_rl tracing 限制）
/opt/venv/bin/python export_onnx_mlp.py \
    --model runs/hierarchical_soccer_chase_hl/model_500.pt \
    --output models/chase_v3_policy.onnx

# 替代方案：标准导出
/opt/venv/bin/python export_onnx.py \
    --model runs/hierarchical_soccer_chase_hl/model_500.pt \
    --output models/soccer_policy.onnx
```

### Booster Studio Sim2Sim 验证

1. 在云实例上安装 Booster Studio：

```bash
bash install_booster_studio.sh
```

2. 通过 noVNC 访问 Booster Studio：

```
https://radeon-global.anruicloud.com/instances/<instance-id>/proxy/6080/vnc.html
```

3. 在 Booster Studio 的 3v3 SoccerSim 中加载 ONNX 模型
4. 与官方 Booster AI 进行对抗

### 3v3 对对抗评估

```bash
# 本地运行对抗评估（规则策略无需 GPU）
/opt/venv/bin/python scripts/match_eval_3v3.py

# 使用 RL 策略
/opt/venv/bin/python scripts/match_eval_3v3.py \
    --checkpoint runs/hierarchical_soccer_chase_hl/model_500.pt
```

### GPU 基准测试收集

```bash
# 后台启动基准测试收集器
/opt/venv/bin/python benchmark_collect.py \
    --log /tmp/train_output.log \
    --output benchmark/ \
    --interval 5 &

# 运行训练（日志输出到 /tmp/train_output.log）
/opt/venv/bin/python train_hierarchical.py --max_iterations 500 \
    2>&1 | tee /tmp/train_output.log

# 训练结束后停止收集器
kill $(cat /tmp/benchmark_pid)
```

输出：`benchmark/gpu_samples.csv` 和 `benchmark/gpu_samples.json`

---

## 🎯 奖励函数设计

奖励函数（`reward.py`）实现了课程学习，针对不同任务使用不同的奖励项组合：

| 任务 | 奖励项 | 目标 |
|------|--------|------|
| `balance` | upright, alive, tracking_vel, feet_swing, feet_slip | 行走时保持平衡 |
| `chase` | balance 项 + approach_ball | 接近球 |
| `chase_hl` | upright, alive, approach_ball, ball_control, ball_to_goal, goal_scored | 分层策略（无步态项 — 冻结模型处理） |

关键奖励塑形技术：

- **指数核**速度跟踪：`exp(-(cmd - actual)² / σ)`
- **距离差值**追球奖励：`prev_dist - current_dist`（接近球时获得奖励）
- **指数邻近度**控球奖励：`exp(-(dist - radius) * 3.0)`
- **惩罚项**：摔倒、能耗、动作抖动

### P0/P1/P2 参数调优

| 参数 | 调整前 | 调整后 | 原因 |
|------|--------|--------|------|
| `hl_clip` (线/角) | 0.05 / 0.05 | 0.8 / 1.0 | 解锁全速行走 |
| `upright` | 5.0 | 0.5 | 冻结模型已能平衡 — 避免奖励饱和 |
| `alive` | 3.0 | 0.0 | 移除被动生存奖励 |
| `approach_ball` | 1.0 | 10.0 | 追球奖励必须占主导 |
| `entropy_coef` | 0.01 | 0.003 | 修复动作标准差爆炸 (5.78→0.07) |
| `learning_rate` | 3e-3 | 1e-3 | 稳定 value loss |

---

## ⚔ 分布式多机器人对抗 (1v1 / 3v3)

由于 Genesis 在 ROCm 上无法在同一场景中处理多个机器人，我们采用多进程分布式架构：

```
┌──────────────────────────────────────────────────┐
│           比赛协调器 (socket)                      │
│  - 50Hz 同步循环                                   │
│  - 向所有 worker 广播球和机器人位置                 │
│  - 成对碰撞检测 + 推回                             │
│  - 结构化 JSON 比赛日志                            │
└──┬──────┬──────┬──────┬──────┬──────┬────────────┘
   │      │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼      ▼
┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐
│RL   ││规则 ││规则 ││规则 ││规则 ││规则 │
│Agent││队友 ││队友 ││对手 ││对手 ││对手 │
│+球   ││     ││     ││     ││     ││     │
└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘
  GPU     GPU    GPU    GPU    GPU    GPU
 (共享 AMD Radeon，6 个进程)
```

**启动 1v1 对抗：**
```bash
bash run_1v1.sh runs/hierarchical_soccer_chase_hl/model_1894.pt 25
```

**启动 3v3 对抗（6 机器人）：**
```bash
bash run_3v3.sh runs/hierarchical_soccer_chase_hl/model_1894.pt 25
```

**结果：**
- 1v1：Agent 119 步，对手 112 步，零 GPU 崩溃
- 3v3：6 个 worker 各 75-84 步，共记录 1240 步，零 GPU 崩溃
- 比赛日志保存到 `match_logs/match_YYYYMMDD_HHMMSS.json`

---

## 📁 项目结构

```
.
├── train_hierarchical.py          # 分层训练入口
├── soccer_env_hierarchical.py     # 分层环境（高层 + 冻结行走模型）
├── soccer_env_v4.py               # 基础足球环境（扁平策略，v4）
├── reward.py                      # 奖励函数（平衡/追球/射门课程）
├── render_hierarchical.py         # 演示视频渲染
├── verify_t1_walk.py              # 验证 t1_walk.pt 行走 30 秒不摔倒
├── export_onnx.py                 # 标准 ONNX 导出
├── export_onnx_mlp.py             # 通过原始 MLP 提取导出 ONNX
├── benchmark_collect.py           # ROCm GPU 基准测试收集器
├── match_coordinator.py           # 分布式比赛协调器（socket 同步）
├── match_worker.py                # 分布式比赛 worker（每进程 1 机器人）
├── run_1v1.sh                     # 启动 1v1 对抗（2 个 worker）
├── run_3v3.sh                     # 启动 3v3 对抗（6 个 worker）
├── match_3v3.py                   # 3v3 比赛模拟运行器（旧版）
├── match_evaluator.py             # 比赛评估逻辑
├── match_scene.py                 # 比赛场景设置
├── soccer_env_1v1.py              # 1v1 环境（Genesis 多实体，WIP）
├── disturbance.py                 # 扰动注入（推力、风）
├── inject_proxy.py                # Agent 框架代理注入
├── configs/
│   ├── hierarchical_agent.yaml    # 分层训练配置
│   ├── curriculum_stage1.yaml     # Stage 1 课程配置
│   ├── soccer_agent.yaml          # 扁平策略训练配置
│   └── match_3v3.yaml            # 比赛模拟配置
├── src/
│   ├── soccer_env/
│   │   └── soccer_scene.py        # Genesis 足球场场景构建器
│   ├── match_3v3/
│   │   ├── __init__.py
│   │   ├── policy.py              # 策略接口（规则 + RL）
│   │   ├── roles.py               # 角色分配（前锋/后卫/守门员）
│   │   ├── scene.py               # 比赛场景和状态定义
│   │   └── result.py              # 比赛结果跟踪
│   └── booster_agent/            # Booster Studio Sim2Sim agent
│       ├── src/main.py            # Agent 入口（ONNX 策略）
│       └── src/rl_playbook.py    # RL 增强战术手册
├── scripts/
│   └── match_eval_3v3.py         # 比赛评估脚本
├── tests/
│   └── test_match_contract.py    # 比赛契约测试
├── docs/                          # 技术报告和文档
├── models/                        # 训练检查点和 ONNX 导出
├── benchmark/                     # GPU 性能数据 + Module E/F 结果
├── training_logs/                 # AMD GPU 训练日志
├── match_logs/                    # 1v1/3v3 比赛轨迹日志（JSON）
├── demos/                         # 演示视频
├── presentations/                 # 海报和幻灯片
├── urdf/t1/                       # T1 人形机器人 URDF + 网格
├── requirements.txt
└── README.md
```

---

## 🛠 技术栈

| 组件 | 技术 | 原因 |
|------|------|------|
| 物理仿真 | Genesis 1.2.3 | GPU 加速，AMD Radeon 兼容，Python 原生 |
| 深度学习 | PyTorch 2.9.1 (ROCm 6.2) | 通过 HIP/ROCm 支持 AMD GPU |
| RL 算法 | rsl_rl 5.4.2 (PPO) | 轻量、成熟、兼容 Genesis |
| 机器人平台 | Booster T1（人形，31 自由度） | RoboCup 足球标准平台 |
| Sim2Sim 验证 | Booster Studio 1.9.4 | 官方 3v3 足球模拟器 |
| 云 GPU | 安睿云 AMD GPU (51 GB 显存) | AMD Radeon GPU + JupyterLab + VNC |

---

## ⚠ 已知限制

1. **Genesis ROCm 多实体崩溃**：Genesis 物理引擎在 AMD ROCm 上当同一场景加载两个或
   以上机器人 URDF 实体时，会触发 `hipErrorLaunchFailure` 崩溃。这是平台级 bug，非显存
   问题（VRAM 占用仅 0.9 GB / 51.5 GB）。

2. **解决方案 — 分布式多进程架构**：每个机器人在独立的 Genesis 进程中运行（已验证 1 个
   机器人稳定）。通过 socket 协调器在进程间同步状态。已验证 6 个并发进程（3v3 对抗）。

3. **近距离控球**：当球距离 2 米以内时，速度指令接口无法表达控球所需的精细动作。
   需要残差关节级策略来实现带球和射门。

---

## 📦 数据来源

本项目**不使用外部数据集**。所有训练数据由 Genesis 物理仿真实时生成：

| 数据 | 来源 | 用途 |
|------|------|------|
| `t1_walk.pt` | [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) 仓库 | 冻结低层行走策略 (720→21) |
| T1 URDF 模型 | [booster_assets](https://github.com/BoosterRobotics/booster_assets) 仓库 | Genesis 机器人物理模型 |
| 足球场 | `src/soccer_env/soccer_scene.py` | 14m × 9m RoboCup 3v3 球场 |
| 奖励函数 | `reward.py` | 课程学习：平衡 → 追球 → 射门 |
| 训练配置 | `configs/hierarchical_agent.yaml` | PPO 超参数、奖励权重 |

---

## 👥 团队信息

- 队伍名称：[提交前填写]
- 队员：[提交前填写]

---

## 📄 License

本项目为 AMD AI DevMaster 黑客马拉松提交作品。请参阅竞赛仓库了解许可条款。
