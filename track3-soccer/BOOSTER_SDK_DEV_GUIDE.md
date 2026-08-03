# Booster Robotics SDK 开发要点整理

> 来源: GitHub repos, booster.tech 官网, PyPI, 技术博客  
> 整理日期: 2026-07-24

---

## 一、机器人硬件概览

### K1 (入门级, RoboCup KidSize 冠军机型)

| 项目 | 规格 |
|------|------|
| 身高 | ~95cm |
| 重量 | ~19.5kg |
| 总自由度 | 22 DoF (腿6×2 + 臂4×2 + 头2) |
| 关节峰值力矩 | 60 N·m |
| 关节编码器 | 双编码器 |
| GPU 算力 | 48 TOPS (Dense) / 117 TOPS / 200 TOPS (可选) |
| 相机 | 立体深度相机 |
| IMU | 9轴 IMU |
| 音频 | 环形6麦克风阵列 + 扬声器 |
| 起售价 | $5,999 |

K1 关节名称 (22个):
```
AAHead_yaw, Head_pitch,
Left_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch, Left_Elbow_Yaw,
Right_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch, Right_Elbow_Yaw,
Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll,
Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw, Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
```

### T1 (标准开发平台)

| 项目 | 规格 |
|------|------|
| 身高 | ~140cm |
| 重量 | ~42-43kg |
| 总自由度 | 31 DoF (腿6×2 + 臂7×2 + 腰3 + 头2) |
| 关节峰值力矩 | 140 N·m |
| 电机 | 高速内转子 PMSM |
| 计算单元 | Thor T5000 |
| 相机 | 头部双目 + 腰部双目 + 腕部相机(可选) |
| 通信 | Wi-Fi 6, 蓝牙 5.2 |

T1 关节名称 (31个):
```
AAHead_yaw, Head_pitch,
Left_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch, Left_Elbow_Yaw,
Right_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch, Right_Elbow_Yaw,
Waist,  # (T1有3个腰部关节)
Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll,
Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw, Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
```

### T2 (旗舰, 2026新款)

| 项目 | 规格 |
|------|------|
| 身高 | ~140cm |
| 重量 | ~42-43kg |
| 总自由度 | 31 DoF (腿6×2 + 臂7×2 + 腰3 + 头2) |
| 关节峰值力矩 | 140 N·m |
| 末端执行器 | 夹爪 / 6-DoF 灵巧手 (可选) |
| 计算 | Thor T5000 |
| 散热 | 局部风冷 |
| 屏幕 | 支持 |

---

## 二、软件架构全景

```
┌─────────────────────────────────────────────────────┐
│                   开发者代码                           │
├──────────────┬──────────────┬────────────────────────┤
│  Booster Gym │  Booster Train│    Booster Deploy      │
│  (RL训练)     │ (Isaac Lab)   │  (Sim2Real/Sim2Sim)    │
│  Isaac Gym   │  Isaac Sim 5.0│  MuJoCo / Webots       │
├──────────────┴──────────────┴────────────────────────┤
│              Booster Robotics SDK                     │
│  ┌─────────────┐  ┌──────────────┐                   │
│  │  C++ Core   │  │  Python SDK  │                   │
│  │ (booster.core)│ │ (pip install) │                   │
│  └──────┬──────┘  └──────┬───────┘                   │
│         │    Channel Facade                           │
│         │    (LowCmd/LowState/Odometer/Hand pub/sub)   │
│         └──────────┬──────────────────────────────────┘
│               DDS Abstraction Layer (Fast DDS)         │
├───────────────────────────────────────────────────────┐
│              ROS 2 Humble 接口层                        │
│  (booster_ros2_interface: 消息+服务定义)                  │
├───────────────────────────────────────────────────────┤
│              机器人控制器 (固件 ≥ v1.4)                  │
└───────────────────────────────────────────────────────┘
```

---

## 三、Booster Robotics SDK

### 3.1 概述

- **GitHub**: https://github.com/BoosterRobotics/booster_robotics_sdk
- **定位**: 控制 Booster 机器人的底层接口
- **C++ SDK**: 在 GitHub 仓库中, 通过 CMake 构建
- **Python SDK**: `pip install booster_robotics_sdk_python` (当前版本 1.3.9)
- **通信层**: 基于 Fast DDS (Data Distribution Service) 实现实时 pub/sub

### 3.2 架构分层

| 层级 | 组件 | 职责 |
|------|------|------|
| DDS 抽象层 (DAL) | `booster.core` 内部 | 管理 DDS participant, QoS, domain, topic 注册 |
| Channel Facade | `booster.core` 内部 | 简化的 pub/sub API, 封装 DDS 细节 |
| IDL 类型 | `booster.idl` | 消息定义: LowCmd, LowState, Odometer, Hand 等 |
| 高层示例 | `booster.exhl` | 运动客户端、轨迹执行器 |
| 低层示例 | `booster.exll` | 最小 pub/sub 示例 |
| Python 绑定 | `booster.pyb` | 镜像 Core 功能, 支持脚本化开发 |

### 3.3 核心数据流

```
开发者代码
    │
    ├── 发布 LowCmd  ──→  Channel  ──→  DDS  ──→  机器人控制器 (执行)
    │
    └── 订阅 LowState ←──  Channel  ←──  DDS  ←──  机器人控制器 (反馈)
         (关节角度/速度/力矩, IMU, 按钮事件, 跌倒状态等)
```

### 3.4 C++ SDK 构建

```bash
# 克隆仓库
git clone https://github.com/BoosterRobotics/booster_robotics_sdk.git
cd booster_robotics_sdk

# 安装依赖
bash install.sh

# 构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行示例
./b1_loco_example_client 127.0.0.1    # 运动控制示例
./b1_low_level_subscriber               # 低层状态订阅示例
```

### 3.5 Python SDK

```bash
pip install booster_robotics_sdk_python

# 验证
python3 -c "import booster_robotics_sdk; print('SDK ready')"
```

Python SDK 提供:
- 关节级别控制 (全部 22/31 个关节)
- 运动模式切换
- 状态读取 (关节角度/速度/力矩, IMU, 按钮, 跌倒状态)
- ROS2 桥接支持

---

## 四、ROS2 接口 (booster_ros2_interface)

### 4.1 概述

- **GitHub**: https://github.com/BoosterRobotics/booster_robotics_sdk_ros2
- **依赖**: ROS 2 Humble
- **用途**: 标准化的消息和服务定义, 用于 ros2_control 硬件接口桥接

### 4.2 消息类型 (msg/)

| 消息类型 | 用途 |
|----------|------|
| `BoosterApiReqMsg` / `BoosterApiRespMsg` | API 请求/响应 |
| `ButtonEventMsg` | 按钮事件 |
| `FallDownState` | 跌倒状态 |
| `HandCommand` / `HandDdsMsg` / `HandParam` | 手部控制 |
| `ImuState` | IMU 传感器数据 |
| `LowCmd` / `LowState` | 低层命令和状态 (核心) |
| `MotorCmd` / `MotorState` | 电机级别命令和状态 |
| `Odometer` | 里程计 |
| `RawBytesMsg` | 原始字节数据 |
| `RemoteControllerState` | 遥控器状态 |

### 4.3 服务类型 (srv/)

| 服务类型 | 用途 |
|----------|------|
| `AgentService` | 字符串消息的请求/响应 |
| `RpcService` | 复杂数据交互的 RPC |

### 4.4 ROS2 桥接启动

```bash
# 安装 ROS2 Humble (Ubuntu 22.04)
sudo apt install ros-humble-desktop ros-humble-ros2-control \
  ros-humble-ros2-controllers ros-humble-joint-state-publisher-gui -y

# 克隆并构建 K1/T1 ROS2 包
# (具体路径参考官方手册)

# 启动桥接节点
ros2 launch booster_bridge k1_bridge.launch.py  # K1
# 或
ros2 launch booster_bridge t1_bridge.launch.py  # T1
```

---

## 五、Booster Gym (RL 训练框架)

### 5.1 概述

- **GitHub**: https://github.com/BoosterRobotics/booster_gym
- **定位**: 人形机器人运动控制强化学习框架
- **仿真引擎**: NVIDIA Isaac Gym (Preview)
- **默认支持**: Booster T1 (开箱即用)
- **流程**: 训练 → 评估 → 部署 (Sim-to-Real 全流程)

### 5.2 核心特性

- 完整的训练到部署 Pipeline
- Sim-to-Real 迁移 (含减少 sim-to-real gap 的技术)
- 可定制环境和算法
- T1 开箱即用预配置

### 5.3 安装

```bash
# 1. 创建 conda 环境
conda create -n booster_gym python=3.8
conda activate booster_gym

# 2. 安装 PyTorch (需匹配 Isaac Gym 的 CUDA 版本)
conda install numpy=1.21.6 pytorch=2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. 安装 Isaac Gym (从 NVIDIA 官网下载)
# https://developer.nvidia.com/isaac-gym

# 4. 安装 Python 依赖
pip install -r requirements.txt
```

### 5.4 训练

```bash
# 启动训练 (T1 默认任务)
python train.py --task=T1

# 关键参数
python train.py \
  --task=T1 \
  --num_envs=4096 \
  --headless \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --seed=42 \
  --max_iterations=10000
```

### 5.5 配置文件

- 训练配置: `envs/<task_name>.yaml`
- 新增任务: 在 `envs/` 下创建 YAML, 在 `envs/__init__.py` 注册
- 训练日志: `logs/<task_name>/` (含 TensorBoard 数据)

```bash
# 可视化训练进度
tensorboard --logdir logs
```

### 5.6 评估与导出

```bash
# 加载 checkpoint 评估
python play.py --task=T1 --checkpoint=<path>

# 使用最新模型
python play.py --task=T1 --checkpoint=-1
```

### 5.7 仿真测试 (无需真机)

```bash
pip install mujoco
git clone https://github.com/BoosterRobotics/booster_gym.git
python booster_gym/examples/walk_sim.py
```

---

## 六、Booster Train (Isaac Lab 训练框架)

### 6.1 概述

- **GitHub**: https://github.com/BoosterRobotics/booster_train
- **定位**: 基于 Isaac Lab 的高级 RL 任务训练框架
- **测试环境**: Isaac Lab 2.2 + Isaac Sim 5.0
- **特色**: 集成 BeyondMimic 动作追踪框架 (适配 K1)

### 6.2 安装

```bash
# 1. 安装 Isaac Lab (推荐 conda 方式)
# 参考: https://isaac-sim.github.io/IsaacLab/

# 2. 克隆仓库 (放在 IsaacLab 目录外)
git clone https://github.com/BoosterRobotics/booster_train.git

# 3. 安装 booster_assets
git clone https://github.com/BoosterRobotics/booster_assets.git
cd booster_assets
pip install -e .

# 4. 安装 booster_train (使用 Isaac Lab 的 Python)
# 用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
python -m pip install -e .
```

### 6.3 动作数据准备

```bash
# 准备 BeyondMimic 动作数据 (使用 Isaac Lab Python)
python scripts/prepare_motion_data.py
```

### 6.4 使用

```bash
# 列出可用任务
python -c "from booster_train import list_tasks; list_tasks()"

# 运行任务 (使用 Isaac Lab Python)
python scripts/train.py --task=beyondmimic_k1

# 播放已训练策略并导出
python scripts/play.py --task=beyondmimic_k1 --checkpoint=<path>
# 导出文件: logs/rsl_rl/<task>/<run>/exported/ (TorchScript/ONNX)
```

### 6.5 ⚠️ AMD GPU 注意事项

Booster Train 依赖 **NVIDIA Isaac Lab/Isaac Sim**, 原生需要 NVIDIA GPU (CUDA)。
在 AMD GPU 环境下:
- Isaac Sim 官方仅支持 NVIDIA GPU
- 可能需要通过 Docker + ROCm 兼容层, 或使用 Booster Gym (Isaac Gym Preview) 作为替代
- **Booster Gym (基于 Isaac Gym Preview) 可能也有同样的 NVIDIA 依赖问题**
- 在安睿云 AMD GPU 实例上, 建议优先使用 **MuJoCo 仿真路径** (Booster Deploy 的 Sim2Sim)

---

## 七、Booster Deploy (部署框架)

### 7.1 概述

- **GitHub**: https://github.com/BoosterRobotics/booster_deploy
- **定位**: 轻量级部署框架, 支持 Sim2Real 和 Sim2Sim
- **仿真后端**: MuJoCo (主), Webots (内部)
- **设计**: 借鉴 IsaacLab 的模块化抽象, 统一策略执行代码

### 7.2 前置要求

| 环境 | 说明 |
|------|------|
| 机器人固件 | ≥ v1.4 (真机部署) |
| Python | 3.10+ (机器人上已预装) |
| ROS 2 Humble | 用于 `/low_state` + `/joint_ctrl` topics (机器人已预装) |
| MuJoCo / Webots | 可选, 用于仿真 |

### 7.3 运行 Sim2Sim (MuJoCo)

```bash
# 安装依赖
pip install -r requirements.txt

# 在 MuJoCo 中启动任务
python scripts/run_mujoco.py --task=<task_name>
```

### 7.4 运行 Sim2Real (真机)

```bash
# 1. 先在 MuJoCo 中测试通过
# 2. 将项目复制到机器人
scp -r my_project robot@<robot_ip>:~/

# 3. 在机器人上安装 SDK
cd booster_robotics_sdk
bash install.sh
cd build && cmake .. && make -j$(nproc) && make install

# 4. 安装 Python 依赖
pip install -r requirements.txt

# 5. 运行部署
python scripts/run_robot.py --task=<task_name>
```

### 7.5 关键设计

- **统一接口**: 同一份策略代码在仿真和真机上运行
- **模块化**: 任务定义与运行时分离
- **导出格式**: 从 Booster Train/Gym 导出的 TorchScript/ONNX 模型可直接加载

### 7.6 ⭐ AMD GPU 适配建议 (全部在安睿云上)

Booster Deploy 的 MuJoCo Sim2Sim 路径**不依赖 NVIDIA GPU**, 是 AMD GPU 环境下的首选开发路径:

```
[安睿云 AMD GPU 实例 (JupyterLab)]
  │
  ├── 编写策略代码 (JupyterLab / Booster Studio)
  ├── 定义 RL 奖励函数
  ├── MuJoCo 仿真运行 (Sim2Sim)
  ├── 加载 ONNX/TorchScript 模型验证
  └── GPU 可用后: ROCm + PyTorch 训练
```

---

## 八、RoboCup Demo (足球比赛参考实现)

### 8.1 概述

- **GitHub**: https://github.com/BoosterRobotics/robocup_demo
- **定位**: RoboCup 3v3 足球比赛完整参考实现
- **支持机型**: K1 (默认), T1 (需修改配置)
- **语言**: C++ (89%) + Python (2.5%) + CUDA (2.4%)

### 8.2 三大模块

| 模块 | 功能 | 技术 |
|------|------|------|
| **vision** | 视觉识别 | YOLO-v8, 检测机器人/球/场地, 几何计算位置 |
| **brain** | 决策 | 读取视觉+比赛数据, 决策并控制机器人动作 |
| **game_controller** | 比赛通信 | 读取裁判机广播数据, 转为 ROS2 消息 |

### 8.3 视觉系统

- 检测模型: YOLO-v8
- 仿真模式: ONNX 模型 (`sim_data_det_0126.onnx`, `sim_data_seg_0126.onnx`)
- 真机模式: TensorRT 模型 (`vision.yaml` 中配置)
- 自动视觉踢球: `RLVisionKick` (仅 K1, 固件 ≥ 1.5.2)

### 8.4 构建

```bash
# 无 CUDA 版本 (仿真/虚拟机器人)
./scripts/build_no_cuda.sh

# 有 CUDA 版本 (真机)
./scripts/build.sh
```

### 8.5 配置

**仿真模式** (`vision_local.yaml`):
- 设置 camera topics (color/depth/intrinsics)
- 使用 ONNX 模型

**真机模式** (`vision.yaml`):
- 使用 TensorRT 模型
- K1: 固件 ≥ 1.5.2 启用 RLVisionKick

---

## 九、Booster Studio (开发 IDE)

### 9.1 概述

- **官网**: https://www.booster.tech/cn/booster-t1 (底部 Booster Studio 介绍)
- **定位**: 专为具身智能开发而生的 IDE
- **平台**: macOS (Apple Silicon), Windows 10/11, Linux (Ubuntu 22/24)
- **最新版本**: 1.9.10

### 9.2 核心功能

| 功能 | 说明 |
|------|------|
| **Vibe Coding** | AI 辅助编码, 意图理解 + 自动代码生成 |
| **高精度物理仿真** | 仿真与真机共享同一套代码, Sim-to-Real 无缝衔接 |
| **调试可视化** | 点云、3D 场景、相机图像、时序图表、URDF 模型实时渲染 |
| **数据回放** | 深度集成 MCAP 和 ROS bag 回放 |
| **Agent 开发** | 覆盖 Agent 开发全流程 (从代码到真机跑通) |

### 9.3 下载

| 平台 | 下载链接 |
|------|----------|
| macOS (Apple Silicon) | https://ci-cdn.booster.tech/release/Booster%20Studio-Setup-1.9.10-darwin-arm64.dmg |
| Windows 10/11 | https://ci-cdn.booster.tech/release/Booster%20Studio-Setup-1.9.10-x64.exe |
| Linux Ubuntu 22/24 | https://ci-cdn.booster.tech/release/Booster%20Studio-Setup-1.9.10-linux-x64.deb |

> ⚠️ 项目脚本中用的是 1.9.4 版本, 官网最新已是 1.9.10, 建议更新。

### 9.4 ⭐ 安睿云上安装 Booster Studio

Booster Studio Linux 版可在安睿云实例上通过 VNC/noVNC 运行:
- 不需要 AMD GPU 也能启动 (使用 Vulkan/OpenGL 软件渲染)
- 用于: URDF 模型查看、仿真调试、Vibe Coding、MCAP/ROS bag 回放
- 项目脚本中已有安装脚本 (`start_all.sh` / `setup_booster.sh`)
- ⚠️ 脚本中使用的是 1.9.4 版本, 官网最新为 1.9.10, 建议更新下载链接

---

## 十、开发手册 (飞书文档)

- **T1 手册**: https://booster.feishu.cn/wiki/XAS3wv4lwiSiXXkDbMrceE6UnHc (需飞书登录)
- **K1 手册**: https://booster.feishu.cn/wiki/E3q5wF5SnitXZgkY18Uc8odBnXb (需飞书登录)

> 飞书文档需要登录才能访问, 建议用飞书账号登录后查看完整 API 文档。

---

## 十一、AMD GPU 环境下的开发策略

### 核心矛盾

| 组件 | NVIDIA 依赖 | AMD GPU 兼容性 |
|------|-------------|-----------------|
| Booster Gym (Isaac Gym) | ❌ 需要 NVIDIA GPU | ⚠️ 不兼容 |
| Booster Train (Isaac Lab) | ❌ 需要 NVIDIA GPU | ⚠️ 不兼容 |
| Booster Deploy (MuJoCo) | ✅ 无 NVIDIA 依赖 | ✅ 兼容 |
| Booster SDK (Python/C++) | ✅ 无 NVIDIA 依赖 | ✅ 兼容 |
| RoboCup Demo (CUDA 版) | ❌ TensorRT 需要 NVIDIA | ⚠️ 用 ONNX 版本替代 |
| Booster Studio | ✅ 无 NVIDIA 依赖 | ✅ 兼容 (Vulkan/OpenGL) |

### 推荐开发路径 (全部在安睿云 AMD GPU 实例上)

```
Phase 1: GPU 不可用期间 (现在, JupyterLab Terminal 可用)
├── pip install booster_robotics_sdk_python — SDK 学习
├── clone booster_deploy + booster_assets — 研究部署框架
├── clone booster_gym — 参考训练逻辑/奖励函数设计
├── 安装 MuJoCo + 跑通 Sim2Sim — 仿真环境验证
├── 安装 Booster Studio (Linux) — 通过 VNC/noVNC 访问 IDE
└── 搭建 ROCm + PyTorch 环境 — 为训练做准备

Phase 2: GPU 可用后
├── MuJoCo + Stable-Baselines3 / RL-Games 自定义 RL 训练
│   (替代 Isaac Gym/Isaac Lab, 因为它们不支持 AMD GPU)
├── 训练步态/平衡/踢球策略
├── 导出 ONNX/TorchScript 模型
└── Booster Deploy Sim2Sim 验证

Phase 3: 真机部署 (如果有真机)
├── 部署到 K1/T1 机器人
├── Sim2Real 验证
└── 比赛准备
```

### ⚠️ AMD GPU RL 训练核心策略

Booster Gym (Isaac Gym) 和 Booster Train (Isaac Lab) **都只支持 NVIDIA GPU**。
在 AMD GPU 环境下, 不能直接用官方训练框架, 需要自建训练 pipeline:

```
替代方案: MuJoCo + ROCm PyTorch + RL 框架

┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
│  MuJoCo     │     │  ROCm PyTorch    │     │  Booster Deploy│
│  (仿真环境)  │ ←→ │  + RL 算法       │ ──→ │  (Sim2Sim 验证) │
│  K1/T1 URDF │     │  (SB3/RL-Games)  │     │  加载 ONNX 模型 │
└─────────────┘     └──────────────────┘     └───────────────┘
       ↑                                            ↑
       └── booster_assets 提供 URDF ─────────────────┘
```

关键步骤:
1. **MuJoCo 仿真**: 加载 booster_assets 中的 K1/T1 URDF 模型
2. **RL 框架**: 使用 Stable-Baselines3 (PPO/SAC) 或 RL-Games, 兼容 ROCm PyTorch
3. **ROCm PyTorch**: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`
4. **模型导出**: 训练完成后导出为 ONNX/TorchScript
5. **部署验证**: 用 booster_deploy 加载模型, 在 MuJoCo 中验证

---

## 十二、关键 GitHub 仓库速查

| 仓库 | 地址 | 说明 |
|------|------|------|
| booster_robotics_sdk | https://github.com/BoosterRobotics/booster_robotics_sdk | C++ SDK, 机器人控制底层接口 |
| booster_robotics_sdk_python | PyPI: `pip install booster_robotics_sdk_python` | Python 绑定, v1.3.9 |
| booster_robotics_sdk_ros2 | https://github.com/BoosterRobotics/booster_robotics_sdk_ros2 | ROS2 消息和服务定义 |
| booster_gym | https://github.com/BoosterRobotics/booster_gym | RL 训练框架 (Isaac Gym) |
| booster_train | https://github.com/BoosterRobotics/booster_train | RL 训练框架 (Isaac Lab) |
| booster_deploy | https://github.com/BoosterRobotics/booster_deploy | 部署框架 (Sim2Real/Sim2Sim) |
| booster_assets | https://github.com/BoosterRobotics/booster_assets | 机器人模型 + 动作数据 |
| robocup_demo | https://github.com/BoosterRobotics/robocup_demo | RoboCup 足球比赛参考实现 |
| BoosterRobotics (组织) | https://github.com/BoosterRobotics | 所有开源仓库 |

---

## 十三、网络配置 (连接真机)

```bash
# 设置 PC 网卡 IP (K1 默认网段)
# K1: 192.168.10.x
sudo ifconfig eth0 192.168.10.10 netmask 255.255.255.0

# 验证连接
ping 192.168.10.x  # K1 的具体 IP

# T1: 可能使用不同网段, 参考飞书手册
```

---

## 十四、下一步行动建议 (全部在安睿云 JupyterLab 上)

### 第一步: 环境搭建 (GPU 不可用, 现在就能做)

在 JupyterLab Terminal 中执行:

```bash
# 1. 安装 Python SDK
/opt/venv/bin/pip install booster_robotics_sdk_python

# 2. clone 关键仓库
cd /workspace
git clone https://github.com/BoosterRobotics/booster_deploy.git
git clone https://github.com/BoosterRobotics/booster_assets.git
git clone https://github.com/BoosterRobotics/booster_gym.git    # 参考奖励函数设计
git clone https://github.com/BoosterRobotics/robocup_demo.git    # 参考比赛实现

# 3. 安装 booster_assets
cd booster_assets && /opt/venv/bin/pip install -e . && cd ..

# 4. 安装 MuJoCo (Sim2Sim 仿真, 不需要 GPU)
/opt/venv/bin/pip install mujoco

# 5. 安装 Booster Deploy 依赖
cd booster_deploy && /opt/venv/bin/pip install -r requirements.txt && cd ..

# 6. 安装 Booster Studio (通过 VNC 访问 GUI)
# 使用项目脚本: bash /workspace/setup_booster.sh
# 或更新到 1.9.10 版本后安装
```

### 第二步: 仿真验证 (不需要 GPU)

```bash
# 用 booster_deploy 在 MuJoCo 中加载机器人模型
cd /workspace/booster_deploy
python scripts/run_mujoco.py --task=<task_name>

# 用 booster_gym 的仿真示例验证
python /workspace/booster_gym/examples/walk_sim.py
```

### 第三步: RL 训练环境准备 (GPU 可用后)

```bash
# 安装 ROCm 版 PyTorch
/opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# 安装 RL 框架
/opt/venv/bin/pip install stable-baselines3

# 参考 booster_gym 的奖励函数, 在 MuJoCo 中自定义训练
```

### 核心挑战

由于 Isaac Gym/Isaac Lab 不支持 AMD GPU, 需要:
1. 参考 booster_gym 的环境定义和奖励函数, 移植到 MuJoCo + Gymnasium
2. 用 Stable-Baselines3 的 PPO/SAC 替代 rsl_rl
3. 训练完成后导出 ONNX, 用 booster_deploy 部署验证
