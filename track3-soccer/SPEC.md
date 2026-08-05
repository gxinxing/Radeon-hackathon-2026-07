# Track 3 Soccer Robot — Spec

> **Spec ID:** T3-SDD-2026-08-06
> **Author:** Simon Xing
> **Date:** 2026-08-06
> **Status:** Active
> **Deadline:** 2026-08-06

---

## 1. 目标

在 AMD Radeon GPU 上交付 3v3 人形机器人足球项目，满足官方 5 大评分维度（100分）。

### 1.1 评分维度

| # | 维度 | 分值 |
|---|------|------|
| 1 | 机器人能力表现（训练收敛+任务指标+奖励分量） | 30 |
| 2 | AMD ROCm 采用（GPU/ROCm/吞吐/延迟/VRAM） | 20 |
| 3 | 创新性（首个 AMD GPU 人形足球管线） | 20 |
| 4 | 实际应用价值（Sim2Sim+部署路径） | 20 |
| 5 | 上游开源贡献（可复用 Genesis 足球环境） | 10 |

### 1.2 交付物

- 技术报告（`docs/technical_report.md`）
- 源代码 + README
- Demo 视频
- 证据：训练日志、GPU 遥测、模型 SHA-256、评估 JSON

---

## 2. 当前状态

### 2.1 已完成（已验证）

| 资产 | 状态 | 数据 |
|------|------|------|
| PPO 训练收敛 | ✅ | 500轮, reward -24→+93, 4618 steps/s |
| t1_walk.pt 行走模型 | ✅ | stance 60步/0跌倒, gait 150步/6.4m |
| chase_v8_policy.onnx | ✅ | 19→3 dim, 0.4ms 推理, 46467 params |
| 单机器人 eval | ✅ | 100步, 0跌倒, 球位移1.04m, 到球0.14m |
| 奖励分量 JSON | ✅ | 5个分量全输出 (approach_ball/ball_control/ball_progress/goal_scored/fall_penalty) |
| 单机器人追球视频 | ✅ | 200步, 0跌倒, 球位移12m, 100帧, 960×540 |
| 基线对比 | ✅ | ONNX vs Rule, 有 fall_count + distance |
| GPU 证据 | ✅ | 612条遥测, 峰值100%利用率, 23.7GB VRAM |
| 本地测试 | ✅ | 151 passed |
| T04 补丁 | ✅ | 碰撞参数4096/1e-4/100 + 踢球逻辑修复 |
| 3v3 场景加载 | ✅ | 6机器人, CG solver, camera, 30帧视频 |
| Booster 策略框架 | ✅ | strategy/{param,player,match}.py 已写 |
| 文档 | ✅ | technical_report.md, README.md, robocup_reference.md |

### 2.2 待解决

| 问题 | 根因 | 修复方案 | 状态 |
|------|------|---------|------|
| **3v3 机器人不动** | `_build_low_level_obs_for_robot()` 用了滤波角速度(初始为零)，walk model 训练时用的是原始角速度 | 改用 `robot.get_ang()` 原始值 | ✅ 已修复 (本地+远端 py_compile PASS) |
| **3v3 walk model 不稳定** | t1_walk.pt 在 3v3 多机器人场景泛化不足，站立约15步(1.5s)后倾斜倒地 | 需接触鲁棒微调 (T07, 超出当前时限) | Known limitation, 单机器人提交 |
| **GitHub 未推送** | 网络超时 | 重试 | ✅ 已推送 |

### 2.3 实例状态

- JupyterLab: `https://radeon-global.anruicloud.com/instances/<REDACTED>/lab`
- Token: `<REDACTED>`
- GPU: AMD Radeon RX 7900 XT, 48GB VRAM, ROCm 7.2.1
- Genesis 1.3.1 已安装, kernel 已缓存
- Persistent: `/workspace/persistent/track3/`

---

## 3. 立即执行的任务

### Task-1: 修复 3v3 obs 角速度（5分钟）

**文件**: 远端 `/workspace/scripts/soccer_env_3v3.py`
**方法**: `_build_low_level_obs_for_robot()`
**改动**: 1 行

```python
# 当前（错误）：
ang_vel = self.all_filtered_ang_vel[:, i, :]

# 改为（正确）：
ang_vel = transform_by_quat(self.robots[i].get_ang(), inv_quat(quat))
```

**依据**: 单机器人 env (soccer_env_v4.py) 用的是 `robot.get_ang()` 原始值，3v3 误用了 EMA 滤波值（初始为零），导致 walk model 收到 out-of-distribution 输入。

### Task-2: 验证 3v3 机器人是否移动（10分钟）

- 跑 30 步
- 检查机器人位移 > 0.01m
- 检查球位移 > 0
- 录视频

### Task-3: 如果机器人动了 → 录完整 3v3 视频

- 跑 100 步
- 录 1280×720 视频
- 检查 kicks > 0, ball_displacement > 0.5m

### Task-4: 如果机器人没动 → 用单机器人成果提交

- 已有视频 + JSON 足够提交
- 诚实标注 3v3 为 known limitation

### Task-6 (P0): 修 3v3 相机渲染 — 出视频

**文件**: 远端 `/workspace/scripts/soccer_env_3v3.py`
**问题**: `'SoccerEnv3v3' object has no attribute 'cam'`，3v3 env 没创建相机，所有 run frames=0
**做法**: 参照单机版 `soccer_env_v4.py` 的相机创建，给 3v3 env 加 cam
**验收**: rule_walk 跑 100 步能输出 mp4（比赛要 3-5 分钟视频）

### Task-7 (P0): 降摔倒 — 步态稳定性

**现状**: v3 已能走+踢球（ball 5.26m, kicks 1），但第 10 步 6 个全倒
**做法**: 步态平衡调优（振幅/频率/踝补偿/支撑相），保持速度同时不倒
**验收**: 100 步 fallen ≤ 2

### Task-8 (P1): 踢球瞄准对方球门

**现状**: 踢球把球往 +y 侧向踢（球滚到 (2.72, 4.50)），没朝对方球门 +x
**做法**: 修踢球方向/追球策略，让球朝对方球门方向位移
**验收**: ball 主要沿 +x 位移，尽量进球（score > 0）

### Task-5: 推送 GitHub

无论 Task-3 还是 Task-4，都要推送。

---

## 4. 系统架构

分层数据流（自上而下）：

```
高层策略 (19 obs → 3 action: vx, vy, wz)
  ↓ 速度指令（clip ±1.2）
低层控制器（二选一）
  ├─ rule_walk：正弦步态（确定性，无模型，稳定优先）
  └─ t1_walk.pt：冻结 RL 行走模型（720 obs → 21 joint）
  ↓ joint targets × action_scale(0.25) + default_pos → PD 控制
Genesis 物理（AMD Radeon GPU，单场景 6 机器人 + 1 球）
```

- 3v3 = **单场景 6 机器人**（左队 3 + 右队 3，共 1 个球），非并行 env
- 决策层：左队 RL 策略 / 右队规则策略；步态层在 rule_walk 模式下全部用规则步态
- Match 控制器每步 `act()` 产出 6×3 指令，`env.step()` 转关节目标
- 踢球：距球 < 0.5m 且冷却 0.5s → 朝对方球门施加 3.0 m/s 冲量

## 5. 模块划分

| 模块 | 路径 | 职责 |
|------|------|------|
| 3v3 环境 | `scripts/soccer_env_3v3.py` | 单场景、6 机器人、步态、踢球、obs、step |
| 单机父类 | `scripts/soccer_env_v4.py` | 720 obs / walk model / 单机器人接口 |
| 策略层 | `strategy/{param,player,match}.py` | 决策、角色分配、比赛控制 |
| 奖励 | `reward.py` | `compute_reward(obs, action, w, task)` |
| 控制工具 | `scripts/control_utils.py` | `compose_full_joint_targets` 等 |
| 比赛入口 | `run_rule_walk_match.py` / `run_booster_match.py` | 跑比赛、写结果 JSON、渲染 |
| 配置 | `configs/hierarchical_agent.yaml` | env/obs/reward/command 参数 |

## 6. 接口

- `SoccerEnv3v3(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, walk_model_path, high_level_decimation=5, show_viewer=False)`
- `step(hl_actions: (1,3) | (1,6,3)) → (obs, reward, done, extras)`；`extras` 含 `kick_events`、`terminal_state`
- `reset() → TensorDict{"policy": (1,19)}`；`get_observations()`
- `Match(env).act() → (1,6,3)`；`check_events(extras) → (kicks:int, scored:bool)`；`get_robot_stats() → [{fallen,...}]`
- `compute_reward(obs: dict, action: Tensor, w: dict, task: str) → Tensor`
- 低层：`_rule_walk_actions(cmd, robot_idx) → (1,21)`；`_run_walk_model(obs_720) → (1,21)`

## 7. 数据模型

- **HL obs 19 维**：lin_vel(3) + ang_vel(3) + gravity_xy(2) + ball_rel_body(2) + ball_vel_body(2) + dist_to_ball(1) + goal_dir(2) + goal_dist(1) + last_hl_action(3)
- **HL action 3 维**：`[vx, vy, wz]` ∈ [-1.2, 1.2]
- **LL obs 720 维 / LL action 21 维**（`POLICY_JOINT_NAMES`）
- **结果 JSON**：`{started_at, gpu, mode, status, steps, frames, kicks, score{left,right}, ball_displacement, robot_displacement, num_robots, ended_at, duration_s}`
- 状态缓冲：`all_base_pos (1,6,3)`、`all_base_quat (1,6,4)`、球位置/速度 `(1,3)`

## 8. 边界条件

| 项 | 值 |
|----|----|
| 场地 | 14 × 9 m，球门宽 2.6 m，球半径 0.11 m，机器人高 0.72 m |
| 摔倒判据 | base 高度 < 0.4 m（`fall_height`） |
| 终止条件 | pitch/roll 30°（rule_walk 模式可关） |
| 踢球 | 距离 < 0.5 m，冷却 0.5 s，冲量 3.0 m/s |
| 指令裁剪 | 线速度 ±1.2 m/s，角速度 ±1.2 rad/s |
| 控制周期 | dt 0.02 s（physics 0.002 × 10），高层 decimation 5 |
| action 缩放 | 0.25（低层 joint target） |
| 初始站位 | 左队 [-1, -3.5, -6.5]，右队 [1, 3.5, 6.5]（x 对称，见代码） |
| 球出界 | 无特殊处理，按物理继续滚 |

## 9. 验收标准（可测量）

| 项 | 标准 |
|----|------|
| 环境加载 | 6 机器人 + 1 球，Genesis 启动 < 60s，干净退出 |
| 低层门禁 | t1_walk stance 60 步 / 0 跌倒 |
| 单机评估 | 100 步 0 跌倒，球位移 ≥ 1m |
| 3v3 rule_walk | 100 步 fallen ≤ 2，robot_disp ≥ 2m，kicks ≥ 1，ball_disp ≥ 2m |
| 视频 | 100 步出 mp4（≥30 帧）；比赛成片 3-5 分钟 |
| 踢球瞄准 | 球主要沿 +x 位移，尽量 score > 0 |
| 结果文件 | 每次 run 输出 JSON（§7 schema）+ 视频 + 进度文件 |

## 10. 策略架构（Booster 风格）

参考 Booster 官方 3v3 基线，已创建三个文件：

```
strategy/
├── param.py     ← 参数（踢球/带球/追球/角色/传球/避障）
├── player.py    ← 动作（chase/attack/dribble/guard/support/defend）
└── match.py     ← 决策（Phase状态机/角色分配/Match控制器）
```

### 4.1 策略核心

- **最近者追球**：不固定前锋，动态选离球最近者（防震荡 0.3m）
- **Guard 守门**：球远守门，球近前出拦截
- **Support 支援**：侧前方接应，失球回防
- **带球推进**：小力度推球，进入射门区再大力踢

### 4.2 运行入口

`run_booster_match.py` → 创建 Match 控制器 → 每步调 `match.act()` → 6 个 Player 计算速度指令 → `env.step()` 转给 walk model

---

## 11. 技术参考（详见链接）

### 5.1 Booster 官方基线

- 代码: `/Users/simon/BoosterStudioProjects/simon3v3-simple-baseline/src/{main,player,param}.py`
- 文档: `docs/booster-3v3-complete-guide.md`（649行，含 Phase 状态机/角色分配/战术/安全/参数表）
- API: `docs/Booster Agent Framework Python API.md`（1404行，含 AgentBase/Context/Player/robot_states）
- 构建: `.agent` 文件是 zip 格式，内含 ROS2 colcon 包 + Python 代码 + 依赖库

### 5.2 RoboCup 参考

- 详见: `docs/robocup_reference.md`
- RobocupGym reward: `reward = ball_displacement`（极简）
- op2 fitness: 跌倒=-1+终止, 向后踢=-100

### 5.3 关键参数（Booster 实战验证）

| 参数 | 值 | 用途 |
|------|-----|------|
| KICK_POWER | 5.0 | 正常踢球 |
| DRIBBLE_KICK | 2.0 | 带球 |
| KICK_ENTER_M | 0.5 | 踢球距离 |
| CHASE_BEHIND_M | 0.35 | 追球站位 |
| PASS_POWER | 3.6 | 传球 |
| GUARD_THREAT_X | -1.0 | 守门威胁区 |
| COUNTER_PRESS_S | 2.5 | 反抢窗口 |
| OPPONENT_RADIUS | 0.55 | 避障半径 |

---

## 12. 工作规则

1. **方向调整必须跟用户确认**
2. **每轮只改一个东西**
3. **产出即备份**到 persistent + 本地
4. **代码改动必须有依据**（参考 Booster 或 RoboCup）
5. **不堆砌文档**，以本 SPEC 为唯一指引
6. **每轮开工先读第 3 节任务清单**，任务完成回写状态+关键数字
7. **所有改动/实验在终端运行，输出可见**（不用 nohup/后台/静默）
8. **时间盒交付**：超时按第 14 节 Go/No-Go 回退备用方案
9. **本 SPEC 同步到远端 `/workspace/SPEC.md`**，codely 每轮必读
10. **进度更新即同步 README**：每完成一个任务，先更新对应 README 的项目状态再跑下一个
11. **清理过时内容**：明显过时/无价值的文件、日志、中间产物及时删除或归档，不留垃圾
12. **文件归位**：产出放固定目录（`demos/`、`acceptance/`、`reports/`、`scripts/`），不散落在 /workspace 根目录

---

## 13. 已有资产清单

| 文件 | 路径 |
|------|------|
| 技术报告 | `docs/technical_report.md` |
| RoboCup+Booster 借鉴指南 | `docs/robocup_reference.md` |
| Booster 完整指南 | Booster 项目内 `docs/booster-3v3-complete-guide.md` |
| Booster API 文档 | Booster 项目内 `docs/Booster Agent Framework Python API.md` |
| 训练日志 | `archive/train_v8.log` |
| GPU 遥测 | `acceptance/performance/gpu_telemetry.csv` |
| 单机器人视频 | `demos/eval_soccer_20260805.mp4` |
| 追球视频 | `demos/match_1v1_20260805.mp4` |
| 奖励分量 | `acceptance/single_agent/eval_reward_components_20260805.json` |
| 策略代码 | `strategy/{param,player,match}.py` |
| 运行脚本 | `run_booster_match.py` |
| GitHub | `gxinxing/Radeon-hackathon-2026-07-track3` (private) |
| GitHub (保底) | `gxinxing/Radeon-hackathon-2026-07-track3-2` (private) |

---

## 14. Go/No-Go 清单

| # | 检查项 | 状态 |
|---|-------|------|
| 1 | 本地测试通过 | ✅ 151 passed |
| 2 | 校验和/commit ID | ✅ SHA-256 |
| 3 | 低层行走门禁 | ✅ stance 60步/0跌倒 |
| 4 | 单 Agent 足球评估 | ✅ 0跌倒, 球1.04m |
| 5 | 多机器人生命周期 | ✅ 10s 干净退出 |
| 6 | 完整演示视频 | ✅ 单机器人(200步/0跌倒/球12m) + 3v3 rule_walk(100步/1踢/球5.77m/50帧/1280×720) |
| 7 | 奖励曲线和任务指标 | ✅ training_curve.csv + 5分量 |
| 8 | ROCm 性能 | ✅ 4618 steps/s, 23.7GB VRAM |
| 9 | README 命令匹配 | ✅ |
| 10 | GitHub 无密钥 | ✅ |

---

## 15. 评分预估

### 路径 A: 3v3 修复成功

| 维度 | 得分 | 满分 |
|------|------|------|
| 机器人能力 | 25 | 30 |
| AMD ROCm | 18 | 20 |
| 创新性 | 18 | 20 |
| 应用价值 | 16 | 20 |
| 开源贡献 | 9 | 10 |
| **总计** | **86** | **100** |

### 路径 B: 单机器人提交

| 维度 | 得分 | 满分 |
|------|------|------|
| 机器人能力 | 22 | 30 |
| AMD ROCm | 18 | 20 |
| 创新性 | 17 | 20 |
| 应用价值 | 14 | 20 |
| 开源贡献 | 8 | 10 |
| **总计** | **79** | **100** |

**先试 A（5分钟），失败转 B（1小时）。**
