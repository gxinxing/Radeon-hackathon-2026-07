# System Context（修订版）：3v3 具身智能足球机器人 — RL Policy 接入与多智能体协同

> **修订说明**：原始结构化文档基于"SharedRLPolicy 是 stub、ONNX 未接入平台"的前提。经核对实例真实代码库（`/workspace/amd-physical-ai-soccer`），该前提**已过时**——ONNX 早已端到端跑通。本版据真实代码修正，可直接作为 Cursor / Claude Code / Codex 的 System Context 复用。
>
> 赛事老师确认的方向（与原始文档一致，保留）：**纯 RL 路线，不用 VLA**；难点在"仿真平台 API 适配" + "多智能体状态表示与训练基建"，不在"RL 能不能追球"。

---

## 1. 项目背景

- **场景**：Booster T1 人形机器人（31 自由度）3v3 足球仿真/半实物对抗，基于 **Genesis** 物理引擎（AMD Radeon GPU）。
- **目标**：将训练好的 RL 追球策略（ONNX）接入 3v3 编排，并升级为多智能体协同策略。
- **当前阶段**：低层行走稳定性修复（让机器人"能站能走不抖"）+ 多智能体训练链路设计。

---

## 2. 真实资产与现状（已核对实例代码）

### 2.1 高层 RL 策略（已接入，非 stub）

- **模型**：`models/chase_v8_policy.onnx`（原始文档写的 `chase_v7` 是旧版）。
- **观测/动作**：19 维 obs（自身 + 球，body frame）→ 3 维动作 `(vx, vy, wz)`。
- **推理封装**：`src/match_3v3/policy.py` 的 `SharedRLPolicy` —— **已实现真实 ONNX Runtime 推理**（`_load_onnx` / `_preprocess_obs` / `_infer` / `_postprocess`），docstring 明确写"no more stub"。
- **已验证打通**：
  - `match_1v1_onnx.py`：单机器人 RL vs 规则对手，完整跑通。
  - `run_3v3_onnx.sh`：启动 `match_coordinator.py` + 6×`match_worker.py` 分布式 3v3（Team A 带 `--onnx`）。
- ONNX 不可用时有 `RulePolicy` 兜底。

### 2.2 编排层（已闭环，非 stub）

- **动作接口**：`PolicyAction(velocity_cmd, should_kick, should_shoot, shoot_dir)`。
- "Booster Studio API（`Player.set_velocity` / `kick`）"在真实代码里是**概念抽象**：实际是 `SharedRLPolicy.compute(player, ball)` → `velocity_cmd` → Genesis `env.step(action)`。action 接口本就是 `velocity_cmd`，未来移植到官方 Booster Studio 只需把 `env.step(velocity_cmd)` 换成 `Player.set_velocity(velocity_cmd)`。
- **分布式架构**：1 个 `match_coordinator`（全局感知融合 + 角色分配 + 比分）+ 6 个 `match_worker`（每机器人一个 Genesis 进程，接收 `MSG_WORLD` 全局状态，本地 `SharedRLPolicy` 推理，下发 `velocity_cmd`）。

### 2.3 多智能体观测（脚手架已就绪）

- `src/match_3v3/multiagent_obs.py`：19 + 5 = **24 维**（队友相对位姿 2 + 对手相对位姿 2 + 球权 flag 1）。
- 通过 `env.use_multiagent_obs` 开关，5 维追加在 19 维**之后**，旧 19 维 policy 不受影响。
- 缺失的是：**用 24 维训练的 policy + 多智能体训练 harness + GPU 资源**。

### 2.4 低层行走（**真正的问题所在**）

- 链路：高层 `velocity_cmd` → `t1_walk.pt`（720 维本体感知 → 21 关节目标）→ PD 控制 → Genesis。
- **问题根因**：渲染视频里机器人"抖到走不了"，是 **`t1_walk.pt` 与机器人本体/物理/obs 尺度不匹配**，不是高层 ONNX。高层 ONNX 输出 `velocity_cmd` 正常。
- **临时修复（Strategy A）**：`soccer_env_hierarchical.py` 的 `use_rule_walk` 开关 —— 跳过 `t1_walk.pt`，改用确定性步态（静止 = 默认站姿；有速度 = 相位驱动左右腿摆动，幅值随速度，**绝不抖**）。`match_worker.py` 已加 `--rule-walk` / `--no-rule-walk`（**默认开**）。

---

## 3. 当前阻塞点（修正版）

| 优先级 | 问题 | 状态 |
|---|---|---|
| **P0（真正）** | 低层行走稳定性：`t1_walk.pt` 抖动 | Strategy A rule-walk 已实现，待实例验证 + 跑 3v3 |
| P1 | 多智能体训练 harness + 用 24 维 obs 训练协同 policy + GPU | 脚手架有，缺 harness + 算力 |
| P2 | 把 24 维 policy 接回 `match_worker` / `match_coordinator` | 已有 obs 维度自动检测逻辑，接回成本低 |

> ⚠️ 原始文档把"编排层未加载 ONNX"列为 P0/P2 阻塞——**这已不成立**（见 2.2）。真正卡点是 P0 的低层行走。

---

## 4. 短期落地方案（方案 A，已落地）

- 在 `soccer_env_hierarchical.py` 加 `use_rule_walk` 开关 + `_rule_walk_actions()`，跳过 `t1_walk.pt`。
- `match_worker.py` 加 `--rule-walk`（默认开）；`run_3v3_onnx.sh` 可全队启用。
- 改动小、不破坏 ONNX 高层路径，先拿到"能走"的 3v3 演示。

---

## 5. 验证思路

- **P0 单机器人稳定性**：`b_verify.py`（stance 6s + gait 30s，测 height / pitch / roll / 位移），输出 `P0_RESULT` + `VERDICT`。
- **3v3**：`run_3v3_onnx.sh` 全队 `--rule-walk`，看机器人是否不抖、能追球、有进攻。

---

## 6. 中长期（竞赛级）

- **24 维 obs 训练协同 policy**：MAPPO / PPO 共享权重，课程式 1v1 → 2v2 → 3v3。
- **coop 奖励**：阵型保持、补位、传球、团队进球。
- **观测一致性**：训练侧归一化 / 坐标 / 噪声必须与部署侧强一致（RL 仿真→实物最大坑）。
- **平台适配**：Genesis `env.step(velocity_cmd)` → 官方 Booster Studio `Player.set_velocity(velocity_cmd)`。

---

## 7. 待办（按优先级）

1. 验证 rule-walk 单机器人不抖（P0）+ 跑一场不抖 3v3 演示。
2. 多智能体训练 harness + GPU（24 维 obs）。
3. 24 维 policy 接回编排层。
4. （可选）观测一致性对齐、仿真→实物迁移。

---

## 参考文档说明

- **用户原始结构化文档**：概念优先级正确，但"stub / 未接入 / 路径（`src/booster_agent`、`chase_v7`、`main.py`、`rl_playbook`）"已过时，勿照字面执行。
- **朋友《问题全景梳理.docx》**：通用 MARL 模板。其 `SharedRLPolicy` 架构图与真实 `policy.py` 方法结构一致（可作自检对照）；多智能体 harness / GPU 折中 / 三大坑（观测一致性、动作空间、时序同步）是有效前瞻 checklist。**但**："stub"前提错、"小型轮式机器人"形态错（实际是 T1 人形）、同样漏掉低层行走真问题。仅作 P1/P2 前瞻参考，且需先纠正两个前提。
