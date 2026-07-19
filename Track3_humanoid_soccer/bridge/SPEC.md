# 桥接蒸馏方案（Genesis → Booster）

> 定位：用户选定 **桥接蒸馏** 作为 AMD GPU / 黑客松 与 Booster 真赛的衔接方式。
> 目标：Genesis 在 AMD Radeon 云上训出的足球策略，提炼成 **可手写进 Booster `main.py` / `player.py` 的 heuristic 规则**，让一次 GPU 投入同时喂饱黑客松交付与真赛实战。

## 为什么权重不能直接搬

- Genesis 动作空间 = **关节角目标**（`control_dofs_position`），训出来的是"每条电机每帧的角度"。
- Booster 动作空间 = **`set_velocity` / `kick`**（高层行为指令），真机 T1 自己解算电机。
- 两者维度、语义、执行体全不同 → **RL 权重无法迁移到真机**。

## 桥怎么架（蒸馏，不是迁移）

RL 当"设计师"，Booster 当"执行者"：

1. **Genesis 训策略**：在 AMD 云上跑 `scripts/train.py`，env 在 `step()` 里把蒸馏观测写进日志（`bridge/genesis_logger.py`）。
2. **提取 heuristic**：`bridge/distill.py` 读日志，对"成功行为"拟合简单阈值（分位数），产出常量。
3. **写进 Booster**：把常量贴进 Booster `param.py`，并在 `main.py` / `player.py` 的对应位置用它们替代拍脑袋的数值。

## 映射表（Genesis 行为 → Booster 落点）

| Genesis 技能（reward 项） | Booster 落点（文件:符号） | 蒸馏产出常量 | 含义 |
|---|---|---|---|
| `approach_ball` / `ball_control` | `main.py:_act_normal` attacker 选择；`player.py:attack` | `ATTACKER_ENGAGE_DIST_M` | 离球多近才认定为"该上抢"（RL 学会的commit距离） |
| `ball_to_goal` / `goal_scored` | `player.py:_behind_ball` / `plan_kick` | `KICK_BEHIND_OFFSET_M`、`KICK_APPROACH_ANGLE_DEG` | 站在球后方多远距离、对准误差容忍 |
| `coop`（多机）角色涌现 | `main.py:_act_normal` guard/support 分配 | `GUARD_HOLD_DIST_M`、`SUPPORT_FOLLOW_DIST_M` | 后卫离己门保持多远、支援位跟进攻者多远 |
| `fall_penalty` / `recovery_bonus` | `player.py:ensure_ready` / `get_up` | `FALL_RECOVERY_PRIORITY` | 倒地后是否优先起身、放弃当前动作 |
| `energy_penalty`（避撞） | `player.py:_heading_clearance`（VFH） | `AVOID_LOOKAHEAD_M`、`AVOID_MIN_CLEAR_M` | 局部避障探距与最小余量 |

> 几何对齐：Genesis env 的 `field` 已改成 **14×9 / 门 2.6 / 圈 1.5**，与 Booster `ADULT_FIELD_DIMENSIONS` 一致，所以蒸馏出的空间阈值在真赛里直接成立，无需缩放。

## 文件

- `bridge/genesis_logger.py` — 蒸馏观测记录器（纯 Python，可被 `soccer_env.step()` 调用；tensor→float 安全）。
- `bridge/distill.py` — 提取器 + `--selftest`（合成数据离线验证 pipeline）。
- `bridge/booster_distilled.py` — 提取产物（常量，由 `distill.py` 生成，勿手改）。

## 用法

```bash
# 1) 云上训练时开启日志：在 train.py 里 env.distill_logger = DistillLogger("bridge/rollout.jsonl")
# 2) 训后提取：
python bridge/distill.py --log bridge/rollout.jsonl --out bridge/booster_distilled.py
# 3) 离线自检 pipeline 是否通：
python bridge/distill.py --selftest
```
