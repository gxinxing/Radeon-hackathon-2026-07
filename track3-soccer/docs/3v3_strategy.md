# 3v3 具身机器人 — 策略层 / 多智能体观测 / 协作奖励

> 赛道三"具身机器人"在无 GPU 情况下可推进的部分（已全部落地，CPU 单测 89 项全 PASS）。
> 代码默认 **向后兼容 v6（19 维）checkpoint**，所有多智能体扩展均 **opt-in**。

## 1. 策略大脑（纯规则，直接套在现有 v6 模型上，无需重训）

文件：`src/match_3v3/strategy.py`（CPU 可测，无需 Genesis）

- `SmartRoleAssigner` — 带迟滞 + 守门员粘性的角色分配。修复了原 `RoleAssigner`
  的**角色抖动 bug**：每 50 步纯 argmin 重分配，两人距离接近时攻击手反复横跳。
  新逻辑：追球手一旦锁定，除非队友近 `switch_margin` 或自己倒地才换手；守门员
  除非倒地否则不动（避免中途换将漏空门）。**倒地球员不再被指派为追球手。**
- `FormationTargets` — 阵型目标点（ possession-aware）：追球手去球、控球时防守者
  前插做接应、丢球时退守球-门连线、守门员贴门线跟球 y。
- `PassPlanner` — 被逼抢时给开放队友传跑。**修复了 lane-clearance 过严 bug**：
  贴身逼抢者（距持球人 < `passer_clear_radius`）不再把整条传球线判定为"被封堵"，
  被逼抢时也能找到出球线路。
- `TeamBrain` / `PlayerCommand` / `build_hl_observation` — 把以上组合成每步给 3 个
  机器人的 velocity command，复用现有 3 维 action 接口（vx, vy, wz）。

**怎么用**：在 `match_coordinator` / `match_3v3` 的 step 循环里，用 `TeamBrain`
替换原 `SharedRLPolicy` 的 stub（当前永远走 rule 回退，checkpoint 被忽略）。这样
v6 模型立刻获得"补位 + 协防 + 传跑"，Sim2Sim / demo 立刻变强。

## 2. 多智能体观测扩展（19 → 24 维，训练时生效）

文件：`soccer_env_hierarchical.py` + `configs/hierarchical_agent.yaml`

- `_multiagent_extra(inv_bq)`：在 19 维后追加 5 维
  `[最近队友相对位姿(2) + 最近对手相对位姿(2) + 球权标志(1)]`，全部 body frame。
- 由 `env_cfg.multiagent_obs` 开关控制，**默认 false** → obs 仍是 19，v6 checkpoint
  完全有效；设为 true → obs 变 24，下一轮训练直接启用。
- 训练 harness 契约：每步必须给 env 喂
  `env.teammate_pos` (num_envs×2×3) 与 `env.opponent_pos` (num_envs×3×3)
  （world frame xyz），`_multiagent_extra` 才能算出扩展维度。

数学逻辑与已测 numpy 参考 `src/match_3v3/multiagent_obs.py` 1:1 对齐。

## 3. 协作奖励（3v3，训练时生效）

文件：`reward.py` + `configs/hierarchical_agent.yaml`

- 新增 `coop_hl` 任务，在 `chase_hl` 基础上加三项：
  - `defensive_position`（0.5）：本方丢球时，机器人待在"球-本门"连线靠门一侧给奖励
    （zonal 防守），控球时该项归零。
  - `support_position`（0.5）：本方控球时，非持球者前插到开阔接应位给奖励，贴脸
    挤占持球者则惩罚。
  - `coop_goal`（10.0）：团队进球时 3 人共享奖励，让支援者学会"做球"而非看戏。
- 这些项 **仅在 `task=coop_hl` 且 `multiagent_obs=true` 且 obs 含团队几何字段时生效**，
  单智能体训练完全不受影响。env 在 `step()` 里自动注入 `self_xy / ball_xy /
  attack_goal_xy / defend_goal_xy / in_possession / scored_my_team`。

## 4. 开多智能体训练的步骤（GPU 回来后）

1. `configs/hierarchical_agent.yaml`：
   - `task: coop_hl`
   - `env.multiagent_obs: true`
2. 编写/接入 3v3 训练 harness：每步把 6 台机器人的位姿拆成
   `teammate_pos` / `opponent_pos` 喂给每个 env 实例。
3. 训练产出的策略 obs 为 24 维 → 导出/Sim2Sim 时用对应 24 维 ONNX 与 obs 归一化。
4. 推理端 `src/match_3v3` 的 `TeamBrain` 负责角色/阵型/传跑编排。

## 5. 测试

```
pytest tests/ -q      # 89 passed（strategy / multiagent_obs / coop_reward）
```
所有逻辑均有 numpy 端口测试，无需 Genesis / torch 即可验证。
