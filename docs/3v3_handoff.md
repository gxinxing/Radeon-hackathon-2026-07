# 赛道三「具身机器人」3v3 — 交接文档

> 适用：下一个 agent / GPU 云实例回来后继续。本文档自包含，读完即可接手。
> 生成时间：2026-07-28。项目根：`~/Documents/01_AI and Code Development/​Radeon-hackathon-2026-07`
> （⚠️ 文件夹名带零宽字符，shell 用 tab 补全或精确路径）

## 0. 一句话现状

训练**已收敛**（v6: 14 进球、std 0.63、ep_len 239），但瓶颈在**单智能体追球范式 + 3v3 编排零团队策略层**。
本次在无 GPU 条件下完成了**策略层 + 多智能体观测 + 协作奖励**三项优化，**全部默认 opt-in、向后兼容 v6（19 维 checkpoint）**，**89 项 CPU 单测全 PASS**。

---

## 1. 本次已落地的改动（已完成、已测）

### 1.1 P0 — 策略大脑 `src/match_3v3/strategy.py`（纯规则，直接套 v6，无需重训）
- `SmartRoleAssigner`：带迟滞 + 守门员粘性的角色分配。**修复角色抖动 bug**（原 `RoleAssigner` 每 50 步纯 argmin，两人接近时攻击手反复横跳）；**倒地球员不再被指派为追球手**。
- `PassPlanner`：**修复 lane-clearance 过严 bug**（贴身逼抢者 < 0.8m 不再把整条传球线判为封堵，被逼抢时也能传出）。
- `FormationTargets`：possession-aware 阵型（追球手去球 / 控球时防守者前插接应 / 丢球退守球-门连线 / 守门员贴门线）。
- `TeamBrain` / `PlayerCommand` / `build_hl_observation`：组合成每步给 3 机器人的 velocity command（vx, vy, wz），复用现有 3 维 action 接口。

### 1.2 P0 — 多智能体观测扩展 `soccer_env_hierarchical.py`
- `_multiagent_extra(inv_bq)`：19 维后追加 5 维 `[最近队友相对位姿(2) + 最近对手相对位姿(2) + 球权标志(1)]`，全部 body frame。
- 由 `env_cfg.multiagent_obs` 开关控制，**默认 false → obs 仍是 19，v6 checkpoint 完全有效**；true → 24 维，下一轮训练启用。
- harness 契约：每步必须喂 `env.teammate_pos`(N×2×3) 与 `env.opponent_pos`(N×3×3)（world frame xyz）。
- 数学与已测 numpy 参考 `src/match_3v3/multiagent_obs.py` 1:1 对齐。

### 1.3 P1 — 协作奖励 `reward.py` + `configs/hierarchical_agent.yaml`
- 新增 `coop_hl` 任务，在 `chase_hl` 上加三项：`r_defensive_position`(0.5) / `r_support_position`(0.5) / `r_coop_goal`(10.0)。
- **仅在 `task=coop_hl` 且 `multiagent_obs=true` 且 obs 含团队几何时生效**，单智能体训练完全不受影响。
- `env.step()` 自动注入 `self_xy / ball_xy / attack_goal_xy / defend_goal_xy / in_possession / scored_my_team`（in_possession 用当前 post-step 位姿算，无滞后）。

### 1.4 测试 & 文档
- 本地 venv（隔离）：`numpy 2.5.1 + pytest 9.1.1`。`pytest tests/` → **89 passed**。
- `src/match_3v3/__init__.py` 已导出策略/观测模块。
- `docs/3v3_strategy.md`：多智能体训练接线手册（config 开关 + harness 契约）。

---

## 2. 关键事实 / 坑（接手前必读）

1. **torch 不在本地 venv**（也不该装 ROCm 版）。torch 训练路径靠"与已测 numpy 逻辑 1:1"保证正确性；本地只跑 numpy 端口测试。
2. **`SharedRLPolicy.compute()` 是 stub**：`src/match_3v3/policy.py` 永远走 rule 回退，加载的 checkpoint 被忽略——所谓"RL vs rule"实际是 rule vs rule。v6 模型**从未真正接入 3v3**。
3. **v6 是 19 维**，新多智能体策略是 24 维 → 开 `multiagent_obs` 后**必须从零重训，不可 resume v6**（维度不匹配）。
4. **策略层是纯规则、CPU 可跑、不依赖 Genesis**——可现在就接进编排循环验证，无需 GPU。
5. Track 2（Crypto Trading Agent）的 handoff 在 `track2-agentic-ai/docs/track2_handoff_prompt.md`（提交阻断项 A/E/B/C/D + DEFER 清单），与本赛道独立。

---

## 3. 下一步（按优先级）

### ★ 下一步 1（无需 GPU，立即做，性价比最高）
**把 `TeamBrain` 接进 3v3 编排循环，替换 `SharedRLPolicy` 的 stub。**
- 位置：`match_3v3.py` / `match_coordinator.py` 的每步策略调用点（原调用 `SharedRLPolicy.compute()` 处）。
- 做法：每个 step 用 `TeamBrain` 产出 3 机器人 command（追球手走 v6 推理，其余走阵型/传球 rule），而非三人同一条追球策略。
- 验证：CPU 即可跑一个无 Genesis 的编排仿真（用 `Scene3v3` 的状态推进），检查角色分配稳定、无横跳、被逼抢会传。
- 收益：现有 v6 模型在 3v3 里立刻获得补位/协防/传跑，**零训练开销**。

### ★ 下一步 2（GPU 回来 — 推理 / Sim2Sim）
- Booster Studio Sim2Sim（VNC 手动）：3v3 加载 `chase_v6_2048_policy.onnx`，用 `TeamBrain` 编排（不是 rule-only），录 `demos/sim2sim_3v3.mp4`。
- 确认 3 机器人不再"像三个陌生人"。

### ★ 下一步 3（GPU 回来 — 多智能体训练，天花板更高，可选）
1. `configs/hierarchical_agent.yaml`：`task: coop_hl` + `env.multiagent_obs: true`。
2. 编写/扩展 3v3 训练 harness：每步把 6 台机器人位姿拆成 `teammate_pos` / `opponent_pos` 喂给每个 env 实例（见 `docs/3v3_strategy.md` 契约）。
3. **从头重训**（勿 resume v6），产出 24 维策略。
4. 导出 24 维 ONNX，更新 Sim2Sim obs 维度与归一化。

### DEFER（ polishing，不阻塞 8/6 锁版）
- 高光集 demo（`render_goal_video.py` 已存在，扩成比赛提交片）。
- Sim2Real 域随机化配置（真机部署时用）。
- coop 奖励权重精调（先跑通看效果再动）。

---

## 4. Done 标准（建议）
- [ ] `TeamBrain` 接入编排，CPU 仿真角色稳定无横跳 ✓（下一步1）
- [ ] Sim2Sim 录出 `sim2sim_3v3.mp4`，3 机器人有可见协同 ✓（下一步2）
- [ ] （可选）24 维多智能体策略训练收敛、进球数 > v6 单智能体 ✓（下一步3）

## 5. 复跑验证
```bash
cd ~/Documents/01_AI\ and\ Code\ Development/​Radeon-hackathon-2026-07
/Users/simon/.workbuddy/binaries/python/envs/default/bin/python -m pytest tests/ -q
# 期望: 89 passed
```
