# 实例代码修复简报（Instance Fix Brief）

> 本地 `track3-soccer/scripts/soccer_env_3v3.py` 已验证修复（py_compile OK）。
> 本简报用于把同样的修复 port 到实例真代码
> （`/workspace/amd-physical-ai-soccer/` 与 `/workspace/radeon-repo/`，
>  结构为 `src/match_3v3/`，与本地 `scripts/` 仅"同架构"）。

## 修复 1 — 方案 A 接触参数（双机器人失稳主因候选）
目标：找到 `RigidOptions(...)` 所在文件:行号，确认当前值。
- 若实例仍是 **`max_collision_pairs=256, tolerance=1e-5`**（旧值）→ 这正是双机器人贴脸相撞后基座被推、walk 720维obs突变失稳的根因，必须改成：
  - `max_collision_pairs=4096`
  - `tolerance=1e-4`
  - `iterations=100`
- 若实例已 = 4096/1e-4/100 → 方案 A 已落地，失稳另有原因（见下方"高层避让"补充）。

## 修复 2 — 踢球逻辑（单机器人能走不能踢）
在实例对应 env 文件中做等价修改：
1. 新增常量（放在字段常量区）：
   ```python
   KICK_DISTANCE = 0.5   # 旧硬编码 0.3，humanoid 脚够到球约 0.45-0.5m 中心距
   KICK_IMPULSE  = 3.0
   KICK_COOLDOWN = 0.5   # 旧 1.0
   ```
2. 规则追球（右队 / rule-based chase）当前 bug：
   `dist <= 0.3` 时机器人**改朝球门跑、放弃追球** → 永远保持不了 <0.3m 接触 → 踢球几乎不触发。
   改为"始终追球、近身提速保持接触"：
   ```python
   close = dist.squeeze(-1) < KICK_DISTANCE
   speed = torch.where(close, torch.tensor(0.55), torch.tensor(0.4))
   vx = direction[:, 0] * speed
   vy = direction[:, 1] * speed
   wz = torch.clamp(torch.atan2(to_ball[:, 1], to_ball[:, 0]) * 0.3, -0.5, 0.5)
   ```
3. `_execute_kick`（rule-based 对所有机器人触发）两处硬编码 0.3 改成 `KICK_DISTANCE`；
   `impulse = goal_dir_norm * 3.0` → `* KICK_IMPULSE`；
   `self.kick_cooldown[can_kick, i] = 1.0` → `= KICK_COOLDOWN`。

## 补充 — 若方案 A 已落地仍失稳（高层避让）
walk 模型是单机器人训的，没做机器人间接触鲁棒。可加：
- 高层速度指令里加机器人间斥力/避让（attacker 间保持 ≥1.5m）；
- 或把两 attacker 初始间距从 2m 拉开到 ≥3m（`LEFT_START/RIGHT_START`）；
- 或 C：在带接触扰动的数据上重训/微调 walk（成本高，留作 P2）。

## 验证
- 本地：`python3 -m py_compile <改后文件>` 已通过。
- 实例：改后跑一段无渲染/短时长 eval，观察 6 机器人能否站稳 + 单机器人能否把球踢向球门。
