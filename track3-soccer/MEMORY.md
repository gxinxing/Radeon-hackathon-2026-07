# MEMORY — Track 3 Session State

> **读 SPEC 前先读本文件。** 本文件记录当前会话的全部诊断数据、决策、断点。
> 最后更新：2026-08-05 17:36 (Asia/Shanghai)

---

## 1. 项目一句话

AMD AI DevMaster Hackathon 2026 Track 3：在 AMD Radeon RX 7900 XT (gfx1100) 上用 Genesis 仿真训练人形机器人踢 3v3 足球。Deadline: 2026-08-06。

---

## 2. 远端实例连接

| 项 | 值 |
|----|-----|
| JupyterLab | `https://radeon-global.anruicloud.com/instances/<REDACTED>/lab` |
| Token | `<REDACTED>` |
| API 基址 | `https://radeon-global.anruicloud.com/instances/<REDACTED>/api/contents/` |
| 工作目录 | `/workspace` |
| GPU | AMD Radeon RX 7900 XT, 48GB VRAM, ROCm 7.2.1 |
| Genesis | 1.3.1 已安装, kernel 已缓存 |
| SSH | 被 anruicloud 网关拦截，只能走 JupyterLab API + WebSocket |

### 连接方式（已验证可用）

```python
# 1. REST API (文件读写)
curl -s -H "Authorization: token <REDACTED>" \
  "https://radeon-global.anruicloud.com/instances/<REDACTED>/api/contents/<path>?content=1"

# 2. WebSocket (执行代码) — 必须禁用代理
NO_PROXY="*" no_proxy="*" python3 -c "
from websocket import create_connection
ws = create_connection(ws_url, http_proxy_host='', http_proxy_port=0, sslopt={'cert_reqs': ssl.CERT_NONE})
"
# 注意: kernel WebSocket stdout 不回传！必须写文件再 GET 读回。
```

---

## 3. 已落地的代码修复（本地 + 远端一致）

| 修复 | 文件 | 远端行号 | 状态 |
|------|------|----------|------|
| ang_vel: `all_filtered` → `robot.get_ang()` | `scripts/soccer_env_3v3.py` `_build_low_level_obs_for_robot()` | L369 | ✅ |
| RigidOptions: 256→4096, 1e-5→1e-4, iter 50→100 | `scripts/soccer_env_3v3.py` | L199-203 | ✅ |
| Kick: KICK_DISTANCE 0.3→0.5, cooldown 1.0→0.5 | `scripts/soccer_env_3v3.py` | L51-53 | ✅ |
| dof_idx_local 替换 dof_start | `scripts/soccer_env_3v3.py` | L149 | ✅ |

---

## 4. 诊断数据（2026-08-05 实测）

### 4.1 30步验证（有 reset，term_pitch/roll=30°）

| 指标 | 值 | 说明 |
|------|-----|------|
| 机器人位移 | 0.338m (robot 0) | ✅ 机器人在动 |
| 球位移 | 0.0m | ❌ 从未触及球 |
| 最近距离 (step 30) | 0.662m | ❌ 未进入 KICK_DISTANCE(0.5m) |
| 步行速度 | ~0.022m/step ≈ 0.22 m/s | 慢但非零 |
| 跌倒 | 0 (step 1-14), reset at step 15 | step 15 某机器人倾斜 >30° |

### 4.2 遥测关键发现

```
Step 1-2:  obs 全零 (历史缓冲空) — 正常
Step 3:    ang_vel=[-0.067,-0.071,0.052] ✅ 非零 (Task-1 修复生效)
           grav=[-0.003,0.003,-0.999] ✅ 正确
           cmd=[0.5,0.0,0.0] ✅ Match 策略输出正确
Step 10:   robot at (-0.870,-0.164), dist=0.886 ← 在接近球
Step 15:   RESET — 某机器人 pitch/roll > 30°, 全场重置, obs 清零
Step 30:   robot at (-0.662,-0.018), dist=0.662 ← 15步走了0.338m
```

### 4.3 100步验证（禁用 reset）

| Step | 跌倒数 | 球位移 |
|------|--------|--------|
| 10 | 0 | 0.0m |
| 20 | **6** (全部倒地) | 0.0m |
| 30-100 | 6 (无法恢复) | 0.0m |

### 4.4 根因链

```
t1_walk.pt 在单机器人 env 能走 150步/0跌倒/6.4m
  → 但在 3v3 env (6机器人在同一场景) 只走 ~15步就倾斜 >30°
  → 原因: 6机器人物理交互产生接触扰动, walk model 从未训练过抗接触
  → 机器人到不了球 (1.0m 距离, 0.22m/s 速度, 15步=1.5s 只走 0.33m)
  → 球位移 = 0
```

### 4.5 速度指令诊断

- attacker_cmd: `[0.5, 0.0, 0.0]` — Match 策略输出正确
- all_cmds: 6 个机器人均有非零指令 — rule-based chase 工作正常
- dead-band: `abs(cmd) < 0.05 → 0` — 0.5 > 0.05 通过

---

## 5. SPEC 任务链状态

| Task | SPEC 要求 | 状态 | 结果 |
|------|----------|------|------|
| Task-1 | 修复 obs ang_vel | ✅ DONE | 本地+远端 py_compile PASS |
| Task-2 | 30步验证 ball_displacement > 0 | ❌ FAIL | 机器人移动0.338m但球位移=0 |
| Task-3 | 100步 3v3 视频 | ⏭️ 跳过 | Task-2 FAIL |
| Task-4 | 单机器人提交 + 3v3 known limitation | ✅ DONE | SPEC Go/No-Go #6 更新 |
| Task-5 | 推送 GitHub | ✅ DONE | `97ca0ac` pushed |

---

## 6. 评分预估

| 路径 | 机器人能力 | ROCm | 创新 | 应用 | 开源 | 总计 |
|------|----------|------|------|------|------|------|
| **B (当前)** 单机器人 | 22 | 18 | 17 | 14 | 8 | **79** |
| **A (目标)** 3v3跑通 | 25 | 18 | 18 | 16 | 9 | **86** |

差距 = **7 分**，唯一变量 = 3v3 是否能跑通。

---

## 7. 下一步：实现 rule_walk（SPEC Strategy A）

### 7.1 依据

- RECOVERY_PLAN: "选 A：规则行走兜底先行…1 天内达成踢完一场比赛的最低可演示目标"
- SYSTEM_CONTEXT: "use_rule_walk 开关 — 跳过 t1_walk.pt，改用确定性步态…绝不抖"
- COMPETITION_ACCEPTANCE: 要求 "reliable independent control and role assignment" — 不要求 walk model 必须是神经网络
- b_verify.py 已为此设计 (`env_cfg['use_rule_walk'] = True`)，但 `_rule_walk_actions()` **从未实现**

### 7.2 实现方案

在 `scripts/soccer_env_3v3.py` 中：

1. `__init__` 加 `self.use_rule_walk` 标志
2. 新增 `_rule_walk_actions(self, cmd, robot_idx)` 方法：
   - 静止 (cmd≈0): 返回默认站姿 (all_default_dof_pos)
   - 有速度: sin 相位驱动左右腿摆动，幅值随速度，保持手臂默认
3. `step()` 的低层循环加分支：
   ```python
   if self.use_rule_walk:
       joint_actions = self._rule_walk_actions(cmd, i)
   else:
       joint_actions = self._run_walk_model(low_obs)
   ```
4. `run_booster_match.py` 设置 `env.use_rule_walk = True`

### 7.3 关键接口

```
_run_walk_model(obs_720) → joint_actions (num_envs, 21)  # 现有
_rule_walk_actions(cmd, i) → joint_actions (num_envs, 21)  # 新增
_low_level_step_robot(i, joint_actions)  # 不变，接受 21 维关节目标
```

### 7.4 T1 机器人关节结构

- 21 个 policy joints (POLICY_JOINT_NAMES in soccer_env_v4.py)
- 核心: 腿部 12 关节 (左右各 Hip yaw/pitch/roll + Knee + Ankle pitch/roll)
- 手臂 9 关节 (保持默认即可)
- action_scale = 0.25, target = action * 0.25 + default_pos

---

## 8. 远端关键文件

| 文件 | 路径 | 用途 |
|------|------|------|
| 3v3 env | `/workspace/scripts/soccer_env_3v3.py` | 主环境 |
| 运行脚本 | `/workspace/run_booster_match.py` | 比赛入口 |
| 策略 | `/workspace/strategy/{param,player,match}.py` | Booster 风格策略 |
| 配置 | `/workspace/configs/hierarchical_agent.yaml` | env/reward/obs 配置 |
| walk model | `/workspace/models/pretrained/t1_walk.pt` | 冻结行走模型 (2.1MB) |
| ONNX | `/workspace/models/chase_v8_policy.onnx` | 高层追球策略 |

---

## 9. 已踩的坑

1. **WebSocket stdout 不回传** — 必须写文件再 GET 读回
2. **本地代理掐断 TLS** — `HTTP_PROXY=127.0.0.1:7890` 会断开 WebSocket，必须 `NO_PROXY="*"`
3. **kernel 残留占满 GPU** — 每次执行前必须清理旧 kernel
4. **env.step() 只接受 robot 0 指令** — robots 1-5 在 step() 内部用 `_compute_rule_actions()`
5. **reset 触发条件** — 任意机器人 pitch/roll > 30° → 全场 reset → obs 清零 → 进度丢失
6. **rule_walk 从未实现** — RECOVERY_PLAN 和 b_verify.py 都引用了它，但 `_rule_walk_actions()` 方法不存在于任何 env 文件中

---

## 10. Git 状态

| 项 | 值 |
|----|-----|
| 分支 | `codex/track3-final-acceptance` |
| 最新 commit | `97ca0ac docs(track3): Task-4/5 — update SPEC with 3v3 diagnosis results` |
| 远端 | `https://github.com/gxinxing/Radeon-hackathon-2026-07.git` |
| Go/No-Go | 10/10 ✅ (第6项标注 3v3 known limitation) |

---

## 11. 会话断点

**当前断点**: rule_walk v3 已跑通 3v3 球位移=5.26m。待修复视频渲染(env.cam 缺失)。

### rule_walk v3 实测结果 (2026-08-05)
- 100步: **1 kick, ball_disp=5.26m, robot_disp=0.86m**
- fallen=6 (步态幅度大导致倒地, 但球被踢走了)
- frames=0 (env.cam 属性不存在, 需要用 scene.camera 或其他方式渲染)
- 球被踢后持续滚动 (kick impulse 有效)

### 待修复
1. **视频渲染**: env 没有 cam 属性, 需要检查 scene.camera 或 Genesis 渲染 API
2. **减少摔倒**: 步态幅度可适当降低, 但当前 ball_disp=5.26m 已满足 Task-3 要求(>0.5m)
3. **增加 kicks**: 当前只有 1 kick, 可通过增加步数或调整 KICK_DISTANCE 来增加

**恢复后应做**:
1. 读 SPEC.md + 本 MEMORY.md
2. 修复视频渲染 (查找 Genesis 渲染 API)
3. 跑 100 步 3v3 + 视频
4. 下载视频 + JSON
5. 更新 SPEC + MEMORY + README
6. 推送 GitHub
