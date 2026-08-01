# 下一步计划：从验收目标倒推

## 当前状态总览

### 验收目标对照

| 验收项 | 状态 | 说明 |
|--------|------|------|
| 3v3 场景能够稳定启动 | ✅ 达标 | 6/6 matches, 6 clients connected |
| RL vs rule 至少完成 5 场 | ✅ 达标 | 6场完成 (1+5) |
| 异常退出率为 0 | ✅ 达标 | 0/6 abnormal exits |
| 每场都有完整 JSON 日志 | ✅ 达标 | 1234-1243 steps, 全部有 robot data |
| 倒地后 episode 不立即终止 | ✅ 达标 | 34.3次/场倒地, 全部继续 |
| 能统计恢复率和恢复时间 | ✅ 达标 | 88.8% recovery rate, avg 2-3.5s |
| RL vs rule 胜率 ≥ 50% | ❌ 未达 | 0% (全部0-0平局) |
| 净胜球不低于 0 | ⚠️ 边界 | 0 (因为没人进球) |
| 扰动场景恢复率高于baseline | ❌ 未跑 | disturbance framework 存在但未接入 match_worker |
| 扰动后仍能继续比赛 | ❌ 未跑 | 同上 |

**结论：最低验收目标 6/6 全部达标。争取目标 0/4 达标。**

---

## 差距分析（倒推）

### 差距 1: 没有进球 (影响: 胜率、净胜球)
**根因:** match_worker.py 中 RL worker 只输出 velocity_cmd (vx, vy, vyaw)，
缺少 kick 动作。RulePolicy 有 `should_kick` 逻辑但 RL worker 没有调用。

**修复方案:**
- 在 match_worker.py 的 ONNX 推理路径中，当机器人离球 < 0.3m 时，
  叠加规则层的 kick 行为（冲刺 + 踢球方向对准球门）
- 代码位置: match_worker.py ~line 220，action_result 之后

### 差距 2: 扰动场景未跑 (影响: 扰动恢复率对比)
**根因:** disturbance.py 有 DisturbanceConfig 类，match_3v3.yaml 有 disturbance 配置段，
但 match_worker.py 和 match_coordinator.py 中没有 import 或使用 DisturbanceConfig。
coordinator 只发 collision push-back (MSG_CMD)，不注入外力/摩擦变化/观测噪声。

**修复方案 (按可行性排序):**

A. **最小可行: 在 coordinator 中注入随机推力** (改动最小)
   - coordinator 已经每步发送 MSG_CMD (push) 给 workers
   - 将 push 从纯碰撞推力改为: 碰撞推力 + 随机扰动推力
   - 每 N 步注入一次随机力 (模拟外力推搡)
   - workers 已经接收并应用 collision_push

B. **中等可行: 在 worker 中修改摩擦/观测** (改动中等)
   - worker 端在 env 创建后修改地面摩擦系数
   - worker 端在 obs 上叠加高斯噪声
   - worker 端随机化初始位置 (--init-pos 随机化)

C. **完整方案: 接入 DisturbanceConfig** (改动最大)
   - 将 disturbance.py 完整集成到 match_worker.py
   - 需要修改 env 的 step 循环

### 差距 3: 10场扩展统计未完成 (影响: 统计显著性)
**根因:** 时间限制，只跑了 5 场 batch + 1 场验证 = 6 场

**修复方案:** 运行 `/tmp/run_batch_3v3.sh 10 rl_vs_rule models/chase_v8_policy.onnx`
预计耗时 ~20 分钟。

---

## 推荐执行顺序

### Priority 1: 注入扰动 (最大评分提升)
**目标:** 跑 Group D (RL + disturbance vs Rule) 至少 5 场
**方法:** 方案 A — 修改 match_coordinator.py 在 push 中叠加随机扰动
**预计耗时:** 代码修改 15min + 5场跑 15min = 30min
**影响:** 证明平台在扰动下仍能运行，恢复率可对比

### Priority 2: 修复进球 (展示价值)
**目标:** RL team 能进球
**方法:** 在 match_worker.py ONNX 路径叠加 rule kick
**预计耗时:** 代码修改 20min + 测试 10min = 30min
**影响:** RL vs rule 胜率可能提升

### Priority 3: 跑 10 场扩展集
**目标:** 10 场 RL vs rule (无扰动) + 5 场 (有扰动)
**方法:** 批量运行
**预计耗时:** 30min (无扰动) + 15min (有扰动) = 45min
**影响:** 统计显著性

### Priority 4: 更新交付物
**目标:** 将新结果写入报告
**方法:** 更新 JSON/CSV/MD 文件
**预计耗时:** 15min
