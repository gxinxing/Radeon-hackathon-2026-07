# Track 3 机器人"抖动/不会走"根因诊断 + 1.5 天作战图

> 生成时间：2026-08-04 03:00（GMT+8）
> 性质：基于本地真实代码（`track3-soccer/` + `BoosterStudioProjects/simon3v3`）的第一性原理诊断，不是空谈。

---

## 一、你现在看到的"训练正常、视频全抖"到底是什么问题

我通读了 `soccer_env_3v3.py`、`policy.py`、`rl_playbook.py`、`render_3v3_match.py`、训练 config，结论很明确：

**抖动不是高层策略（chase ONNX）的问题，是底层行走模型 `t1_walk.pt` 与"本体/物理/观测尺度"不匹配导致的关节指令噪声。**

### 控制链路（两层架构）
```
高层策略(ONNX chase, 19维obs → 3维速度指令 vx,vy,wz @10Hz)
        │  velocity command
        ▼
底层行走模型 t1_walk.pt (720维obs → 关节目标角度)
        │  joint targets
        ▼
Genesis/Mujoco 物理引擎执行 (DECIMATION=10 → 底层 50Hz)
```

### 三个致命的不匹配证据（代码里白纸黑字）

| # | 不匹配点 | 代码位置 | 后果 |
|---|---------|---------|------|
| 1 | **`t1_walk.pt` 不是在你的 K1/T1 本体上训的** | `models/pretrained/t1_walk.pt` 是通用预训练行走模型；底层 obs 是 720 维 = 10帧×(ang_vel3+grav3+cmd3+dof_pos21+dof_vel21+last_act21)，**21 个电机映射对你的机器人构型必须一一对应** | 关节目标角度错配 → 一迈步就抖、站不稳 |
| 2 | **训练 obs 用真实滤波速度/IMU，部署用"位姿差分"估速度** | `rl_playbook._build_observation` 注释自己写了 `zeros here would be a major obs mismatch`；lin_vel/ang_vel 全靠有限差分，**且没做训练时的滤波** | obs 分布偏了 → ONNX 输出速度抖动 → 底层 `t1_walk` 收到高频振荡指令 |
| 3 | **没有"渲染一致"验证门禁** | `render_3v3_match.py` 只检查 `frames<100 / std<0.01 / nan`，**完全没检查机器人是否站得稳、是否抖动** | 视频能生成、reward 日志正常，但根本没人核验"动作质量" |

### 为什么"训练日志正常"
- 训练日志的 reward 是**高层**指标（接近球、把球推往球门），高层策略确实学到了 chase；
- 但**底层行走质量的抖动**在 reward 里被 `action_rate=-1.0`（惩罚突变）这种软项稀释，且训练时底层 `t1_walk` 是 frozen、在训练环境（真机参数）里跑，和**部署/demo 渲染用的本体参数不一致**就暴露了。

---

## 二、1.5 天达成"机器人踢完一场比赛"的作战图

按"先能站→再能走→再能踢→能踢整场"分层，每阶段有**硬退出条件**。目标不是"RL 完美"，是"**用最稳的路径让 6 个机器人在场上跑完整场 90 秒不倒、能追球、偶尔进球**"。

### 阶段预算（36 小时 ≈ 1.5 天）

| 阶段 | 目标 | 退出条件（必须真满足） | 预算 |
|------|------|---------------------|------|
| **P0 本体对齐** | 确认/替换 `t1_walk.pt` 为**你的机器人构型**对应的行走模型；统一 obs 尺度与训练一致 | 单机器人闭环 10s 在 Genesis 里**站得稳、抖动 std<阈值** | 4h |
| **P1 单机器人走** | 高层 + 底层在 Genesis 渲染里稳定行走、追球 | render_1v1 视频里机器人**连续走 20s 不抖、能接近球** | 6h |
| **P2 2v2 稳定** | 两队各 2 机器人，无碰撞自杀、能踢几脚 | verify 脚本：2v2 跑 30s，倒地次数=0、球进区≥1次 | 8h |
| **P3 3v3 整场** | 6 机器人完整一场（90s 模拟） | `verify_g0` 连续 3 场干净赛：无 RPC-flood、无倒地崩溃、比分有变化 | 12h |
| **P4 渲染+交付** | 产出 demo 视频 + 元数据 + 评测报告 | 视频过质量门禁（稳+能踢）；technical_report 更新 | 6h |

> 关键策略：**P0 用规则行走兜底（RulePolicy 直接给关节 PD 目标）先把"能站能走"跑通**，再叠加 RL。不要在"本体没对齐"的情况下花 12h 调 RL——那正是你现在卡住的地方。

### P0 决策已定（2026-08-04 03:22，用户拍板走 A）
- **选 A：规则行走兜底先行**。deploy 端暂不依赖 `t1_walk.pt` 的 ONNX 高层，改用 `policy.py` 的 `RulePolicy`（几何规则直接给速度指令 vx,vy,vz）+ 一个**固定步态/PD 关节控制**兜底，先把"6 机器人能站稳走完一场"跑通。
- 这是 1 天内达成"踢完一场比赛"**最低可演示目标**的最快路径。等整场跑通保底后，再评估是否重训真机 `t1_walk`（路径 B）作为增强。
- **规则行走兜底实现路线**（P0 节点动作，待在 Radeon 上 verify）：
  1. 在 `soccer_env_3v3.py` 的 `_low_level_step_robot` 增加分支：若 `use_rule_walk=True`，跳过 `_run_walk_model`（t1_walk），改由 `RuleWalk` 根据当前 `velocity_cmd`(vx,vy,vz) 直接算 21 个关节目标（如：步态相位 sin + 速度→步频/步幅映射 + 默认站立姿态）。
  2. 高层仍可用 `RulePolicy`（几何追球）或已训好的 `chase_vN` ONNX（obs 已对齐则更好）；底层恒定规则行走，消除"行走模型错配抖动"。
  3. verify：Genesis 单机器人 10s 闭环，记录 base 高度 std 与 roll/pitch std，阈值内才过 P0。

---

## 三、Graph Engine 设计（自动化编排）

把上面 5 个阶段做成 **DAG 状态机**，每个节点 = 一个可验证、可回滚的步骤。引擎负责：
1. 按依赖顺序唤醒节点
2. 每个节点执行后跑 `verify()`，不过门禁不进下一节点
3. 记录 checkpoint，断点可续
4. 实时 dashboard 给你看进度（Ego Lite 浏览器里能看着跑）

### 节点图（DAG）

```
[check_env] ──▶ [align_walk] ──▶ [single_stable]
                                      │
                                      ▼
                                 [two_v_two]
                                      │
                                      ▼
                                 [three_v_three] ──▶ [render_and_report]
```

节点说明（每个节点都在 Radeon 实例上跑，通过 Ego Lite/JupyterLab 驱动）：

| 节点 | 动作 | verify（可读文本输出，不发截图） |
|------|------|-------------------------------|
| check_env | 查 GPU(rocm-smi)、代码同步、模型清单 | 打印 GPU 状态 + 模型 sha + 代码 commit |
| align_walk | 确认 t1_walk 对应本体；否则切 RulePolicy 兜底行走 | 单机器人 10s rollout 的 base 高度 std |
| single_stable | 1v1 Genesis 渲染，调 obs/deadzone/clip | 视频 metadata 的"稳定度"字段 |
| two_v_two | 2v2 评估脚本 | 倒地次数、进球数 |
| three_v_three | 3v3 + verify_g0 ×3 | RPC-flood 次数、干净赛判定 |
| render_and_report | 出 demo 视频 + 报告 | 文件清单 + 质量门禁结果 |

### 为什么用 Graph Engine 而不是"一步步手动"
- 你能**从浏览器实时看 DAG 走到哪、哪个节点失败**（符合你说的"这样我也能看到过程"）
- 失败的节点不污染后续，自动停在门禁前，**不浪费 GPU 时间**
- 断点续跑：实例重启/代码改了，从 checkpoint 接着跑，不重头来

---

## 四、立即要做的第一刀（P0 关键决策）

**不要在没对齐本体的情况下继续调 RL。** 两个选择，必须二选一：

- **A（推荐，最快）**：deploy 端暂时**不用 `t1_walk.pt` 的 ONNX 高层**，改用 `policy.py` 里已写好的 `RulePolicy`（几何规则直接给速度指令）+ 一个**规则行走兜底**（固定步态/PD 控制），先把"6 个机器人能站稳走完一场"跑通。这能 1 天内达成"踢完一场比赛"的**最低可演示目标**。
- **B（更优但更慢）**：找到/重训一个**真正对应你机器人构型**的 `t1_walk`，再接 RL 高层。需要 GPU 跑 base 行走训练（你之前 GPU 云不可用，是阻塞点）。

我先按 **A** 在 Radeon 上验证"规则行走兜底能不能让单机器人站稳"，这是决定后面一切的前提。如果 A 在 Genesis 里能站稳，整条路就通了。

---

## 五、Ego Lite 浏览器驱动说明（重要约束）

- 外部 SSH 被 anruicloud 网关拦截（已诊断），走 **JupyterLab（实例自己的 shell）** 才是正路。
- **JupyterLab Terminal 是 canvas 渲染，当前模型读不到截图** → 改用 **Notebook cell 执行**（`print()` 输出在 DOM 里，能读到文本）。
- 所以 Graph Engine 的每个节点在 Radeon 上以 **`.ipynb` cell / `!python -c`** 形式跑，输出走文本，绕过截图不可读的限制。
