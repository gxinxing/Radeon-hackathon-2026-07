# RoboCup + Booster 官方基线借鉴指南 — 策略与设计参考

> 本文档记录从 RoboCup 3D Simulation League 代码和 Booster 官方 3v3 基线中提取的可借鉴内容，
> 用于指导 Track 3 项目的后续优化和 3v3 比赛实现。
>
> 来源：
> - RobocupGym (Michael-Beukman): https://github.com/Michael-Beukman/RobocupGym
> - UT Austin Villa op2: https://github.com/robocup3d/op2
> - RoboCup 3D Simulation League Rules 2025
> - **Booster 官方 3v3 基线 Agent**: `/Users/simon/BoosterStudioProjects/simon3v3-simple-baseline`

---

## 1. 奖励函数设计（优先级：高，改动小）

### RoboCup 的做法

RoboCup 的 reward 设计极其简洁，核心就三个维度：

```python
# RobocupGym: env_simple_kick.py
def _get_reward(self, action):
    if self.has_completed:
        return np.linalg.norm(current_ball_pos - start_ball_pos)  # 球位移
    return 0

# RobocupGym: velocity_kick.py (进阶版)
def _get_reward(self, action):
    if self.has_completed:
        dist = current_ball_pos[0] - last_ball_pos[0]    # X方向位移
        vel = (xv**2 + zv**2) ** 0.5                      # 球速
        y_penalty = abs(y_pos)                            # Y偏差惩罚
        return a * dist + b * vel - c * y_penalty         # 可调系数
    return 0
```

```cpp
// op2: optimizationbehaviors.cc — 踢球优化 fitness
double fitness = distance;           // 球X方向移动距离
if (backwards || distance <= 0.1)    // 向后踢或距离不足
    fitness = -100;                  // 严重惩罚
// 跌倒
if (worldModel->isFallen())
    totalFitness += -1;              // 小惩罚 + 终止
```

### 我们当前的问题

我们的 `reward.py` 有 8 个加权项，`rl_scoring.md` 有 56 个评分项。
RoboCup 的经验告诉我们：**简洁的 reward 更容易训练出有效行为**。

### 借鉴建议

| 项目 | 当前 | 建议改为 | 依据 |
|------|------|---------|------|
| 主 reward | 8维加权 | `ball_displacement` 为主 | RoboCup: 球位移是核心指标 |
| 跌倒惩罚 | -5 (reward.py) / -15~-25 (评分) | -5 + 终止 episode | RoboCup: 跌倒=终止+小惩罚 |
| 方向惩罚 | 无 | 球向后移动 = -大分 | RoboCup: backwards = -100 |
| 速度奖励 | 无 | `+ ball_speed * 系数` | RoboCup: velocity_kick |
| Y偏差惩罚 | 无 | `- abs(ball_y_offset)` | RoboCup: 鼓励直线踢球 |
| 评分表 | 56项 | 简化到 10 项以内 | RoboCup: 极简设计 |

---

## 2. Obs 设计（优先级：中，改动中等）

### RoboCup Player 状态

RoboCup 的 `player.py` 包含以下信息：

```python
# RobocupGym: player.py
class Player:
    # 位置与朝向
    real_pos_x, real_pos_y, real_pos_z        # 全局位置
    real_angle_x, real_angle_y, real_angle_z  # 朝向角度（欧拉角）
    
    # 球信息
    real_ball_pos                              # 球全局位置
    ball_velocity                              # 球速度 ← 我们缺这个
    
    # 关节状态
    joint_angles[22]                           # 关节角度
    joint_speeds[22]                           # 关节速度
    
    # 力传感器
    left_foot_force, right_foot_force          # 足底力 ← 我们缺这个
    
    # 身体姿态
    is_fallen                                  # 是否跌倒
    gyro_rate[3]                                # 陀螺仪 ← 我们用 filtered_ang_vel 替代
    
    # 比赛状态
    play_mode                                   # 比赛模式（开球/比赛中/进球后）
    side                                       # 球场方向（左/右）
```

### 我们的 19-dim obs

```
ball_pos(3) + ball_vel(无) + goal_dir(3) + dist_to_ball(1) + dist_to_goal(1)
+ base_ang_vel(3) + projected_gravity(3) + last_cmd(2)
= 19 dims
```

### 缺失项

| 缺失信息 | RoboCup 有 | 影响 | 补充难度 |
|---------|-----------|------|---------|
| 球速度 | ✅ `ball_velocity` | 无法预判球轨迹 | 低（Genesis 有 `ball.get_vel()`） |
| 机器人 yaw | ✅ `real_angle_z` | 不知道面朝哪个方向 | 低（从 quat 提取） |
| 队友位置 | ✅ 相对坐标 | 无法配合 | 中（3v3 需要） |
| 对手位置 | ✅ 相对坐标 | 无法避障 | 中（3v3 需要） |
| 足底力 | ✅ `foot_force` | 无法判断支撑脚 | 高（Genesis 可能不支持） |

### 借鉴建议

短期（3v3 修复）：加球速度到 obs，从 quat 提取 yaw
长期（3v3 训练）：加队友/对手相对位置（19→24 dim，已有代码支持）

---

## 3. 行为决策（优先级：高，收益大）

### RoboCup 的决策逻辑

```cpp
// op2: naobehavior.cc — act() 方法
void NaoBehavior::act() {
    // 1. 根据 play_mode 决定行为
    switch (worldModel->getPlayMode()) {
        case PM_PLAY_ON:    // 比赛中
            // 2. 判断谁该追球（最近者追球，不是固定角色）
            if (I_am_closest_to_ball()) {
                chase_ball();
            } else {
                maintain_formation();  // 保持阵型
            }
            break;
        case PM_KICKOFF:    // 开球
            go_to_kickoff_position();
            break;
        // ...
    }
    
    // 3. 碰撞避免（三级）
    VecPosition target = getTarget();
    target = avoidCollision(target);  // 前方扇形区域检测
    
    // 4. 头部追踪球
    setHeadAngle(ball_direction);
    
    // 5. 调用技能
    selectSkill();  // walk / kick / turn / getup
}
```

### 关键设计

1. **最近者追球**：不是固定前锋追，而是计算全队谁离球最近，最近者追
2. **三级碰撞避免**：
   - 前方 0.5m 内有障碍 → 减速
   - 前方 0.3m 内有障碍 → 转向 30°
   - 前方 0.1m 内有障碍 → 停止
3. **头部追踪**：head_yaw 跟随球方向，保持感知
4. **动态角色**：追球者=前锋，其他人保持阵型位置

### 我们当前的决策

```python
# soccer_env_3v3.py — step() 方法
# 只有 robot 0 (attacker) 用 ONNX 策略
# robots 1-5 用 _compute_rule_actions() 固定规则
# 没有动态角色切换
# 没有碰撞避免
# 没有头部追踪
```

### 借鉴建议

```python
# 建议在 soccer_env_3v3.py 中实现：

def _assign_roles_dynamically(self):
    """距离球最近的前场机器人追球，其他人保持阵型"""
    distances = [dist(robot, ball) for robot in self.robots]
    chaser = argmin(distances)
    for i, robot in enumerate(self.robots):
        if i == chaser:
            self.role[i] = CHASE
        elif i < 3:  # left team
            self.role[i] = FORMATION  # 保持阵型位置
        else:
            self.role[i] = DEFEND

def _avoid_collision(self, target_pos, robot_idx):
    """三级碰撞避免"""
    for other_idx in range(self.num_robots):
        if other_idx == robot_idx: continue
        dist = norm(self.all_base_pos[other_idx] - target_pos)
        if dist < 0.1: return STOP
        elif dist < 0.3: return TURN_30
        elif dist < 0.5: return SLOW_DOWN
    return target_pos
```

---

## 4. 训练策略（优先级：中，长期参考）

### RoboCup 的分阶段训练

```
阶段1: 行走优化 (walk optimization)
  - 目标: 直线行走 10m，不跌倒
  - reward: -行走时间 (越快越好)
  - 跌倒惩罚: time_cost = 1000 * fallen_count
  
阶段2: 踢球优化 (kick optimization)  
  - 目标: 球踢得越远越好
  - reward: ball_displacement (X方向)
  - 向后踢: -100
  - 跌倒: -1 + 终止
  
阶段3: 组合 (walk + kick + strategy)
  - 先走到球旁，再踢球
  - 加入碰撞避免和角色分配
```

### 我们当前

```
端到端: 高层 PPO (19→3) + 底层冻结 walk model
  - 一次训练 500 轮
  - reward 包含追球+踢球+进球
  - 没有单独的踢球训练阶段
```

### 借鉴建议

可以考虑增加踢球专项训练：
1. 固定球位置在机器人前方 0.2m
2. reward = ball_displacement
3. 训练一个专门的 kick policy
4. 比赛时根据距离球切换 walk/kick

---

## 5. 比赛规则（优先级：低，已有代码支持）

### RoboCup 的 playMode 系统

```cpp
enum PlayMode {
    PM_BEFORE_KICKOFF,      // 开场前
    PM_KICKOFF_LEFT,        // 左队开球
    PM_KICKOFF_RIGHT,       // 右队开球
    PM_PLAY_ON,            // 比赛中
    PM_GOAL_LEFT,          // 左队进球
    PM_GOAL_RIGHT,         // 右队进球
    PM_THROW_IN_LEFT,      // 左队边线球
    PM_THROW_IN_RIGHT,     // 右队边线球
    PM_CORNER_KICK_LEFT,   // 左队角球
    PM_CORNER_KICK_RIGHT,  // 右队角球
    PM_PENALTY_KICK_LEFT,  // 左队点球
    PM_PENALTY_KICK_RIGHT, // 右队点球
};
```

### 借鉴建议

在 `soccer_env_3v3.py` 中加一个 `play_mode` 状态机：
- 进球后重置到中圈
- 球出界后放到出界点
- 开球时机器人回到各自半场

这个改动不大但能让比赛更真实。

---

## 6. 代码架构（优先级：低，参考价值）

### RoboCup 的三层分离

```
决策层 (naobehavior.cc::act())
  → 决定做什么（追球/传球/射门/防守）
  
技能层 (skills.cc)
  → walk / kick / turn / getup / dive
  
运动层 (utwalk/)
  → 关节级 PD 控制
```

### 我们的对应

```
决策层: ONNX policy (19→3) + rule_actions
技能层: walk model (720→21) ← 没有单独的 kick skill
运动层: control_dofs_position (PD control)
```

### 缺失

我们缺一个**踢球技能**。当前踢球靠 walk model 走过去撞球，没有专门的踢球动作。
RoboCup 有专门的 kick skill（抬腿+摆动+触球）。

长期可以训练一个 kick policy，短期可以用 `_execute_kick` 的 impulse 模拟。

---

## 总结：优先级排序

| 优先级 | 借鉴内容 | 预期改动 | 预期收益 |
|--------|---------|---------|---------|
| P0 | 简化 reward（球位移为主，跌倒-5+终止） | reward.py 修改 | 更稳定的训练信号 |
| P0 | 最近者追球 + 碰撞避免 | soccer_env_3v3.py 修改 | 3v3 机器人能动起来 |
| P1 | obs 加球速度 + yaw | obs 构建修改 | 更好的感知 |
| P1 | 方向惩罚（向后=-大分） | reward.py 修改 | 鼓励正确方向 |
| P2 | playMode 状态机 | soccer_env_3v3.py | 更真实的比赛 |
| P2 | 踢球专项训练 | 新训练脚本 | 更好的踢球效果 |
| P3 | 队友/对手 obs（24 dim） | 训练新模型 | 多智能体配合 |

---

## 7. Booster 官方 3v3 基线 Agent（最直接可借鉴）

> 来源: `/Users/simon/BoosterStudioProjects/simon3v3-simple-baseline`
> 这是 Booster 官方提供的 3v3 足球比赛 Agent 基线代码，直接运行在 Booster T1 机器人上。

### 7.1 整体架构

```
main.py (策略层)
  ├── Phase 状态机: NORMAL / KICKOFF / SET_PLAY / READY / STOPPED
  ├── 角色分配: 最近者追球 + Guard + Support
  ├── 传球决策: 安全通道检测 + 持球人压力触发
  └── 丢球反抢: 检测控球权丢失 → 最近者冲抢

player.py (技能层)
  ├── walk_to(target, face, avoid_ball, avoid_robots)  → 全局路径规划
  ├── attack(target, power)  → 踢球/射门
  ├── dribble()  → 带球推进（小力度踢球保持控球）
  ├── guard()  → 守门员防守
  ├── support(attacker_pose)  → 支援走位
  └── ensure_ready()  → 跌倒起身 + 模式切换

param.py (参数层)
  └── 所有可调参数集中管理: 踢球力度/走位/避障/传球/防守
```

### 7.2 关键设计模式（可直接移植到 Genesis 3v3）

#### A. Phase 状态机

```python
class Phase(Enum):
    NORMAL = "normal"              # 正常拼抢
    OUR_KICKOFF = "our_kickoff"    # 我方开球
    OPP_KICKOFF = "opp_kickoff"    # 对方开球
    OUR_SET_PLAY = "our_set_play"  # 我方定位球
    OPP_SET_PLAY = "opp_set_play"  # 对方定位球
    READY = "ready"                # 走位
    STOPPED = "stopped"            # 停止
```

**借鉴价值**: 我们的 `soccer_env_3v3.py` 没有 Phase 状态机。加入状态机可以让比赛更真实。

#### B. 最近者追球（不是固定前锋）

```python
def _select_closest_attacker(context, players, preferred_id=None):
    ranked = [(p, dist_to_ball(p)) for p in players]
    best, best_dist = min(ranked, key=lambda x: x[1])
    if preferred and preferred_dist <= best_dist + 0.3:  # 防震荡
        return preferred
    return best
```

**借鉴价值**: 我们固定 robot 0 追球。Booster 动态选择最近者，加 `ATTACKER_KEEP_DIST_MARGIN_M=0.3` 防震荡。

#### C. 丢球反抢

```python
our_possession = our_closest < opp_closest + 0.5
if prev_our and not our_possession and closest_press < 4.0:
    store.counter_press_active = True
    store.counter_press_until = now + 2.5  # 2.5秒窗口
```

**借鉴价值**: 我们没有反抢逻辑。加入后比赛更真实。

#### D. 带球推进（Dribble）

```python
def dribble(self):
    if dist_to_ball < 0.5 and dist_to_goal > 4.0:
        kick(power=2.0)  # 小力度推球
        return True
    return False
```

**借鉴价值**: 我们只有大力踢球（KICK_IMPULSE=3.0），没有带球。

#### E. 三级路径规划

```python
# 1. 全局规划: A* 栅格搜索 (GLOBAL_GRID_RESOLUTION_M = 0.35)
# 2. 局部规划: VFH 方向扫描 (PLAN_STEP = 15°, PLAN_LOOKAHEAD = 1.2m)
# 3. 避障: 前方扇形检测 (BALL_OBSTACLE_RADIUS=0.5, OPPONENT_RADIUS=0.55)
```

**借鉴价值**: 我们完全没有路径规划，机器人直线走向目标不避障。

#### F. 传球决策门控（5层安全检查）

```python
def _pass_target(context, attacker, receiver, store):
    if not _passer_controls_ball(context, attacker): return None      # 1.持球
    if receiver_distance > 3.2: return None                            # 2.距离
    if any(dist(opp, receiver) < 1.0 for opp in opponents): return None # 3.接球人安全
    if any(segment_clearance(opp, ball, receiver) < 0.65): return None  # 4.通道净空
    if shot_blocked or passer_pressured: return (receiver.x, receiver.y) # 5.射门被挡才传
    return None
```

**借鉴价值**: 最完整的传球决策逻辑，我们没有传球功能。

#### G. 守门员智能防守

```python
# 1. 球远时：守在球门区域中心
# 2. 球进入威胁区 (< -1.0m)：前出拦截
# 3. 球速朝球门 > 0.35 m/s：预判轨迹扑救
# 4. 球离开威胁区 (> -0.7m)：退回门线
```

#### H. 安全自适应

```python
# 落后且时间不多 → 激进（提高速度上限）
# 领先且有人倒地 → 保守（降低速度）
# 正常 → 标准参数
```

### 7.3 参数表（经过实战验证，直接可用）

| 参数 | 值 | 用途 |
|------|-----|------|
| KICK_POWER_DEFAULT | 5.0 | 正常踢球力度 |
| KICK_POWER_NEAR_GOAL | 4.5 | 近门射门 |
| DRIBBLE_KICK_POWER | 2.0 | 带球推球 |
| DRIBBLE_SHOOT_RANGE_M | 4.0 | 进入射门区域距离 |
| KICK_ENTER_M | 0.5 | 进入踢球状态距离 |
| CHASE_BEHIND_M | 0.35 | 追球站到球后方距离 |
| PASS_POWER | 3.6 | 传球力度 |
| PASS_RECEIVER_MAX_DISTANCE_M | 3.2 | 最大传球距离 |
| PASS_LANE_CLEARANCE_M | 0.65 | 传球通道最小净空 |
| GUARD_THREAT_ENTER_X | -1.0 | 守门员威胁区入口 |
| COUNTER_PRESS_WINDOW_S | 2.5 | 反抢窗口期 |
| COUNTER_PRESS_MAX_DISTANCE_M | 4.0 | 最大反抢距离 |
| OPPONENT_RADIUS | 0.55 | 对手避障半径 |
| TEAMMATE_RADIUS | 0.48 | 队友避障半径 |
| BALL_OBSTACLE_RADIUS | 0.5 | 球的避障半径 |
| GLOBAL_GRID_RESOLUTION_M | 0.35 | A*栅格分辨率 |
| PLAN_LOOKAHEAD | 1.2 | 局部避障探测距离 |
| PLAN_CLEARANCE | 0.35 | 最小安全余量 |
| PLAN_STEP | 15° | 方向扫描步长 |
| ATTACKER_KEEP_DIST_MARGIN_M | 0.3 | 追球者切换防震荡 |

### 7.4 借鉴优先级更新

| 优先级 | 借鉴内容 | 来源 | 预期收益 |
|--------|---------|------|---------|
| **P0** | 最近者追球 + 防震荡 | Booster main.py | 3v3 机器人能动 |
| **P0** | Phase 状态机 | Booster main.py | 比赛流程完整 |
| **P0** | 三级路径规划（A*+VFH+避障） | Booster player.py | 防止卡死 |
| **P1** | 带球推进 | Booster player.py | 保持控球权 |
| **P1** | 传球决策门控（5层安全检查） | Booster main.py | 安全传球 |
| **P1** | 守门员智能防守 | Booster player.py | 有效防守 |
| **P1** | 丢球反抢 | Booster main.py | 比赛更激烈 |
| **P2** | 安全自适应 | Booster safety_adaptation.py | 动态策略 |
| **P2** | 参数表移植 | Booster param.py | 经过验证的参数值 |
