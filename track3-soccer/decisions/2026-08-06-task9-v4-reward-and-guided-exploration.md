## Decision: Task-9 P1 v4 — 新增 vel_to_ball 奖励 + 引导探索打破站桩死锁

## Context
- P1 三轮训练（R1 240iter / R2 500iter / R3 100iter）后 300 步评测：fallen=0 但 robot_disp=0.125m、kicks=0。
- 根因 1（奖励结构）：`approach_ball = tanh(Δdist)` 权重 7，每步逼近 2cm 仅值 ~0.14，而站桩白拿 alive(0.35)+upright(1.3)=1.65/步 —— 站桩收益是追球的 12 倍。
- 根因 2（反转奖励 bug）：首次实现 `vel_to_ball = -dot(v, dir)` 符号反了，等于奖励远离球，策略学出 vx 均值 -0.056（后退）。
- 根因 3（walk 模型物理上限）：诊断显示冻结 t1_walk.pt 在任何移动指令（vx 0.25~0.8）下摔倒率 3~6%/步、最高速度 ~0.10~0.15 m/s；规则 walk（rule_walk_v3）0.086 m/s 但更稳。300 步评测位移 ≥1.8m 的门槛在物理上极紧。

## Alternatives considered
- 只调 yaml 权重（approach_ball 7→20）：tanh 上限 1.0，逐步收益仍 <1/步，站桩 margin 太薄。
- 加 tracking_lin_vel 奖励：`exp(-(cmd-actual)²/σ)` 在 (0,0) 处满分，可被“命令 0”反撸，放弃。
- 重训/替换 walk 模型：时间不够（冻结 12:00），放弃。

## Reasoning
- v4 参数：alive 0.35→0.10、upright 1.3→0.6、approach_ball→15、vel_to_ball=20（朝球实际线速度，正比于真实位移，站桩必为 0）、fall_penalty -12→-10。
- 引导探索（--guide）：25% 概率用“朝球 vx=0.42+转向”启发式替代策略动作，250 iter 内衰减到 0；引导期 fall_penalty 临时降到 -4（干净正向信号）。直接给策略示范“移动=高回报”，打破 μ≈0 死锁。

## Trade-offs accepted
- 引导期摔倒率升高（per_ep 0.5~0.8），样本含较多摔倒轨迹。
- 位移物理上限 ~0.1 m/s：即使策略完美，300 步评测 robot_disp 最多 ~1.5~3m，crit2/3 仍紧张。
- 最终若 P1 评测仍不达标，保底提交线（单机射门视频 Path-B）作为主交付。
