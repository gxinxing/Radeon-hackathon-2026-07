# 任务 #01（本地，Codely 执行）— 起草双机器人接触失稳修复（方案 A）

## 背景
单机器人能正常走路、双机器人在场上站不稳。读本地 `scripts/soccer_env_3v3.py` 后，
根因假设：6 机器人放在**同一个 Genesis 场景**、全部共用 `t1_walk.pt`；
`RigidOptions(max_collision_pairs=256, tolerance=1e-5)` 碰撞对上限过低、容差极小；
两 attacker 初始仅距 2m，抢球会贴脸撞 → 基座被推 → walk 模型 720 维观测突变 →
walk 模型没在接触扰动下训练过 → 输出失稳 → 双双倒地。根因 = 缺接触鲁棒性。

## 你要做的（全是本地，不要连实例 / 不要跑 GPU）
1. 在本地副本 `scripts/soccer_env_3v3.py`（及 `soccer_env_hierarchical.py` 若有同名 RigidOptions）
   中搜索 `RigidOptions` / `max_collision_pairs`，定位场景构建处，确认当前值。
2. 起草**方案 A** 修复补丁（接触参数调优，不动模型）：
   - `max_collision_pairs` 提到 **≥4096**（6 人形 + 球，留余量）
   - `tolerance` 放宽到 **~1e-4**
   - 加接触阻尼：查 Genesis `gs.options.RigidOptions` 是否有 `contact_damping` /
     `contact_restitution` 字段，有则设合理值（阻尼如 1e2~1e3 量级，需查 Genesis 默认/文档），
     没有则在汇报里注明"Genesis 无该字段，建议改用提高 `solver_iters`"
   - 可选：提高求解器迭代 `solver_iters`（查字段名，如 `solver_iters` / `iterations`）
3. 用 `python3 -m py_compile <改后文件>` 验证语法（本地可能没装 genesis，编译只查语法，不要 import 运行）。
4. 产出：在 `OUT/result_01.md` 写
   - 当前 `RigidOptions` 原文 vs 修改后
   - 完整 diff / 修改片段（标明文件:行号）
   - 是否需同步到实例（是），以及注意事项（如 Genesis 版本字段差异）
   - 若发现 Genesis 有更优接触/求解参数，也一并建议

## 不要做
- 不要连实例、不要跑 GPU、不要删文件、不要 git push。
- 这是诊断性起草，不是最终落地；最终由 Dream 中台拍板并转发实例验证。
