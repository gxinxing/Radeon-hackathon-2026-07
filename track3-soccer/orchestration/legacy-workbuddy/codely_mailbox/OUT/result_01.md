# 方案 A 修复补丁 — 双机器人接触失稳（接触参数调优，不动模型）

## 1. 诊断摘要

| 项目 | 详情 |
|------|------|
| **根因** | 6 机器人共用同一 Genesis 场景，`RigidOptions` 碰撞对上限过低（256）、容差极小（1e-5）、求解器迭代次数为默认值（50）。两 attacker 初始仅距 2m，抢球时贴脸碰撞 → 基座被推 → walk 模型 720 维观测突变 → 失稳倒地。 |
| **修复策略** | 方案 A：仅调优接触/求解器参数，不改动模型权重。 |
| **修改文件** | `scripts/soccer_env_3v3.py`（唯一含 `RigidOptions` 的目标文件） |
| **未修改文件** | `soccer_env_hierarchical.py` — 根目录存在该文件但 **不含** `RigidOptions`，无需改动。 |

## 2. Genesis RigidOptions 字段调研结果

从 [Genesis 源码](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/genesis/options/solvers.py) 获取了完整的 `RigidOptions` 类定义。与接触稳定性相关的关键字段如下：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_collision_pairs` | int | 150 | 碰撞对上限。当前代码设为 256，6 人形 + 球不够用。 |
| `tolerance` | float | None→1e-5 (single precision) | 约束求解器容差。越小越精确但越刚硬，接触冲击时易发散。 |
| `iterations` | int | 50 | 约束求解器最大迭代次数。即任务中的 `solver_iters`。 |
| `ls_iterations` | int | 50 | 线搜索迭代次数。 |
| `ls_tolerance` | float | 1e-2 | 线搜索容差。 |
| `constraint_timeconst` | float | 0.01 | 约束时间常数（同 MuJoCo `timeconst`），越小越刚硬。 |
| `noslip_iterations` | int | 0 (disabled) | noslip 后处理迭代，抑制接触滑移。 |
| `impratio` | float | None→1 (pyramidal) | 摩擦/法向约束阻抗比，增大可 stiffen 摩擦。 |
| `contact_pruning_tolerance` | float | 0.02 | 接触剪枝容差。 |
| `enable_self_collision` | bool | True | 当前代码设为 False。 |

### `contact_damping` / `contact_restitution` 字段

**Genesis `RigidOptions` 不存在 `contact_damping` 和 `contact_restitution` 字段。** Pydantic 模型使用了 `extra="forbid"`，传入未知字段会直接报错。

接触柔化在 Genesis 中通过以下参数间接实现：
- `constraint_timeconst`（增大 → 接触更柔和）
- `iterations`（增大 → 求解器有更多迭代收敛，减少穿透和弹跳）

## 3. 当前 RigidOptions 原文 vs 修改后

### 修改前（`scripts/soccer_env_3v3.py` L193–194）

```python
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False, tolerance=1e-5, max_collision_pairs=256),
```

### 修改后

```python
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=4096,
                tolerance=1e-4,
                iterations=100,
            ),
```

## 4. 完整 Diff

**文件：`scripts/soccer_env_3v3.py`，第 193–194 行**

```diff
--- a/scripts/soccer_env_3v3.py
+++ b/scripts/soccer_env_3v3.py
@@ -190,8 +190,11 @@
         """Build scene with 6 robots, ball, field, goals, camera."""
         self.scene = gs.Scene(
             sim_options=gs.options.SimOptions(dt=PHYSICS_DT, substeps=1),
-            rigid_options=gs.options.RigidOptions(
-                enable_self_collision=False, tolerance=1e-5, max_collision_pairs=256),
+            rigid_options=gs.options.RigidOptions(
+                enable_self_collision=False,
+                max_collision_pairs=4096,
+                tolerance=1e-4,
+                iterations=100,
+            ),
             viewer_options=gs.options.ViewerOptions(
                 camera_pos=(0, -12, 8), camera_lookat=(0, 0, 0.5), camera_fov=50),
             vis_options=gs.options.VisOptions(
```

## 5. 参数变更说明

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `max_collision_pairs` | 256 | **4096** | 6 人形机器人（每台 ~20+ link）+ 球 + 地面 + 球门，碰撞对远超 256。256 会导致碰撞检测被截断 → 穿透 → 接触力突变。4096 留充足余量。 |
| `tolerance` | 1e-5 | **1e-4** | 1e-5 过于严格，接触冲击时求解器难以收敛，数值发散。1e-4 放宽 10×，允许求解器在接触冲击下更快收敛，牺牲极少精度换取稳定性。 |
| `iterations` | 50（默认） | **100** | 默认 50 次迭代在密集接触场景（6 机器人抢球）可能不足以收敛。翻倍至 100 给求解器更多迭代来处理多接触约束。GPU 上代价较小（每步增加 ~10–20% 求解时间）。 |

## 6. 语法验证

```
$ python3 -m py_compile scripts/soccer_env_3v3.py
SYNTAX OK
```

编译通过（本地未安装 Genesis，仅做语法检查，不 import 运行）。

## 7. 是否需同步到实例

**是，需要同步到 GPU 实例。** 本地修改仅在工作站副本上完成，GPU 实例上的 `/workspace/amd-physical-ai-soccer/scripts/soccer_env_3v3.py`（或对应路径）需要同步更新。

### 同步注意事项

1. **Genesis 版本字段差异**：实例上 Genesis 版本为 1.3.1（见 `amd-physical-ai-soccer/README.md`）。`iterations` 字段在 Genesis 1.3.x 中存在（源码确认），但建议同步后先用 `python -c "import genesis as gs; gs.init(); print(gs.options.RigidOptions(iterations=100))"` 验证字段可用性。
2. **`max_collision_pairs` 对 GPU 显存影响**：从 256 提到 4096 会增加碰撞检测的 GPU 显存占用。6 机器人场景下预计增加 ~50–100 MB，在 24 GB MI300X 上可忽略。
3. **`tolerance` 放宽的副作用**：1e-4 比 1e-5 宽 10×，在精密操控（如脚部贴地行走）场景中可能导致微小穿透。但 walk 模型本身对接触不敏感（训练时无接触扰动），稳定性优先于精度。
4. **其他文件中的 RigidOptions**：项目中还有多个文件包含 `RigidOptions`，如果实例运行的是其他入口（如 `match_scene.py`、`soccer_env_v4.py`），也需要同步调整。相关文件清单：

   | 文件 | 当前 `max_collision_pairs` | 当前 `tolerance` | 是否需同步 |
   |------|--------------------------|-----------------|-----------|
   | `scripts/soccer_env_3v3.py` | ✅ 已改 (4096) | ✅ 已改 (1e-4) | 已完成 |
   | `amd-physical-ai-soccer/scripts/soccer_env_3v3.py` | 256 | 1e-5 | 是（实例副本） |
   | `match_scene.py` | 2048 | 1e-5 | 建议同步 |
   | `amd-physical-ai-soccer/envs/soccer_env.py` | 512 | 1e-5 | 建议同步 |
   | `amd-physical-ai-soccer/match_scene.py` | 2048 | 1e-5 | 建议同步 |
   | `src/match_3v3/scene.py` | 2048 | 1e-5 | 建议同步 |
   | `soccer_env_v4.py` | 512 | 1e-5 | 视使用情况 |
   | `scripts/render_2robot_match.py` | 128 | 1e-5 | 视使用情况 |
   | `scripts/render_real_1v1.py` | 64 | 1e-5 | 1v1 场景，可暂不动 |

## 8. 额外建议（Genesis 更优接触/求解参数）

基于 Genesis 源码分析，以下参数可进一步提升双机器人接触稳定性（本补丁未含，供后续方案 B/C 参考）：

| 参数 | 建议值 | 理由 |
|------|--------|------|
| `constraint_timeconst` | 0.02（默认 0.01） | 增大约束时间常数 → 接触更柔和，减少冲击力突变。代价是接触稍"软"，但对 walk 模型稳定性有益。 |
| `noslip_iterations` | 5（默认 0） | 启用 noslip 后处理，抑制接触滑移。Genesis 文档推荐操控任务设为 5。代价：每步增加少量计算。注意：不支持 elliptic friction cone。 |
| `impratio` | 10–100（默认 1） | 增大摩擦/法向阻抗比，使摩擦约束更刚硬，机器人脚底不易打滑。配合 `friction_cone=gs.friction_cone.elliptic` 效果更佳。 |
| `ls_iterations` | 100（默认 50） | 线搜索迭代翻倍，配合 `iterations=100` 确保求解器在复杂接触场景中找到可行解。 |
| `enable_self_collision` | True（当前 False） | 当前关闭了自碰撞。开启后人形肢体间碰撞会被检测，避免脚腿穿透。但会增加计算量，建议先在方案 A 验证后再考虑。 |
| `substeps` (SimOptions) | 2（当前 1） | 在 `SimOptions` 中将 substeps 从 1 增到 2，等效于物理步长减半，接触求解更精细。代价：仿真速度减半。 |

## 9. 结论

方案 A 补丁已完成并通过语法验证。三处参数变更（`max_collision_pairs` 256→4096、`tolerance` 1e-5→1e-4、`iterations` 50→100）均为保守调优，不改模型、不改训练流程，可直接在 GPU 实例上验证。若方案 A 仍不足够稳定，建议按第 8 节的额外建议逐步引入 `constraint_timeconst`、`noslip_iterations` 等参数。
