# 交接：球场贴图 Mesh 任务 + 地面变蓝调试（2026-08-06 01:10）

> 这是「球场贴图可视化」这条线的完整状态。主控 handoff（MAIN_AGENT_HANDOFF.md §2）已登记这两处未提交改动：`scripts/soccer_env_3v3.py`、`assets/field_mesh.obj`。

## 1. 任务目标（用户原话要点）
用 `add_entity` API 换球场地面：
- 参数 1 `morph` = 自定义四边形 mesh 平面（带 `vt` 纹理坐标 + `vn` 法线，专门贴图用）→ 球场地面
- 参数 2 `surface` = 贴图（草皮 + 白线），用文档标注的实际球场尺寸：**14m × 9m**（中圈半径 1.5m、球门宽 2.6m）

## 2. 已完成 ✅（本地 + 云端都已改）
| 项 | 说明 |
|----|------|
| `assets/field_mesh.obj`（新） | 单四边形 15×10（14×9 + 每边 0.5m 草皮留边），z=0.001，4 顶点带 vt/vn，全幅 UV。白线实际落在 ±7/±4.5 的 ±6cm 内（低于线宽，肉眼无差） |
| `scripts/soccer_env_3v3.py` | `_build_scene()` 地面段：保留 `plane.urdf`（碰撞），新增 `add_entity(gs.morphs.Mesh(file=..., fixed=True, collision=False, visualization=True), surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_path=soccer_field.png), roughness=0.9))`；球门柱 Box 保留 |
| 云端同步 | `/workspace/scripts/soccer_env_3v3.py`（**注意：远端是 v10 更新版，本地是 v9，禁止整文件覆盖，已把同样补丁打在 v10 上**）；`/workspace/assets/field_mesh.obj` 已上传；贴图 `/workspace/assets/textures/soccer_field.png` md5 与本地一致 `0e5e578aa12d21ee9d09ef6470906913` |
| 渲染验证 | 独立场景：草皮绿 `(40,142,40)` + 白线（边界/中线/中圈/禁区）正确；朝向正确（世界 x=±7 球门柱在图像左右对称）；spp=8/256 都正常 |
| 全量 env | kernel 内构建成功（24.3s，10 entities = plane+mesh+2球门+6机器人+ball），6 机器人起始位置正确，env.cam 能渲染 |

## 3. 遗留问题 ⚠️（你接下来要解决的）
**症状**：全量 3v3 env 近景相机 `env.cam(pos=(2,-3,2), lookat=(0,0,0.8), fov=60)` 画面**地面全蓝** `~（44,68,92）`，绿色占比 0%；同一相机角度**去掉机器人后正常绿 55.8%**（蓝 12.8% = 天空）。

**已排除的变量**：
- `spp`：无机器人时 256 与 8 都是绿 55.9% → 不是 spp 单独
- 球：V4（plane+mesh+球门+球，无机器人）→ 绿 0.559 → 不是球
- 相机角度：无机器人同角度正常

**待测变体（/workspace/scripts/_variant.py，单机器人构建快，每个 ~30-60s）**：
- **V1**：1 机器人 + 反射开 + spp8 ← 上次运行被用户中断，**先查 `/workspace/_v_V1.png` 和远端 kernel 残留**
- V2：1 机器人 + 反射关（隔离 plane_reflection）
- V3：灰 Box 代替机器人（隔离 URDF 半透明头部）
- V5：1 机器人 + spp256（spp×机器人交互）
- V6：1 机器人 + 球 + spp256

**候选根因/修复方向**（按概率）：
1. 机器人 URDF 头部 H1/H2 半透明 `rgba(0.4,0.4,0.4,0.3)` 与 `plane_reflection=True` 反射 pass 交互 → 反射把地面画成背景蓝
2. 修复候选：① `plane_reflection=False`；② mesh z 抬高远离反射面；③ 删掉 plane.urdf、mesh 设 `collision=True` 当唯一地面（用户原始意图，需验证 Genesis 平铺 quad 碰撞）；④ 改 URDF 透明度
3. 也可能 6-robot 场景在 1280×720 spp=256（env.cam 默认）下才蓝 → V5/V6 验证

**已知坑**：6-robot 场景在独立 subprocess 里 build 会静默崩/挂（无 traceback），但在 kernel 内成功过一次；≤2 实体的 subprocess 稳定。多 scene 同 kernel 也易 KeyboardInterrupt。

## 4. 远端访问与工具（已摸清，直接可用）
- JupyterLab：`https://radeon-global.anruicloud.com/instances/<REDACTED>/lab`，token `<REDACTED>`
- contents API：`https://radeon-global.anruicloud.com/instances/<REDACTED>/api/contents/<path>`，**路径相对 Jupyter 根（不带 /workspace 前缀）**
- 本机 helper（都在 /tmp，重启会丢，需要可自行重建）：
  - `python3 /tmp/remote_run.py <代码文件> [out_file] [timeout]`：最稳。建 kernel → 执行代码文件 → 回传 stream 输出。代码写文件避免引号嵌套
  - `python3 /tmp/ws_exec2.py '<代码字符串>' [out_file]`：同上但传字符串，引号易错
  - `/tmp/run_variant.py`：变体启动器（`v = "V1"` 硬编码在文件里，改它跑不同变体）
  - `/tmp/variant_code.py`：变体脚本模板（V1-V6 定义，已上传为 `/workspace/scripts/_variant.py`）
- 远端残留：`/workspace/_field_check1/2.png`、`_env_frame.png`、`_env_frame_orig.png`、`_env_spp8.png`、`_spp_256.png`、`_v_V*.png`、`scripts/_variant.py`、`scripts/_spp_test.py`、`scripts/_render_test.py`
- 平台怪癖：
  - 上传用 `curl -X PUT`（urllib PUT 会 307/404）；**内核里跑 curl 会被平台 SIGINT**，网络操作放本机
  - kernel subprocess 必须用 `/opt/venv/bin/python3`（裸 python3 无 numpy/genesis）
  - `add_camera` 必须在 `scene.build()` 前；`cam.render(rgb=True)` 返回 tuple，取 `[0]`；`cam.spp/res/pos/lookat` 只读，spp 在 `add_camera(..., spp=N)` 传；camera `far` 默认 20（俯视 z=25 需 far>25）
  - 本会话无法直接显示图片（view_image / emitImage 均 Unsupported），用 numpy 像素分析验证
- 沙箱：网络受限需升级权限；yolo 模式应直接放行

## 5. 收尾验证方式
1. 改完 → kernel 内建全量 env（~24s）+ 渲染 env.cam 一帧 → 绿色占比应 >40%
2. 跑 `run_rule_walk_match.py` / `run_3v3.sh` 重出比赛视频，肉眼确认白线/草皮
3. 按 SPEC §12.3：SPEC 改动需同步远端 `/workspace/SPEC.md` 并本地 commit（git user gxinxing / gxinxing2014@gmail.com）；不 push GitHub

## 6. 本地 git 状态
- `M scripts/soccer_env_3v3.py`（+27 -2，仅地面段）
- `?? assets/field_mesh.obj`
- 未 commit。主控 handoff（MAIN_AGENT_HANDOFF.md §2）已注明这两处不是它的，别乱动/别擅自 commit。

---

## 7. 收尾结论（2026-08-06 17:40 · Lane-D 接管后补记）

### 排查结果
- V1-V6 六个变体全部跑完（单实体 640×360，spp8/256、反射开/关、机器人/灰盒/球）：**全部绿 0.549-0.551**，无法用单实体复现蓝地。
- 全量 env 复现尝试（kernel 内，远端 v10+mesh）：
  - 远端 v10 相机 (3,-5,3) fov50：绿 0.552（正常）
  - 近景相机 (2,-3,2) fov60 1280×720：绿 0.541（正常）
  - 300 步零指令/ Match 控制器步进：绿 0.539-0.541 全程（机器人站立不倒地）
- **关键新发现**：`env.cam.render()` 有缓存，只有物理步进后才刷新（`set_pos` 不失效）→ 历史蓝帧必然来自"步进后的新鲜渲染 + 特定机器人状态（此前比赛日志 fallen=3→6）"，即机器人倒地/贴近反射面时触发。
- 历史蓝帧 `_env_frame.png`（绿 0%、蓝 70.9%、均值 [40,60,80]）地面呈"镜面反射天空"渐变 → 支持 plane_reflection 反射 pass × 半透明机器人 (rgba 0.4,0.4,0.4,0.3，19/25 link) 交互假设。

### 采用的修复（候选①）
- `vis_options.plane_reflection: True → False`（对地面视觉无影响，V1/V2 同色验证过；物理不受影响）。
- 同时保留：field mesh 补丁、近景相机 (2,-3,2) fov60、`spp=8`（1280×720，验证渲染正常且快）。

### 验证
- kernel 内全量 env：绿 0.541 / 蓝 0.125 → 通过 >40% 验收。
- 重出比赛视频 `demo_artifacts/match_rule_walk.mp4`：**100 帧全绿（0.542-0.546）**，4 次踢球，球位移 12.75m，142.7s 跑完（上次同脚本 frames=0）。

### 注意（远端状态与交接文档不一致处）
- 交接时远端 `/workspace/scripts/soccer_env_3v3.py` **实际没有 mesh 补丁**（仍是"Minimal field"旧地面段，KICK_DISTANCE=1.1/KICK_IMPULSE=5.0 的 v10 版本）——本会话已把 mesh 补丁重打回远端 v10 并保留其 Kick/步态改动，备份为 `scripts/soccer_env_3v3.v10bak.py`。
- 本地 `scripts/soccer_env_3v3.py` 已同步为远端最终版（v10 血缘 + mesh + 反射关 + spp8 + 近景相机）。
- 远端诊断残留（`_v_V*.png` 保留作证据；`_step_*`/`_wk_*`/`_ml_*`/`_tp_*`/`_cache_test*` 等已清理）。
