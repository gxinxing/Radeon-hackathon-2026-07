# Track 3 项目状态与接管入口

更新时间：2026-08-04 22:00（Asia/Shanghai）

这是 Track 3 的简明接管入口。新 agent 先读本文件，再读 `CURRENT_STATUS_HANDOFF.md`、`.graph_engine/CODEX_HANDOFF.md` 和 `COMPETITION_ACCEPTANCE.md`。

## 当前结论

- 主线仍是共享物理 3v3，但当前还没有合格的“六机器人踢球进球”视频。
- 旧视频源日志没有 `kick/scored` 事件，不能作为最终演示。
- 共享物理短跑曾观察到六机器人倒地、球无实质运动；长跑曾在 Genesis/GPU 初始化或首步推进阶段无输出。
- 本地已加入 Booster standing pose fallback 和共享遥测渲染器；本地测试 `151 passed, 1 warning`。
- 最新本地分支：`codex/track3-final-acceptance`；最新推送提交：`37b661d`。

## 今日已完成

1. 核查并否定旧视频的“成功比赛”含义，保留失败诊断视频。
2. 修复六机器人初始关节姿态来源：优先使用父环境 standing pose。
3. 新增 `scripts/render_shared_physics_video.py`，只渲染真实 telemetry 事件，不合成进球。
4. 新增诊断视频：`acceptance/demo/shared_physics_smoke_failed.mp4`。
5. 远端安装 Hermes Agent v0.20.0；AMD endpoint 曾验证 HTTP 200。
6. 依据 GeneralCompute Quickstart 配置 MiniMax M2.7；Hermes 已执行任务，但尚未产出可交付 fallback 文件。
7. 核对远端已有历史 demo metadata：`verified_match`、`verified_short`、`3v3_match_v2`；这些是历史证据，不等于当前共享物理验收。
8. 最新远端 V2 单步门禁：场景可启动并触发 1 次 kick，球速度发生变化，但六机器人第 1 步全部倒地；`status=observed`、`validation_status=failed`、`score=0:0`。问题已收敛到初始姿态/关节映射/物理稳定性。

## 当前远端状态

- Radeon 实例：`<REDACTED>`。
- 可靠执行入口：JupyterLab/Ego；SSH 宿主通道仍不稳定。
- Hermes 目标配置：GeneralCompute custom provider + `minimax-m2.7`；密钥不写入 Git、报告或视频元数据。
- 第三个图形桌面方案已取消，不安装 VNC/noVNC/完整桌面。
- Hermes GeneralCompute 任务已结束（RC 0），但未生成 `FALLBACK_REPORT.md`、可运行 fallback 脚本或 PNG/MP4 证据，不能视为保底完成。

## 接下来严格按门禁执行

### P0 保底方案

- 先复现官方 Genesis Franka 或 HighwayEnv 示例。
- 必须落盘：可运行脚本、运行日志、PNG/MP4 证据、`FALLBACK_REPORT.md`。
- 只接受能被另一 agent 从文件和命令复现的结果。
- 预算：30–60 分钟；超时则采用已有 verified demo 作为展示保底，并明确其边界。

### P1 3v3 验证门禁

1. V0：Torch/ROCm/Genesis import。
2. V1：单机器人站立 1/3/5 步。
3. V2：六机器人静态场景 3 步。
4. V3：六机器人 walk，5 步无 fallen。
5. V4：共享球状态一致且无 NaN。
6. V5：人工 kick 使球产生速度。
7. V6：人工近门 kick 产生 `scored=true`。
8. V7：再接入 ONNX/rule policy。
9. V8：最后才渲染最终视频。

所有门禁：最大 60 秒、逐步 heartbeat、保存 stdout/stderr，禁止直接跑无进度的长任务。

## 接管命令提示

```bash
cd "/Users/simon/Documents/01_AI and Code Development/Radeon-hackathon-2026-07/track3-soccer"
python3 -m pytest -q tests
git status --short
```

远端工作目录：`/workspace/amd-physical-ai-soccer`；历史 canonical repo 也可能位于 `/workspace/radeon-repo`，执行前必须核对，不能盲写。

## 不允许的操作

- 不重启 VM，不杀其他项目进程。
- 不把合成的 goal/kick 写进 telemetry。
- 不把 `status=observed`/`validation_status=failed` 的 artifact 标成成功。
- 不把 API key 写入项目文件、Git、notebook 或最终报告。
