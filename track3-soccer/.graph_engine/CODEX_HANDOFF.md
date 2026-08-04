# CODEX_HANDOFF.md — 主控交接汇总（2026-08-04）

> 最新项目状态请优先查看上级目录的 [PROJECT_STATUS.md](../PROJECT_STATUS.md)；本文件保留 Graph Engine 历史任务与实例背景。

> 给 **Codex CLI（新主控 / 编排）** 的当前状态汇总。Dream 已转为执行 / 中转层，听你安排。
> 配套"宪法"：`agent.md`；任务看板：`tasks/Txx.md`；信箱：`mailbox/`。

## 0. 一句话现状
**代码侧全就绪（T04 补丁 3742B 待 apply），唯一硬阻塞 = 实例执行通道全死（Jupyter kernel + terminal 都起不来），需用户在控制台 Stop→Start 或绑 SSH 公钥才能解。**

## 1. 角色反转
- **Codex CLI = 主控 / 编排**：分解任务、派给各 agent、审核结果、决策重试 / 修复。
- **Dream = 执行 / 中转层**：听 Codex 安排；可做本地编排 / 中转 / 监控 / 文档 / 审核；实例操作经 Ego 浏览器（但当前通道死）。
- **agent 池不变**：claude(c1) / hermes(c3) / codely(c5) 本地 + 实例执行；trae(c2) 曾因鉴权失败暂弃用。
- **信箱收件箱**：Codex 写任务给 Dream → `mailbox/IN/d1/task_NN.md`；Dream 执行后回 `mailbox/OUT/result_NN.md`。

## 2. 实例现状（关键阻塞）
- 实例：`<REDACTED>`（控制台显示名 **simon-robot**），AMD Radeon Cloud。
  访问：`https://radeon-global.anruicloud.com/instances/<REDACTED>/lab?token=<REDACTED>`
- **Jupyter HTTP 活（GET 200）**：contents / kernel / terminal API 都能创建对象。
- **但两个执行通道全死（DEAD_EXEC 顽固，2026-08-04 09:00 确诊 → 18:0x 复测仍 true）**：
  - kernel：新建后轮询 24s 永远 `execution_state: starting`，不进 idle；
  - terminal：WS 能建连但**数据面死**（发命令 20s 零回显）。
  - → 实例上**任何命令都跑不了**，T05 及之后 GPU 任务全卡。
- **SSH 不可用**：`~/.ssh/config` 的 `Host radeon` 引用私钥 `radeon_ssh/id_ed25519` 本地已缺失；`ssh radeon` 被网关 `Connection closed`（anruicloud 网关前置鉴权，需控制台先绑定公钥）。
- **根因推断**：实例端 kernel / terminal 启动环境坏（疑似残留进程占 GPU / 内存，或 ipykernel 环境损坏）；用户此前重启仅恢复了 Jupyter HTTP，没救回执行后端。

## 3. 解阻塞（需用户操作，均 ≠ 销毁）
- **A（推荐）**：控制台对实例 **Stop → Start**（彻底停再起，清残留 / 资源），最可能复活 kernel / terminal。
- **B**：控制台**绑定 SSH 公钥**（本机 `~/.ssh/id_ed25519_kandong.pub`），Dream 改 config 指向该私钥后用 SSH 进实例 `pkill` 残留 + 重启 jupyter 修复。
- **恢复判断标准**：新建 kernel 能进 `idle`、terminal 能回显 → 即可跑 T05。

## 4. 任务链 T01–T09（看板摘要）
| ID | 内容 | 状态 |
|----|------|------|
| T01 | 双机接触失稳 RigidOptions 调参 | 本地已修(4096/1e-4/100)；实例读回确认仍是旧值 256/1e-5 |
| T02 | 踢球逻辑修复 | 本地已修；文档并入 T04 |
| T03 | 扫描清理错误 demo 视频 | 实例预置 `_disc.py`，待可靠通道跑 |
| **T04** | **port T01/T02 到实例** | **✅ 补丁就绪** `patches/instance_soccer_env_3v3.patch`(3742B/6 hunk)，claude 起草 + Dream 审核，py_compile OK |
| **T05** | **apply 补丁 + 短 eval + 视频扫描** | **阻塞**：等实例执行通道恢复 |
| T06 | 全量 3v3 ONNX 演示 | todo（前置 T05 PASS） |
| T07 | 行走接触鲁棒性微调 | todo（前置 T06） |
| T08 | 多智能体协作训练 | todo（前置 T07） |
| T09 | 扩展评测量化指标 | todo（前置 T08） |

## 5. 信箱协议（Codex 派活）
- 给我（Dream）：`mailbox/IN/d1/task_NN.md` → 我执行 / 中转 → `mailbox/OUT/result_NN.md`
- 给 claude：`IN/c1/`；hermes：`IN/c3/`；codely：`IN/c5/`（兜底）
- 实例执行经 Ego 浏览器：`tools/read_instance.mjs`（只读）、`tools/ego_exec.mjs`（写 + 执行）——**当前通道死，须先解 §3 阻塞**
- 闭环：Codex 写任务 → agent 执行 → 写 OUT → Codex 审核 → PASS 标记 done / FAIL 写 fix 重跑

## 6. 惨痛教训（务必遵守）
- **🔴 禁止盲发 `POST /api/shutdown` 远程重启 Jupyter**：曾误发导致 Jupyter 被彻底关掉、平台不自动重启，反而逼用户进控制台。重启只能由用户在控制台点（≠销毁）。

## 7. 给 Codex 的建议起手
1. 先判断 §3 阻塞是否已由用户解除（让 Dream 经 Ego 复测 kernel 是否进 idle）。
2. 若已解：派 T05（apply 补丁 + 短 eval + 视频扫描）→ 串行 T06→T09。
3. 若未解：提醒用户执行 §3 的 A 或 B；期间可推进纯本地任务（如 T04 补丁复审、文档）。
