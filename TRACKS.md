# 仓库布局说明 (TRACKS.md)

本仓库是 AMD AI DevMaster Hackathon 2026 的**混合 monorepo**，同时包含赛道2 与赛道3 的代码。两者已物理分离到不同顶层目录（2026-08-03 完成）。

## 目录划分

| 路径 | 赛道 | 说明 | 处理 |
|------|------|------|------|
| `track2-agentic-ai/` | 赛道2 Agentic AI | 含约 32G 模型权重文件（`models/` 下 `*.pth/*.pt/*.onnx` 等已被 `.gitignore` 忽略，不进版本库） | **严禁删除**，也不要随意移动 |
| `track3-soccer/` | 赛道3 Physical AI（人形机器人足球） | 全部足球训练/仿真/渲染/评测代码，含根 `README.md`/`Dockerfile`/Booster 平台脚本，以及 v3 编排（`match_worker_v3.py`/`match_coordinator_v3.py`）+ 当前 `demos/` | **本目录已是赛道3 的规范（canonical）源**；2026-08-03 已将原本地副本 B（`Documents/Codex/track3-publish-local`）内容合并进来并退役 B |
| `.git` `.gitignore` `.codely*` `.pytest_cache` `__pycache__` `.DS_Store` `sync.*` `setup_codely_cloud.sh` | 共享 / meta | 工具与元数据 | 不动 |

## 重要约束

- **赛道2 的 32G 模型文件严禁删除**（已 gitignore，不会进版本库，也不会被误提交）。
- 赛道3 的**规范源就是本仓库的 `track3-soccer/`**（2026-08-03 已将 B = `Documents/Codex/track3-publish-local` 内容完整合并进来，B 本地副本已移至废纸篓退役；B 在 GitHub 上的远程仓库 `Radeon-hackathon-2026-07-track3` 保留）。活跃开发直接在 `track3-soccer/` 进行。
- 运行赛道3 脚本请先 `cd track3-soccer` 再执行（脚本内的相对路径基于该目录）。

## 归档（archive）约定

为保持活跃开发树聚焦于主线路（`soccer_env_curriculum` + `train_curriculum` + `match_3v3`），冗余/历史文件已从活跃树移出：

- **A（本仓库）**：`track3-soccer/archive/` —— 旧环境版本（`soccer_env_v3/1v1/1v1_virtual`）、旧训练脚本（`train_pretrained/_v2`、`train_1v1` + 日志）、多余渲染器、`audit/benchmark/verify` 工具、被取代的运行脚本，以及体积较大的 `match_logs/`、`training_logs/`、`results/`、`reports/`、`bridge/`、`remote_backup/`。
- **B（规范源 `track3-publish-local`）**：`archive/`（仓库根，因 B 本身就是赛道3 专属仓库）—— 内容与 A 对齐；另把 1v1 遗留渲染脚本（`scripts/render_*_1v1.py`）连同其依赖的 `soccer_env_1v1` 一并归档，避免活跃树出现悬空 import。

归档文件**均可恢复**：git 跟踪的文件保留在 git 历史中，`git mv` 的目录结构完整；被 gitignore 的大视频/日志仅在磁盘上移入 `archive/`，未删除。

> 注意：B 的 `demos/` 中保留的是**当前** demo 证据（`verified_match`、`curriculum`、`3v3_coop`、`2robot` 等），不属于冗余，故保留在活跃树；A 中同名旧 demo 已删除/归档。

## 变更历史

- 2026-08-03：将赛道3 从仓库顶层归集到 `track3-soccer/`，实现赛道2 / 赛道3 物理分离；清理 1 个坏拷贝产生的、文件名含换行符的 0 字节损坏条目（显示为 `ema_slow, short: null}` / `exit: {long: ema_fast` 两行）；新增本说明文件。
- 2026-08-03：将 A 与 B 中赛道3 的冗余/历史文件归档至各自 `archive/`（A=`track3-soccer/archive/`，B=`archive/`）；B 已提交并推送到赛道3 专属远程（`e7c5fda..ebeec00`，0 领先/0 落后）。A 的归档提交留在本机（分支 `codex/track2-eval-improvement`，未推送）。
