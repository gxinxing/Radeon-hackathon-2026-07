# 仓库布局说明 (TRACKS.md)

本仓库是 AMD AI DevMaster Hackathon 2026 的**混合 monorepo**，同时包含赛道2 与赛道3 的代码。两者已物理分离到不同顶层目录（2026-08-03 完成）。

## 目录划分

| 路径 | 赛道 | 说明 | 处理 |
|------|------|------|------|
| `track2-agentic-ai/` | 赛道2 Agentic AI | 含约 32G 模型权重文件（`models/` 下 `*.pth/*.pt/*.onnx` 等已被 `.gitignore` 忽略，不进版本库） | **严禁删除**，也不要随意移动 |
| `track3-soccer/` | 赛道3 Physical AI（人形机器人足球） | 全部足球训练/仿真/渲染/评测代码，含根 `README.md`/`Dockerfile`/Booster 平台脚本 | 规范开发源为 `Documents/Codex/track3-publish-local`（即 B）；本目录为 monorepo 内整洁副本，内容与之对齐 |
| `.git` `.gitignore` `.codely*` `.pytest_cache` `__pycache__` `.DS_Store` `sync.*` `setup_codely_cloud.sh` | 共享 / meta | 工具与元数据 | 不动 |

## 重要约束

- **赛道2 的 32G 模型文件严禁删除**（已 gitignore，不会进版本库，也不会被误提交）。
- 赛道3 的**规范源是 B = `Documents/Codex/track3-publish-local`**；本仓库 `track3-soccer/` 与其内容一致，仅作 monorepo 内归档，活跃开发以 B 为准。
- 运行赛道3 脚本请先 `cd track3-soccer` 再执行（脚本内的相对路径基于该目录）。

## 变更历史

- 2026-08-03：将赛道3 从仓库顶层归集到 `track3-soccer/`，实现赛道2 / 赛道3 物理分离；清理 1 个坏拷贝产生的、文件名含换行符的 0 字节损坏条目（显示为 `ema_slow, short: null}` / `exit: {long: ema_fast` 两行）；新增本说明文件。
