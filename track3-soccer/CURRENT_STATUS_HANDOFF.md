# Current Status Handoff

> Canonical status entry: [PROJECT_STATUS.md](PROJECT_STATUS.md). 新 agent 先读该文件。

更新时间：2026-08-04

## 当前操作原则

- 实例上仍有任务运行时，不移动、不重命名 `/workspace` 下正在使用的目录。
- `/workspace/persistent` 作为跨实例生命周期的归档目标；先复制，任务结束后再做最终同步和校验。
- 不销毁实例。若必须恢复宿主执行通道，优先考虑重启 VM；销毁前必须完成本地、GitHub、persistent 三端备份。

## Track 3

- 本地项目：`track3-soccer`
- 本地分支：`codex/track3-final-acceptance`
- 本地提交：`37b661d`
- 本地测试：151 passed，1 warning。
- 已有 AMD 训练、收敛曲线、ROCm 性能、ONNX/规则单体对比、分布式 3v3 生命周期和视频证据。
- 共享物理 smoke 已在 AMD GPU 上构造并推进 6 机器人 + 1 球单场景 5 步。
- 最新共享物理报告必须诚实视为 `status=observed`、`validation_status=failed`：六台机器人均低于配置的 0.8 m 跌倒阈值，球没有实质运动。它不是成功比赛证据。
- 共享物理报告中的 policy、环境、评估器、配置、ONNX、walk model 六项 SHA 已与本地当前文件一致。
- GitHub 推送尚未确认成功；此前遇到 GitHub DNS/网络阻塞。

## Track 2 Dify

- 本地 `.env` 已设置 `MARKETPLACE_ENABLED=false`、`NEXT_TELEMETRY_DISABLED=1`。
- 本地 compose 已加入 web/api 内存护栏：web 1 GiB、api 2 GiB，并通过 `docker compose config --quiet`。
- 远端实际 Dify 版本与本地不同，远端约为 1.7.1；不能直接覆盖整份 1.16.1 compose。
- Jupyter 容器没有 Docker CLI、Docker socket 或宿主 namespace 权限；SSH 宿主通道仍未可信恢复。
- 尚未同步远端或重启 Dify 容器，避免半部署。

## 后续顺序

1. 等当前训练/评估任务结束，不改变其运行路径。
2. 将已完成的源码、模型、日志和验收文件复制到 `/workspace/persistent/backups/`，再做 SHA-256 清单。
3. 确认本地和 GitHub 公开仓库备份成功。
4. 只有确认数据安全后，才考虑重启 VM 恢复宿主 Docker/SSH；不直接销毁实例。
5. 赛道 2 按远端实际 Dify 版本做最小配置补丁，只重启 web/api 容器。
