# Next Main Window Prompt

继续 Track3 时，先读取 `PROJECT_STATUS.md`、`CURRENT_STATUS_HANDOFF.md`、`.graph_engine/CODEX_HANDOFF.md`、`remote_reports/README.md` 和 `acceptance/diagnostics/README.md`，再查看最近的 `git log` 与未提交 diff。不要重启 Radeon VM；先核对远端产物和报告，再执行新的短门禁。Hermes 只能执行有明确文件范围、timeout、报告和 diff 的窄任务；主控 agent 必须审查测试、日志和 diff 后才允许合并。所有进度写回 `PROJECT_STATUS.md` 与 `CURRENT_STATUS_HANDOFF.md`，并提交 GitHub。
