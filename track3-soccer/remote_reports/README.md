# Remote Hermes Reports

所有 Radeon/Hermes 远端任务必须在本目录留下可追踪报告，不能只保留在聊天窗口或临时 notebook 中。

## 命名

```text
hermes_<task-id>_report.md
```

## 报告必须包含

- 任务目标与允许修改的文件
- 实际修改文件与摘要
- 远端执行命令、超时和退出码
- 测试/验证结果与原始 artifact 路径
- 失败原因、未解决风险
- `merge_status: pending|approved|rejected`
- 下一步接手动作

## 合并规则

Hermes 的文字总结不是验收证据。只有报告、diff、日志和测试结果都归档后，主控 agent 才能把 `merge_status` 改为 `approved` 并同步到本地/GitHub。

临时 notebook 在确认上述文件已复制后才可删除；API key 不得进入报告、diff、日志或 Git。
