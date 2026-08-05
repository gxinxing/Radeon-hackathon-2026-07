## Decision: 清除公开仓库中的实例凭据并重写分支历史
## Context: 2026-08-06 检查发现公开仓库 gxinxing/Radeon-hackathon-2026-07 的 3 个远端分支
(codex/track2-cn-market, codex/track2-sync-20260802, codex/track3-final-acceptance)
历史与当前树含 AMD 实例访问 token（形如 `amd-*`）与实例 ID（形如 `u-*`），违反"公开仓库绝不出现 token/实例 URL"硬规则。
## Alternatives considered: 1) 仅在新提交中删除文件（历史仍泄露）；2) 历史重写+force-push（已选）；3) 删除分支重建（丢失全部历史与证据链）。
## Reasoning: main 分支历史干净，泄露仅存在于 3 个 agent 工作分支；git filter-branch --index-filter 逐 blob 替换
凭据字符串为 <REDACTED>（跳过二进制），181 个提交重写后全树/全历史无命中，force-push 替换远端。
## Trade-offs accepted: force-push 使旧提交 SHA 失效（仅影响 agent 分支，无协作方）；GitHub 可能保留悬空对象，
建议轮换实例 token 或设仓库为 private；本地已 gc + 删除 backup tag 清理旧对象。
