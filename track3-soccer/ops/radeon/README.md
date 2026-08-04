# Radeon operations

This is the canonical home for Radeon GPU operational material.

## Three-way source policy

| Location | Role | Authority |
|---|---|---|
| Local `track3-soccer/` | Development and integration | Canonical working tree |
| GitHub | Reviewed source history and recovery | Canonical published revision |
| Radeon `/workspace/amd-physical-ai-soccer/` | Current GPU execution workspace | Runtime mirror only |

`/workspace/radeon-repo` appears in legacy material but is not the current
synchronization target. Resolve commands and artifacts relative to
`/workspace/amd-physical-ai-soccer/` unless a new audited status record says
otherwise.

Remote-only checkpoints and videos are allowed because of size, but each final
artifact must be copied locally or recorded with its absolute remote path,
SHA-256, producing commit, config path and validation result.

## Safety

- SSH private keys stay outside this project. Never commit them.
- Do not use `StrictHostKeyChecking=no` for active synchronization.
- The recorded host key changed on 2026-08-03. Verify the new fingerprint with
  the Radeon provider before reconnecting or updating `known_hosts`.
- The former backup script was not retained: it suppressed synchronization
  failures, disabled host-key checking, and could expose a GitHub token in a
  configured remote URL.
- `legacy/gpu_task_loop.UNSAFE.txt` is non-executable historical evidence. It
  did not check child exit codes and could falsely print `ALL TASKS COMPLETE`.

## Before GPU work

1. Confirm the remote host fingerprint through a trusted channel.
2. Record local `git rev-parse HEAD` and `git status --short`.
3. Compare the local and remote hashes for files in the task scope.
4. Synchronize only reviewed files; do not overwrite remote checkpoints/runs.
5. Run the task and copy back structured results.
6. Re-run local tests and update `SYNC_STATUS.md`.

See `SYNC_STATUS.md` for the presently known state and blockers.

The 2026-08-04 status records a passing remote A/B check for the checked-in
`models/pretrained/t1_walk.pt`, a failing rule-walk fallback, and 100 passing
local tests. It does not record a 3v3 pass: launcher review is still open.
