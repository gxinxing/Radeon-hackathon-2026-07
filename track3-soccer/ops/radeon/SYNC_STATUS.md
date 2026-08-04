# Local / Radeon / GitHub synchronization status

Last local audit: 2026-08-04 (Asia/Shanghai)

## Known state

- Local canonical directory: this `track3-soccer/` tree.
- Local branch: `codex/track2-eval-improvement`.
- Audited local HEAD: `98ff035`.
- Local worktree contains uncommitted Track 3 changes; therefore neither the
  remote runtime nor GitHub can currently be assumed identical to local.
- GitHub remote: `gxinxing/Radeon-hackathon-2026-07`.
- The current branch has no configured upstream in the audited branch listing.
- Radeon runtime path used by the 2026-08-04 validation:
  `/workspace/amd-physical-ai-soccer/`. References to `/workspace/radeon-repo`
  are legacy and must not be used for current synchronization.
- After the AMD GPU returned to idle, remote A/B validation ran successfully
  against `models/pretrained/t1_walk.pt` (SHA-256
  `ef1d61e19082b83405f4320a08f4cfc2d7d7f003ed3790dab013778ba442dec7`).
- Rule-walk failed: stance fell at step 11 and gait fell at step 11.
- Pretrained-policy A/B passed: stance completed 60 steps with 0 falls; gait
  completed 150 steps with 0 falls and +6.423 m displacement.
- This A/B check exercised the `.pt` model directly; no `.onnx` export was
  revalidated in this run. Existing ONNX results are separate historical evidence.
- The current local test run reports 100 passed. The launcher remains in final
  review; 3v3 has not passed end-to-end validation.

## Required reconciliation

- [ ] Record the verified Radeon endpoint/fingerprint used for future syncs.
- [ ] Inventory remote Git commit and dirty files (the validated model hash is
      recorded above).
- [ ] Compare task-scope files against the local canonical tree.
- [ ] Review and commit the current local changes.
- [ ] Push the reviewed commit and configure/record its upstream.
- [ ] Deploy that exact commit or reviewed patch set to Radeon.
- [ ] Copy back validation JSON, logs and final artifact hashes.
- [x] Confirm the current local working tree tests pass (100 passed).
- [ ] Confirm local tests pass from the final committed revision.
- [ ] Complete launcher review and run 3v3 end-to-end acceptance; until then,
      do not describe 3v3 as passed.

Until every applicable item is checked, reports must state which location they
describe rather than saying the project is globally “up to date.”
