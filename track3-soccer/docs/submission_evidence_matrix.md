# Submission Evidence Matrix

Status vocabulary:

- **VERIFIED** — reproduced during the current acceptance run with an artifact or log.
- **REPORTED** — present in existing project documentation but not yet reproduced in the current run.
- **PENDING** — required before submission.
- **OBSERVED / HEALTH GATE FAILED** — execution succeeded but behavior missed acceptance thresholds.
- **OPTIONAL** — useful only after all required gates pass.

| Evidence | Current status | Existing evidence | Acceptance action |
|---|---|---|---|
| AMD GPU availability | VERIFIED | Remote Radeon GPU detected with about 51.5 GB VRAM; idle before acceptance tests | Record GPU/ROCm versions and timestamp in final report |
| Low-level rule-walk baseline | VERIFIED | Stance and gait each fell after 11 steps | Preserve log and use only as a failed baseline |
| Frozen `t1_walk.pt` locomotion | VERIFIED | Stance: 60 steps, 0 falls; gait: 150 steps, 0 falls, +6.423 m | Preserve output, model SHA-256, and command |
| Local automated tests | VERIFIED | 151 tests pass after shared-physics additions | Preserve final test output |
| Reward convergence | REPORTED | Technical report claims training results and 1,358 goals | Export TensorBoard scalars and identify exact run/checkpoint |
| Training throughput | REPORTED | Technical report claims 4,618 steps/s | Reproduce with command, environment count, duration, and GPU telemetry |
| Baseline vs RL chase | REPORTED | `docs/module_e_verification_report.md` contains four-scenario comparison | Verify referenced JSON, checkpoints, and commands exist and match the report |
| Short single-agent football gate | VERIFIED | Fixed-seed ONNX and rule JSON under `acceptance/single_agent/` | Preserve commands and hashes |
| 3v3 launcher lifecycle | VERIFIED | 10 s run: six exit 0, all consumed END at step 20, no orphan process | Preserve raw log and match JSON |
| Shared-physics 3v3 smoke | OBSERVED / HEALTH GATE FAILED | AMD GPU: one Genesis scene, six robots, one ball, 5/5 steps; CG solver; all-six-fallen and no-ball-motion health checks failed | Preserve `acceptance/shared_physics/`; do not claim match success |
| Multi-robot role assignment | VERIFIED | Fixed A/B attacker, defender, keeper identities with controller/source hashes | Do not claim learned communication |
| Inference performance | VERIFIED | Fixed-seed ONNX latency recorded in single-agent acceptance artifact | Preserve raw JSON |
| Peak VRAM | PENDING | Only idle VRAM has been observed | Capture peak during training and 3v3 match |
| Demo artifact | PENDING | Existing demo metadata may exist | Produce a reproducible video plus raw match log |
| Learned robot communication | OPTIONAL | Not required by teacher guidance | Do not implement during the two-day critical path |

## Claim-control rules

1. A statement in an older report is not automatically current evidence.
2. Do not label the full 3v3 system “verified” based only on launcher unit tests or low-level locomotion.
3. Every headline metric must point to a command, configuration, model, timestamp, and artifact.
4. Failed baselines are valuable when conditions are identical and honestly reported.
5. If a metric cannot be reproduced before submission, label it as a previous result instead of a current verified result.

## Required artifact set

- `acceptance/locomotion/`: baseline and pretrained locomotion logs.
- `acceptance/training/`: TensorBoard scalar export, selected checkpoint metadata, and training configuration.
- `acceptance/evaluation/`: fixed-seed baseline/RL comparison JSON.
- `acceptance/performance/`: ROCm environment, throughput, inference, and VRAM measurements.
- `acceptance/3v3/`: launcher log, match trajectory/score, process-cleanup check, and video metadata.

The directories above describe the final artifact contract. Create them only when the corresponding evidence is generated; do not add placeholder results.
