# Track 3 orchestration control plane

This directory is the durable, Git-trackable home for multi-agent planning and
handoff material. The project root (`track3-soccer/`) is the only canonical
source for Track 3 work.

## Layout

- `legacy-workbuddy/`: preserved WorkBuddy Graph Engine, recovery plans, Codely
  mailbox tasks, and results. This is historical input, not the active executor.
- `../.graph_engine/`: local runtime state (logs, scans, transient mailboxes and
  instance snapshots). It is intentionally ignored by Git.
- `../ops/radeon/`: remote GPU runbooks and scripts. Credentials never belong
  in this repository.

## Current controller

Codex is the primary controller. Luna Worker and other agents are workers with
bounded ownership. A worker result is not considered complete until the primary
controller runs or inspects its acceptance checks.

The legacy `graph_engine.py` must not be used as an acceptance authority. Its
default actions only print descriptions and its default verifiers return `True`.
It is retained for its DAG/checkpoint design and historical context.

The source-of-truth decision is stored in the repository at
`../decisions/2026-08-04-track3-single-source-and-sync.md`; a second copy remains
in the home-workspace `/decisions/` directory to satisfy workspace-wide policy.

## Working rules

1. Every task names its owner, inputs, output files, acceptance command, timeout,
   and whether GPU execution is required.
2. No two agents edit the same file concurrently.
3. Local code is changed first. Remote changes are synchronized from a reviewed
   local revision, never treated as an undocumented source of truth.
4. GitHub records reviewed source and durable evidence. Large models, videos,
   secrets, runtime logs and temporary scans remain outside Git and are recorded
   by path plus SHA-256 when relevant.
5. Remote results are not accepted from prose alone; retain command output,
   metadata, model/config hashes, and the exact local Git commit used.

## Acceptance snapshot (2026-08-04)

- Remote AMD GPU validation was run after the GPU returned to idle, using the
  current mirror at `/workspace/amd-physical-ai-soccer/`; `/workspace/radeon-repo`
  is a legacy path, not the current execution target.
- `models/pretrained/t1_walk.pt` (SHA-256
  `ef1d61e19082b83405f4320a08f4cfc2d7d7f003ed3790dab013778ba442dec7`)
  passed A/B validation: stance 60 steps/0 falls; gait 150 steps/0 falls and
  +6.423 m displacement.
- Rule-walk failed both trials at step 11 (stance and gait).
- The A/B run validated the `.pt` model, not a high-level `.onnx` export; do not
  conflate earlier ONNX evidence with this acceptance result.
- The local suite currently reports 100 tests passed. Distributed launcher
  review is not final, and 3v3 must not be marked passed.
