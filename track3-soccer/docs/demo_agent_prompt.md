# Demo Video Agent Prompt

Copy-paste this prompt to an AI agent (or follow it yourself) AFTER the
`-B 2048` training run finishes. It produces the competition demo video
and all supporting evidence clips. Keep everything in English.

---

**Prompt (paste to the agent):**

> You are producing the final submission demo for an AMD AI DevMaster
> Hackathon Track-3 (Physical AI) project: a humanoid robot soccer policy
> trained on an AMD Radeon GPU with Genesis + ROCm PyTorch + rsl_rl.
> The repo is at `/workspace/amd-physical-ai-soccer`. All outputs MUST be
> in English. The final deliverable is a 3–5 minute narrative demo video
> that proves the policy works end-to-end.
>
> Work strictly in this order:
>
> 1. **Goal showreel (the 30-point evidence).** Run
>    `python render_goal_video.py --model runs/hierarchical_soccer_chase_hl/model_300.pt --max_steps 1500`
>    (use the actual last checkpoint from the `-B 2048` run). Confirm
>    `demos/goal_showreel.mp4` was created. If it falls back to
>    `demos/chase_full.mp4` (no goal), STOP and report that the policy did
>    not score — the training needs a fix before submitting.
>
> 2. **Training telemetry.** Export these TensorBoard curves as PNGs into
>    `demos/`: `Episode/mean_reward`, `Episode/mean_episode_length`,
>    `Episode/goal_per_1k_steps`, and `Policy/mean_action_std`. These are
>    the quantitative proof that training converged.
>
> 3. **ROCm evidence.** While a short training snippet runs, record the
>    terminal with `rocm-smi` visible in the same frame. This is the
>    mandatory proof that training runs on AMD Radeon / ROCm. Save as
>    `demos/rocm_training.mp4`.
>
> 4. **Sim2Sim.** Follow `docs/sim2sim_runbook.md` — **first re-export a valid ONNX
>    from the 2048-envs checkpoint (Step 0) and pass the size + param gates**, then load
>    `models/chase_v6_2048_policy.onnx` into Booster Studio 3v3 SoccerSim via VNC and
>    record a match. Save as `demos/sim2sim_3v3.mp4`.
>    Do NOT use the committed `chase_v3/v4/v5_policy.onnx` — they are 2 KB stubs.
>
> 5. **Assemble the narrative (3–5 min).** Cut one video with this structure
>    and on-screen captions: (a) problem — humanoid soccer RL is locked to
>    NVIDIA Isaac Gym; (b) our approach — AMD Radeon + Genesis + ROCm
>    pipeline (use rocm_training.mp4 + architecture caption); (c) training
>    results — the TensorBoard PNGs with a one-line takeaway each;
>    (d) policy in action — goal_showreel.mp4; (e) Sim2Sim validation —
>    sim2sim_3v3.mp4; (f) closing — judging-criteria recap. Keep captions
>    concise and English.
>
> 6. Report the final file path and a 3-sentence summary of what each
>    clip proves. Do not fabricate results — only use clips you actually
>    generated.

---

## Notes for the human operator

- Step 1 guarantees a goal clip even if a single short rollout is unlucky
  (the script runs a long rollout and trims around the first goal).
- If `goal_showreel.mp4` is not produced, the policy is not yet scoring —
  revisit `configs/hierarchical_agent.yaml` (ball_progress / ball_contact
  weights) before submitting.
- The 3–5 minute length is a hard requirement in `CLAUDE.md`.
