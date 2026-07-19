# Auto-distilled from Genesis rollouts by bridge/distill.py.
# Do NOT hand-edit -- re-run distill.py to regenerate.
# Source log: /workspace/amd-physical-ai-soccer/bridge/rollout.jsonl
# Mapping of each constant -> Booster code location: see bridge/SPEC.md

# ---- attacker / kicking (main.py:_act_normal, player.py:plan_kick) ----
KICK_WHEN_DIST_TO_BALL_LT = None  # commit to kick when ball within this (m)
KICK_BEHIND_OFFSET_M = None        # stand this far behind ball (m)
KICK_APPROACH_ANGLE_DEG = 18.0     # plan_kick tolerance (default; tighten after coop)

# ---- recovery (player.py:ensure_ready / get_up) ----
FALL_RECOVERY_PRIORITY = None     # 1.0 = always try get_up first

# ---- multi-agent (fill after 'coop' training) ----
GUARD_HOLD_DIST_M = None
SUPPORT_FOLLOW_DIST_M = None
AVOID_LOOKAHEAD_M = None
AVOID_MIN_CLEAR_M = None
