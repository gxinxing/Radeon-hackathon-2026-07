"""Distill a trained Genesis policy into Booster behavioral-rule constants.

Reads a distillation log produced by bridge/genesis_logger.DistillLogger (JSONL),
fits simple heuristics from the data, and emits paste-ready constants for Booster's
param.py plus guidance for main.py. Pure stdlib -> runs offline (no torch/genesis).

Usage
-----
  # after cloud training, extract from a real rollout log:
  python bridge/distill.py --log bridge/rollout.jsonl --out bridge/booster_distilled.py
  # prove the pipeline works without a GPU (synthesizes a fake rollout):
  python bridge/distill.py --selftest
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def _percentile(values, q):
    """Nearest-rank percentile. values: list[float], q in [0,100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[rank]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- #
# core: fit heuristics from logged rows
# --------------------------------------------------------------------------- #
def analyze(rows):
    """Return a dict of distilled constants derived from rollout rows."""
    const = {}
    report = []

    kicked = [r for r in rows if r.get("kicked")]
    # --- kicking: distance at which the policy commits to a kick ---
    if kicked:
        d = [r["dist_to_ball"] for r in kicked]
        const["KICK_WHEN_DIST_TO_BALL_LT"] = round(_percentile(d, 90), 2)
        behind = [r["dist_to_ball"] for r in kicked if r.get("behind_ball")]
        if behind:
            const["KICK_BEHIND_OFFSET_M"] = round(_mean(behind), 2)
        report.append(
            f"  kicks fired on {len(kicked)} steps; commit distance p90 = "
            f"{const['KICK_WHEN_DIST_TO_BALL_LT']} m"
        )
    else:
        const["KICK_WHEN_DIST_TO_BALL_LT"] = None
        const["KICK_BEHIND_OFFSET_M"] = None
        report.append("  no kick events logged -> train 'shoot' task, then re-run")

    # --- recovery: how reliably the policy gets up ---
    fallen = [r for r in rows if r.get("fell")]
    recovered_next = 0
    for i, r in enumerate(rows):
        if r.get("fell") and i + 1 < len(rows) and rows[i + 1].get("recovered"):
            recovered_next += 1
    if fallen:
        rate = recovered_next / len(fallen)
        const["FALL_RECOVERY_PRIORITY"] = round(rate, 2)
        report.append(
            f"  fell {len(fallen)} times; recovered next step {rate*100:.0f}% of the time"
        )
    else:
        const["FALL_RECOVERY_PRIORITY"] = None

    # --- coop placeholders (fill after training the 'coop' task) ---
    const["GUARD_HOLD_DIST_M"] = None
    const["SUPPORT_FOLLOW_DIST_M"] = None
    const["AVOID_LOOKAHEAD_M"] = None
    const["AVOID_MIN_CLEAR_M"] = None
    report.append(
        "  coop (GUARD_HOLD_DIST_M / SUPPORT_FOLLOW_DIST_M) + avoidance"
        " (AVOID_*) fill after 'coop' training with 3 robots"
    )
    return const, report


def render_module(const, report, log_path):
    lines = [
        "# Auto-distilled from Genesis rollouts by bridge/distill.py.",
        "# Do NOT hand-edit -- re-run distill.py to regenerate.",
        f"# Source log: {log_path}",
        "# Mapping of each constant -> Booster code location: see bridge/SPEC.md",
        "",
        "# ---- attacker / kicking (main.py:_act_normal, player.py:plan_kick) ----",
        f"KICK_WHEN_DIST_TO_BALL_LT = {const['KICK_WHEN_DIST_TO_BALL_LT']!r}  # commit to kick when ball within this (m)",
        f"KICK_BEHIND_OFFSET_M = {const['KICK_BEHIND_OFFSET_M']!r}        # stand this far behind ball (m)",
        "KICK_APPROACH_ANGLE_DEG = 18.0     # plan_kick tolerance (default; tighten after coop)",
        "",
        "# ---- recovery (player.py:ensure_ready / get_up) ----",
        f"FALL_RECOVERY_PRIORITY = {const['FALL_RECOVERY_PRIORITY']!r}     # 1.0 = always try get_up first",
        "",
        "# ---- multi-agent (fill after 'coop' training) ----",
        f"GUARD_HOLD_DIST_M = {const['GUARD_HOLD_DIST_M']!r}",
        f"SUPPORT_FOLLOW_DIST_M = {const['SUPPORT_FOLLOW_DIST_M']!r}",
        f"AVOID_LOOKAHEAD_M = {const['AVOID_LOOKAHEAD_M']!r}",
        f"AVOID_MIN_CLEAR_M = {const['AVOID_MIN_CLEAR_M']!r}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# offline self-test: synthesize a plausible "trained" rollout
# --------------------------------------------------------------------------- #
def _selftest():
    random.seed(7)
    rows = []
    ball_x, ball_y = 2.0, 0.4
    rx, ry = 0.0, 0.0
    for t in range(900):
        dtb = math.hypot(rx - ball_x, ry - ball_y)
        behind = rx < ball_x
        if dtb > 0.5:
            # approach the ball
            ux, uy = (ball_x - rx) / dtb, (ball_y - ry) / dtb
            rx += 0.06 * ux + random.uniform(-0.01, 0.01)
            ry += 0.06 * uy + random.uniform(-0.01, 0.01)
            kicked = False
            btg = 0.0
        else:
            # close enough: a well-trained policy commits a kick when within ~0.55 m
            behind = True
            kicked = dtb < 0.55
            btg = 1.6 if kicked else 0.1
            if kicked:
                ball_x += 0.12  # ball travels toward +x goal
        fell = (t % 151 == 0)
        recovered = (t % 151 == 1)
        rows.append(
            dict(
                t=round(t * 0.02, 3), robot_x=round(rx, 3), robot_y=round(ry, 3),
                robot_yaw=0.0, ball_x=round(ball_x, 3), ball_y=round(ball_y, 3),
                ball_vx=round(btg, 3), ball_vy=0.0,
                dist_to_ball=round(dtb, 3),
                dist_to_own_goal=round(math.hypot(rx + 7.0, ry), 3),
                dist_to_opp_goal=round(math.hypot(rx - 7.0, ry), 3),
                behind_ball=behind, ball_to_goal=btg,
                fell=fell, recovered=recovered, kicked=kicked,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="Distill Genesis rollouts into Booster constants.")
    p.add_argument("--log", help="path to DistillLogger JSONL")
    p.add_argument("--out", help="path to write the distilled param.py module")
    p.add_argument("--selftest", action="store_true", help="synthesize a fake rollout and run")
    args = p.parse_args(argv)

    if args.selftest:
        rows = _selftest()
        const, report = analyze(rows)
        print("== bridge/distill.py --selftest ==")
        print("Synthesized %d steps of a plausible trained policy." % len(rows))
        for line in report:
            print(line)
        print("\n--- would write to bridge/booster_distilled.py ---")
        print(render_module(const, report, "<synthetic>"))
        return 0

    if not args.log:
        p.error("either --log PATH or --selftest is required")
    try:
        from bridge.genesis_logger import DistillLogger
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from bridge.genesis_logger import DistillLogger

    rows = DistillLogger.load(args.log)
    if not rows:
        print("ERROR: no rows in %s" % args.log, file=sys.stderr)
        return 1
    const, report = analyze(rows)
    out = args.out or "bridge/booster_distilled.py"
    text = render_module(const, report, os.path.abspath(args.log))
    with open(out, "w") as f:
        f.write(text)
    print("Distilled %d steps -> %s" % (len(rows), out))
    for line in report:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
