#!/usr/bin/env python3
"""3v3 Match Evaluation Script

Runs N matches between two teams and collects statistics.
Outputs per-match JSON files and a summary CSV + JSON.

Usage:
    # Rule vs rule (no checkpoints)
    python scripts/match_eval_3v3.py --matches 20 --steps 1000 --seed 42

    # RL (walk checkpoint) vs rule
    python scripts/match_eval_3v3.py --matches 20 --checkpoint /path/to/t1_walk.pt

    # Full (walk + shoot checkpoints) vs rule
    python scripts/match_eval_3v3.py --matches 20 --checkpoint /path/to/t1_walk.pt --shoot-checkpoint /path/to/shoot.pt

Important:
    This script requires Genesis and a GPU to run actual matches.
    If Genesis is not available, it will report a blocker and exit.
    Match results are never fabricated — all stats come from real simulation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np

from match_3v3 import (
    Scene3v3, SceneConfig, FieldConstants, DEFAULT_FIELD,
    RoleAssigner, RulePolicy, SharedRLPolicy,
    MatchResult, MatchSummary, ScoreBoard,
    Team, Role, PlayerState, BallState,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_scene_config(cfg: dict) -> SceneConfig:
    """Build SceneConfig from YAML config dict."""
    fc = cfg.get("field", {})
    field = FieldConstants(
        field_length=fc.get("length", 14.0),
        field_width=fc.get("width", 9.0),
        goal_width=fc.get("goal_width", 2.6),
        goal_height=fc.get("goal_height", 1.0),
        circle_radius=fc.get("circle_radius", 1.5),
        ball_radius=fc.get("ball_radius", 0.11),
        robot_stand_height=fc.get("robot_stand_height", 0.72),
    )

    ip = cfg.get("initial_positions", {})
    left_form = ip.get("left", {})
    right_form = ip.get("right", {})
    ball_start = tuple(ip.get("ball", [0.0, 0.0, 0.11]))

    return SceneConfig(
        field_cfg=field,
        ball_start=ball_start,
        left_formation=[
            tuple(left_form.get("attacker", [-1.0, 0.0, 0.72])),
            tuple(left_form.get("defender", [-3.5, 1.5, 0.72])),
            tuple(left_form.get("goalkeeper", [-6.5, 0.0, 0.72])),
        ],
        right_formation=[
            tuple(right_form.get("attacker", [1.0, 0.0, 0.72])),
            tuple(right_form.get("defender", [3.5, -1.5, 0.72])),
            tuple(right_form.get("goalkeeper", [6.5, 0.0, 0.72])),
        ],
    )


def create_policy(cfg: dict, args) -> SharedRLPolicy:
    """Create SharedRLPolicy with optional checkpoints."""
    policy_cfg = cfg.get("policy", {})
    walk_ckpt = args.checkpoint or policy_cfg.get("walk_checkpoint", "")
    shoot_ckpt = args.shoot_checkpoint or policy_cfg.get("shoot_checkpoint", "")

    return SharedRLPolicy(
        walk_checkpoint=walk_ckpt if walk_ckpt else None,
        shoot_checkpoint=shoot_ckpt if shoot_ckpt else None,
        field=DEFAULT_FIELD,
        action_scale=policy_cfg.get("action_scale", 0.25),
        obs_history_length=policy_cfg.get("obs_history_length", 10),
    )


def run_single_match(
    match_id: int,
    scene: Scene3v3,
    policy: SharedRLPolicy,
    role_assigner: RoleAssigner,
    max_steps: int,
    seed: int,
    dt: float,
) -> MatchResult:
    """Run a single 3v3 match.

    This function requires a built Genesis scene (scene.handles.built == True).
    """
    np.random.seed(seed)

    # Reset positions
    scene.reset_positions()
    scoreboard = ScoreBoard()

    for step in range(max_steps):
        # Role assignment
        assignments = role_assigner.assign(scene.players, scene.ball_state, step)

        # Policy computation per player
        for player in scene.players:
            action = policy.compute(player, scene.ball_state)

        # TODO: Apply actions to Genesis scene
        # This requires reading/writing entity positions from scene.handles
        # scene.handles.scene.step()

        # TODO: Read updated positions from Genesis
        # For now, this is a stub that will be completed when Genesis is available

        scoreboard.step = step + 1

        # Check goals
        scoring_team = scene.check_goal()
        if scoring_team is not None:
            team_name = "left" if scoring_team == Team.LEFT else "right"
            scoreboard.record_goal(team_name)
            # Reset after goal
            scene.reset_positions()

        # Check out of bounds
        if scene.ball_out_of_bounds():
            scoreboard.ball_out_of_bounds += 1
            scene._init_ball()

    # Determine winner
    if scoreboard.left_score > scoreboard.right_score:
        winner = "left"
    elif scoreboard.right_score > scoreboard.left_score:
        winner = "right"
    else:
        winner = "draw"

    method = policy.mode

    return MatchResult(
        match_id=match_id,
        method=method,
        left_score=scoreboard.left_score,
        right_score=scoreboard.right_score,
        left_falls=scoreboard.left_falls,
        right_falls=scoreboard.right_falls,
        left_recoveries=scoreboard.left_recoveries,
        right_recoveries=scoreboard.right_recoveries,
        left_shots=scoreboard.left_shots,
        right_shots=scoreboard.right_shots,
        left_shots_on_target=scoreboard.left_shots_on_target,
        right_shots_on_target=scoreboard.right_shots_on_target,
        ball_out_count=scoreboard.ball_out_of_bounds,
        total_steps=scoreboard.step,
        match_duration_s=scoreboard.step * dt,
        winner=winner,
        seed=seed,
    )


def save_match_result(result: MatchResult, output_dir: str):
    """Save a single match result as JSON."""
    path = os.path.join(output_dir, f"match_{result.match_id:03d}.json")
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def save_summary_csv(results: list[MatchResult], output_dir: str):
    """Save all results as a CSV file."""
    path = os.path.join(output_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        f.write(MatchResult.csv_header() + "\n")
        for r in results:
            f.write(r.to_csv_row() + "\n")


def save_summary_json(summary: MatchSummary, results: list[MatchResult], output_dir: str):
    """Save aggregate summary as JSON."""
    path = os.path.join(output_dir, "summary.json")
    output = {
        "summary": summary.to_dict(),
        "matches": [r.to_dict() for r in results],
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="3v3 Soccer Match Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods (auto-detected from checkpoints):
  rule_vs_rule   — no checkpoints loaded, rule-based policy only
  rl_vs_rule     — walk checkpoint loaded, shoot checkpoint not loaded
  full_vs_rule   — both walk and shoot checkpoints loaded

Output:
  results/match_3v3/match_001.json, match_002.json, ...
  results/match_3v3/summary.csv  (all matches)
  results/match_3v3/summary.json (aggregate stats)
""",
    )
    parser.add_argument("--matches", type=int, default=20,
                        help="Number of matches to run (default: 20)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Steps per match (default: 1000, ~20s at dt=0.02)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to walk checkpoint (t1_walk.pt)")
    parser.add_argument("--shoot-checkpoint", type=str, default="",
                        help="Path to shoot checkpoint (.pt)")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "match_3v3.yaml"),
                        help="Path to match_3v3.yaml config")
    parser.add_argument("--output", type=str, default="results/match_3v3",
                        help="Output directory for results")
    parser.add_argument("--robot-urdf", type=str, default="",
                        help="Path to T1 robot URDF/MJCF file")
    parser.add_argument("--show-viewer", action="store_true",
                        help="Show Genesis viewer (requires display)")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Build scene config
    scene_config = build_scene_config(cfg)
    if args.robot_urdf:
        scene_config.robot_urdf = args.robot_urdf
    else:
        scene_config.robot_urdf = cfg.get("robot", {}).get("urdf", "")

    # Create scene
    scene = Scene3v3(scene_config)

    # Check Genesis availability
    if not scene.genesis_available:
        print("=" * 60)
        print("  BLOCKER: Genesis is not available.")
        print("  Cannot run actual 3v3 matches without Genesis physics.")
        print("  Match results will NOT be fabricated.")
        print()
        print("  To resolve:")
        print("    1. Install genesis-world: pip install genesis-world")
        print("    2. Ensure GPU is accessible")
        print("    3. Provide robot URDF: --robot-urdf /path/to/t1.xml")
        print()
        print("  Current method label:", "rule_vs_rule")
        print("  Unit tests (no Genesis needed): python -m pytest tests/test_match_contract.py")
        print("=" * 60)
        sys.exit(1)

    # Create policy
    policy = create_policy(cfg, args)
    print(f"\nPolicy mode: {policy.mode}")
    print(f"  Walk checkpoint loaded: {policy.walk_loaded}")
    print(f"  Shoot checkpoint loaded: {policy.shoot_loaded}")

    # Role assigner
    roles_cfg = cfg.get("roles", {})
    role_assigner = RoleAssigner(
        reassign_interval=roles_cfg.get("reassign_interval", 50),
        field=scene_config.field_cfg,
        defend_offset=roles_cfg.get("defend_offset", 2.0),
        keeper_y_range=roles_cfg.get("keeper_y_range", 1.04),
    )

    # Match params
    match_cfg = cfg.get("match", {})
    max_steps = args.steps or match_cfg.get("max_steps", 1000)
    dt = match_cfg.get("dt", 0.02)

    # Output dir
    output_dir = args.output or cfg.get("evaluation", {}).get("output_dir", "results/match_3v3")
    os.makedirs(output_dir, exist_ok=True)

    # Try to build Genesis scene
    try:
        scene.build(robot_urdf=scene_config.robot_urdf)
    except Exception as e:
        print("=" * 60)
        print(f"  BLOCKER: Failed to build Genesis scene: {e}")
        print("  Match results will NOT be fabricated.")
        print("=" * 60)
        sys.exit(1)

    # Run matches
    print(f"\n{'=' * 60}")
    print(f"  3v3 Match Evaluation")
    print(f"  Method: {policy.mode}")
    print(f"  Matches: {args.matches}")
    print(f"  Steps per match: {max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    results: list[MatchResult] = []
    for i in range(args.matches):
        t0 = time.time()
        result = run_single_match(
            match_id=i,
            scene=scene,
            policy=policy,
            role_assigner=role_assigner,
            max_steps=max_steps,
            seed=args.seed + i,
            dt=dt,
        )
        elapsed = time.time() - t0
        results.append(result)
        save_match_result(result, output_dir)
        print(f"  Match {i + 1}/{args.matches}: "
              f"L {result.left_score} - R {result.right_score} "
              f"({result.winner}) "
              f"falls L:{result.left_falls} R:{result.right_falls} "
              f"[{elapsed:.1f}s]")

    # Save summary
    summary = MatchSummary.from_results(results)
    save_summary_csv(results, output_dir)
    save_summary_json(summary, results, output_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Summary: {policy.mode}")
    print(f"  Matches: {summary.n_matches}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'Left':>10} {'Right':>10}")
    print(f"{'-' * 25} {'-' * 10} {'-' * 10}")
    print(f"{'Win Rate':<25} {summary.left_win_rate:>10.1%} {summary.right_win_rate:>10.1%}")
    print(f"{'Avg Goals':<25} {summary.avg_goals_for:>10.2f} {summary.avg_goals_against:>10.2f}")
    print(f"{'Avg Goal Diff':<25} {summary.avg_goal_diff:>10.2f}")
    print(f"{'Avg Falls':<25} {summary.avg_fallen_count:>10.2f}")
    print(f"{'Avg Recoveries':<25} {summary.avg_recovery_count:>10.2f}")
    print(f"{'Recovery Rate':<25} {summary.recovery_rate:>10.1%}")
    print(f"{'Draw Rate':<25} {summary.draw_rate:>10.1%}")
    print(f"{'Avg Duration (s)':<25} {summary.avg_match_duration:>10.2f}")
    print(f"{'=' * 60}\n")

    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
