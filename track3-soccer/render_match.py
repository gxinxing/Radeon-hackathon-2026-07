#!/usr/bin/env python3
"""Render 3v3 match video from JSON match log.

Reads match_logs/match_*.json (produced by match_coordinator.py) and
renders a top-down 2D animation showing all 6 robots + ball trajectory.

Usage:
    python render_match.py                          # auto-find latest match log
    python render_match.py --log match_logs/match_xxx.json
    python render_match.py --output demos/3v3_match.mp4
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors


def find_latest_log(match_dir="match_logs"):
    files = sorted(glob.glob(os.path.join(match_dir, "match_*.json")))
    if not files:
        print(f"No match logs found in {match_dir}/")
        sys.exit(1)
    return files[-1]


def load_match_log(path):
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded: {path}")
    print(f"  Duration: {data['duration']:.1f}s | Steps: {data['steps']} | Robots: {data['n_clients']}")
    return data


def render_match(data, output_path, fps=30):
    log = data["log"]
    n_steps = len(log)
    if n_steps == 0:
        print("Empty match log, nothing to render.")
        return

    # Soccer field dimensions (must match env config)
    field_x, field_y = 14.0, 9.0
    goal_half = 1.3
    circle_r = 1.5

    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor="#0a1a0a")
    ax.set_facecolor("#0d3a1a")
    ax.set_xlim(-field_x / 2 - 1, field_x / 2 + 1)
    ax.set_ylim(-field_y / 2 - 1, field_y / 2 + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Field lines
    field_rect = patches.Rectangle(
        (-field_x / 2, -field_y / 2), field_x, field_y,
        linewidth=2, edgecolor="white", facecolor="none", zorder=1)
    ax.add_patch(field_rect)
    ax.plot([0, 0], [-field_y / 2, field_y / 2], "w-", linewidth=1, zorder=1)
    center_circle = plt.Circle((0, 0), circle_r, fill=False, color="white", linewidth=1.5, zorder=1)
    ax.add_patch(center_circle)
    ax.add_patch(plt.Circle((0, 0), 0.08, color="white", zorder=1))

    # Goals
    for gx in [-field_x / 2, field_x / 2]:
        ax.plot([gx, gx], [-goal_half, goal_half], "w-", linewidth=3, zorder=1)

    # Team colors
    team_a_color = "#4FC3F7"  # blue
    team_b_color = "#FF5252"  # red
    ball_color = "#FFD700"

    # Robot labels: first 3 = Team A, last 3 = Team B
    n_robots = data["n_clients"]
    n_team_a = n_robots // 2
    robot_names = list(log[0].get("robots", {}).keys())

    # Pre-allocate trail arrays
    ball_trail_x, ball_trail_y = [], []

    # Dynamic elements (will be updated each frame)
    robot_dots = []
    robot_labels = []
    ball_dot = None
    ball_trail_line = None
    score_text = None

    def init():
        nonlocal ball_dot, ball_trail_line, score_text
        for d in robot_dots:
            d.remove()
        robot_dots.clear()
        for l in robot_labels:
            l.remove()
        robot_labels.clear()

        if ball_dot:
            ball_dot.remove()
        if ball_trail_line:
            ball_trail_line.remove()

        ball_dot = plt.Circle((0, 0), 0.15, color=ball_color, zorder=10, ec="white", lw=0.5)
        ax.add_patch(ball_dot)
        ball_trail_line, = ax.plot([], [], "-", color=ball_color, alpha=0.3, linewidth=1.5, zorder=5)
        ball_trail_x.clear()
        ball_trail_y.clear()

    def update(frame):
        nonlocal ball_dot, ball_trail_line, score_text
        if frame >= n_steps:
            return

        entry = log[frame]
        # Clear previous frame elements
        for d in robot_dots:
            d.remove()
        robot_dots.clear()
        for l in robot_labels:
            l.remove()
        robot_labels.clear()

        # Draw robots
        for i, (name, state) in enumerate(entry.get("robots", {}).items()):
            x = state.get("x", 0)
            y = state.get("y", 0)
            pitch = state.get("pitch", 0)
            fallen = abs(pitch) > 30

            if i < n_team_a:
                color = team_a_color
                label = f"A{i+1}"
            else:
                color = team_b_color
                label = f"B{i - n_team_a + 1}"

            marker = "o" if not fallen else "x"
            size = 180 if not fallen else 100
            dot = ax.scatter(x, y, c=color, s=size, marker=marker, zorder=8,
                             edgecolors="white", linewidths=0.8)
            robot_dots.append(dot)

            txt = ax.annotate(label, (x, y), fontsize=7, ha="center", va="center",
                              color="white", fontweight="bold", zorder=9)
            robot_labels.append(txt)

            # Fallen indicator
            if fallen:
                fall_txt = ax.annotate("💥", (x, y + 0.3), fontsize=6, ha="center", zorder=9)
                robot_labels.append(fall_txt)

        # Draw ball
        ball = entry.get("ball", {})
        bx, by = ball.get("x", 0), ball.get("y", 0)
        ball_dot.center = (bx, by)

        # Ball trail (last 50 frames)
        ball_trail_x.append(bx)
        ball_trail_y.append(by)
        if len(ball_trail_x) > 50:
            ball_trail_x.pop(0)
            ball_trail_y.pop(0)
        ball_trail_line.set_data(ball_trail_x, ball_trail_y)

        # Score / time display
        if score_text:
            score_text.remove()
        elapsed = entry.get("t", 0)
        collisions = entry.get("collisions", 0)
        score_text = ax.text(
            0, -field_y / 2 - 0.6,
            f"Time: {elapsed:.1f}s  |  Collisions: {collisions}",
            ha="center", fontsize=10, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", alpha=0.8))

    # Title
    ax.text(0, field_y / 2 + 0.5, "3v3 Soccer Match — RL Team A (Blue) vs Rule Team B (Red)",
            ha="center", fontsize=12, color="white", fontweight="bold")

    # Sample every Nth frame for desired FPS
    sample_step = max(1, n_steps // (fps * int(data["duration"])))
    frames = list(range(0, n_steps, sample_step))

    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         blit=False, interval=1000 / fps, repeat=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anim.save(output_path, writer="ffmpeg", fps=fps,
              savefig_kwargs={"facecolor": "#0a1a0a"})
    plt.close(fig)
    print(f"Video saved: {output_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="Match log JSON path")
    parser.add_argument("--output", default="demos/3v3_match.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    data = load_match_log(log_path)
    render_match(data, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
