"""Distillation logger: records the exact observables Booster's behavioral rules
consume, so a trained Genesis policy can be *distilled* into Booster main.py heuristics.

Pure-python (no torch/genesis) so it imports and unit-tests on macOS. At runtime on
the AMD cloud, soccer_env.step() feeds it tensors; we convert to floats here.

Schema (one row per env-step):
  t                float  episode time (s)
  robot_x,y        float  robot base position (m)
  robot_yaw        float  heading (rad)
  ball_x,y         float  ball position (m)
  ball_vx,vy       float  ball velocity (m/s)
  dist_to_ball     float  |robot - ball| in XY (m)
  dist_to_own_goal float  |robot - own goal| (m)   (own goal at -field_x/2)
  dist_to_opp_goal float  |robot - opp goal| (m)   (opp goal at +field_x/2)
  behind_ball      bool    robot is on the own-goal side of the ball (good for kicking toward opp goal)
  ball_to_goal     float   x-component of ball velocity (opp goal is +x); >0 = moving toward goal
  fell             bool    robot considered fallen this step
  recovered        bool    just got up this step
  kicked           bool    ball moving toward opp goal fast AND robot behind ball
"""
from __future__ import annotations

import json
import math

FIELDS = [
    "t", "robot_x", "robot_y", "robot_yaw",
    "ball_x", "ball_y", "ball_vx", "ball_vy",
    "dist_to_ball", "dist_to_own_goal", "dist_to_opp_goal",
    "behind_ball", "ball_to_goal", "fell", "recovered", "kicked",
]


def _f(x):
    """Tensor-or-float -> float. Tolerates Genesis tensors and plain numbers."""
    try:
        if hasattr(x, "detach"):
            return float(x.detach().cpu().item())
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float(x)


def _as_bool(x):
    try:
        if hasattr(x, "detach"):
            return bool(float(x.detach().cpu().item()) > 0.5)
        if hasattr(x, "item"):
            return bool(float(x.item()) > 0.5)
        return bool(x)
    except Exception:
        return bool(x)


class DistillLogger:
    """Accumulates step rows; save()/load() as JSONL."""

    def __init__(self, path):
        self.path = path
        self.rows = []

    def log_step(self, **kw):
        self.rows.append({k: kw.get(k) for k in FIELDS})

    def save(self):
        with open(self.path, "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        return len(self.rows)

    @classmethod
    def load(cls, path):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def record_step(logger, env, soccer):
    """Pull distillation features from a SoccerEnv + its soccer state dict (tensors ok).

    Called from soccer_env.step() when ``env.distill_logger`` is set. Opponent goal
    is at +x (env.goal_x), own goal at -x; this matches Booster's x-axis convention
    once the Genesis field is aligned to Booster's 14x9 pitch.
    """
    bp = env.base_pos
    ball = env.ball_pos
    bvel = env.ball_vel
    own_gx = -env.goal_x
    opp_gx = env.goal_x
    n = env.num_envs
    for i in range(n):
        rx, ry = _f(bp[i, 0]), _f(bp[i, 1])
        bx, by = _f(ball[i, 0]), _f(ball[i, 1])
        bvx, bvy = _f(bvel[i, 0]), _f(bvel[i, 1])
        dtb = math.hypot(rx - bx, ry - by)
        dist_own = math.hypot(rx - own_gx, ry - 0.0)
        dist_opp = math.hypot(rx - opp_gx, ry - 0.0)
        behind = rx < bx  # robot on own-goal side of ball -> can strike toward +x goal
        ball_to_goal = bvx  # opp goal is +x, so x-velocity component = toward goal
        kicked = (ball_to_goal > 0.3) and behind
        logger.log_step(
            t=_f(env.episode_length_buf[i]) * env.dt,
            robot_x=rx, robot_y=ry, robot_yaw=0.0,
            ball_x=bx, ball_y=by, ball_vx=bvx, ball_vy=bvy,
            dist_to_ball=dtb, dist_to_own_goal=dist_own, dist_to_opp_goal=dist_opp,
            behind_ball=behind, ball_to_goal=ball_to_goal,
            fell=_as_bool(soccer["fallen"][i]),
            recovered=_as_bool(soccer["just_recovered"][i]),
            kicked=kicked,
        )
