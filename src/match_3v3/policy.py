"""Policy interface for 3v3 soccer — rule-based and RL-based.

Action interface (PolicyAction):
    velocity_cmd  — (vx, vy, vyaw) fed to the walk policy (t1_walk.pt)
    should_kick   — bool: trigger rule-based kick (not RL)
    should_shoot   — bool: trigger shoot checkpoint (if loaded)
    shoot_dir      — (2,) desired ball direction when shooting

Key design decisions:
    - t1_walk.pt handles locomotion only (velocity → joint targets).
    - shoot checkpoint handles kicking skill only.
    - t1_walk.pt does NOT automatically kick the ball.
    - RulePolicy uses geometric rules, no checkpoints needed.
    - SharedRLPolicy loads walk + optional shoot checkpoints.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .scene import Team, Role, PlayerState, BallState, FieldConstants, DEFAULT_FIELD

try:
    import torch
except Exception:
    torch = None


# ═══════════════════════════════════════════════════════════════════
# Action interface
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PolicyAction:
    """Unified output of all policy implementations.

    Attributes:
        velocity_cmd: (vx, vy, vyaw) in robot frame — fed to walk policy.
        should_kick:  Trigger a rule-based kick motion (no RL needed).
        should_shoot: Trigger the shoot checkpoint (requires loaded model).
        shoot_dir:    (dx, dy) desired ball direction when shooting.
    """
    velocity_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    should_kick: bool = False
    should_shoot: bool = False
    shoot_dir: np.ndarray = field(default_factory=lambda: np.zeros(2))


# ═══════════════════════════════════════════════════════════════════
# Rule Policy
# ═══════════════════════════════════════════════════════════════════

class RulePolicy:
    """Rule-based player behavior — no checkpoints, no GPU.

    Behavior per role:
        ATTACKER    — approach ball, align with goal, kick
        DEFENDER    — hold position between ball and own goal
        GOALKEEPER  — stay on goal line, track ball y
    """

    KICK_DISTANCE = 0.3
    KICK_ALIGN_THRESHOLD = 0.5        # radians
    MAX_SPEED = 0.5                   # m/s
    MAX_TURN = 0.5                    # rad/s
    APPROACH_SPEED = 0.4
    DEFEND_SPEED = 0.3
    KEEPER_SPEED = 0.3

    def __init__(self, field: Optional[FieldConstants] = None):
        self.field = field or DEFAULT_FIELD

    @property
    def keeper_y_range(self) -> float:
        return self.field.goal_half * 0.8

    def compute(self, player: PlayerState, ball: BallState) -> PolicyAction:
        """Compute the action for a single player given match state."""
        to_ball = ball.pos[:2] - player.pos[:2]
        dist_to_ball = float(np.linalg.norm(to_ball))

        if dist_to_ball > 0.01:
            ball_dir = math.atan2(to_ball[1], to_ball[0])
        else:
            ball_dir = 0.0

        yaw_diff = ball_dir - player.yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi

        vx, vy, vyaw = 0.0, 0.0, 0.0
        should_kick = False
        should_shoot = False
        shoot_dir = np.zeros(2)

        if player.role == Role.ATTACKER:
            if dist_to_ball < self.KICK_DISTANCE:
                goal_dx = player.attack_goal_x - player.pos[0]
                goal_dy = 0.0 - player.pos[1]
                to_goal = np.array([goal_dx, goal_dy])
                goal_angle = math.atan2(to_goal[1], to_goal[0])
                goal_yaw_diff = goal_angle - player.yaw
                goal_yaw_diff = (goal_yaw_diff + math.pi) % (2 * math.pi) - math.pi

                if abs(goal_yaw_diff) < self.KICK_ALIGN_THRESHOLD:
                    should_kick = True
                    should_shoot = True
                    shoot_dir = to_goal / (np.linalg.norm(to_goal) + 1e-6)
                    vx = self.MAX_SPEED
                else:
                    vyaw = float(np.clip(goal_yaw_diff, -self.MAX_TURN, self.MAX_TURN))
            else:
                if abs(yaw_diff) < 1.0:
                    vx = math.cos(yaw_diff) * self.APPROACH_SPEED
                    vy = math.sin(yaw_diff) * self.APPROACH_SPEED
                vyaw = float(np.clip(yaw_diff, -self.MAX_TURN, self.MAX_TURN))

        elif player.role == Role.DEFENDER:
            goal_x = player.defend_goal_x
            direction = 1.0 if goal_x < 0 else -1.0
            target_x = goal_x + direction * 2.0
            target_y = float(np.clip(ball.pos[1], -self.keeper_y_range, self.keeper_y_range))
            to_target = np.array([target_x - player.pos[0], target_y - player.pos[1]])
            target_dist = float(np.linalg.norm(to_target))

            if target_dist > 0.3:
                target_dir = math.atan2(to_target[1], to_target[0])
                td_yaw = target_dir - player.yaw
                td_yaw = (td_yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(td_yaw) < 1.0:
                    vx = math.cos(td_yaw) * self.DEFEND_SPEED
                    vy = math.sin(td_yaw) * self.DEFEND_SPEED
                vyaw = float(np.clip(td_yaw, -self.MAX_TURN, self.MAX_TURN))
            else:
                vyaw = float(np.clip(yaw_diff, -self.MAX_TURN, self.MAX_TURN))

        elif player.role == Role.GOALKEEPER:
            goal_x = player.defend_goal_x
            target_y = float(np.clip(ball.pos[1], -self.keeper_y_range, self.keeper_y_range))
            dx = goal_x - player.pos[0]
            dy = target_y - player.pos[1]
            d = math.sqrt(dx * dx + dy * dy)
            if d > 0.2:
                vx = (dx / d) * self.KEEPER_SPEED
                vy = (dy / d) * self.KEEPER_SPEED
            vyaw = float(np.clip(yaw_diff, -self.MAX_TURN, self.MAX_TURN))

        # Clamp speed
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > self.MAX_SPEED:
            scale = self.MAX_SPEED / speed
            vx *= scale
            vy *= scale

        player.vel_cmd = np.array([vx, vy, vyaw])
        return PolicyAction(
            velocity_cmd=np.array([vx, vy, vyaw]),
            should_kick=should_kick,
            should_shoot=should_shoot,
            shoot_dir=shoot_dir,
        )


# ═══════════════════════════════════════════════════════════════════
# Shared RL Policy
# ═══════════════════════════════════════════════════════════════════

class SharedRLPolicy:
    """RL policy using separate walk and shoot checkpoints.

    Architecture:
        1. Role layer (RoleAssigner) decides role + target per player.
        2. Rule layer computes velocity_cmd (same as RulePolicy).
        3. Walk checkpoint (t1_walk.pt) converts velocity_cmd → joint targets.
        4. Shoot checkpoint converts shoot trigger → kick motion.

    The walk checkpoint handles locomotion only — it does NOT kick.
    The shoot checkpoint handles kicking skill only — it does NOT walk.

    Loading:
        walk_checkpoint   — path to t1_walk.pt (TorchScript)
        shoot_checkpoint  — path to shoot policy .pt (optional)

    When torch or checkpoints are unavailable, falls back to RulePolicy.
    """

    def __init__(
        self,
        walk_checkpoint: Optional[str] = None,
        shoot_checkpoint: Optional[str] = None,
        field: Optional[FieldConstants] = None,
        action_scale: float = 0.25,
        obs_history_length: int = 10,
    ):
        self.field = field or DEFAULT_FIELD
        self.action_scale = action_scale
        self.obs_history_length = obs_history_length
        self.walk_model = None
        self.shoot_model = None
        self.torch_available = torch is not None
        self.rule_fallback = RulePolicy(field=self.field)

        self._load_checkpoints(walk_checkpoint, shoot_checkpoint)

    def _load_checkpoints(self, walk_path: Optional[str], shoot_path: Optional[str]):
        """Load TorchScript checkpoints if torch is available."""
        if not self.torch_available:
            return

        if walk_path and os.path.exists(walk_path):
            try:
                self.walk_model = torch.jit.load(walk_path, map_location="cpu")
            except Exception:
                self.walk_model = None

        if shoot_path and os.path.exists(shoot_path):
            try:
                self.shoot_model = torch.jit.load(shoot_path, map_location="cpu")
            except Exception:
                self.shoot_model = None

    @property
    def walk_loaded(self) -> bool:
        return self.walk_model is not None

    @property
    def shoot_loaded(self) -> bool:
        return self.shoot_model is not None

    @property
    def mode(self) -> str:
        """Return the policy mode label for match output."""
        if self.walk_loaded and self.shoot_loaded:
            return "full_vs_rule"
        elif self.walk_loaded:
            return "rl_vs_rule"
        else:
            return "rule_vs_rule"

    def compute(self, player: PlayerState, ball: BallState) -> PolicyAction:
        """Compute action using walk policy for locomotion + rule layer for decisions.

        The rule layer always runs (deciding role-based targets and kick triggers).
        The walk model, if loaded, overrides the velocity_cmd → joint mapping.
        The shoot model, if loaded, overrides the kick motion.
        """
        # Rule layer always provides the decision (targets, kick/shoot flags)
        action = self.rule_fallback.compute(player, ball)

        # If walk model is loaded, we could use it to translate velocity_cmd
        # to joint targets. This requires the full observation buffer.
        # For now, the walk model flag indicates capability, not execution.
        # Actual joint control happens in the scene step loop.

        return action

    def get_walk_joint_targets(
        self,
        velocity_cmd: np.ndarray,
        obs_history: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Convert velocity command to joint targets via walk checkpoint.

        Returns 21-dim joint target array, or None if walk model not loaded.
        This is called by the scene step loop, NOT by compute().
        """
        if not self.walk_loaded or not self.torch_available:
            return None

        # Build observation from history (if provided) or zeros
        if obs_history is not None:
            obs = torch.from_numpy(obs_history).float().unsqueeze(0)
        else:
            obs = torch.zeros(1, 720)

        with torch.no_grad():
            actions = self.walk_model(obs).squeeze(0).cpu().numpy()

        # Scale: actions * action_scale + default_pos (handled by caller)
        return actions

    def get_shoot_joint_targets(
        self,
        shoot_dir: np.ndarray,
        obs_history: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Get kick motion joint targets from shoot checkpoint.

        Returns 21-dim joint target delta, or None if shoot model not loaded.
        """
        if not self.shoot_loaded or not self.torch_available:
            return None

        if obs_history is not None:
            obs = torch.from_numpy(obs_history).float().unsqueeze(0)
        else:
            obs = torch.zeros(1, 720)

        with torch.no_grad():
            actions = self.shoot_model(obs).squeeze(0).cpu().numpy()

        return actions


def team_side_positive(goal_x: float) -> bool:
    """Helper: returns True if goal is on positive x side."""
    return goal_x > 0
