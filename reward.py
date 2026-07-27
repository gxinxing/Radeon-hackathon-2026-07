"""Booster-derived reward curriculum for humanoid soccer.

Adds gait-related rewards from booster_gym:
    tracking_lin_vel_x/y  -> velocity tracking (exponential kernel)
    tracking_ang_vel      -> angular velocity tracking
    feet_swing            -> rewards correct swing phase per gait cycle
    feet_slip             -> penalize foot sliding when in contact
"""
from __future__ import annotations

import torch


def r_upright(torso_up: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torso_up, min=0.0)


def r_alive(fallen: torch.Tensor) -> torch.Tensor:
    return (~fallen).float()


def r_tracking_lin_vel_x(command, actual, sigma):
    return torch.exp(-torch.square(command - actual) / sigma)


def r_tracking_lin_vel_y(command, actual, sigma):
    return torch.exp(-torch.square(command - actual) / sigma)


def r_tracking_ang_vel(command, actual, sigma):
    return torch.exp(-torch.square(command - actual) / sigma)


def r_feet_swing(gait_process, gait_frequency, feet_contact, swing_period):
    left_swing = (torch.abs(gait_process - 0.25) < 0.5 * swing_period) & (gait_frequency > 1e-8)
    right_swing = (torch.abs(gait_process - 0.75) < 0.5 * swing_period) & (gait_frequency > 1e-8)
    return (left_swing & ~feet_contact[:, 0]).float() + (right_swing & ~feet_contact[:, 1]).float()


def r_feet_slip(feet_pos, last_feet_pos, feet_contact, dt, episode_length_buf):
    vel_sq = torch.square((last_feet_pos - feet_pos) / dt).sum(dim=-1)
    return (vel_sq * feet_contact.float()).sum(dim=-1) * (episode_length_buf > 1).float()


def r_approach_ball(dist_to_ball, prev_dist):
    return (prev_dist - dist_to_ball)


def r_ball_control(dist_to_ball, radius):
    return torch.exp(-torch.clamp(dist_to_ball - radius, min=0.0) * 3.0)


def r_ball_to_goal(ball_vel_to_goal):
    return torch.clamp(ball_vel_to_goal, min=0.0)


def r_goal(scored):
    return scored.float()


def r_fall(fallen):
    return fallen.float()


def r_recovery(just_recovered):
    return just_recovered.float()


def r_energy(action):
    return torch.sum(action ** 2, dim=-1)


def r_lin_vel_z(base_lin_vel_z):
    return torch.square(base_lin_vel_z)


def r_ang_vel_xy(base_ang_vel_xy):
    return torch.sum(torch.square(base_ang_vel_xy), dim=-1)


def r_orientation(projected_gravity_xy):
    return torch.sum(torch.square(projected_gravity_xy), dim=-1)


def r_action_rate(last_actions, actions):
    return torch.sum(torch.square(last_actions - actions), dim=-1)


def r_command_penalty(actions):
    """Penalize non-zero velocity commands — encourages 'do nothing' first."""
    return torch.sum(torch.square(actions), dim=-1)


def r_dof_acc(last_dof_vel, dof_vel, dt):
    return torch.sum(torch.square((last_dof_vel - dof_vel) / dt), dim=-1)


TASK_TERMS = {
    "balance": {"upright", "alive", "tracking_lin_vel_x", "tracking_lin_vel_y", "tracking_ang_vel",
                "feet_swing", "feet_slip", "lin_vel_z", "ang_vel_xy", "orientation",
                "fall", "recovery", "energy", "action_rate", "dof_acc"},
    "chase":   {"upright", "alive", "tracking_lin_vel_x", "tracking_lin_vel_y", "tracking_ang_vel",
                "feet_swing", "feet_slip", "approach_ball",
                "lin_vel_z", "ang_vel_xy", "orientation",
                "fall", "recovery", "energy", "action_rate", "dof_acc"},
    "dribble": {"upright", "alive", "tracking_lin_vel_x", "tracking_lin_vel_y", "tracking_ang_vel",
                "feet_swing", "feet_slip", "approach_ball", "ball_control",
                "lin_vel_z", "ang_vel_xy", "orientation",
                "fall", "recovery", "energy", "action_rate", "dof_acc"},
    "shoot":   {"upright", "alive", "tracking_lin_vel_x", "tracking_lin_vel_y", "tracking_ang_vel",
                "feet_swing", "feet_slip", "ball_control", "ball_to_goal", "goal_scored",
                "lin_vel_z", "ang_vel_xy", "orientation",
                "fall", "recovery", "energy", "action_rate", "dof_acc"},
    "coop":    {"upright", "alive", "tracking_lin_vel_x", "tracking_lin_vel_y", "tracking_ang_vel",
                "feet_swing", "feet_slip", "ball_control", "ball_to_goal", "goal_scored",
                "lin_vel_z", "ang_vel_xy", "orientation",
                "fall", "recovery", "energy", "action_rate", "dof_acc"},
    # Hierarchical: frozen low-level handles gait; high-level outputs velocity commands.
    # Drops tracking/feet terms (low-level concern), keeps ball-focused + safety terms.
    "chase_hl": {"upright", "alive", "approach_ball", "ball_control", "ball_to_goal", "goal_scored",
                 "lin_vel_z", "ang_vel_xy", "orientation",
                 "fall", "recovery", "action_rate"},
    "balance_hl": {"upright", "alive", "lin_vel_z", "ang_vel_xy", "orientation",
                   "fall", "recovery", "action_rate", "command_penalty"},
}


def compute_reward(obs: dict, action: torch.Tensor, w: dict, task: str) -> torch.Tensor:
    terms = TASK_TERMS.get(task, TASK_TERMS["chase"])
    total = torch.zeros_like(obs["torso_up"])
    sigma = w.get("tracking_sigma", 0.25)

    if "upright" in terms:
        total += w["upright"] * r_upright(obs["torso_up"])
    if "alive" in terms:
        total += w["alive"] * r_alive(obs["fallen"])
    if "tracking_lin_vel_x" in terms:
        total += w.get("tracking_lin_vel_x", 1.0) * r_tracking_lin_vel_x(
            obs["commands"][:, 0], obs["base_lin_vel_x"], sigma)
    if "tracking_lin_vel_y" in terms:
        total += w.get("tracking_lin_vel_y", 1.0) * r_tracking_lin_vel_y(
            obs["commands"][:, 1], obs["base_lin_vel_y_y"], sigma)
    if "tracking_ang_vel" in terms:
        total += w.get("tracking_ang_vel", 0.5) * r_tracking_ang_vel(
            obs["commands"][:, 2], obs["base_ang_vel_z"], sigma)
    if "feet_swing" in terms:
        total += w.get("feet_swing", 3.0) * r_feet_swing(
            obs["gait_process"], obs["gait_frequency"], obs["feet_contact"],
            w.get("swing_period", 0.2))
    if "feet_slip" in terms:
        total += w.get("feet_slip", -0.1) * r_feet_slip(
            obs["feet_pos"], obs["last_feet_pos"], obs["feet_contact"],
            w.get("dt", 0.02), obs["episode_length_buf"])
    if "approach_ball" in terms:
        total += w["approach_ball"] * r_approach_ball(obs["dist_to_ball"], obs["prev_dist_to_ball"])
    if "ball_control" in terms:
        total += w["ball_control"] * r_ball_control(obs["dist_to_ball"], w.get("_ball_radius", 0.11))
    if "ball_to_goal" in terms:
        total += w["ball_to_goal"] * r_ball_to_goal(obs["ball_vel_to_goal"])
    if "goal_scored" in terms:
        total += w["goal_scored"] * r_goal(obs["scored"])
    if "lin_vel_z" in terms:
        total += w.get("lin_vel_z", -2.0) * r_lin_vel_z(obs.get("base_lin_vel_z", torch.zeros_like(obs["torso_up"])))
    if "ang_vel_xy" in terms:
        total += w.get("ang_vel_xy", -0.2) * r_ang_vel_xy(obs.get("base_ang_vel_xy", torch.zeros_like(obs["torso_up"]).unsqueeze(1).expand(-1, 2)))
    if "orientation" in terms:
        total += w.get("orientation", -5.0) * r_orientation(obs.get("projected_gravity_xy", torch.zeros_like(obs["torso_up"]).unsqueeze(1).expand(-1, 2)))

    total += w["fall_penalty"] * r_fall(obs["fallen"])
    total += w["recovery_bonus"] * r_recovery(obs["just_recovered"])
    total += w["energy_penalty"] * r_energy(action)

    if "action_rate" in terms and "last_actions" in obs:
        total += w.get("action_rate", -1.0) * r_action_rate(obs["last_actions"], action)
    if "command_penalty" in terms:
        total += w.get("command_penalty", -1.0) * r_command_penalty(action)
    if "dof_acc" in terms and "last_dof_vel" in obs:
        total += w.get("dof_acc", -1e-7) * r_dof_acc(obs["last_dof_vel"], obs.get("dof_vel", obs["last_dof_vel"]), w.get("dt", 0.02))

    return total
