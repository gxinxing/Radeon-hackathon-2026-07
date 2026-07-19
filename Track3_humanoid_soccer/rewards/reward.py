"""Booster-derived reward curriculum for humanoid soccer.

Real Booster T1 RoBoLeague 3v3 skills -> RL reward terms:

    stay upright / footwork      -> upright, alive
    walk forward                 -> lin_vel (booster_gym core reward)
    chase the ball               -> approach_ball
    dribble / keep control       -> ball_control
    shoot toward goal            -> ball_to_goal, goal_scored
    get up after a fall          -> fall_penalty, recovery_bonus
    efficient movement           -> energy_penalty

Each helper is a pure function of tensors so it runs batched on the AMD GPU.
`compute_reward` selects/weights terms per the active task, letting you run a
curriculum: balance -> chase -> dribble -> shoot -> coop.
"""
from __future__ import annotations

import torch


def r_upright(torso_up: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torso_up, min=0.0)


def r_alive(fallen: torch.Tensor) -> torch.Tensor:
    return (~fallen).float()


def r_lin_vel_x(base_lin_vel_x: torch.Tensor) -> torch.Tensor:
    """Reward forward motion (+x). Core reward from booster_gym."""
    return torch.clamp(base_lin_vel_x, min=-0.5, max=2.0)


def r_approach_ball(dist_to_ball: torch.Tensor, prev_dist: torch.Tensor) -> torch.Tensor:
    return (prev_dist - dist_to_ball)


def r_ball_control(dist_to_ball: torch.Tensor, radius: float) -> torch.Tensor:
    return torch.exp(-torch.clamp(dist_to_ball - radius, min=0.0) * 3.0)


def r_ball_to_goal(ball_vel_to_goal: torch.Tensor) -> torch.Tensor:
    return torch.clamp(ball_vel_to_goal, min=0.0)


def r_goal(scored: torch.Tensor) -> torch.Tensor:
    return scored.float()


def r_fall(fallen: torch.Tensor) -> torch.Tensor:
    return fallen.float()


def r_recovery(just_recovered: torch.Tensor) -> torch.Tensor:
    return just_recovered.float()


def r_energy(action: torch.Tensor) -> torch.Tensor:
    return torch.sum(action ** 2, dim=-1)


# tasks -> which terms are active (curriculum gating)
TASK_TERMS = {
    "balance": {"upright", "alive", "lin_vel", "fall", "recovery", "energy"},
    "chase":   {"upright", "alive", "lin_vel", "approach_ball", "fall", "recovery", "energy"},
    "dribble": {"upright", "alive", "lin_vel", "approach_ball", "ball_control", "fall", "recovery", "energy"},
    "shoot":   {"upright", "alive", "lin_vel", "ball_control", "ball_to_goal", "goal_scored", "fall", "recovery", "energy"},
    "coop":    {"upright", "alive", "lin_vel", "ball_control", "ball_to_goal", "goal_scored", "fall", "recovery", "energy"},
}


def compute_reward(obs: dict, action: torch.Tensor, w: dict, task: str) -> torch.Tensor:
    terms = TASK_TERMS.get(task, TASK_TERMS["chase"])
    total = torch.zeros_like(obs["torso_up"])

    if "upright" in terms:
        total += w["upright"] * r_upright(obs["torso_up"])
    if "alive" in terms:
        total += w["alive"] * r_alive(obs["fallen"])
    if "lin_vel" in terms:
        total += w["lin_vel"] * r_lin_vel_x(obs["base_lin_vel_x"])
    if "approach_ball" in terms:
        total += w["approach_ball"] * r_approach_ball(obs["dist_to_ball"], obs["prev_dist_to_ball"])
    if "ball_control" in terms:
        total += w["ball_control"] * r_ball_control(obs["dist_to_ball"], w.get("_ball_radius", 0.11))
    if "ball_to_goal" in terms:
        total += w["ball_to_goal"] * r_ball_to_goal(obs["ball_vel_to_goal"])
    if "goal_scored" in terms:
        total += w["goal_scored"] * r_goal(obs["scored"])

    total += w["fall_penalty"] * r_fall(obs["fallen"])
    total += w["recovery_bonus"] * r_recovery(obs["just_recovered"])
    total += w["energy_penalty"] * r_energy(action)
    return total
