"""Humanoid soccer reward curriculum (v2).

Reward design:
  安全推进: +2    (lin_vel_x forward motion)
  保持控球: +1    (ball within control radius)
  进入有利进攻区域: +3  (ball in opponent half)
  成功传球: +5    (multi-agent, skip for single robot)
  丢球: -10      (ball moving away from goal)
  进入对手高压区域: -4  (ball in own penalty area)
  碰撞/摔倒: -20  (fall penalty)
  进球: +100     (goal scored)
"""
from __future__ import annotations

import torch


def r_upright(torso_up: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torso_up, min=0.0)


def r_alive(fallen: torch.Tensor) -> torch.Tensor:
    return (~fallen).float()


def r_lin_vel_x(base_lin_vel_x: torch.Tensor) -> torch.Tensor:
    """安全推进: 奖励向前移动 (+2 per step at 1 m/s)."""
    return torch.clamp(base_lin_vel_x, min=-0.5, max=2.0)


def r_approach_ball(dist_to_ball: torch.Tensor, prev_dist: torch.Tensor) -> torch.Tensor:
    """接近球: 距离减小的差值."""
    return (prev_dist - dist_to_ball)


def r_ball_control(dist_to_ball: torch.Tensor, radius: float) -> torch.Tensor:
    """保持控球: +1 when ball within control radius."""
    return torch.exp(-torch.clamp(dist_to_ball - radius, min=0.0) * 3.0)


def r_ball_to_goal(ball_vel_to_goal: torch.Tensor) -> torch.Tensor:
    return torch.clamp(ball_vel_to_goal, min=0.0)


def r_goal(scored: torch.Tensor) -> torch.Tensor:
    """进球: +100."""
    return scored.float()


def r_fall(fallen: torch.Tensor) -> torch.Tensor:
    """碰撞/摔倒: -20."""
    return fallen.float()


def r_recovery(just_recovered: torch.Tensor) -> torch.Tensor:
    return just_recovered.float()


def r_energy(action: torch.Tensor) -> torch.Tensor:
    return torch.sum(action ** 2, dim=-1)


def r_advancing_zone(ball_x: torch.Tensor, field_x: float) -> torch.Tensor:
    """进入有利进攻区域: ball in opponent half (+3)."""
    half = field_x / 2.0
    in_opp_half = (ball_x > 0).float()
    return in_opp_half


def r_danger_zone(ball_x: torch.Tensor, field_x: float, penalty_len: float) -> torch.Tensor:
    """进入对手高压区域: ball in own penalty area (triggers -4)."""
    own_penalty_edge = -field_x / 2.0 + penalty_len
    in_danger = (ball_x < own_penalty_edge).float()
    return in_danger


def r_lose_ball(ball_vel_to_goal: torch.Tensor, prev_ball_vel_to_goal: torch.Tensor) -> torch.Tensor:
    """丢球: ball moving away from goal (-10)."""
    return torch.clamp(-ball_vel_to_goal, min=0.0) * (prev_ball_vel_to_goal > 0).float()


TASK_TERMS = {
    "balance": {"upright", "alive", "lin_vel", "fall", "recovery", "energy"},
    "chase":   {"upright", "alive", "lin_vel", "approach_ball", "ball_control", "fall", "recovery", "energy", "advancing_zone", "danger_zone"},
    "dribble": {"upright", "alive", "lin_vel", "approach_ball", "ball_control", "fall", "recovery", "energy", "advancing_zone", "danger_zone"},
    "shoot":   {"upright", "alive", "lin_vel", "ball_control", "ball_to_goal", "goal_scored", "fall", "recovery", "energy", "advancing_zone", "danger_zone"},
    "coop":    {"upright", "alive", "lin_vel", "ball_control", "ball_to_goal", "goal_scored", "fall", "recovery", "energy", "advancing_zone", "danger_zone"},
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
    if "advancing_zone" in terms:
        total += w["advancing_zone"] * r_advancing_zone(obs["ball_x"], w.get("_field_x", 14.0))
    if "danger_zone" in terms:
        total += w["danger_zone"] * r_danger_zone(obs["ball_x"], w.get("_field_x", 14.0), w.get("_penalty_len", 3.0))

    # penalties
    total += w["fall_penalty"] * r_fall(obs["fallen"])
    total += w["recovery_bonus"] * r_recovery(obs["just_recovered"])
    total += w["energy_penalty"] * r_energy(action)
    return total
