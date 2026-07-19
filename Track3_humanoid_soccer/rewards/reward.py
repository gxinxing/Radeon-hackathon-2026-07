"""Humanoid soccer reward (v3).

总奖励 = 控球奖励 + 进攻空间奖励 + 安全传球奖励 + 射门/进球奖励
       - 丢球惩罚 - 碰撞惩罚 - 危险动作惩罚

拆分逻辑:
  控球奖励 = ball_control(保持控球 +1) + approach_ball(接近球)
  进攻空间奖励 = lin_vel(安全推进 +2) + advancing_zone(进入有利区域 +3)
  安全传球奖励 = ball_to_goal(球向球门移动) (单机器人无传球,用球方向替代)
  射门/进球奖励 = goal_scored(+100)
  丢球惩罚 = ball moving away from goal (-10)
  碰撞惩罚 = 碰撞/摔倒 (-5, 不太重,鼓励尝试)
  危险动作惩罚 = energy penalty + danger zone (-1)
"""
from __future__ import annotations

import torch


def r_upright(torso_up: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torso_up, min=0.0)


def r_alive(fallen: torch.Tensor) -> torch.Tensor:
    return (~fallen).float()


def r_lin_vel_x(base_lin_vel_x: torch.Tensor) -> torch.Tensor:
    """安全推进: 奖励向前移动."""
    return torch.clamp(base_lin_vel_x, min=-0.5, max=2.0)


def r_approach_ball(dist_to_ball: torch.Tensor, prev_dist: torch.Tensor) -> torch.Tensor:
    """控球: 接近球."""
    return (prev_dist - dist_to_ball)


def r_ball_control(dist_to_ball: torch.Tensor, radius: float) -> torch.Tensor:
    """保持控球: +1 when ball within control radius."""
    return torch.exp(-torch.clamp(dist_to_ball - radius, min=0.0) * 3.0)


def r_ball_to_goal(ball_vel_to_goal: torch.Tensor) -> torch.Tensor:
    """安全传球/射门: 球向球门移动."""
    return torch.clamp(ball_vel_to_goal, min=0.0)


def r_goal(scored: torch.Tensor) -> torch.Tensor:
    """进球: +100."""
    return scored.float()


def r_fall(fallen: torch.Tensor) -> torch.Tensor:
    """碰撞惩罚: 摔倒 -5 (不重,鼓励尝试移动)."""
    return fallen.float()


def r_recovery(just_recovered: torch.Tensor) -> torch.Tensor:
    return just_recovered.float()


def r_energy(action: torch.Tensor) -> torch.Tensor:
    """危险动作惩罚: 动作过大."""
    return torch.sum(action ** 2, dim=-1)


def r_advancing_zone(ball_x: torch.Tensor, field_x: float) -> torch.Tensor:
    """进攻空间: 进入有利进攻区域 +3."""
    return (ball_x > 0).float()


def r_danger_zone(ball_x: torch.Tensor, field_x: float, penalty_len: float) -> torch.Tensor:
    """危险区域: 球在我方禁区 -1."""
    own_penalty_edge = -field_x / 2.0 + penalty_len
    return (ball_x < own_penalty_edge).float()


def r_lose_ball(ball_vel_to_goal: torch.Tensor) -> torch.Tensor:
    """丢球惩罚: 球远离球门 -10."""
    return torch.clamp(-ball_vel_to_goal, min=0.0)


TASK_TERMS = {
    "balance": {"upright", "alive", "lin_vel", "fall", "recovery", "energy"},
    "chase":   {"upright", "alive", "lin_vel", "approach_ball", "ball_control", "ball_to_goal", "fall", "recovery", "energy", "advancing_zone", "danger_zone", "lose_ball"},
    "dribble": {"upright", "alive", "lin_vel", "approach_ball", "ball_control", "ball_to_goal", "fall", "recovery", "energy", "advancing_zone", "danger_zone", "lose_ball"},
    "shoot":   {"upright", "alive", "lin_vel", "ball_control", "ball_to_goal", "goal_scored", "fall", "recovery", "energy", "advancing_zone", "danger_zone", "lose_ball"},
    "coop":    {"upright", "alive", "lin_vel", "ball_control", "ball_to_goal", "goal_scored", "fall", "recovery", "energy", "advancing_zone", "danger_zone", "lose_ball"},
}


def compute_reward(obs: dict, action: torch.Tensor, w: dict, task: str) -> torch.Tensor:
    terms = TASK_TERMS.get(task, TASK_TERMS["chase"])
    total = torch.zeros_like(obs["torso_up"])

    # 正向奖励
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

    # 负向惩罚
    if "fall" in terms:
        total += w["fall_penalty"] * r_fall(obs["fallen"])
    if "danger_zone" in terms:
        total += w["danger_zone"] * r_danger_zone(obs["ball_x"], w.get("_field_x", 14.0), w.get("_penalty_len", 3.0))
    if "lose_ball" in terms:
        total += w["lose_ball"] * r_lose_ball(obs["ball_vel_to_goal"])
    total += w["recovery_bonus"] * r_recovery(obs["just_recovered"])
    total += w["energy_penalty"] * r_energy(action)
    return total
