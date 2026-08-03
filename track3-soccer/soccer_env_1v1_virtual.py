"""1v1 Soccer Environment — RL agent vs virtual opponent (kinematic).

No second Genesis entity needed. Opponent position is computed kinematically
each step (rule-based chase ball), injected into agent's observation as 2 extra
dims (opponent relative xy in body frame). Total obs: 19 + 2 = 21 dims.

This avoids the Genesis ROCm multi-entity crash (hipErrorLaunchFailure)
by NOT loading a second URDF entity in the same scene.

Architecture:
    RL Agent: 21-dim obs → 3-dim action (vx, vy, wz) → frozen t1_walk.pt
    Virtual Opponent: kinematic position update (no physics, no Genesis entity)
"""
from __future__ import annotations
import math, os, torch

try:
    import genesis as gs
except Exception:
    gs = None

from genesis.utils.geom import inv_quat, transform_by_quat

try:
    from envs.soccer_env_hierarchical import SoccerEnvHierarchical
except ImportError:
    from soccer_env_hierarchical import SoccerEnvHierarchical

try:
    from rewards.reward import compute_reward
except ImportError:
    from reward import compute_reward


class SoccerEnv1v1Virtual(SoccerEnvHierarchical):
    """1v1 with virtual (kinematic) opponent — no second Genesis entity.

    Obs: 21 dims (19 base + 2 opponent relative xy in body frame)
    Action: 3 dims (vx, vy, wz)
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 walk_model_path, high_level_decimation=5, show_viewer=False,
                 opponent_speed=0.4, opponent_init_pos=(-3.0, 0.0)):
        self._is_1v1 = False
        self.opponent_speed = opponent_speed
        self.opponent_init_xy = opponent_init_pos
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                         walk_model_path, high_level_decimation, show_viewer)
        self._is_1v1 = True
        # Override obs dim to 21
        self.hl_obs_dim = 21
        self.obs_buf = torch.empty((self.num_envs, 21),
                                   dtype=gs.tc_float, device=self.device)
        # Virtual opponent position (xy only, z=0.7 fixed)
        self.opp_pos = torch.zeros((self.num_envs, 3),
                                   dtype=gs.tc_float, device=self.device)
        self.opp_pos[:, 0] = opponent_init_pos[0]
        self.opp_pos[:, 1] = opponent_init_pos[1]
        self.opp_pos[:, 2] = 0.7
        # Opponent fall state (kinematic, never falls)
        self.opp_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        print(f"[1v1-virtual] Opponent speed: {opponent_speed} m/s")
        print(f"[1v1-virtual] Obs dim: 21 (19 base + 2 opponent relative)")

    def _obs_dim(self):
        return 21

    def _update_virtual_opponent(self):
        """Kinematic opponent: move toward ball at fixed speed."""
        ball_rel = self.ball_pos[:, :2] - self.opp_pos[:, :2]
        dist = torch.norm(ball_rel, dim=1, keepdim=True) + 1e-6
        direction = ball_rel / dist
        # Move at opponent_speed, but don't overshoot
        move = torch.clamp(dist, max=self.opponent_speed * self.high_level_dt)
        self.opp_pos[:, 0] += direction[:, 0] * move.squeeze(-1)
        self.opp_pos[:, 1] += direction[:, 1] * move.squeeze(-1)
        # Keep opponent on field
        self.opp_pos[:, 0] = torch.clamp(self.opp_pos[:, 0],
                                          -self.field_x/2 + 0.5, self.field_x/2 - 0.5)
        self.opp_pos[:, 1] = torch.clamp(self.opp_pos[:, 1],
                                          -self.field_y/2 + 0.5, self.field_y/2 - 0.5)

    def _soccer_state(self):
        """Extend soccer state with opponent info for reward computation."""
        state = super()._soccer_state()
        state["opp_pos"] = self.opp_pos
        state["opp_fallen"] = self.opp_fallen
        # Agent-to-ball distance vs opponent-to-ball distance (possession)
        agent_ball_dist = state["dist_to_ball"]
        opp_ball_dist = torch.norm(self.opp_pos[:, :2] - self.ball_pos[:, :2], dim=1)
        state["opp_ball_dist"] = opp_ball_dist
        state["has_possession"] = (agent_ball_dist < opp_ball_dist).float()
        return state

    def step(self, hl_actions):
        """High-level step: agent acts, virtual opponent moves, both sync."""
        # Agent step (same as parent)
        self.hl_actions = torch.stack([
            torch.clamp(hl_actions[:, 0], -self.hl_clip_lin, self.hl_clip_lin),
            torch.clamp(hl_actions[:, 1], -self.hl_clip_lin, self.hl_clip_lin),
            torch.clamp(hl_actions[:, 2], -self.hl_clip_ang, self.hl_clip_ang),
        ], dim=1)
        self.hl_actions = torch.where(
            torch.abs(self.hl_actions) < 0.05,
            torch.zeros_like(self.hl_actions), self.hl_actions)
        self.commands[:] = self.hl_actions

        super(SoccerEnvHierarchical, self)._update_observation()

        for _ in range(self.high_level_decimation):
            low_obs = self._build_low_level_obs()
            joint_actions = self._run_walk_model(low_obs)
            self._low_level_step(joint_actions)

        # Virtual opponent moves (kinematic, no physics)
        self._update_virtual_opponent()

        # Compute reward with opponent info
        soccer = self._soccer_state()
        w = dict(self.reward_scales)
        w["_ball_radius"] = self.ball_radius
        w["dt"] = self.high_level_dt
        for _k in list(w.keys()):
            if isinstance(w[_k], (int, float)) and _k not in [
                "_ball_radius", "dt", "tracking_sigma", "swing_period",
                "only_positive_rewards", "fall_penalty", "recovery_bonus",
            ]:
                w[_k] *= self.high_level_dt
        soccer["last_actions"] = self.last_hl_actions

        # Use chase_hl reward + possession bonus
        self.rew_buf = compute_reward(soccer, self.hl_actions, w, self.task)

        # Add possession bonus (agent closer to ball than opponent)
        possession_reward = soccer["has_possession"] * 2.0 * self.high_level_dt
        self.rew_buf += possession_reward

        # Metrics
        self._m_goals += soccer["scored"].float().sum()
        self._m_dist += soccer["dist_to_ball"].mean()
        self._m_steps += 1
        if self._m_steps >= 10:
            self.extras["episode"] = {
                "goal_per_1k_steps": (self._m_goals / (10.0 * self.num_envs) * 1000.0).item(),
                "mean_dist_to_ball": (self._m_dist / 10.0).item(),
                "goals_total": float(self.goals_scored),
                "possession_rate": soccer["has_possession"].mean().item(),
            }
            self._m_goals.zero_()
            self._m_dist.zero_()
            self._m_steps = 0

        self._resample_ball_if_needed()

        # Termination
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= soccer["scored"]
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.term_pitch
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.term_roll
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)
        self._update_observation()
        self.last_hl_actions.copy_(self.hl_actions)
        self.fallen_prev.copy_(soccer["fallen"])

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _update_observation(self):
        """Build 21-dim obs: 19 base + 2 opponent relative (body frame)."""
        if not self._is_1v1:
            super()._update_observation()
            return

        inv_bq = inv_quat(self.base_quat)

        # Base 19 dims (same as parent)
        ball_rel = self.ball_pos - self.base_pos
        ball_rel_body = transform_by_quat(ball_rel, inv_bq)
        ball_vel_body = transform_by_quat(self.ball_vel, inv_bq)
        goal_pos = torch.zeros_like(self.base_pos)
        goal_pos[:, 0] = self.goal_x
        goal_rel = goal_pos - self.base_pos
        goal_rel_body = transform_by_quat(goal_rel, inv_bq)
        goal_dist = torch.norm(goal_rel_body[:, :2], dim=1, keepdim=True)
        goal_dir = goal_rel_body[:, :2] / (goal_dist + 1e-6)
        dist_to_ball = torch.norm(ball_rel_body[:, :2], dim=1, keepdim=True)

        # Opponent relative position in body frame (2 dims)
        opp_rel = self.opp_pos - self.base_pos
        opp_rel_body = transform_by_quat(opp_rel, inv_bq)

        self.obs_buf = torch.cat([
            self.filtered_lin_vel,              # 3
            self.filtered_ang_vel,              # 3
            self.projected_gravity[:, :2],      # 2
            ball_rel_body[:, :2],               # 2
            ball_vel_body[:, :2],               # 2
            dist_to_ball,                        # 1
            goal_dir,                            # 2
            goal_dist,                           # 1
            self.last_hl_actions,                # 3
            opp_rel_body[:, :2],                 # 2 (opponent relative xy)
        ], dim=-1)  # Total: 21

    def _reset_idx(self, envs_idx=None):
        super()._reset_idx(envs_idx)
        if self._is_1v1:
            if envs_idx is None:
                self.opp_pos[:, 0] = self.opponent_init_xy[0]
                self.opp_pos[:, 1] = self.opponent_init_xy[1]
                self.opp_pos[:, 2] = 0.7
            else:
                self.opp_pos[envs_idx, 0] = self.opponent_init_xy[0]
                self.opp_pos[envs_idx, 1] = self.opponent_init_xy[1]
                self.opp_pos[envs_idx, 2] = 0.7
