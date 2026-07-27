"""Hierarchical soccer environment — frozen low-level walking + trainable high-level velocity policy.

Architecture:
    High-level PPO policy (19-dim obs → 3-dim action: vx, vy, wz)
        ↓ velocity commands injected into obs history
    Frozen t1_walk.pt (720-dim proprioception → 21-dim joint actions)
        ↓ PD control
    Genesis physics (AMD Radeon GPU)

The high-level policy can SEE the ball (ball position, velocity, goal direction
in body frame). The low-level frozen model only sees proprioception.

This solves the fundamental problem in v4: the RL policy had no ball info in its
720-dim observation but was rewarded for approaching the ball. With the hierarchical
split, the high-level policy directly observes ball state and outputs velocity
commands, while the frozen walking model handles balance and gait.
"""
from __future__ import annotations
import math, os, torch

try:
    import genesis as gs
except Exception:
    gs = None

try:
    from tensordict import TensorDict
except Exception:
    TensorDict = None

from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

# Import parent env (works both locally and on remote)
try:
    from envs.soccer_env import SoccerEnv, POLICY_JOINT_NAMES, DECIMATION, PHYSICS_DT
except ImportError:
    from soccer_env_v4 import SoccerEnv, POLICY_JOINT_NAMES, DECIMATION, PHYSICS_DT

# Import reward (works both locally and on remote)
try:
    from rewards.reward import compute_reward
except ImportError:
    from reward import compute_reward


class SoccerEnvHierarchical(SoccerEnv):
    """Hierarchical env: frozen t1_walk + trainable high-level velocity policy.

    High-level action: 3 dims [vx, vy, wz] ∈ [-1, 1]
    High-level obs: 19 dims (ball-aware, body frame)

    Low-level: frozen t1_walk.pt (720 → 21), runs at 50 Hz
    High-level: PPO, runs at 5–10 Hz (configurable via decimation)
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 walk_model_path, high_level_decimation=5, show_viewer=False):
        # Pre-set flag so overridden methods behave correctly during super().__init__
        self._hl_initialized = False
        self.high_level_decimation = high_level_decimation

        # Pre-allocate high-level action buffers (needed by overridden _reset_idx)
        device = gs.device if gs is not None else "cpu"
        self.hl_actions = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=device)
        self.last_hl_actions = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=device)

        # Call parent init — sets up physics, scene, buffers, calls reset()
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer)

        # === Override for high-level ===
        self._hl_initialized = True
        self.num_actions = 3                     # vx, vy, wz
        self.hl_clip_lin = 0.8                      # Stage 2: full walking speed (was 0.05 Stage-1 crawl)
        self.hl_clip_ang = 1.0                      # Stage 2: full turn rate (was 0.05 Stage-1 crawl)
        self.high_level_dt = self.dt * high_level_decimation

        # Resize obs buffer for high-level (19 dims, not 720)
        self.hl_obs_dim = 19
        self.obs_buf = torch.empty((self.num_envs, self.hl_obs_dim),
                                   dtype=gs.tc_float, device=self.device)

        # Load frozen walking model
        self.walk_model = torch.jit.load(walk_model_path, map_location=self.device)
        self.walk_model.eval()

        # Extract normalizer tensors for efficient batch inference
        try:
            _norm = self.walk_model.obs_normalizer
            self._norm_mean = _norm._mean.to(self.device)
            self._norm_std = torch.clamp(_norm._std, min=1e-8).to(self.device)
            print(f"[hierarchical] Walk model normalizer loaded: mean={self._norm_mean.shape}")
        except Exception as e:
            self._norm_mean = None
            self._norm_std = None
            print(f"[hierarchical] Walk model normalizer not available: {e}")

        print(f"[hierarchical] Frozen walk model loaded from {walk_model_path}")
        print(f"[hierarchical] HL obs dim={self.hl_obs_dim}, HL action dim={self.num_actions}")
        print(f"[hierarchical] HL dt={self.high_level_dt:.3f}s, decimation={high_level_decimation}")
    def set_curriculum_stage(self, clip_lin, clip_ang, task=None):
        """Switch curriculum stage: adjust action clip and optionally switch task."""
        self.hl_clip_lin = clip_lin
        self.hl_clip_ang = clip_ang
        if task is not None:
            self.task = task
        print(f"[curriculum] clip_lin={clip_lin}, clip_ang={clip_ang}, task={self.task}")


        # Re-update observation with high-level format
        self._update_observation()

    # ─── Public API ───────────────────────────────────────────────

    def _obs_dim(self):
        return self.hl_obs_dim

    def step(self, hl_actions):
        """High-level step: set velocity commands, run N low-level steps with frozen model.

        Args:
            hl_actions: (num_envs, 3) velocity commands [vx, vy, wz]
        Returns:
            obs, reward, done, extras (same as standard env interface)
        """
        # Clip and store high-level actions
        # Per-dimension clip (no filter)
        self.hl_actions = torch.stack([
            torch.clamp(hl_actions[:, 0], -self.hl_clip_lin, self.hl_clip_lin),
            torch.clamp(hl_actions[:, 1], -self.hl_clip_lin, self.hl_clip_lin),
            torch.clamp(hl_actions[:, 2], -self.hl_clip_ang, self.hl_clip_ang),
        ], dim=1)
        # Deadzone: commands below 0.05 are treated as zero (scaled up with clip range)
        self.hl_actions = torch.where(torch.abs(self.hl_actions) < 0.05, torch.zeros_like(self.hl_actions), self.hl_actions)
        self.commands[:] = self.hl_actions

        # Ensure obs_buf is 720-dim (from parent _update_observation)
        super(SoccerEnvHierarchical, self)._update_observation()

        # Run N low-level steps with frozen walking model
        for _ in range(self.high_level_decimation):
            low_obs = self._build_low_level_obs()      # 720-dim proprioception
            joint_actions = self._run_walk_model(low_obs)  # 21-dim joint targets
            self._low_level_step(joint_actions)

        # ── Compute high-level reward ──
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

        # Use high-level actions for reward (action_rate, energy)
        soccer["last_actions"] = self.last_hl_actions
        self.rew_buf = compute_reward(soccer, self.hl_actions, w, self.task)

        # Update prev_dist for next step
        self._resample_ball_if_needed()

        # Termination (episode_length_buf counts low-level steps, max_episode_length too)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= soccer["scored"]  # episode ends on goal — no goal-camping, clean success stats
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.term_pitch
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.term_roll
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        # Reset terminated envs
        self._reset_idx(self.reset_buf)

        # Build high-level observation
        self._update_observation()

        # Save last high-level actions
        self.last_hl_actions.copy_(self.hl_actions)
        self.fallen_prev.copy_(soccer["fallen"])

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ─── Low-level inference ──────────────────────────────────────

    def _build_low_level_obs(self):
        return self.obs_buf

    def _run_walk_model(self, obs_720):
        """Run frozen walking model: normalize obs → actor → 21-dim joint actions."""
        with torch.no_grad():
            if self._norm_mean is not None:
                obs_normed = (obs_720 - self._norm_mean) / self._norm_std
            else:
                obs_normed = obs_720
            return self.walk_model.actor(obs_normed)

    def _low_level_step(self, joint_actions):
        """Execute one low-level control step (DECIMATION physics substeps)."""
        self.actions = torch.clip(joint_actions, -self.clip_actions, self.clip_actions)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # Map 21 policy actions → 23 motor targets (head keeps default)
        target_dof_pos = self.default_dof_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        policy_targets = exec_actions * self.action_scale + self.policy_default_pos.unsqueeze(0)
        target_dof_pos[:, self.policy_joint_indices] = policy_targets

        self.robot.control_dofs_position(
            target_dof_pos[:, self.actions_dof_idx],
            slice(self.base_dof_start, self.base_dof_start + self.num_motors),
        )

        for _ in range(DECIMATION):
            self.scene.step()

        self.episode_length_buf += 1
        self._read_state()

        # Update low-level action history
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        super(SoccerEnvHierarchical, self)._update_observation()

    # ─── High-level observation ───────────────────────────────────

    def _update_observation(self):
        """Build high-level observation: ball-aware, body-frame, 19 dims.

        Layout:
            filtered_lin_vel(3)    — robot velocity in body frame
            filtered_ang_vel(3)    — robot angular velocity in body frame
            projected_gravity(2)   — orientation indicator (xy)
            ball_rel_body(2)       — ball position relative to robot, body frame (xy)
            ball_vel_body(2)       — ball velocity in body frame (xy)
            dist_to_ball(1)        — Euclidean distance to ball
            goal_dir(2)            — goal direction in body frame (xy, normalized)
            goal_dist(1)           — distance to goal
            last_hl_actions(3)     — last velocity command [vx, vy, wz]
        """
        if not self._hl_initialized:
            # During parent __init__, use parent's observation
            super()._update_observation()
            return

        inv_bq = inv_quat(self.base_quat)

        # Ball position relative to robot, in body frame
        ball_rel = self.ball_pos - self.base_pos
        ball_rel_body = transform_by_quat(ball_rel, inv_bq)

        # Ball velocity in body frame
        ball_vel_body = transform_by_quat(self.ball_vel, inv_bq)

        # Goal direction in body frame
        goal_pos = torch.zeros_like(self.base_pos)
        goal_pos[:, 0] = self.goal_x
        goal_rel = goal_pos - self.base_pos
        goal_rel_body = transform_by_quat(goal_rel, inv_bq)
        goal_dist = torch.norm(goal_rel_body[:, :2], dim=1, keepdim=True)
        goal_dir = goal_rel_body[:, :2] / (goal_dist + 1e-6)

        # Distance to ball
        dist_to_ball = torch.norm(ball_rel_body[:, :2], dim=1, keepdim=True)

        self.obs_buf = torch.cat([
            self.filtered_lin_vel,              # 3
            self.filtered_ang_vel,             # 3
            self.projected_gravity[:, :2],     # 2
            ball_rel_body[:, :2],              # 2
            ball_vel_body[:, :2],              # 2
            dist_to_ball,                       # 1
            goal_dir,                           # 2
            goal_dist,                          # 1
            self.last_hl_actions,               # 3
        ], dim=-1)  # Total: 19

    # ─── Overrides for hierarchical behavior ──────────────────────

    def _resample_commands(self, envs_idx=None):
        """No-op: high-level policy provides velocity commands, not random sampling."""
        if not self._hl_initialized:
            super()._resample_commands(envs_idx)

    def _reset_idx(self, envs_idx=None):
        """Reset environments, zeroing high-level actions."""
        super()._reset_idx(envs_idx)
        if self._hl_initialized:
            if envs_idx is None:
                self.hl_actions.zero_()
                self.last_hl_actions.zero_()
            else:
                self.hl_actions.masked_fill_(envs_idx[:, None], 0.0)
                self.last_hl_actions.masked_fill_(envs_idx[:, None], 0.0)
