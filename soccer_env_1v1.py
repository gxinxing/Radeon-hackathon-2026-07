"""1v1 Soccer Environment — RL agent vs rule-based opponent.

Extends SoccerEnvHierarchical with a second robot (opponent) controlled
by a simple rule-based policy. The RL agent's observation is extended
to include opponent relative position.

Architecture:
    RL Agent: 19+2=21 dim obs → 3 dim action (vx, vy, wz) → frozen t1_walk.pt
    Opponent: Rule-based velocity command → frozen t1_walk.pt

The opponent uses the same frozen t1_walk.pt for locomotion, with
velocity commands from a simple chase-ball rule.
"""
from __future__ import annotations
import math, os, torch

try:
    import genesis as gs
except Exception:
    gs = None

try:
    from envs.soccer_env import SoccerEnv, POLICY_JOINT_NAMES, DECIMATION, PHYSICS_DT, KP_23, KD_23, DEFAULT_POS_23, INIT_POS, INIT_QUAT
except ImportError:
    from soccer_env_v4 import SoccerEnv, POLICY_JOINT_NAMES, DECIMATION, PHYSICS_DT, KP_23, KD_23, DEFAULT_POS_23, INIT_POS, INIT_QUAT

from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

try:
    from envs.soccer_env_hierarchical import SoccerEnvHierarchical
except ImportError:
    from soccer_env_hierarchical import SoccerEnvHierarchical


class SoccerEnv1v1(SoccerEnvHierarchical):
    """1v1 environment: RL agent vs rule-based opponent.

    Obs: 21 dims (19 base + 2 opponent relative xy in body frame)
    Action: 3 dims (vx, vy, wz) — same as hierarchical
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 walk_model_path, high_level_decimation=5, show_viewer=False,
                 opponent_init_pos=(-3.0, 0.0, 0.7)):
        self.opponent_init_pos = opponent_init_pos
        self._is_1v1 = False
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                         walk_model_path, high_level_decimation, show_viewer)
        self._is_1v1 = True

    def _build_scene(self, show_viewer):
        """Build scene with TWO robots: intercept parent's scene.build to add opponent first."""
        # We can't monkey-patch before scene exists.
        # Instead, call parent _build_scene but intercept scene.build by
        # wrapping it after scene creation but before build call.
        # The trick: temporarily replace gs.Scene.build at class level.
        original_build = gs.Scene.build
        opp_holder = {}

        def patched_build(self_scene, *args, **kwargs):
            if 'opponent' not in opp_holder:
                # Add opponent robot before building
                robot_path = self.cfg["robot_urdf"]
                if not os.path.isabs(robot_path):
                    from envs.soccer_env import _genesis_asset
                    ga = _genesis_asset(robot_path)
                    robot_path = ga if os.path.exists(ga) else os.path.abspath(robot_path)

                opp_pos = list(self.opponent_init_pos)
                self.opponent = self_scene.add_entity(
                    gs.morphs.URDF(file=robot_path, pos=opp_pos, quat=INIT_QUAT,
                                  fixed=False, merge_fixed_links=False))

                # PD gains will be set after build
                opp_holder['opponent'] = True
            return original_build(self_scene, *args, **kwargs)

        gs.Scene.build = patched_build
        try:
            super(SoccerEnvHierarchical, self)._build_scene(show_viewer)
        finally:
            gs.Scene.build = original_build

        # Now scene is built — set opponent PD gains and init buffers
        opp_motor_joints = [j for j in self.opponent.joints[1:] if j.n_dofs > 0]
        self.opp_motors_dof_idx = torch.tensor([j.dof_start for j in opp_motor_joints],
                                                dtype=gs.tc_int, device=self.device)
        self.opp_base_dof_start = int(self.opp_motors_dof_idx[0].item())
        self.opp_num_motors = len(opp_motor_joints)
        self.opponent.set_dofs_kp(KP_23, self.opp_motors_dof_idx)
        self.opponent.set_dofs_kv(KD_23, self.opp_motors_dof_idx)

        self.opp_motors_dof_idx = opp_motors_dof_idx
        self.opp_base_dof_start = int(opp_motors_dof_idx[0].item())
        self.opp_num_motors = len(opp_motor_joints)
        self.opp_policy_joint_indices = torch.tensor(
            [all_joint_names.index(n) for n in POLICY_JOINT_NAMES]
            if False else  # we need opponent's joint names, but they're same as agent
            [self.opponent.joints[1:][i].dof_start - self.opp_base_dof_start
             for i in range(len(opp_motor_joints)) if opp_motor_joints[i].n_dofs > 0][:21],
            dtype=gs.tc_int, device=self.device
        ) if False else None  # We'll compute it properly below

        # Store opponent joint indices properly
        all_opp_joints = self.opponent.joints[1:]  # skip world
        opp_joint_names = []
        for j in all_opp_joints:
            if j.n_dofs > 0:
                opp_joint_names.extend([j.name] * j.n_dofs)

        # Map policy joint names to opponent dof indices
        policy_joint_names_list = [
            "Left_Shoulder_Pitch", "Right_Shoulder_Pitch",
            "Left_Shoulder_Roll", "Right_Shoulder_Roll",
            "Left_Elbow_Pitch", "Right_Elbow_Pitch",
            "Left_Elbow_Yaw", "Right_Elbow_Yaw",
            "Left_Hip_Pitch", "Right_Hip_Pitch",
            "Left_Hip_Roll", "Right_Hip_Roll",
            "Left_Hip_Yaw", "Right_Hip_Yaw",
            "Left_Knee_Pitch", "Right_Knee_Pitch",
            "Left_Ankle_Pitch", "Right_Ankle_Pitch",
            "Left_Ankle_Roll", "Right_Ankle_Roll",
        ]

        # Build opponent default pos and joint mapping
        self.opp_default_dof_pos = torch.tensor(DEFAULT_POS_23, dtype=gs.tc_float, device=self.device)
        self.opp_policy_default_pos = self.opp_default_dof_pos[
            torch.tensor([2, 6, 3, 7, 4, 8, 5, 9, 10, 14, 11, 15, 12, 16, 13, 17, 18, 19, 20, 21, 22],
                         dtype=gs.tc_int, device=self.device)
        ]

        # Opponent state buffers
        self.opp_base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.opp_base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=self.device)
        self.opp_base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.opp_dof_pos = torch.empty((self.num_envs, self.opp_num_motors), dtype=gs.tc_float, device=self.device)
        self.opp_dof_vel = torch.empty((self.num_envs, self.opp_num_motors), dtype=gs.tc_float, device=self.device)
        self.opp_actions = torch.zeros((self.num_envs, 21), dtype=gs.tc_float, device=self.device)
        self.opp_last_actions = torch.zeros((self.num_envs, 21), dtype=gs.tc_float, device=self.device)
        self.opp_commands = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=self.device)

        # Opponent init qpos
        opp_qpos_template = self.opponent.get_qpos()[0].clone()
        opp_qpos_template[0] = self.opponent_init_pos[0]
        opp_qpos_template[1] = self.opponent_init_pos[1]
        opp_qpos_template[2] = self.opponent_init_pos[2]
        opp_qpos_template[self.opp_base_dof_start:self.opp_base_dof_start + self.opp_num_motors] = self.opp_default_dof_pos
        self.opp_init_qpos = opp_qpos_template.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Extend obs dim: 19 (base) + 2 (opponent relative xy) = 21
        self.hl_obs_dim = 21

    def _read_state(self):
        super()._read_state()
        # Read opponent state
        opp_trunk = self.opponent.links[1]
        self.opp_base_pos = opp_trunk.get_pos()
        self.opp_base_quat = opp_trunk.get_quat()
        self.opp_base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.opp_base_quat),
            rpy=True, degrees=True)
        self.opp_dof_pos = self.opponent.get_dofs_position(self.opp_motors_dof_idx)
        self.opp_dof_vel = self.opponent.get_dofs_velocity(self.opp_motors_dof_idx)

    def _compute_opponent_command(self):
        """Simple rule-based opponent: chase ball, avoid going too fast."""
        # Ball relative to opponent in world frame
        ball_rel = self.ball_pos - self.opp_base_pos
        dist = torch.norm(ball_rel[:, :2], dim=1, keepdim=True)

        # Normalize direction (world frame)
        direction = ball_rel[:, :2] / (dist + 1e-6)

        # Simple proportional control, clamped to [-0.2, 0.2]
        cmd = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        cmd[:, 0] = torch.clamp(direction[:, 0] * 0.2, -0.2, 0.2)
        cmd[:, 1] = torch.clamp(direction[:, 1] * 0.2, -0.2, 0.2)
        # Turn toward ball
        angle_to_ball = torch.atan2(ball_rel[:, 1], ball_rel[:, 0])
        cmd[:, 2] = torch.clamp(angle_to_ball * 0.1, -0.2, 0.2)

        self.opp_commands[:] = cmd

    def _step_opponent(self):
        """Run one low-level step for the opponent using frozen walk model."""
        self._compute_opponent_command()

        # Build opponent 720-dim obs (same format as agent)
        policy_dof_pos = self.opp_dof_pos[:, [2, 6, 3, 7, 4, 8, 5, 9, 10, 14, 11, 15, 12, 16, 13, 17, 18, 19, 20, 21, 22]]
        policy_dof_vel = self.opp_dof_vel[:, [2, 6, 3, 7, 4, 8, 5, 9, 10, 14, 11, 15, 12, 16, 13, 17, 18, 19, 20, 21, 22]]

        s_ang_vel = self.obs_scales.get("ang_vel", 0.25)
        s_dof_pos = self.obs_scales.get("dof_pos", 1.0)
        s_dof_vel = self.obs_scales.get("dof_vel", 0.05)

        opp_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=self.device)  # simplified
        opp_gravity = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        opp_gravity[:, 2] = -1.0  # pointing down

        base_obs = torch.cat([
            opp_ang_vel * s_ang_vel,
            opp_gravity,
            self.opp_commands,
            (policy_dof_pos - self.opp_policy_default_pos.unsqueeze(0)) * s_dof_pos,
            policy_dof_vel * s_dof_vel,
            self.opp_last_actions,
        ], dim=-1)

        # We don't maintain full 10-frame history for opponent (simplification)
        # Just repeat current frame 10 times
        opp_obs_720 = base_obs.unsqueeze(1).expand(-1, 10, -1).reshape(self.num_envs, -1)

        with torch.no_grad():
            if self._norm_mean is not None:
                opp_obs_normed = (opp_obs_720 - self._norm_mean) / self._norm_std
            else:
                opp_obs_normed = opp_obs_720
            joint_actions = self.walk_model.actor(opp_obs_normed)

        # Apply to opponent
        exec_actions = self.opp_last_actions if self.simulate_action_latency else joint_actions
        target_dof_pos = self.opp_default_dof_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        policy_targets = exec_actions * self.action_scale + self.opp_policy_default_pos.unsqueeze(0)

        # Map 21 policy joints to 23 motor targets
        opp_policy_indices = torch.tensor(
            [2, 6, 3, 7, 4, 8, 5, 9, 10, 14, 11, 15, 12, 16, 13, 17, 18, 19, 20, 21, 22],
            dtype=gs.tc_int, device=self.device)
        target_dof_pos[:, opp_policy_indices] = policy_targets

        opp_actions_dof_idx = torch.argsort(self.opp_motors_dof_idx)
        self.opponent.control_dofs_position(
            target_dof_pos[:, opp_actions_dof_idx],
            slice(self.opp_base_dof_start, self.opp_base_dof_start + self.opp_num_motors),
        )

        self.opp_last_actions.copy_(joint_actions)

    def step(self, hl_actions):
        """High-level step: clip agent commands, run N low-level steps for both robots."""
        # Clip and store agent actions
        self.hl_actions = torch.stack([
            torch.clamp(hl_actions[:, 0], -self.hl_clip_lin, self.hl_clip_lin),
            torch.clamp(hl_actions[:, 1], -self.hl_clip_lin, self.hl_clip_lin),
            torch.clamp(hl_actions[:, 2], -self.hl_clip_ang, self.hl_clip_ang),
        ], dim=1)
        self.commands[:] = self.hl_actions

        # Ensure obs_buf is 720-dim for low-level
        super(SoccerEnvHierarchical, self)._update_observation()

        # Run N low-level steps for BOTH robots
        for _ in range(self.high_level_decimation):
            # Agent step
            low_obs = self._build_low_level_obs()
            joint_actions = self._run_walk_model(low_obs)
            self._low_level_step(joint_actions)

            # Opponent step (interleaved in same physics substeps)
            self._step_opponent()
            for _ in range(DECIMATION):
                self.scene.step()

        # Compute reward (same as parent + opponent proximity bonus)
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
        self.rew_buf = compute_reward(soccer, self.hl_actions, w, self.task)

        # Update prev_dist
        self._resample_ball_if_needed()

        # Termination
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.term_pitch
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.term_roll
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

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

        # Ball info (same as parent)
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

        # Opponent relative position in agent body frame
        opp_rel = self.opp_base_pos - self.base_pos
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
        if self._is_1v1 and hasattr(self, 'opponent'):
            # Reset opponent
            self.opponent.set_qpos(self.opp_init_qpos, envs_idx=envs_idx,
                                   zero_velocity=True, skip_forward=True)
            if envs_idx is None:
                self.opp_actions.zero_()
                self.opp_last_actions.zero_()
                self.opp_commands.zero_()
            else:
                self.opp_actions.masked_fill_(envs_idx[:, None], 0.0)
                self.opp_last_actions.masked_fill_(envs_idx[:, None], 0.0)
                self.opp_commands.masked_fill_(envs_idx[:, None], 0.0)
