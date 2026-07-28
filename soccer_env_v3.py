"""Genesis humanoid-soccer environment aligned with booster_deploy t1_walk.pt.

Key alignment with pre-trained policy:
  - 21 policy joints (excludes AAHead_yaw, Head_pitch)
  - obs per frame: ang_vel(3) + projected_gravity(3) + commands(3) + dof_pos(21) + dof_vel(21) + last_action(21) = 72
  - 10-frame history → 720 input dims
  - action_scale = 0.25, obs_dof_vel_scale = 1.0
  - PD gains from T1WalkControllerCfg
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
from rewards.reward import compute_reward


def torch_rand_float(low, high, shape, device):
    return (high - low) * torch.rand(shape, device=device) + low


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


def _genesis_asset(*parts):
    return os.path.join(os.path.dirname(gs.__file__), "assets", *parts)


# 21 policy joints (from booster_deploy T1WalkControllerCfg)
POLICY_JOINT_NAMES = [
    "Left_Shoulder_Pitch", "Right_Shoulder_Pitch", "Waist",
    "Left_Shoulder_Roll", "Right_Shoulder_Roll",
    "Left_Hip_Pitch", "Right_Hip_Pitch",
    "Left_Elbow_Pitch", "Right_Elbow_Pitch",
    "Left_Hip_Roll", "Right_Hip_Roll",
    "Left_Elbow_Yaw", "Right_Elbow_Yaw",
    "Left_Hip_Yaw", "Right_Hip_Yaw",
    "Left_Knee_Pitch", "Right_Knee_Pitch",
    "Left_Ankle_Pitch", "Right_Ankle_Pitch",
    "Left_Ankle_Roll", "Right_Ankle_Roll",
]

# PD gains from booster_deploy T1WalkControllerCfg
DEFAULT_KP = [50,50,50,50,50,50,50,50,50,50,200,200,200,200,200,50,50,200,200,200,200,50,50]
DEFAULT_KD = [1,1,1,1,1,1,1,1,1,1,5,5,5,5,5,2,2,5,5,5,5,2,2]

# Default standing pose from booster_deploy
DEFAULT_JOINT_POS = [0, 0, 0.2, -1.3, 0, -0.5, 0.2, 1.3, 0, 0.5, 0.0,
                     -0.2, 0, 0, 0.4, -0.2, 0.0, -0.2, 0, 0, 0.4, -0.2, 0.0]


class SoccerEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        if gs is None:
            raise RuntimeError("Genesis not available.")
        self.num_envs = num_envs
        self.device = gs.device
        self.cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.task = env_cfg.get("task", "chase")
        self.distill_logger = None

        self.dt = env_cfg["dt"]
        self.substeps = env_cfg["substeps"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.action_scale = env_cfg.get("action_scale", 0.25)
        self.clip_actions = env_cfg["clip_actions"]
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)

        self.ball_radius = env_cfg["ball_radius"]
        self.field_x, self.field_y = env_cfg["field"]
        self.goal_half = env_cfg["goal_width"] / 2.0
        self.goal_x = self.field_x / 2.0
        self.circle_radius = env_cfg.get("circle_radius", 1.5)
        self.fall_height = env_cfg["fall_height"]
        self.term_pitch = math.radians(env_cfg["termination_pitch_deg"])
        self.term_roll = math.radians(env_cfg["termination_roll_deg"])

        self.obs_history_length = env_cfg.get("obs_history_length", 10)
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg

        self._build_scene(show_viewer)

        # ---- auto-discover motor DOFs ----
        self.motor_joints = [j for j in self.robot.joints[1:] if j.n_dofs > 0]
        self.motors_dof_idx = torch.tensor([j.dof_start for j in self.motor_joints], dtype=gs.tc_int, device=self.device)
        self.base_dof_start = int(self.motors_dof_idx[0].item())
        self.num_actions = len(self.motor_joints)  # 23 total, but we control 21

        # Build policy joint mapping: which of the 23 motor joints are in POLICY_JOINT_NAMES
        all_joint_names = [j.name for j in self.motor_joints]
        self.policy_joint_indices = torch.tensor(
            [all_joint_names.index(n) for n in POLICY_JOINT_NAMES],
            dtype=gs.tc_int, device=self.device
        )
        self.num_policy_actions = len(POLICY_JOINT_NAMES)  # 21

        # Default standing pose for 23 joints, then extract policy subset
        full_default = self.robot.get_dofs_position(self.motors_dof_idx)[0].clone()
        self.default_dof_pos = full_default
        self.policy_default_pos = full_default[self.policy_joint_indices].clone()

        # PD gains
        kp_list = [DEFAULT_KP[i] for i in range(len(DEFAULT_KP))]
        kd_list = [DEFAULT_KD[i] for i in range(len(DEFAULT_KD))]
        self.robot.set_dofs_kp(kp_list, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd_list, self.motors_dof_idx)

        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # goal direction
        self.command = torch.tensor(
            command_cfg.get("goal_dir", [1.0, 0.0, 0.0]), dtype=gs.tc_float, device=self.device
        ).expand(self.num_envs, -1).clone()

        # ---- buffers ----
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=self.device)
        self.init_base_pos = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=self.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=self.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)

        qpos_template = self.robot.get_qpos()[0].clone()
        qpos_template[self.base_dof_start:self.base_dof_start + self.num_actions] = self.default_dof_pos
        self.init_qpos = qpos_template.unsqueeze(0).expand(self.num_envs, -1).clone()
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # policy obs: 72 per frame, 10 history = 720 + ball info
        self.obs_dim_per_frame = 3 + 3 + 3 + self.num_policy_actions + self.num_policy_actions + self.num_policy_actions  # 72
        self.obs_history = torch.zeros((self.num_envs, self.obs_history_length, self.obs_dim_per_frame), dtype=gs.tc_float, device=self.device)

        # Action buffer uses 21 policy actions
        self.actions = torch.zeros((self.num_envs, self.num_policy_actions), dtype=gs.tc_float, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        # State buffers
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=self.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.dof_pos = torch.empty((self.num_envs, self.num_actions), dtype=gs.tc_float, device=self.device)
        self.dof_vel = torch.empty((self.num_envs, self.num_actions), dtype=gs.tc_float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.ball_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.ball_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.prev_dist_to_ball = torch.empty((self.num_envs,), dtype=gs.tc_float, device=self.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=self.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=self.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=self.device)
        self.extras = dict()

        # Gait phase generator
        self.gait_frequency = torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=gs.tc_int, device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=self.device)
        self.feet_pos = torch.zeros(self.num_envs, 2, 3, dtype=gs.tc_float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, 2, dtype=gs.tc_bool, device=self.device)
        self.filtered_lin_vel = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=self.device)
        self.filtered_ang_vel = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=self.device)
        self.fallen_prev = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=self.device)
        self.shot_prev_vel = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device)
        self.shots_taken = 0
        self.shots_on_target = 0
        self.goals_scored = 0

        # obs_buf: 720 (history) + 6 (ball) = 726
        self.obs_buf = torch.empty((self.num_envs, self._obs_dim()), dtype=gs.tc_float, device=self.device)

        self.reset()

    def _build_scene(self, show_viewer):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True, tolerance=1e-5, max_collision_pairs=512),
            viewer_options=gs.options.ViewerOptions(camera_pos=(6, 6, 4), camera_lookat=(0, 0, 0.5), camera_fov=40),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=True,
                ambient_light=(0.7, 0.7, 0.7),
                shadow=True,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=show_viewer,
        )

        # Ground
        self.scene.add_entity(gs.morphs.URDF(file=_genesis_asset("urdf", "plane", "plane.urdf"), fixed=True))

        # Soccer field (only for single-env rendering)
        if self.num_envs == 1:
            import math as _m
            _w = gs.surfaces.Rough(color=(1, 1, 1), roughness=0.8)
            _gs_s = gs.surfaces.Rough(color=(0.95, 0.95, 0.95), roughness=0.5)
            _FL = self.field_x; _FW = self.field_y; _HL = _FL/2; _HW = _FW/2
            _LH = 0.005; _LW = 0.12; _GW = self.goal_half*2; _GH = 1.0; _PR = 0.05; _CR = self.circle_radius
            for _x, _y in [(0, -_HW), (0, _HW)]:
                self.scene.add_entity(morph=gs.morphs.Box(size=(_FL, _LW, _LH), pos=(_x, _y, _LH/2), fixed=True), surface=_w)
            for _x, _y in [(-_HL, 0), (_HL, 0), (0, 0)]:
                self.scene.add_entity(morph=gs.morphs.Box(size=(_LW, _FW, _LH), pos=(_x, _y, _LH/2), fixed=True), surface=_w)
            for _i in range(32):
                _a = 2 * _m.pi * _i / 32
                self.scene.add_entity(morph=gs.morphs.Box(size=(0.3, _LW, _LH), pos=(_CR*_m.cos(_a), _CR*_m.sin(_a), _LH/2), euler=(0, 0, _m.degrees(_a)), fixed=True), surface=_w)
            _pw = 3.0
            for _px in [-_HL+1.5, _HL-1.5]:
                self.scene.add_entity(morph=gs.morphs.Box(size=(_LW, 6.0, _LH), pos=(_px, -_pw, _LH/2), fixed=True), surface=_w)
                self.scene.add_entity(morph=gs.morphs.Box(size=(_LW, 6.0, _LH), pos=(_px, _pw, _LH/2), fixed=True), surface=_w)
            for _sx in [-_HL, _HL]:
                self.scene.add_entity(morph=gs.morphs.Box(size=(3.0, _LW, _LH), pos=(_sx, _pw, _LH/2), fixed=True), surface=_w)
                self.scene.add_entity(morph=gs.morphs.Box(size=(3.0, _LW, _LH), pos=(_sx, -_pw, _LH/2), fixed=True), surface=_w)
            _hg = _GW/2; _pw2 = _PR*2
            for _gx in [-_HL, _HL]:
                self.scene.add_entity(morph=gs.morphs.Box(size=(_pw2, _pw2, _GH), pos=(_gx, -_hg, _GH/2), fixed=True), surface=_gs_s)
                self.scene.add_entity(morph=gs.morphs.Box(size=(_pw2, _pw2, _GH), pos=(_gx, _hg, _GH/2), fixed=True), surface=_gs_s)
                self.scene.add_entity(morph=gs.morphs.Box(size=(_pw2, _GW+_pw2, _pw2), pos=(_gx, 0, _GH), fixed=True), surface=_gs_s)

        # Robot
        robot_path = self.cfg["robot_urdf"]
        if not os.path.isabs(robot_path):
            ga = _genesis_asset(robot_path)
            robot_path = ga if os.path.exists(ga) else os.path.abspath(robot_path)
        if robot_path.endswith(".xml") or robot_path.endswith(".mjcf"):
            self.robot = self.scene.add_entity(gs.morphs.MJCF(file=robot_path, pos=self.cfg["base_init_pos"], quat=self.cfg["base_init_quat"]))
        else:
            self.robot = self.scene.add_entity(gs.morphs.URDF(file=robot_path, pos=self.cfg["base_init_pos"], quat=self.cfg["base_init_quat"]))

        # Ball
        ball_path = os.path.join(os.path.dirname(__file__), "..", "assets", "ball.urdf")
        self.ball = self.scene.add_entity(gs.morphs.URDF(file=os.path.abspath(ball_path)))

        # Camera for rendering
        self.scene.add_camera(res=(960, 540), pos=(6, -8, 4), lookat=(0, 0, 0.5), fov=50, GUI=False)
        self.scene.build(n_envs=self.num_envs)

    def _obs_dim(self):
        n = self.num_policy_actions  # 21
        return self.obs_dim_per_frame * self.obs_history_length + 6  # 720 + 6 = 726

    def get_stats(self):
        return {"shots_taken": self.shots_taken, "shots_on_target": self.shots_on_target,
                "goals_scored": self.goals_scored}

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    def step(self, actions):
        # actions: (num_envs, 21) policy actions
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # Map 21 policy actions to 23 motor joints (head joints keep default)
        target_dof_pos = self.default_dof_pos.clone()
        # Update only policy joints
        policy_targets = exec_actions * self.action_scale + self.policy_default_pos
        target_dof_pos[:, self.policy_joint_indices] = policy_targets

        self.robot.control_dofs_position(
            target_dof_pos[:, self.actions_dof_idx],
            slice(self.base_dof_start, self.base_dof_start + self.num_actions),
        )
        self.scene.step()
        self.episode_length_buf += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)
        self._read_state()

        soccer = self._soccer_state()
        w = dict(self.reward_scales)
        w["_ball_radius"] = self.ball_radius
        w["dt"] = self.dt
        for _k in list(w.keys()):
            if isinstance(w[_k], (int, float)) and _k not in ["_ball_radius", "dt", "tracking_sigma", "swing_period", "only_positive_rewards"]:
                w[_k] *= self.dt
        self.rew_buf = compute_reward(soccer, self.actions, w, self.task)

        self._resample_ball_if_needed()
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.term_pitch
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.term_roll
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)
        self._update_observation()
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        self.fallen_prev.copy_(soccer["fallen"])
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _read_state(self):
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True)
        inv_bq = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_bq)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_bq)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_bq)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.ball_pos = self.ball.get_pos()
        self.ball_vel = self.ball.get_vel()
        fw = 0.1
        self.filtered_lin_vel[:] = self.base_lin_vel * fw + self.filtered_lin_vel * (1.0 - fw)
        self.filtered_ang_vel[:] = self.base_ang_vel * fw + self.filtered_ang_vel * (1.0 - fw)
        self.last_feet_pos.copy_(self.feet_pos)
        for _i, _n in enumerate(["left_foot_link", "right_foot_link"]):
            try:
                _link = self.robot.get_link(_n)
                self.feet_pos[:, _i, :] = _link.get_pos()
                self.feet_contact[:, _i] = self.feet_pos[:, _i, 2] < 0.06
            except Exception:
                self.feet_contact[:, _i] = False

    def _soccer_state(self):
        torso_up = torch.clamp(-self.projected_gravity[:, 2], min=-1.0, max=1.0)
        fallen = (self.base_pos[:, 2] < self.fall_height) | (torch.abs(self.base_euler[:, 1]) > 45) | (torch.abs(self.base_euler[:, 0]) > 45)
        dist_to_ball = torch.norm(self.base_pos[:, :2] - self.ball_pos[:, :2], dim=1)
        goal_dir = torch.stack([self.goal_x - self.ball_pos[:, 0], -self.ball_pos[:, 1], torch.zeros_like(self.ball_pos[:, 0])], dim=1)
        goal_dir = goal_dir / (torch.norm(goal_dir, dim=1, keepdim=True) + 1e-6)
        ball_vel_to_goal = torch.sum(self.ball_vel[:, :2] * goal_dir[:, :2], dim=1)
        scored = (self.ball_pos[:, 0] > self.goal_x) & (torch.abs(self.ball_pos[:, 1]) < self.goal_half)
        just_recovered = self.fallen_prev & (~fallen)

        shooting_now = (ball_vel_to_goal > 1.0) & (self.shot_prev_vel < 1.0)
        for i in range(self.num_envs):
            if shooting_now[i]:
                self.shots_taken += 1
                if abs(self.ball_pos[i, 1].item()) < self.goal_half:
                    self.shots_on_target += 1
        for i in range(self.num_envs):
            if scored[i] and self.shot_prev_vel[i] > 0.5:
                self.goals_scored += 1
        self.shot_prev_vel.copy_(ball_vel_to_goal)

        return {
            "torso_up": torso_up, "fallen": fallen,
            "base_lin_vel_x": self.filtered_lin_vel[:, 0],
            "base_lin_vel_y_y": self.filtered_lin_vel[:, 1],
            "base_lin_vel_z": self.filtered_lin_vel[:, 2],
            "base_ang_vel_z": self.filtered_ang_vel[:, 2],
            "base_ang_vel_xy": self.base_ang_vel[:, :2],
            "projected_gravity_xy": self.projected_gravity[:, :2],
            "ball_x": self.ball_pos[:, 0],
            "dist_to_ball": dist_to_ball,
            "prev_dist_to_ball": self.prev_dist_to_ball,
            "ball_vel_to_goal": ball_vel_to_goal,
            "scored": scored, "just_recovered": just_recovered,
            "commands": self.commands,
            "gait_process": self.gait_process,
            "gait_frequency": self.gait_frequency,
            "feet_contact": self.feet_contact,
            "feet_pos": self.feet_pos, "last_feet_pos": self.last_feet_pos,
            "episode_length_buf": self.episode_length_buf,
            "last_actions": self.last_actions, "last_dof_vel": self.last_dof_vel, "dof_vel": self.dof_vel,
        }

    def _resample_ball_if_needed(self):
        self.prev_dist_to_ball = torch.norm(self.base_pos[:, :2] - self.ball_pos[:, :2], dim=1).clone()

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
        ball_qpos = self._sample_ball_qpos()
        self.ball.set_qpos(ball_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos)
            self.base_lin_vel.zero_(); self.base_ang_vel.zero_(); self.dof_vel.zero_()
            self.actions.zero_(); self.last_actions.zero_(); self.last_dof_vel.zero_()
            self.obs_history.zero_(); self.episode_length_buf.zero_()
            self.reset_buf.fill_(True); self.fallen_prev.zero_()
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
            torch.where(envs_idx[:, None], self.default_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
            self.fallen_prev.masked_fill_(envs_idx, False)

        self._read_state()
        self.prev_dist_to_ball = torch.norm(self.base_pos[:, :2] - self.ball_pos[:, :2], dim=1).clone()
        self._resample_commands(envs_idx)

    def _sample_ball_qpos(self):
        lo = torch.tensor([-self.field_x / 2 + 0.5, -self.goal_half, self.ball_radius], device=self.device)
        hi = torch.tensor([self.goal_x - 1.0, self.goal_half, self.ball_radius], device=self.device)
        pos = gs_rand(lo, hi, (self.num_envs,))
        quat = torch.zeros((self.num_envs, 4), device=self.device)
        quat[:, 0] = 1.0
        return torch.cat([pos, quat], dim=1)

    def _resample_commands(self, envs_idx):
        if envs_idx is None:
            n = self.num_envs
            self.commands[:, 0] = torch_rand_float(-1.0, 1.0, (n,), device=self.device)
            self.commands[:, 1] = torch_rand_float(-1.0, 1.0, (n,), device=self.device)
            self.commands[:, 2] = torch_rand_float(-1.0, 1.0, (n,), device=self.device)
            self.gait_frequency[:] = torch_rand_float(1.0, 2.0, (n,), device=self.device)
            _still = torch.randperm(n)[:int(0.1 * n)]
            self.commands[_still, :] = 0.0
            self.gait_frequency[_still] = 0.0
            self.cmd_resample_time[:] = self.episode_length_buf + torch.randint(int(8.0 / self.dt), int(12.0 / self.dt), (n,), device=self.device).to(dtype=gs.tc_int)
        else:
            n = envs_idx.sum().item()
            if n > 0:
                _idx = envs_idx.nonzero(as_tuple=False).flatten()
                self.commands[_idx, 0] = torch_rand_float(-1.0, 1.0, (n,), device=self.device)
                self.commands[_idx, 1] = torch_rand_float(-1.0, 1.0, (n,), device=self.device)
                self.commands[_idx, 2] = torch_rand_float(-1.0, 1.0, (n,), device=self.device)
                self.gait_frequency[_idx] = torch_rand_float(1.0, 2.0, (n,), device=self.device)
                self.cmd_resample_time[_idx] = self.episode_length_buf[_idx] + torch.randint(int(8.0 / self.dt), int(12.0 / self.dt), (n,), device=self.device).to(dtype=gs.tc_int)
        _need = self.episode_length_buf >= self.cmd_resample_time
        if _need.any():
            _idx = _need.nonzero(as_tuple=False).flatten()
            _n = len(_idx)
            self.commands[_idx, 0] = torch_rand_float(-1.0, 1.0, (_n,), device=self.device)
            self.commands[_idx, 1] = torch_rand_float(-1.0, 1.0, (_n,), device=self.device)
            self.commands[_idx, 2] = torch_rand_float(-1.0, 1.0, (_n,), device=self.device)
            self.gait_frequency[_idx] = torch_rand_float(1.0, 2.0, (_n,), device=self.device)
            self.cmd_resample_time[_idx] += torch.randint(int(8.0 / self.dt), int(12.0 / self.dt), (_n,), device=self.device).to(dtype=gs.tc_int)

    def _update_observation(self):
        # Build per-frame obs aligned with booster_deploy: ang_vel(3) + grav(3) + cmd(3) + dof_pos(21) + dof_vel(21) + act(21) = 72
        policy_dof_pos = self.dof_pos[:, self.policy_joint_indices]
        policy_dof_vel = self.dof_vel[:, self.policy_joint_indices]

        base_obs = torch.cat([
            self.base_ang_vel,
            self.projected_gravity,
            self.commands,
            (policy_dof_pos - self.policy_default_pos) * 1.0,
            policy_dof_vel * 1.0,
            self.last_actions,
        ], dim=-1)

        self.obs_history = torch.cat([self.obs_history[:, 1:], base_obs.unsqueeze(1)], dim=1)
        self.obs_buf = torch.cat([
            self.obs_history.reshape(self.num_envs, -1),
            (self.ball_pos - self.base_pos) * 2.0,
            self.ball_vel * 2.0,
        ], dim=-1)
