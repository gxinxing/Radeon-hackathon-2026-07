"""Worker process: runs one robot in its own Genesis scene.

Connects to MatchCoordinator via socket. Sends robot state, receives
opponent state + ball state. For the authority process (has_ball=True),
ball physics is simulated. For non-authority, ball position is set from network.

Usage:
    python match_worker.py --role agent --has-ball --model runs/.../model_1894.pt
    python match_worker.py --role opponent --port 9876
"""
import argparse, socket, struct, time, sys, os, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import genesis as gs

# ─── Protocol (must match match_coordinator.py) ───
MSG_STATE = 1
MSG_BALL = 2
MSG_CMD = 3
MSG_END = 4

DEFAULT_PORT = 9876
N_STEPS = 1000  # 20s at 50Hz (HL is 10Hz, but we sync at 50Hz for physics)
HL_DECIMATION = 5  # 50Hz physics / 5 = 10Hz HL


def pack_state(msg_type, data):
    payload = struct.pack(f'<{len(data)}f', *data)
    return struct.pack('<BI', msg_type, len(data)) + payload


def recv_all(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def recv_msg(sock):
    header = recv_all(sock, 5)
    if not header:
        return None, None
    msg_type, length = struct.unpack('<BI', header)
    if length == 0:
        return msg_type, []
    data = recv_all(sock, length * 4)
    if not data:
        return None, None
    return msg_type, struct.unpack(f'<{length}f', data)


class MatchWorker:
    """One robot in one Genesis scene, synced via socket."""

    def __init__(self, role, has_ball, port, model_path, init_pos):
        self.role = role
        self.has_ball = has_ball
        self.port = port
        self.model_path = model_path
        self.init_pos = init_pos
        self.running = False
        self.opp_pos = np.array([0, 0, 0.7])
        self.ball_pos = np.array([0, 0, 0.11])
        self.ball_vel = np.array([0, 0, 0])
        self.collision_push = np.array([0, 0, 0])

        # Load config
        with open('configs/hierarchical_agent.yaml') as f:
            self.cfg = yaml.safe_load(f)
        self.env_cfg = dict(self.cfg['env'])
        self.env_cfg['task'] = 'chase_hl'
        self.hl_cfg = self.cfg.get('high_level', {})

    def setup(self):
        gs.init(backend=gs.gpu, logging_level='warning')

        from envs.soccer_env_hierarchical import SoccerEnvHierarchical
        self.env = SoccerEnvHierarchical(
            num_envs=1, env_cfg=self.env_cfg, obs_cfg=self.cfg['obs'],
            reward_cfg=self.cfg['reward'], command_cfg=self.cfg['command'],
            walk_model_path=self.hl_cfg.get('walk_model_path'),
            high_level_decimation=self.hl_cfg.get('decimation', 5),
            show_viewer=False)

        # Set initial position
        self.env.init_base_pos = torch.tensor(self.init_pos, dtype=gs.tc_float, device=gs.device)

        # Load policy if agent
        self.policy = None
        if self.model_path:
            from rsl_rl.runners import OnPolicyRunner
            runner = OnPolicyRunner(self.env, self.cfg['train'],
                                   'runs/hierarchical_soccer_chase_hl', device=gs.device)
            runner.load(self.model_path)
            self.policy = runner.get_inference_policy(device=gs.device)

        # Connect to coordinator
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(('localhost', self.port))
        print(f'[{self.role}] Connected to coordinator')

    def run(self):
        self.running = True
        obs = self.env.reset()
        step = 0
        last_ball_pos = None

        while self.running and step < N_STEPS:
            # ─── Receive state from coordinator ───
            for _ in range(3):  # drain messages
                msg_type, data = recv_msg(self.sock)
                if msg_type is None:
                    self.running = False
                    break
                if msg_type == MSG_END:
                    print(f'[{self.role}] Received END signal')
                    self.running = False
                    break
                elif msg_type == MSG_STATE and data:
                    self.opp_pos = np.array(data[:3])
                elif msg_type == MSG_BALL and data:
                    if not self.has_ball:
                        self.ball_pos = np.array(data[:3])
                        self.ball_vel = np.array(data[3:6])
                elif msg_type == MSG_CMD and data:
                    self.collision_push = np.array(data[:3])

            if not self.running:
                break

            # ─── Compute action ───
            if self.policy:
                with torch.no_grad():
                    action = self.policy(obs)
            else:
                # Rule-based: move toward ball
                ball_rel = self.ball_pos - self.env.base_pos[0, :3].cpu().numpy()
                dist = np.linalg.norm(ball_rel[:2])
                if dist > 0.01:
                    direction = ball_rel[:2] / dist
                    action = torch.tensor([[
                        np.clip(direction[0] * 0.2, -0.2, 0.2),
                        np.clip(direction[1] * 0.2, -0.2, 0.2),
                        np.clip(np.arctan2(ball_rel[1], ball_rel[0]) * 0.1, -0.2, 0.2),
                    ]], dtype=gs.tc_float, device=gs.device)
                else:
                    action = torch.zeros((1, 3), dtype=gs.tc_float, device=gs.device)

            # ─── Step environment ───
            obs, rew, done, extras = self.env.step(action)
            step += 1

            # ─── Send our state to coordinator ───
            agent_pos = self.env.base_pos[0].cpu().numpy()
            agent_pitch = self.env.base_euler[0, 1].item()
            agent_roll = self.env.base_euler[0, 0].item()
            self.sock.sendall(pack_state(MSG_STATE, [
                agent_pos[0], agent_pos[1], agent_pos[2],
                agent_pitch, agent_roll
            ]))

            # If authority, send ball state
            if self.has_ball:
                ball_pos = self.env.ball_pos[0].cpu().numpy()
                ball_vel = self.env.ball_vel[0].cpu().numpy() if hasattr(self.env, 'ball_vel') else np.zeros(3)
                self.sock.sendall(pack_state(MSG_BALL, [
                    ball_pos[0], ball_pos[1], ball_pos[2],
                    ball_vel[0], ball_vel[1], ball_vel[2]
                ]))

            # Log
            if step % 50 == 0:
                ball_d = np.linalg.norm(agent_pos[:2] - self.ball_pos[:2])
                opp_d = np.linalg.norm(agent_pos[:2] - self.opp_pos[:2])
                print(f'[{self.role}] step {step}: h={agent_pos[2]:.3f} ball_d={ball_d:.2f} '
                      f'opp_d={opp_d:.2f} rew={rew.mean().item():.3f}')

        print(f'[{self.role}] Finished after {step} steps')
        self.sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', required=True, choices=['agent', 'opponent'])
    parser.add_argument('--has-ball', action='store_true')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--model', default=None)
    parser.add_argument('--init-pos', type=float, nargs=3, default=[0, 0, 0.7])
    args = parser.parse_args()

    worker = MatchWorker(
        role=args.role, has_ball=args.has_ball, port=args.port,
        model_path=args.model, init_pos=args.init_pos)
    worker.setup()
    worker.run()
