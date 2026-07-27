"""Multi-process 1v1 match: RL agent (process 1, has ball) vs rule opponent (process 2).

Architecture:
  - Process 1 (RL_AGENT): Genesis scene with RL robot + ball (authority)
  - Process 2 (RULE_OPP): Genesis scene with rule robot only (no ball physics)
  - Coordinator: socket server syncing state between processes

Each process runs its own Genesis instance with 1 robot (proven stable).
The coordinator broadcasts ball position from P1 to P2, and robot positions both ways.
Collision is approximated: if robots are within 0.3m, apply push-back velocity.
"""
import argparse, socket, struct, json, threading, time, sys, os
import numpy as np

# ─── Protocol ───
MSG_STATE = 1    # robot state: x, y, z, pitch, roll
MSG_BALL = 2      # ball state: x, y, z, vx, vy, vz
MSG_CMD = 3       # velocity command: vx, vy, wz
MSG_END = 4       # match end

DEFAULT_PORT = 9876
MATCH_DURATION = 20.0  # seconds
COLLISION_DIST = 0.3   # meters


def pack_state(msg_type, data):
    """Pack a message: [type:1][len:4][data:N]"""
    payload = struct.pack(f'<{len(data)}f', *data)
    header = struct.pack('<BI', msg_type, len(data))
    return header + payload


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
    data = recv_all(sock, length * 4)
    if not data:
        return None, None
    values = struct.unpack(f'<{length}f', data)
    return msg_type, values


class MatchCoordinator:
    """Central coordinator that syncs state between two processes."""

    def __init__(self, port=DEFAULT_PORT):
        self.port = port
        self.running = False
        self.state = {
            'agent': {'x': 0, 'y': 0, 'z': 0.7, 'pitch': 0, 'roll': 0},
            'opp': {'x': -3, 'y': 0, 'z': 0.7, 'pitch': 0, 'roll': 0},
            'ball': {'x': 0, 'y': 0, 'z': 0.11, 'vx': 0, 'vy': 0, 'vz': 0},
        }
        self.lock = threading.Lock()
        self.clients = {}

    def run(self):
        self.running = True
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('localhost', self.port))
        server.listen(2)
        print(f'[Coordinator] Listening on port {self.port}')

        # Accept 2 clients
        for i in range(2):
            conn, addr = server.accept()
            name = f'client_{i}'
            self.clients[name] = conn
            print(f'[Coordinator] {name} connected from {addr}')
            threading.Thread(target=self._handle_client, args=(conn, name), daemon=True).start()

        # Broadcast loop
        start = time.time()
        while self.running and time.time() - start < MATCH_DURATION:
            with self.lock:
                agent_state = self.state['agent'].copy()
                opp_state = self.state['opp'].copy()
                ball_state = self.state['ball'].copy()

            # Check collision
            dx = agent_state['x'] - opp_state['x']
            dy = agent_state['y'] - opp_state['y']
            dist = (dx**2 + dy**2) ** 0.5
            collision = dist < COLLISION_DIST

            # Send to client_0 (agent): opp state + ball
            if 'client_0' in self.clients:
                self._send(self.clients['client_0'], MSG_STATE,
                          [opp_state['x'], opp_state['y'], opp_state['z'],
                           opp_state['pitch'], opp_state['roll']])
                self._send(self.clients['client_0'], MSG_BALL,
                          [ball_state['x'], ball_state['y'], ball_state['z'],
                           ball_state['vx'], ball_state['vy'], ball_state['vz']])
                if collision:
                    self._send(self.clients['client_0'], MSG_CMD, [-dx/dist*0.1, -dy/dist*0.1, 0])

            # Send to client_1 (opponent): agent state + ball
            if 'client_1' in self.clients:
                self._send(self.clients['client_1'], MSG_STATE,
                          [agent_state['x'], agent_state['y'], agent_state['z'],
                           agent_state['pitch'], agent_state['roll']])
                self._send(self.clients['client_1'], MSG_BALL,
                          [ball_state['x'], ball_state['y'], ball_state['z'],
                           ball_state['vx'], ball_state['vy'], ball_state['vz']])
                if collision:
                    self._send(self.clients['client_1'], MSG_CMD, [dx/dist*0.1, dy/dist*0.1, 0])

            time.sleep(0.02)  # 50 Hz

        # Signal end
        for conn in self.clients.values():
            try:
                self._send(conn, MSG_END, [])
            except:
                pass

        elapsed = time.time() - start
        print(f'[Coordinator] Match ended after {elapsed:.1f}s')
        self.running = False

    def _send(self, conn, msg_type, data):
        try:
            conn.sendall(pack_state(msg_type, data))
        except:
            pass

    def _handle_client(self, conn, name):
        while self.running:
            msg_type, data = recv_msg(conn)
            if msg_type is None:
                break
            with self.lock:
                if msg_type == MSG_STATE and data:
                    if name == 'client_0':
                        self.state['agent'] = {'x': data[0], 'y': data[1], 'z': data[2],
                                              'pitch': data[3], 'roll': data[4]}
                    else:
                        self.state['opp'] = {'x': data[0], 'y': data[1], 'z': data[2],
                                             'pitch': data[3], 'roll': data[4]}
                elif msg_type == MSG_BALL and data:
                    self.state['ball'] = {'x': data[0], 'y': data[1], 'z': data[2],
                                          'vx': data[3], 'vy': data[4], 'vz': data[5]}
        print(f'[Coordinator] {name} disconnected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--duration', type=float, default=MATCH_DURATION)
    args = parser.parse_args()
    MATCH_DURATION = args.duration
    coord = MatchCoordinator(args.port)
    coord.run()
