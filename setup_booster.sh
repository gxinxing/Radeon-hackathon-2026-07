#!/bin/bash
set -e
echo "=== Booster Studio Setup ==="

# 1. Install deps
echo "[1/5] Installing dependencies..."
apt-get update -qq
apt-get install -y wget curl libgl1 libglx-mesa0 libegl1 libgbm1 \
  libdrm2 libxkbcommon0 libwayland-client0 vulkan-tools mesa-vulkan-drivers \
  libvulkan1 xdg-utils libnotify4 libatspi2.0-0 libsecret-1-0 libgtk-3-0 \
  libnss3 libnspr4 libasound2t64 libxss1 xvfb mesa-utils dbus-x11 \
  docker.io tigervnc-standalone-server xterm openbox novnc websockify \
  2>&1 | tail -3
echo "  OK"

# 2. Install Booster Studio
echo "[2/5] Installing Booster Studio..."
if [ ! -f /usr/share/booster-studio/booster-studio ]; then
  wget -q -O /tmp/booster-studio.deb \
    "https://ci-cdn.booster.tech/release/Booster%20Studio-Setup-1.9.4-release-0720f659-linux-x64.deb"
  dpkg -i /tmp/booster-studio.deb 2>&1 | tail -3
  apt-get install -f -y -qq 2>&1 | tail -3
fi
echo "  OK"

# 3. Start Docker
echo "[3/5] Starting Docker..."
if ! pgrep -x dockerd >/dev/null 2>&1; then
  nohup dockerd --iptables=false --bridge=none &>/tmp/dockerd.log 2>&1 &
  sleep 5
fi
docker info >/dev/null 2>&1 && echo "  OK" || echo "  WARN"

# 4. Start VNC + Booster Studio
echo "[4/5] Starting VNC + Booster Studio..."
mkdir -p ~/.vnc
echo "***REMOVED***" | vncpasswd -f > ~/.vnc/passwd
chmod 600 ~/.vnc/passwd
cat > ~/.vnc/xstartup << 'XEOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
openbox &
sleep 2
DISPLAY=:99 /usr/share/booster-studio/booster-studio --no-sandbox --disable-gpu-sandbox &
exec xterm
XEOF
chmod +x ~/.vnc/xstartup
pkill -f Xtigervnc 2>/dev/null; sleep 1
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null
vncserver :99 -geometry 1920x1080 -depth 24 -localhost no 2>&1 | tail -3
sleep 3
echo "  OK (port 5999, password: ***REMOVED***)"

# 5. Start noVNC + install proxy
echo "[5/5] Starting noVNC + Jupyter proxy..."
pkill -f websockify 2>/dev/null; sleep 1
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5999 &>/tmp/websockify.log 2>&1 &
sleep 2

# Install jupyter-server-proxy
/opt/venv/bin/pip install -q jupyter-server-proxy 2>&1 | tail -1
/opt/venv/bin/jupyter server extension enable jupyter_server_proxy 2>&1 | tail -1

# Write proxy config
mkdir -p /opt/venv/etc/jupyter/jupyter_server_config.d/
cat > /opt/venv/etc/jupyter/jupyter_server_config.d/jupyter_server_proxy.json << 'JEOF'
{
  "ServerProxy": {
    "servers": {
      "vnc": {
        "command": ["websockify", "--web=/usr/share/novnc/", "6080", "localhost:5999"],
        "port": 6080,
        "launcher_entry": {"title": "Booster Studio (VNC)"}
      }
    }
  }
}
JEOF

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "VNC: port 5999 (password: ***REMOVED***)"
echo "noVNC: port 6080"
echo ""
echo ">>> IMPORTANT <<<"
echo "You need to RESTART the instance from the"
echo "Anrui Cloud console for the proxy to work."
echo "After restart, run this script again:"
echo "  bash /workspace/setup_booster.sh"
echo ""
echo "Then open:"
echo "  https://radeon-global.anruicloud.com/instances/u-9230-e9a5adc6/proxy/6080/vnc.html"
echo ""
echo "Or if proxy doesn't work, open JupyterLab"
echo "Terminal and run:"
echo "  websockify --web=/usr/share/novnc/ 8889 localhost:5999 &"
echo "Then access port 8889"
echo ""
