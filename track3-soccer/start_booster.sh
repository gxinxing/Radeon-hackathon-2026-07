#!/bin/bash
# ============================================================
# Booster Studio 一键启动脚本 (AMD GPU)
# 在 JupyterLab Terminal 中运行: bash /workspace/start_booster.sh
# ============================================================
set -e

echo "============================================"
echo "  Booster Studio 启动脚本"
echo "============================================"

# 1. 安装依赖
echo "[1/5] 安装系统依赖..."
apt-get update -qq 2>/dev/null
apt-get install -y -qq wget curl libgl1 libglx-mesa0 libegl1 libgbm1 \
  libdrm2 libxkbcommon0 libwayland-client0 vulkan-tools mesa-vulkan-drivers \
  libvulkan1 xdg-utils libnotify4 libatspi2.0-0 libsecret-1-0 libgtk-3-0 \
  libnss3 libnspr4 libasound2 libxss1 xvfb mesa-utils dbus-x11 \
  docker.io tigervnc-standalone-server xterm openbox novnc websockify \
  2>&1 | tail -3
echo "✅ 依赖安装完成"

# 2. 安装 jupyter-server-proxy
echo "[2/5] 安装 Jupyter 代理扩展..."
/opt/venv/bin/pip install -q jupyter-server-proxy 2>&1 | tail -2
/opt/venv/bin/jupyter server extension enable jupyter_server_proxy 2>&1 | tail -2
echo "✅ 代理扩展已安装"

# 3. 下载安装 Booster Studio
if ! command -v booster-studio &>/dev/null && [ ! -f /usr/share/booster-studio/booster-studio ]; then
  echo "[3/5] 下载 Booster Studio..."
  wget -q --show-progress -O /tmp/booster-studio.deb \
    "https://ci-cdn.booster.tech/release/Booster%20Studio-Setup-1.9.4-release-0720f659-linux-x64.deb" 2>&1 | tail -3
  dpkg -i /tmp/booster-studio.deb 2>&1 | tail -3 || apt-get install -f -y -qq 2>&1 | tail -3
  echo "✅ Booster Studio 安装完成"
else
  echo "[3/5] ✅ Booster Studio 已安装"
fi

# 4. 启动 Docker
echo "[4/5] 启动 Docker..."
if ! pgrep -x dockerd >/dev/null 2>&1; then
  nohup dockerd --iptables=false --bridge=none &>/tmp/dockerd.log 2>&1 &
  sleep 4
fi
docker info >/dev/null 2>&1 && echo "✅ Docker 运行中" || echo "⚠️ Docker 启动失败"

# 5. 启动 VNC + Booster Studio + noVNC
echo "[5/5] 启动 VNC 和 Booster Studio..."
pkill -f Xtigervnc 2>/dev/null; sleep 1
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null

# VNC 密码
mkdir -p ~/.vnc
echo "***REMOVED***" | vncpasswd -f > ~/.vnc/passwd
chmod 600 ~/.vnc/passwd

# xstartup
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
openbox &
sleep 2
DISPLAY=:99 /usr/share/booster-studio/booster-studio --no-sandbox --disable-gpu-sandbox &
exec xterm
EOF
chmod +x ~/.vnc/xstartup

# 启动 VNC
vncserver :99 -geometry 1920x1080 -depth 24 -localhost no 2>&1 | tail -3
sleep 3

# 启动 websockify (noVNC)
pkill -f websockify 2>/dev/null; sleep 1
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5999 &>/tmp/websockify.log 2>&1 &
sleep 2

echo ""
echo "============================================"
echo "  ✅ 全部启动完成！"
echo "============================================"
echo ""
echo "VNC 端口: 5999 (密码: ***REMOVED***)"
echo "noVNC 端口: 6080"
echo ""
echo "⚠️ 重要: 需要重启 Jupyter 实例使代理扩展生效"
echo "  重启后在 JupyterLab Terminal 中重新运行此脚本"
echo "  然后访问 noVNC Web 界面"
echo ""
echo "进程状态:"
ps aux | grep -E "dockerd|Xtigervnc|booster-studio|websockify" | grep -v grep | awk '{print $2, $11, $12}'
