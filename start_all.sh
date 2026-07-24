#!/bin/bash
# Booster Studio 一键启动脚本
# 在 JupyterLab Terminal 中运行: bash /workspace/start_all.sh
set -e

echo "=== 1/4 安装依赖 ==="
apt-get update -qq 2>/dev/null
apt-get install -y wget tigervnc-standalone-server xterm openbox novnc websockify \
  libgl1 libglx-mesa0 libegl1 libgbm1 libdrm2 libxkbcommon0 libwayland-client0 \
  libvulkan1 libnotify4 libatspi2.0-0 libsecret-1-0 libgtk-3-0 libnss3 libnspr4 \
  libasound2t64 libxss1 mesa-utils dbus-x11 2>&1 | tail -3

echo "=== 2/4 安装 Booster Studio ==="
if [ ! -f /usr/share/booster-studio/booster-studio ]; then
  wget -q -O /tmp/bs.deb "https://ci-cdn.booster.tech/release/Booster%20Studio-Setup-1.9.4-release-0720f659-linux-x64.deb"
  dpkg -i /tmp/bs.deb 2>&1 | tail -3 || apt-get install -f -y -qq 2>&1 | tail -3
fi
echo "  OK"

echo "=== 3/4 启动 VNC ==="
mkdir -p ~/.vnc
echo "***REMOVED***" | vncpasswd -f > ~/.vnc/passwd
chmod 600 ~/.vnc/passwd
pkill -9 Xtigervnc 2>/dev/null; sleep 1
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
nohup Xtigervnc :99 -localhost no -rfbport 5999 -PasswordFile ~/.vnc/passwd \
  -geometry 1920x1080 -depth 24 -SecurityTypes VncAuth >/tmp/vnc.log 2>&1 &
sleep 3

export DISPLAY=:99
nohup openbox &>/dev/null 2>&1 &
sleep 1
nohup /usr/share/booster-studio/booster-studio --no-sandbox --disable-gpu-sandbox &>/tmp/bs.log 2>&1 &
sleep 3

echo "=== 4/4 启动 noVNC ==="
pkill -f websockify 2>/dev/null; sleep 1
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5999 &>/tmp/ws.log 2>&1 &
sleep 2

echo ""
echo "========================================"
echo "  完成！"
echo "========================================"
echo ""
echo "在浏览器中打开:"
echo "  https://radeon-global.anruicloud.com/instances/<REDACTED>/proxy/6080/vnc.html"
echo ""
echo "VNC 密码: ***REMOVED***"
echo ""
echo "端口状态:"
ss -tlnp | grep -E "5999|6080"
