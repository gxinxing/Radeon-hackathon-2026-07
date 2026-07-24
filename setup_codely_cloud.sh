#!/bin/bash
# ============================================================
# Codely CLI 云电脑安装+启动脚本
# 在 JupyterLab Terminal 中运行: bash setup_codely_cloud.sh
# ============================================================
set -e

CODELY_VERSION="1.0.0-release.41"
CODELY_URL="https://codesearch-plugins.cdn.tuanjie.cn/codely-cli-binary/codely-linux-x64-${CODELY_VERSION}"
CODELY_BIN="/usr/local/bin/codely"
CODELY_HOME="$HOME/.codely"

echo "============================================"
echo "  Codely CLI 云电脑安装脚本"
echo "  版本: ${CODELY_VERSION}"
echo "============================================"
echo ""

# ---- Step 1: 检查/安装依赖 ----
echo "[1/6] 检查系统依赖..."
apt-get update -qq 2>/dev/null

# VNC 和终端相关依赖
DEPS=(
  wget curl tigervnc-standalone-server xterm openbox novnc websockify
  libgl1 libglx-mesa0 libegl1 libgbm1 libdrm2 libxkbcommon0
  libwayland-client0 libvulkan1 libnotify4 libatspi2.0-0
  libsecret-1-0 libgtk-3-0 libnss3 libnspr4 libasound2t64
  libxss1 mesa-utils dbus-x11
)
for dep in "${DEPS[@]}"; do
  dpkg -s "$dep" &>/dev/null || apt-get install -y -qq "$dep" 2>/dev/null || true
done
echo "  ✅ 依赖就绪"

# ---- Step 2: 下载安装 Codely CLI ----
echo ""
echo "[2/6] 下载 Codely CLI (Linux x64)..."
if [ -f "$CODELY_BIN" ] && "$CODELY_BIN" --version &>/dev/null; then
  CURRENT=$("$CODELY_BIN" --version 2>/dev/null || echo "unknown")
  echo "  已安装: $CURRENT"
  if [ "$CURRENT" != "$CODELY_VERSION" ]; then
    echo "  更新到 $CODELY_VERSION ..."
    wget -q --show-progress -O /tmp/codely-linux "$CODELY_URL" 2>&1 | tail -3
    chmod +x /tmp/codely-linux
    mv /tmp/codely-linux "$CODELY_BIN"
  fi
else
  wget -q --show-progress -O /tmp/codely-linux "$CODELY_URL" 2>&1 | tail -3
  chmod +x /tmp/codely-linux
  mv /tmp/codely-linux "$CODELY_BIN"
fi
echo "  ✅ Codely CLI: $("$CODELY_BIN" --version 2>/dev/null)"

# ---- Step 3: 配置认证 Token ----
echo ""
echo "[3/6] 配置认证 Token..."
echo ""
echo "  ┌──────────────────────────────────────────────────────┐"
echo "  │  请粘贴你的 CODELY_TOKEN (从本地终端获取)             │"
echo "  │  获取方法: 在本地 Mac 终端运行 echo \$CODELY_TOKEN    │"
echo "  │  粘贴后按回车                                         │"
echo "  └──────────────────────────────────────────────────────┘"
echo ""
read -r -p "  CODELY_TOKEN: " CODELY_TOKEN_INPUT

if [ -z "$CODELY_TOKEN_INPUT" ]; then
  echo "  ⚠️  未输入 Token，Codely CLI 可能无法正常认证"
  echo "  你可以稍后手动设置: export CODELY_TOKEN=\"your-token\""
else
  echo "  ✅ Token 已接收 (${#CODELY_TOKEN_INPUT} 字符)"
fi

# ---- Step 4: 创建配置文件 ----
echo ""
echo "[4/6] 创建配置文件..."
mkdir -p "$CODELY_HOME/.configs"

# 远程模型配置 (不含密钥)
cat > "$CODELY_HOME/.configs/remote-codely-cloud.yaml" << 'YAMLEOF'
context:
-   provider: code
-   provider: diff
-   provider: terminal
-   provider: problems
-   provider: folder
-   provider: codebase
-   provider: repo-map
-   provider: currentFile
data:
-   destination: https://codely.tuanjie.cn/api/codely/storage
    level: all
    name: Local Data
    schema: 0.2.0
models:
-   apiBase: https://codely.tuanjie.cn/v1
    capabilities:
    - tool_use
    defaultCompletionOptions:
        contextLength: 202752
        maxTokens: 16384
        temperature: 0.7
        topP: 1
    description: Balanced reasoning and speed. Recommended for daily use.
    model: codely-core
    name: 'Core (GLM-5.2-MAX)'
    provider: openai
    roles:
    - chat
-   apiBase: https://codely.tuanjie.cn/v1
    capabilities:
    - tool_use
    defaultCompletionOptions:
        contextLength: 202752
        maxTokens: 16384
        temperature: 0.7
        topP: 1
    description: Quick responses, lightweight tasks. The cheapest option.
    model: codely-basic
    name: Basic
    provider: openai
    roles:
    - chat
    - edit
    - apply
name: Codely Assistant
schema: v1
version: 1.0.0
YAMLEOF

# 基础 config.yaml
cat > "$CODELY_HOME/config.yaml" << 'YAMLEOF'
name: Cloud Agent
version: 1.0.0
schema: v1
models: []
YAMLEOF

# cowork-settings.json (Token 从用户输入填充)
if [ -n "$CODELY_TOKEN_INPUT" ]; then
  cat > "$CODELY_HOME/cowork-settings.json" << JSONEOF
{
  "ContinueAccessToken": "$CODELY_TOKEN_INPUT",
  "ContinueAccountId": "",
  "ContinueAccountLabel": "cloud-instance",
  "ThemeMode": "dark",
  "enableTunnel": "true",
  "keepAwake": "true"
}
JSONEOF
fi

# 创建启动包装脚本
cat > "$CODELY_HOME/run_codely.sh" << 'SHEOF'
#!/bin/bash
# Codely CLI 启动包装
export CODELY_TOKEN="${CODELY_TOKEN:-$(cat ~/.codely/.token 2>/dev/null)}"
export CODELY_CLIENT_TYPE="cowork"
export GEMINI_CLI=1
export CODELY_EXTENSION_VERSION=1.1.1
export CODELY_SIDECHAT=false

# 工作目录
WORKDIR="${1:-/workspace}"
cd "$WORKDIR" 2>/dev/null || cd "$HOME"

echo "╔══════════════════════════════════════════╗"
echo "║       Codely CLI (Cloud Instance)        ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  工作目录: $(pwd)"
echo "  版本: $(codely --version 2>/dev/null)"
echo ""

exec codely
SHEOF
chmod +x "$CODELY_HOME/run_codely.sh"

# 保存 token 供启动脚本使用
if [ -n "$CODELY_TOKEN_INPUT" ]; then
  echo -n "$CODELY_TOKEN_INPUT" > "$CODELY_HOME/.token"
  chmod 600 "$CODELY_HOME/.token"
fi

echo "  ✅ 配置文件已创建"
echo "     - $CODELY_HOME/.configs/remote-codely-cloud.yaml"
echo "     - $CODELY_HOME/config.yaml"
echo "     - $CODELY_HOME/run_codely.sh"

# ---- Step 5: 启动 VNC ----
echo ""
echo "[5/6] 启动 VNC 桌面..."
mkdir -p ~/.vnc
echo "***REMOVED***" | vncpasswd -f > ~/.vnc/passwd
chmod 600 ~/.vnc/passwd

# xstartup: 打开 xterm 运行 codely
cat > ~/.vnc/xstartup << 'XEOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
openbox &
sleep 2
# 打开 xterm 窗口运行 Codely CLI
xterm -fs 14 -fa "Monospace" -bg "#1e1e2e" -fg "#cdd6f4" \
  -title "Codely CLI" -maximized -e "$HOME/.codely/run_codely.sh" &
exec xterm -fs 14 -fa "Monospace" -bg "#1e1e2e" -fg "#cdd6f4" -title "Terminal"
XEOF
chmod +x ~/.vnc/xstartup

# 杀掉旧的 VNC 进程
pkill -9 Xtigervnc 2>/dev/null; sleep 1
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null

# 启动 VNC
nohup Xtigervnc :99 -localhost no -rfbport 5999 \
  -PasswordFile ~/.vnc/passwd \
  -geometry 1920x1080 -depth 24 \
  -SecurityTypes VncAuth >/tmp/vnc.log 2>&1 &
sleep 3
echo "  ✅ VNC 已启动 (端口 5999, 密码: ***REMOVED***)"

# ---- Step 6: 启动 noVNC ----
echo ""
echo "[6/6] 启动 noVNC Web 界面..."
pkill -f websockify 2>/dev/null; sleep 1
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5999 &>/tmp/ws.log 2>&1 &
sleep 2
echo "  ✅ noVNC 已启动 (端口 6080)"

# ---- 完成 ----
echo ""
echo "============================================"
echo "  ✅ 全部完成！"
echo "============================================"
echo ""
echo "  在浏览器中打开:"
echo "    https://radeon-global.anruicloud.com/instances/u-9230-e9a5adc6/proxy/6080/vnc.html"
echo ""
echo "  VNC 密码: ***REMOVED***"
echo ""
echo "  打开后你会看到两个窗口:"
echo "    1. Codely CLI - 自动启动的 Codely 终端"
echo "    2. Terminal - 备用终端"
echo ""
echo "  进程状态:"
ps aux | grep -E "Xtigervnc|websockify|openbox" | grep -v grep | awk '{print $2, $11, $12, $13}'
echo ""
echo "  如需重新启动 Codely CLI:"
echo "    ~/.codely/run_codely.sh /workspace"
echo ""
