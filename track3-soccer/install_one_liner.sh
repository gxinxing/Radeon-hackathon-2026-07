#!/bin/bash
# ============================================================
# Booster Studio 一键安装 (AMD GPU Linux)
# 在 Web 终端中粘贴此命令执行:
# bash install_one_linerr.sh
# ============================================================

# Step 1: 检查环境
echo "=== 系统环境 ==="
cat /etc/os-release | head -5
echo ""
echo "=== GPU 检测 ==="
lspci | grep -iE 'vga|3d|display|amd' || echo "(lspci 无GPU信息)"
echo ""
ls -la /dev/kfd /dev/dri/render* 2>/dev/null || echo "未检测到AMD GPU设备节点"
echo ""
rocm-smi 2>/dev/null || echo "(rocm-smi 未安装或不可用)"
echo ""

# Step 2: 安装依赖
echo "=== 安装系统依赖 ==="
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq wget curl libgl1 libglx-mesa0 libegl1 libgles2 libgbm1 \
    libdrm2 libxkbcommon0 libwayland-client0 libwayland-server0 \
    vulkan-tools mesa-vulkan-drivers libvulkan1 xdg-utils libnotify4 \
    libatspi2.0-0 libsecret-1-0 libgtk-3-0 libnss3 libnspr4 libasound2 \
    libxss1 xvfb 2>&1 | tail -3
echo "✅ 依赖安装完成"
echo ""

# Step 3: 下载 Booster Studio
echo "=== 下载 Booster Studio ==="
DEB_URL="https://ci-cdn.booster.tech/release/Booster%20Studio-Setup-1.9.4-release-0720f659-linux-x64.deb"
wget -q --show-progress -O /tmp/booster-studio.deb "$DEB_URL"
ls -lh /tmp/booster-studio.deb
echo ""

# Step 4: 安装
echo "=== 安装 Booster Studio ==="
dpkg -i /tmp/booster-studio.deb 2>&1 || {
    echo "修复依赖..."
    apt-get install -f -y -qq 2>&1 | tail -3
    dpkg -i /tmp/booster-studio.deb 2>&1
}
echo ""

# Step 5: 检查安装
echo "=== 安装结果 ==="
dpkg -l | grep -i booster || echo "(未找到已安装的booster包)"
echo ""
echo "=== 可执行文件 ==="
find /opt /usr/bin /usr/local/bin -iname '*booster*' 2>/dev/null || echo "(未找到可执行文件)"
echo ""

# Step 6: 检查 NVIDIA 依赖
echo "=== 检查 NVIDIA/CUDA 依赖 ==="
mkdir -p /tmp/booster-check
dpkg-deb -x /tmp/booster-studio.deb /tmp/booster-check 2>/dev/null
NVIDIA_FOUND=$(find /tmp/booster-check -type f -exec grep -l -iE 'nvidia|cuda|libcuda' {} \; 2>/dev/null | head -5)
if [[ -n "$NVIDIA_FOUND" ]]; then
    echo "⚠️  发现 NVIDIA/CUDA 引用:"
    echo "$NVIDIA_FOUND"
    echo "部分GPU加速功能可能需要NVIDIA GPU"
else
    echo "✅ 未发现硬性NVIDIA依赖"
fi
echo ""

# Step 7: 检查渲染后端
echo "=== 渲染后端 ==="
echo "Vulkan:"
vulkaninfo --summary 2>/dev/null | head -15 || echo "Vulkan 不可用"
echo ""
echo "OpenGL:"
apt-get install -y -qq mesa-utils 2>/dev/null
glxinfo 2>/dev/null | grep -i 'opengl renderer' || echo "(glxinfo 不可用或无DISPLAY)"
echo ""

# Step 8: 设置虚拟显示（如需要）
echo "=== 虚拟显示 ==="
if [[ -z "$DISPLAY" ]]; then
    echo "无图形显示环境，配置虚拟显示..."
    Xvfb :99 -screen 0 1920x1080x24 &
    export DISPLAY=:99
    echo "✅ DISPLAY=:99 已设置"
else
    echo "✅ DISPLAY=$DISPLAY"
fi
echo ""

echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "运行 Booster Studio:"
echo "  booster-studio --no-sandbox"
echo ""
echo "或查找可执行文件:"
echo "  find /opt -name 'booster*' -type f -executable"
echo ""
echo "如需VNC远程桌面:"
echo "  apt-get install -y tigervnc-standalone-server"
echo "  vncserver :99 -geometry 1920x1080"
echo ""
