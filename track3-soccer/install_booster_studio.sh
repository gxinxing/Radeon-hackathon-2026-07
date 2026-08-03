#!/bin/bash
#=============================================================================
# Booster Studio 安装脚本 (AMD GPU Linux)
# 适用: Ubuntu 22.04 / 24.04 + AMD GPU (ROCm)
#=============================================================================
set -e

echo "============================================"
echo "  Booster Studio Installer for AMD GPU"
echo "============================================"

# ---- Step 1: 系统环境检查 ----
echo ""
echo "[1/6] 检查系统环境..."

OS_ID=$(grep '^ID=' /etc/os-release | cut -d'"' -f2)
OS_VERSION=$(grep '^VERSION_ID=' /etc/os-release | cut -d'"' -f2)
echo "  OS: $OS_ID $OS_VERSION"

if [[ "$OS_ID" != "ubuntu" ]]; then
    echo "  ⚠️  警告: 官方支持 Ubuntu 22/24，当前系统为 $OS_ID"
    echo "  继续安装，但可能遇到兼容性问题..."
fi

echo "  Kernel: $(uname -r)"
echo "  CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "  RAM: $(free -h | awk '/^Mem:/{print $2}')"

# ---- Step 2: GPU 检查 ----
echo ""
echo "[2/6] 检查 GPU..."

echo "  --- lspci GPU ---"
lspci | grep -iE 'vga|3d|display' || echo "  (未检测到GPU设备)"

echo ""
echo "  --- AMD GPU (ROCm) ---"
if command -v rocm-smi &>/dev/null; then
    rocm-smi --showproductname 2>/dev/null || rocm-smi 2>/dev/null || echo "  rocm-smi 存在但无法获取信息"
    echo ""
    echo "  ROCm 版本:"
    rocm-smi --showversion 2>/dev/null || true
else
    echo "  ⚠️  rocm-smi 未安装，尝试检测 AMD GPU 驱动..."
    if ls /dev/kfd 2>/dev/null; then
        echo "  ✅ 检测到 /dev/kfd (AMD GPU 驱动节点)"
    else
        echo "  ❌ 未检测到 /dev/kfd，AMD GPU 驱动可能未安装"
    fi
    if ls /dev/dri/render* 2>/dev/null; then
        echo "  ✅ 检测到 DRI 渲染节点"
    fi
fi

echo ""
echo "  --- Vulkan (GPU 渲染后端) ---"
if command -v vulkaninfo &>/dev/null; then
    vulkaninfo --summary 2>/dev/null | head -20 || echo "  vulkaninfo 存在但无法获取信息"
else
    echo "  ⚠️  vulkaninfo 未安装"
fi

# ---- Step 3: 安装系统依赖 ----
echo ""
echo "[3/6] 安装系统依赖..."

apt-get update -qq

# 基础依赖
DEPS=(
    wget
    curl
    libgl1
    libglx-mesa0
    libegl1
    libgles2
    libgbm1
    libdrm2
    libxkbcommon0
    libwayland-client0
    libwayland-server0
    vulkan-tools
    mesa-vulkan-drivers
    libvulkan1
    xdg-utils
    libnotify4
    libatspi2.0-0
    libsecret-1-0
    libgtk-3-0
    libnss3
    libnspr4
    libasound2
    libxss1
    libgconf-2-4
)

# Ubuntu 24.04 中某些包名可能变化
if [[ "$OS_VERSION" == "24.04" ]] || [[ "$OS_VERSION" == "24.10" ]]; then
    DEPS+=(
        libasound2t64
    )
    # 移除可能不存在的包
    DEPS=(${DEPS[@]/libgconf-2-4/})
fi

echo "  安装依赖: ${DEPS[*]}"
apt-get install -y -qq "${DEPS[@]}" 2>&1 | tail -3 || {
    echo "  ⚠️  部分依赖安装失败，尝试逐个安装..."
    for dep in "${DEPS[@]}"; do
        apt-get install -y -qq "$dep" 2>/dev/null || echo "    跳过: $dep"
    done
}

echo "  ✅ 系统依赖安装完成"

# ---- Step 4: 下载 Booster Studio ----
echo ""
echo "[4/6] 下载 Booster Studio..."

DEB_URL="https://ci-cdn.booster.tech/release/Booster%20Studio-Setup-1.9.4-release-0720f659-linux-x64.deb"
DEB_FILE="/tmp/booster-studio.deb"

echo "  下载地址: $DEB_URL"
wget -q --show-progress -O "$DEB_FILE" "$DEB_URL" 2>&1 | tail -5

if [[ ! -f "$DEB_FILE" ]] || [[ ! -s "$DEB_FILE" ]]; then
    echo "  ❌ 下载失败"
    echo "  尝试使用 curl..."
    curl -L -o "$DEB_FILE" "$DEB_URL" 2>&1 | tail -5
fi

if [[ -f "$DEB_FILE" ]] && [[ -s "$DEB_FILE" ]]; then
    FILE_SIZE=$(du -h "$DEB_FILE" | cut -f1)
    echo "  ✅ 下载完成: $DEB_FILE ($FILE_SIZE)"
else
    echo "  ❌ 下载彻底失败，请检查网络连接"
    exit 1
fi

# ---- Step 5: 安装 Booster Studio ----
echo ""
echo "[5/6] 安装 Booster Studio..."

# 先检查依赖
echo "  检查 .deb 依赖..."
MISSING_DEPS=$(dpkg-deb -f "$DEB_FILE" Depends 2>/dev/null || echo "")
if [[ -n "$MISSING_DEPS" ]]; then
    echo "  声明的依赖: $MISSING_DEPS"
fi

echo "  安装中..."
dpkg -i "$DEB_FILE" 2>&1 || {
    echo "  依赖问题，尝试修复..."
    apt-get install -f -y -qq 2>&1 | tail -5
    dpkg -i "$DEB_FILE" 2>&1 || {
        echo "  ❌ 安装失败"
        exit 1
    }
}

echo "  ✅ Booster Studio 安装完成"

# 检查安装位置
echo ""
echo "  --- 安装文件位置 ---"
find /opt -name "*booster*" -o -name "*Booster*" 2>/dev/null | head -10 || true
find /usr/share/applications -name "*booster*" -o -name "*Booster*" 2>/dev/null | head -5 || true
which booster-studio 2>/dev/null || true
dpkg -L booster-studio 2>/dev/null | head -20 || dpkg -L "booster-studio" 2>/dev/null | head -20 || true

# ---- Step 6: 运行环境检查 ----
echo ""
echo "[6/6] 运行环境检查..."

echo ""
echo "  --- 检查 NVIDIA 依赖 ---"
echo "  扫描 .deb 内容中的 NVIDIA/CUDA 引用..."
mkdir -p /tmp/booster-extract
dpkg-deb -x "$DEB_FILE" /tmp/booster-extract 2>/dev/null || true

NVIDIA_REFS=$(find /tmp/booster-extract -type f -exec grep -l -iE 'nvidia|cuda|libcuda' {} \; 2>/dev/null | head -10)
if [[ -n "$NVIDIA_REFS" ]]; then
    echo "  ⚠️  发现 NVIDIA/CUDA 引用:"
    echo "$NVIDIA_REFS"
    echo ""
    echo "  这可能意味着 Booster Studio 部分 GPU 加速功能依赖 NVIDIA GPU。"
    echo "  但基本功能（UI、代码编辑、基础仿真）可能仍可在 AMD GPU 上运行。"
else
    echo "  ✅ 未发现硬性 NVIDIA 依赖"
fi

echo ""
echo "  --- 检查渲染后端 ---"
VULKAN_REFS=$(find /tmp/booster-extract -type f -exec grep -l -i 'vulkan' {} \; 2>/dev/null | head -5)
OPENGL_REFS=$(find /tmp/booster-extract -type f -exec grep -l -i 'opengl\|mesa\|egl' {} \; 2>/dev/null | head -5)
if [[ -n "$VULKAN_REFS" ]]; then
    echo "  ✅ 检测到 Vulkan 渲染后端（AMD GPU 兼容）"
fi
if [[ -n "$OPENGL_REFS" ]]; then
    echo "  ✅ 检测到 OpenGL/Mesa 渲染后端（AMD GPU 兼容）"
fi

echo ""
echo "  --- 虚拟显示检查 ---"
if echo $DISPLAY | grep -q ':' 2>/dev/null; then
    echo "  ✅ DISPLAY=$DISPLAY (有图形显示环境)"
else
    echo "  ⚠️  无 DISPLAY 环境变量，可能需要虚拟显示"
    echo "  安装 Xvfb 用于无头运行..."
    apt-get install -y -qq xvfb 2>/dev/null || true
    echo ""
    echo "  如需无头运行 Booster Studio，可执行:"
    echo "    Xvfb :99 -screen 0 1920x1080x24 &"
    echo "    export DISPLAY=:99"
    echo "    booster-studio --no-sandbox &"
fi

# ---- 清理 ----
rm -rf /tmp/booster-extract
# 不删除 .deb 文件，以备重装需要

# ---- 完成报告 ----
echo ""
echo "============================================"
echo "  安装完成！"
echo "============================================"
echo ""
echo "  下一步:"
echo "  1. 如果有图形界面: 直接运行 booster-studio"
echo "  2. 如果是无头服务器:"
echo "     Xvfb :99 -screen 0 1920x1080x24 &"
echo "     export DISPLAY=:99"
echo "     booster-studio --no-sandbox"
echo ""
echo "  3. 如果需要 VNC 远程访问:"
echo "     apt-get install -y tigervnc-standalone-server"
echo "     vncserver :99 -geometry 1920x1080"
echo "     export DISPLAY=:99"
echo "     booster-studio --no-sandbox"
echo ""
echo "  4. 检查 GPU 加速是否正常:"
echo "     vulkaninfo --summary"
echo "     glxinfo | grep 'OpenGL renderer'"
echo ""
