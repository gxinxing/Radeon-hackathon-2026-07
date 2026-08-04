#!/usr/bin/env bash
# W1 诊断脚本 — 不装任何东西，只摸清环境现状
# 在 Radeon Cloud JupyterLab Terminal 或 SSH 里：bash w1_diag.sh

set +e
echo "================================================"
echo "  W1 ENVIRONMENT DIAGNOSTIC  $(date -u +%FT%TZ)"
echo "================================================"

echo ""
echo "### 1. WHO / WHERE ###"
whoami
hostname
pwd
echo "uname: $(uname -a)"

echo ""
echo "### 2. ROCM / GPU ###"
which rocm-smi && rocm-smi --showproductname --showmeminfo vram 2>&1 | head -30

echo ""
echo "### 3. PYTHON VENVS ###"
echo "--- /opt/venv ---"
ls -la /opt/venv/bin/python* 2>&1 | head -5
/opt/venv/bin/python --version 2>&1
echo "--- system python ---"
/usr/bin/python3 --version 2>&1
which conda mamba 2>&1

echo ""
echo "### 4. TORCH (in /opt/venv) ###"
/opt/venv/bin/python - <<'PY' 2>&1 | head -20
import torch
print("torch:", torch.__version__)
print("cuda.is_available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("HIP version:", torch.version.hip)
PY

echo ""
echo "### 5. GENESIS ###"
/opt/venv/bin/python -c "import genesis as gs; print('genesis version:', gs.__version__); print('genesis path:', gs.__file__)" 2>&1 | head -5
if [ $? -ne 0 ]; then
    echo ">>> genesis NOT installed yet (expected — need to install)"
fi

echo ""
echo "### 6. GENESIS ASSETS / URDF ###"
GENESIS_DIR=$(/opt/venv/bin/python -c "import genesis, os; print(os.path.dirname(genesis.__file__))" 2>/dev/null)
if [ -n "$GENESIS_DIR" ]; then
    echo "genesis package dir: $GENESIS_DIR"
    echo "--- urdf files (depth 4) ---"
    find "$GENESIS_DIR" -maxdepth 4 -iname "*.urdf" 2>/dev/null | head -20
    echo "--- assets dir if exists ---"
    ls -la "$GENESIS_DIR/assets" 2>&1 | head -10
    if [ -d "$GENESIS_DIR/assets" ]; then
        find "$GENESIS_DIR/assets" -maxdepth 3 -iname "*.urdf" 2>/dev/null | head -20
    fi
else
    echo "(genesis not importable, skip URDF probe)"
fi

echo ""
echo "### 7. /workspace CONTENTS ###"
ls -la /workspace/ 2>&1 | head -30

echo ""
echo "### 8. PERSISTENT VOLUME CHECK ###"
df -h /workspace 2>&1 | head -5
mount | grep -i workspace 2>&1 | head -5

echo ""
echo "================================================"
echo "  DIAGNOSTIC COMPLETE"
echo "================================================"
