#!/usr/bin/env python3
"""ARCHIVED WorkBuddy probe; retained for reference and not executable.

Its hard-coded `/workspace/radeon-repo` path is obsolete. Use the current
controller and `ops/radeon/` runbook instead.
"""
from __future__ import annotations

"""P0 节点：Radeon 实例环境核查 + 行走模型对齐验证。

设计为在 JupyterLab Notebook cell 里直接 `%run` 或 `!python` 执行，
所有关键结果用 print() 输出到 DOM 文本（当前模型读不到截图，故不依赖图像）。

跑法（在 Radeon 实例，WorkBuddy/Ego Lite 浏览器驱动 JupyterLab）：
    !cd /workspace/radeon-repo && python /workspace/persistent/track3_graph/nodes/p0_check_align.py
"""
import os, sys, subprocess, hashlib, json, datetime

REPO = "/workspace/radeon-repo"
MODELS = os.path.join(REPO, "models")
WALK = os.path.join(MODELS, "pretrained", "t1_walk.pt")

def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout,
                           cwd=REPO)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"<err: {e}>"

def sha256(path):
    if not os.path.exists(path):
        return "<missing>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()[:16]

def main():
    raise SystemExit(
        "ARCHIVED: p0_check_align.py uses an obsolete remote path and must not run"
    )
    print("=" * 60)
    print("P0 CHECK_ENV + ALIGN_WALK  —", datetime.datetime.now())
    print("=" * 60)

    # 1) GPU 状态
    print("\n[1] GPU (rocm-smi)")
    gpu = sh("rocm-smi --showid 2>/dev/null | head -20 || echo 'rocm-smi 不可用'")
    print(gpu[:1500])

    # 2) 代码同步状态
    print("\n[2] 代码 commit")
    print(sh("git -C %s rev-parse HEAD 2>/dev/null || echo '非 git'" % REPO))
    print("本地改动:", sh("git -C %s status --short 2>/dev/null | head" % REPO) or "（无）")

    # 3) 关键模型清单
    print("\n[3] 模型清单 (models/ + pretrained/)")
    for f in sorted(os.listdir(MODELS)) if os.path.isdir(MODELS) else []:
        p = os.path.join(MODELS, f)
        if os.path.isfile(p):
            print(f"  {f:32s} {sha256(p)}  {os.path.getsize(p)//1024}KB")
    pre = os.path.join(MODELS, "pretrained")
    if os.path.isdir(pre):
        for f in sorted(os.listdir(pre)):
            p = os.path.join(pre, f)
            if os.path.isfile(p):
                print(f"  pretrained/{f:24s} {sha256(p)}  {os.path.getsize(p)//1024}KB")

    # 4) 行走模型对齐判定（关键）
    print("\n[4] 行走模型对齐诊断")
    print(f"  t1_walk.pt: {WALK}")
    print(f"  存在: {os.path.exists(WALK)}  大小: {os.path.getsize(WALK)//1024 if os.path.exists(WALK) else 0}KB")
    print("  ⚠️ 判定：t1_walk.pt 是通用预训练行走模型，必须确认它对应你的 K1/T1 机器人构型")
    print("     21 个电机映射若与本体不一致 → 关节目标错配 → 一迈步就抖（当前视频抖动根因）。")
    print("     验证方法：在 Genesis 里跑 10s 单机器人闭环，记录 base 高度 std。")

    # 5) 单机器人稳定度快速探针（若 repo 有 render/random rollout 脚本）
    print("\n[5] 建议下一步（align_walk 节点动作）")
    print("  A. 若 t1_walk 本体不匹配 → 暂切 RulePolicy 规则行走兜底，先把'能站能走'跑通")
    print("  B. 若本体匹配 → 检查 obs 尺度/滤波是否与训练一致，再调 deadzone/clip")

    print("\n" + "=" * 60)
    print("P0 文本报告结束（无需截图，以上为可读结果）")
    print("=" * 60)

if __name__ == "__main__":
    main()
