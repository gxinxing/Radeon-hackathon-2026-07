# 在浏览器里跑 Track3 P0（你或 Ego Lite 都可照做）

> 目的：用 Ego Lite / 你自己登录的浏览器驱动 Radeon 实例，跑通 Graph Engine 的 P0 节点，
> 实时看到"环境核查 + 行走模型对齐诊断"的过程。所有输出是**文本**（Notebook cell print），
> 不依赖截图，所以看得到、读得懂。

## 为什么走 JupyterLab 而不是 SSH
- 外部 SSH（端口 31151）被 anruicloud 网关前置，只认控制台绑定的公钥 → 我们的 key 被拒（已诊断）。
- JupyterLab 是实例自己的 shell（host `<REDACTED>`，user `root`），绕过网关。
- JupyterLab Terminal 是 canvas 渲染、截图读不到；**所以改用 Notebook cell**，文本输出在 DOM 里可读。

## 操作步骤（照抄即可）

1. 打开浏览器，登录并进入 JupyterLab：
   `https://radeon-global.anruicloud.com/instances/<REDACTED>/lab`

2. 在 JupyterLab 里：顶部菜单 `File → New → Notebook`（选 Python 3 内核）。
   在第一个 cell 里贴下面这段，**Shift+Enter 运行**：

```python
# === P0: 环境核查 + 行走模型对齐诊断 ===
import os, sys, subprocess, hashlib, datetime

REPO = "/workspace/radeon-repo"
MODELS = os.path.join(REPO, "models")
WALK = os.path.join(MODELS, "pretrained", "t1_walk.pt")

def sh(c, t=30):
    try:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, cwd=REPO)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"<err {e}>"

def sha(p):
    if not os.path.exists(p): return "<missing>"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8192), b""): h.update(b)
    return h.hexdigest()[:16]

print("="*60)
print("P0 CHECK_ENV + ALIGN_WALK", datetime.datetime.now())
print("="*60)

print("\n[1] GPU"); print(sh("rocm-smi --showid 2>/dev/null | head -20 || echo NO_ROCM").[:1500])
print("\n[2] commit"); print(sh("git -C %s rev-parse HEAD 2>/dev/null || echo NOGIT" % REPO))
print("\n[3] 模型清单")
for f in (sorted(os.listdir(MODELS)) if os.path.isdir(MODELS) else []):
    p = os.path.join(MODELS, f)
    if os.path.isfile(p): print(f"  {f:32s} {sha(p)}  {os.path.getsize(p)//1024}KB")
pre = os.path.join(MODELS, "pretrained")
if os.path.isdir(pre):
    for f in sorted(os.listdir(pre)):
        p = os.path.join(pre, f)
        if os.path.isfile(p): print(f"  pretrained/{f:24s} {sha(p)}  {os.path.getsize(p)//1024}KB")
print("\n[4] 行走对齐诊断：t1_walk.pt 存在=%s 大小=%dKB" % (
    os.path.exists(WALK), os.path.getsize(WALK)//1024 if os.path.exists(WALK) else 0))
print("  ⚠ 必须确认 t1_walk.pt 对应你的 K1/T1 构型(21电机映射)，否则=关节错配=抖动根因")
print("  ✅ 验证法：Genesis 跑 10s 单机器人闭环，记录 base 高度 std")
print("\n[5] 下一步：A)本体不匹配→切 RulePolicy 兜底行走先跑通；B)匹配→查 obs 尺度/滤波一致性")
print("="*60)
```

3. 把上面的输出**贴回给我**（或截图反正读不到，就复制文本），我据此判断 P0 是否通过、决定进 P1 还是先修行走。

## 如果 P0 显示 t1_walk 本体不匹配（大概率）
下一步动作（在 Notebook 下一个 cell）：
- 确认 deploy 端是否能在 **不依赖 t1_walk ONNX 高层** 的情况下，用 `policy.py` 的 `RulePolicy`（几何规则直接给速度指令）+ 规则行走兜底先把"6 机器人能站稳走完一场"跑通。
- 这是 1 天内达成"踢完一场比赛"最低可演示目标的最快路径（作战图 P0 策略 A）。

## 文件位置
- 诊断与作战图：`/Users/simon/WorkBuddy/Claw/track3_graph_engine/TRACK3_RECOVERY_PLAN.md`
- Graph Engine 骨架：`/Users/simon/WorkBuddy/Claw/track3_graph_engine/graph_engine.py`
- P0 脚本（可上传到 Radeon）：`/Users/simon/WorkBuddy/Claw/track3_graph_engine/nodes/p0_check_align.py`
