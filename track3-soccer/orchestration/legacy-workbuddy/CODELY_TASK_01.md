# Codely 任务包 #1 — 训练/demo 视频盘点 + 标记明显错误

> 派发方：Dream（中台）  |  执行方：Codely（远端 AMD GPU 实例）
> 最后更新：2026-08-04
> **本任务只盘点 + 标记，不要删除任何文件。** 删除是后续任务，需 Dream 收敛确认后下发。

---

## 0. 连接与坑（详见 CODEXLY_HANDOFF.md）
- 实例：`https://radeon-global.anruicloud.com/instances/<REDACTED>`，工作目录 `/workspace/amd-physical-ai-soccer/`
- ⚠️ kernel WebSocket **stdout 不回传** → 代码结果必须写文件，再用 GET 读回。
- ⚠️ 本机若有 `HTTP_PROXY` 会掐 TLS → 走浏览器/Ego 网络，或对该域名禁用代理。
- 写文件轮询模式：`subprocess`/kernel 内 `open('/workspace/_x.json','w').write(...)` → `GET /api/contents/_x.json?content=1&format=text`

---

## 1. 盘点全部 mp4
用下面的写文件版脚本扫 `/workspace` 全部 `.mp4`，写 `/workspace/_codely_inv.json`：
```python
import os, json
inv = []
for rt in ['/workspace']:
    for dp, dn, fn in os.walk(rt):
        if '/.git' in dp or 'node_modules' in dp: continue
        for f in fn:
            if f.lower().endswith('.mp4'):
                p = os.path.join(dp, f)
                try: sz = os.path.getsize(p)
                except: sz = -1
                inv.append({'path': p, 'size': sz})
inv.sort(key=lambda x: x['path'])
with open('/workspace/_codely_inv.json','w') as fo:
    json.dump({'mp4': inv}, fo, indent=2)
```

## 2. 取元数据（duration / fps / 分辨率）
对每个 mp4 用 `ffprobe` 取时长等；**若 ffprobe 缺失**：
```
subprocess.run([sys.executable,'-m','pip','install','-q','imageio-ffmpeg'])
# 或 opencv-python-headless，二选一
```
把 `duration`(秒)、`fps`、`w x h` 补进清单。

## 3. 标记「明显错误」
满足任一即标记 `flagged=true` 并写 `reason`：
- `size < 5 KB` → 空/损坏
- `duration < 1.0 s` → 几乎无内容
- 抽中间帧分析（用 ffmpeg 抽 1 帧 PNG 到 `/workspace/_frames/`，文件名同视频）：
  - 全黑 / 接近全黑（平均亮度过低）
  - 机器人已倒地（后续可由 Dream 看帧判断；本步先抽帧存盘，不强制判）
  - 画面异常（分辨率异常小 / 全白）

## 4. 产出报告（写文件，不删除）
写 `/workspace/_codely_result_01.json`：
```json
{
  "total": 12,
  "flagged": [
    {"path": "/workspace/.../x.mp4", "size": 0, "duration": 0.0, "reason": "size<5KB"}
  ],
  "kept": 11,
  "frame_dir": "/workspace/_frames"
}
```

## 5. 回传
结果写 `/workspace/_codely_result_01.json` 即完成本任务。**不要删除任何视频。**
Dream 会读回该文件收敛，确认 `flagged` 清单后下发删除任务 #2。

---

## 备注（给 Codely）
- 不要修改任何训练产物/代码，只做视频盘点与分析。
- 若 `_frames/` 抽帧耗时过长，可只对 `flagged` 候选抽帧，正常视频不必全抽。
- 完成后在实例留好 `_codely_inv.json` / `_codely_result_01.json` / `_frames/`，供 Dream 收敛与后续删除任务复用。
