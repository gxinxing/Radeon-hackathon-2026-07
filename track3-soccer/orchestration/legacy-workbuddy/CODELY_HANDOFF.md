# Codely 接管 AMD GPU 实例 — 操作交接手册

> 用途：让终端里的 Codely（或任何编码 Agent）接手远端 AMD GPU 实例的执行工作；
> Dream（本会话）退为监控 + 策略层，不再直接驱动 GPU。
> 最后更新：2026-08-04

---

## 1. 实例连接方式（必读）

- 实例基础 URL：`https://radeon-global.anruicloud.com/instances/<REDACTED>`
- JupyterLab 界面：`{base}/lab`
- REST API 根：`{base}/api/contents/`  ← 根目录映射到远端 `/workspace`
- 内核管理：`POST {base}/api/kernels`（建）、`DELETE {base}/api/kernels/{id}`（删）、`GET {base}/api/kernels/{id}`（查状态）
- 内核通道（执行代码）：WebSocket `{base}/api/kernels/{id}/channels`，发 `execute_request`

**工作目录（已重建，可直接用）：**
- `/workspace/amd-physical-ai-soccer/` ← 主工作区
  - `soccer_env_hierarchical.py`（低层行走 + rule-walk 分支）
  - `match_worker.py`（单 worker，已加 `--rule-walk` 开关）
  - `run_3v3_onnx.sh` / `match_coordinator.py` / `match_worker.py`（分布式 3v3 编排）
  - `src/match_3v3/`（`policy.py`=SharedRLPolicy 真实 ONNX 推理；`multiagent_obs.py`=24维 obs）
  - `models/chase_v8_policy.onnx`（高层追球 ONNX，19维 obs→vx,vy,vyaw）
- `/workspace/persistent/radeon-track3-private/` ← 源码备份（49 文件）
- `/workspace/booster_assets/robots/T1/meshes/` ← 机器人 STL（urdf 已 symlink）
- `/workspace/assets/ball.urdf` ← 球

---

## 2. ⚠️ 最关键的一个坑：kernel WebSocket 的 stdout 不回传

**现象**：`execute_request` 确实执行了（kernel 状态 busy→idle），但 **stream 消息收不到**，拿不到任何 `print` 输出。Dream 在本会话实测多次确认。

**已验证可用的 workaround**：让代码把结果写成文件，再用 GET 读回。
```
# 代码里这样写（不要依赖 print 回传）：
import json
with open('/workspace/_out.json','w') as f:
    json.dump(result, f, indent=2)

# 然后另一次请求读取：
GET {base}/api/contents/_out.json?content=1&format=text
→ 返回 {"content": "<文件文本>"}
```

**发代码的 WebSocket 报文格式**：
```json
{
  "header": {"msg_id":"1","username":"","session":"s","msg_type":"execute_request","version":"5.3"},
  "parent_header": {}, "metadata": {},
  "content": {"code": "<python 源码>", "silent": false, "store_history": true,
              "user_expressions": {}, "allow_stdin": false, "stop_on_error": true}
}
```
注意：发送前等 kernel 就绪（建完等 ~3s 再发），在 `onopen` 后用 `setTimeout` 延迟 ~600ms 发送，否则请求会被丢弃。

**本地代理坑**：本机有 `HTTP_PROXY=127.0.0.1:7890`，会掐断 Node→实例的 TLS（报 502 / TLS 断开）。
**必须绕过本地代理**——通过浏览器网络（Ego 浏览器）访问，或在 Node 里对实例域名禁用代理。

---

## 3. 当前项目状态与诊断（已修正早期误判，很重要）

赛道三：Booster T1 人形机器人 3v3 足球（Genesis 仿真，AMD GPU）。

**用户最新症状（2026-08-04 澄清）：**
- 单个机器人：能正常走路，但**不能踢球**。
- 两个机器人在场上：**站不稳**（发抖/摔倒）。

**据此修正的诊断（之前误判"低层 t1_walk 抖动"已推翻）：**
1. **单机器人能走** → 低层 `t1_walk.pt` 本身正常。之前做的 rule-walk 回退（Strategy A）大概率**不需要**。
2. **双机器人站不稳（最致命）** → 6 机器人放在**同一个 Genesis 场景**，全部共用 `t1_walk.pt`。
   - `RigidOptions(max_collision_pairs=256, tolerance=1e-5)`：6 人形碰撞对上限仅 256、容差极小。
   - 两 attacker 初始位只相距 **2m**（(-1,0) 与 (1,0)），抢球会贴脸撞。
   - 相撞 → 基座被推 → walk 模型 720 维观测突变 → **walk 模型没在接触扰动下训练过** → 输出失稳 → 双双倒地。
   - 根因 = 缺接触鲁棒性，不是行走模型坏。
3. **不能踢球** → `_execute_kick` 要求 `距球<0.3m` 且冷却结束才给球冲量；追球策略进不了 0.3m，或 1v1 课程环境里踢球是 ONNX 第 4 维动作未触发。

**文档状态澄清（用户那份 System Context 和朋友那份 docx 都已过时）：**
- 实例里 `SharedRLPolicy.compute()` **不是 stub**，ONNX 已端到端跑通（`match_1v1_onnx.py`、`run_3v3_onnx.sh`）。
- 多智能体 24 维 obs（`multiagent_obs.py`，19+5）已实现，只差"用 24 维训练的 policy + 训练 harness + GPU"。
- 真实阻塞就是上面三点，**不是**高层 ONNX 没接入。

---

## 4. 待决策（用户 / Dream 拍板，Codely 执行）

**双机器人失稳 — 4 选 1（或组合）：**
- **A. 接触参数调优（最快）**：上调 `max_collision_pairs`（如 4096）、放宽 `tolerance`、加接触阻尼。不动模型。
- **B. 高层避让（中等）**：高层 command 给机器人间加"保持距离/绕行"，避免贴脸撞。改 `match_3v3` 高层逻辑。
- **C. walk 接触鲁棒化（最稳最慢）**：接触域随机化重训/微调 `t1_walk.pt`。最贴根因，但 1.5 天内未必完。
- **D. 先拉开初始距离 + 降密度**：先验证"稳走+能踢"再逐步加压。

**踢球**：先定位是"追不到 0.3m"还是"踢球逻辑没接"，再决定放大 `KICK_DISTANCE` 还是修 ONNX 第 4 维。

---

## 5. 用户要求的待办

- 扫描实例上全部训练 / demo 视频（`.mp4`），识别**明显错误**的（0 字节 / 0 时长 / 全黑 / 机器人瞬间倒地），列出清单后删除。
- Dream 正用后台任务扫视频清单，结果会交给 Codely 执行删除（或 Codely 自行扫描）。

---

## 6. 若通过 Ego 浏览器驱动（Dream 的已验证模式，供参考）

```
const task = await useOrCreateTaskSpace('radeon-track3-p0')
await openOrReuseTab(LAB, { wait: true, timeout: 60 })
// 页面内用 fetch / WebSocket（绕过本地代理）
// 读文件：fetch(base+'/api/contents/<path>?content=1&format=text')
// 写文件：PUT base+'/api/contents/<path>'  body {type:'file',format:'base64',content:<b64>}
// 建内核：fetch(base+'/api/kernels',{method:'POST',body:'{}'})
// 执行：WS base+'/api/kernels/<id>/channels' 发 execute_request
// 删内核：fetch(base+'/api/kernels/<id>',{method:'DELETE'})
```

---

## 7. 快速自检脚本（写文件版，可直接跑）

把下面 python 写到实例 `/workspace/_diag.py` 并执行，结果在 `/workspace/_diag.json`：
```python
import os, json, subprocess
rep = {'mp4': [], 'bins': {}}
for rt in ['/workspace']:
    for dp, dn, fn in os.walk(rt):
        if '/.git' in dp or 'node_modules' in dp: continue
        for f in fn:
            if f.lower().endswith('.mp4'):
                p = os.path.join(dp, f)
                try: sz = os.path.getsize(p)
                except: sz = -1
                rep['mp4'].append({'path': p, 'size': sz})
rep['mp4'].sort(key=lambda x: x['path'])
for b in ['ffprobe','ffmpeg','python3']:
    try: rep['bins'][b] = subprocess.run(['which',b],capture_output=True,text=True).stdout.strip() or 'MISSING'
    except: rep['bins'][b] = 'MISSING'
with open('/workspace/_diag.json','w') as fo: json.dump(rep, fo, indent=2)
```
