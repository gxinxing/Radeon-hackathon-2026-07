# Track 3 Review Prompt — 人形机器人足球策略训练

> 复制以下 prompt 到 Codex 中执行。在项目目录 `track3-soccer/` 下运行。

---

## Review Prompt

你是一个资深 RL + 机器人代码审查专家。请对以下人形机器人足球策略训练项目进行全面 review。

### 项目背景

- 赛事：AMD AI DevMaster Hackathon 2026 — Track 3: Physical AI
- 目标：在 AMD Radeon GPU 上用 Genesis 物理引擎 + ROCm PyTorch 训练人形机器人足球策略
- 评分维度（100分）：
  - 机器人能力表现（30分）：balance/chase/shoot 策略训练效果
  - AMD ROCm 采用（20分）：Genesis + PyTorch 在 ROCm 上的使用
  - 创新性（20分）：首个 AMD GPU 人形足球训练管线（Isaac Gym 替代）
  - 实际应用价值（20分）：Sim2Sim 验证、真实机器人部署路径
  - 上游开源贡献（10分）：可复用的 Genesis 足球环境

### Review 要求

请按以下 5 个维度逐项审查，每个维度给出评分（满分见上）和具体问题：

**1. 机器人能力表现（30分）**
- 检查 `reward.py`：奖励函数设计是否合理？是否有奖励作弊（reward hacking）风险？
- 检查 `soccer_env_hierarchical.py`：层级策略（高层 PPO + 底层冻结行走）设计是否正确？
- 检查 `train_hierarchical.py`：训练超参数是否合理？PPO 配置有无明显问题？
- 检查 `train_curriculum.py`：课程学习阶段设计是否合理？
- 19维观测空间是否充分？有无关键信息缺失？

**2. AMD ROCm 采用（20分）**
- 代码中是否真正依赖 ROCm？有无隐藏的 CUDA 依赖？
- Genesis 物理引擎是否正确使用 AMD GPU？
- 训练日志/benchmark 是否有 AMD GPU 证据？
- ONNX 导出和推理是否在 AMD GPU 上验证？

**3. 创新性（20分）**
- 相比 Isaac Gym 方案，这个项目的架构创新点在哪里？
- 层级策略设计（19→3 dim 高层 + 720→21 dim 底层）是否有技术深度？
- 分布式多机器人比赛架构（socket 同步）是否是合理的工程方案？

**4. 实际应用价值（20分）**
- Sim2Sim 验证路径是否完整？从训练到 ONNX 到比赛的链路是否可复现？
- 1v1 比赛结果（200步，球位移20米）是否可信？
- 3v3 比赛的"独立物理场景"限制是否影响可信度？
- 已知限制（close-range ball control、Genesis ROCm multi-entity solver）是否诚实披露？

**5. 代码质量与可复现性（10分）**
- 代码是否有明显 bug？检查 `notify_order` 类的逻辑、观测历史对齐、T+1 类约束
- 测试覆盖是否充分？`tests/` 目录的测试质量如何？
- README 的复现步骤是否真的能一步步跑通？
- Docker 配置是否可用？

### 输出格式

对每个维度，输出：
```
## 维度X: [名称] (XX/YY分)
### 评分理由
- ...
### 发现的问题
- 🔴 严重: ...
- 🟡 中等: ...
- 🟢 建议: ...
### 改进建议
- ...
```

最后给出总分和一句话总结。

### 重要约束

- 只读不写，不要修改任何代码
- 如果发现某个文件不存在或无法读取，标注后跳过
- 评分要严格但公平，这是黑客松项目不是生产系统
- 特别关注：是否有声称做了但实际没做的功能（README vs 代码不一致）
