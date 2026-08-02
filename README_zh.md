# AMD ROCm 本地量化智能体

**AMD AI DevMaster Hackathon 2026 — 赛道二：私有 AI Agent 本地部署**

这是一个运行在 AMD Radeon GPU / ROCm 上的本地投资与量化助理。它同时支持普通知识问答和可审计的量化工作流：用户问题经过意图路由后，可进入 RAG、多 Agent 推理、DSL 生成、校验、回测和风险报告链路。

## 当前能力

- ReAct 推理循环：思考、规划、工具调用、观察和最终回答。
- 多 Agent：检索 Agent → 推理 Agent → 独立风控 Agent；风控拥有否决权。
- 三层记忆：工作记忆、情景记忆、语义记忆。
- 多路 RAG：关键词、BM25、重排序和置信度门控。
- AMD 本地推理：Qwen2.5-7B、FP16 LoRA、ROCm vLLM。
- 结构化闭环：自然语言 → DSL → 规范化 → 校验 → 回测 → 风险报告。
- Dify 六节点编排：用户输入、RAG、LLM、代码校验、回测、风险报告。
- 普通问题走通用助理回复，不会被强制转换成策略 DSL。

## 已验证结果

| 项目 | 结果 |
|---|---:|
| GPU | gfx1100，ROCm 7.2.1 |
| LoRA | 400 条国内市场样本，39 步，615 秒 |
| 训练质量 | loss 0.2848，token accuracy 98.1% |
| 国内市场评估 | 修复后 24/24 |
| Dify 工作流 | 6 节点，3 个演示案例 |
| 测试 | 232 passed，另有 2 个已知异步测试问题 |

演示回测使用确定性合成历史数据，便于评委复现；不执行真实交易，也不构成投资建议。

## 快速开始

```bash
bash scripts/setup.sh
python -m uvicorn src.api:app --host 0.0.0.0 --port 8080
python src/chat_app.py
```

```bash
bash scripts/verify_e2e.sh
python -m pytest tests/ -v
```

详细文档：

- [技术报告](./docs/technical_report.md)
- [最终状态](./docs/track2_final_status.md)
- [DSL 规范](./docs/dsl_specification.md)
- [Dify 编排指南](./dify/workflows/SETUP_GUIDE.md)
