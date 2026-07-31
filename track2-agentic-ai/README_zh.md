# 🤖 基于 AMD Radeon GPU 的量化交易 Agent

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%207.2-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![vLLM](https://img.shields.io/badge/vLLM-ROCm-blue)](https://docs.vllm.ai/)
[![Qwen2.5](https://img.shields.io/badge/Qwen2.5-7B-6E49C8?logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Gradio](https://img.shields.io/badge/Gradio-Chat%20UI-FF7300)](https://gradio.app/)
[![Tests](https://img.shields.io/badge/Tests-224%20passed-brightgreen)](#-测试)
[![License](https://img.shields.io/badge/License-Hackathon-lightgrey)](#license)

[English](./README.md) | [中文](./README_zh.md)

> 在 AMD ROCm GPU 上构建的全链路量化交易 Agent：ReAct 推理循环、多Agent管道（含风控否决权）、
> 三层记忆架构、RL奖励优化、多路RAG检索、Dify集成 — 全部由 AMD Radeon GPU 上的
> Qwen2.5-7B 微调模型驱动。

**AMD AI DevMaster 黑客松 2026 — 赛道二：Agentic AI**

---

## 📋 目录

- [概述](#-概述)
- [架构](#-架构)
- [管道流程](#-管道流程)
- [Agent能力](#-agent能力)
- [核心组件](#-核心组件)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [微调管道](#-微调管道)
- [DSL规范](#-dsl规范)
- [测试](#-测试)
- [评审对标](#-评审对标)
- [License](#license)

---

## 🌟 概述

本项目构建了一个**自主量化交易Agent**，具备五大AI Agent核心能力：推理、规划、工具调用、
记忆管理和任务执行。

用户用自然语言描述交易策略，Agent自主推理市场状况、生成策略DSL、回测历史数据、
通过独立的风控Agent进行风险评估（拥有一票否决权），并可选执行模拟交易 —
同时通过RL奖励系统从每次回测中学习优化。

**核心亮点：**
- **ReAct Agent循环** — LLM自主推理（Thought）、选择工具（Action）、观察结果、迭代直至目标完成
- **多Agent管道** — 检索Agent → 推理Agent → 风控Agent（一票否决权）；LLM只产生交易意向，风控Agent决定是否执行
- **三层记忆架构** — 工作记忆（RAM）+ 情景记忆（会话JSON）+ 语义记忆（跨会话JSON）实现持久化学习
- **RL奖励系统** — 8维奖励函数 [-1, +1]，三层反馈：即时prompt注入、经验规则提取、DPO训练对生成
- **多路RAG检索** — 关键词 + BM25 + 稠密向量 + CrossEncoder重排序，置信度门控（得分<0.45强制neutral，防止幻觉）
- **意图路由 + 人格化** — 交易查询走完整Agent管道；闲聊走"小R"人格化回复
- **Dify集成** — 单次HTTP调用（`/api/agent/run`）运行完整管道；3节点Chatflow替代12节点手工配置
- **100% AMD GPU** — vLLM推理 + LoRA/DPO微调全部在ROCm上运行

---

## 🏗 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      Gradio Chat UI (port 7860)                   │
│                                                                   │
│  "BTC放量突破前高，帮我做一个突破策略，止损3%"                     │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                   意图路由 (personality.py)                        │
│                                                                    │
│  交易意图? ──是──▶ ReAct Agent循环 (core.py)                      │
│      │              ├─ Thought → Action → Observe ↻               │
│      │              ├─ 8个工具, 三层记忆, RL奖励                    │
│      │              └─ 最终回答 + 指标 + 推理轨迹                   │
│      │                                                             │
│      └──否──▶ 人格化回复 (小R)                                    │
│                └─ 带记忆的自然对话                                   │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│              多Agent管道 (orchestrator.py)                        │
│              (通过 /api/agent/run 或 MULTI_AGENT_MODE=true)        │
│                                                                    │
│  ┌─ 检索Agent ────────────────────────────────────────────┐       │
│  │  多路RAG: 关键词 + BM25 + 稠密向量 + 重排序            │       │
│  │  置信度门控: 得分<0.45 → has_valid_docs=false          │       │
│  │  → 无有效文档时短路到neutral                            │       │
│  └────────────────────────┬───────────────────────────────┘       │
│                           │                                        │
│  ┌─ 推理Agent ─────────────▼──────────────────────────────┐      │
│  │  LoRA + RAG上下文 → 交易意向JSON                        │      │
│  │  {view, confidence, position_ratio, stop_loss, reason}   │      │
│  │  RAG不足时强制neutral                                   │      │
│  └────────────────────────┬───────────────────────────────┘       │
│                           │                                        │
│  ┌─ 风控Agent ─────────────▼──────────────────────────────┐      │
│  │  硬规则校验 (代码级，不受模型影响)                       │      │
│  │  ✓ 仓位上限 (总≤30%, 单品种≤10%)                       │      │
│  │  ✓ 止损距离 (0.5%–15%)                                  │      │
│  │  ✓ 置信度阈值 (≥0.30)                                   │      │
│  │  ✓ 理由完整性                                            │      │
│  │  ✗ 一票否决: allow_execute=false → 不执行               │      │
│  └────────────────────────────────────────────────────────┘      │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                AMD ROCm GPU (51 GB 显存)                          │
│                                                                    │
│  ┌────────────────────────┐    ┌────────────────────────────────┐ │
│  │ vLLM (ROCm V1引擎)     │    │ 微调                           │ │
│  │ Qwen2.5-7B (已合并)    │    │ • LoRA: 2,000条NL→DSL训练对    │ │
│  │ FP16, ~32 t/s          │    │ • DPO: reward排序偏好对        │ │
│  │ ~20 GB显存             │    │ • PEFT合并 → vLLM部署          │ │
│  └────────────────────────┘    └────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ 工具服务 (FastAPI + CCXT + TA-Lib)                          │ │
│  │ • 行情数据 • 回测 (多仓位, 滑点) • Walk-Forward分析         │ │
│  │ • 模拟交易 (Binance Testnet) • 知识库RAG • RL奖励计算       │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 管道流程

### 模式1: ReAct Agent循环 (默认, `AGENT_MODE=true`)

```
① 意图路由       交易? → ReAct循环 | 闲聊? → 人格化回复 (小R)
    │
    ▼ (交易)
② Thought        LLM推理下一步 (UI中实时可见)
③ Action         LLM选择8个工具之一调用
④ Observe        工具结果存入三层记忆
    │  ↻ 重复直到Final Answer或最大8轮
    ▼
⑤ RL奖励         回测指标 → 8维reward [-1,+1], 评级A+到F
    │             reward注入下一轮推理的prompt
    ▼
⑥ 最终回答        报告 + 指标表 + DSL + 推理轨迹
```

### 模式2: 多Agent管道 (`MULTI_AGENT_MODE=true` 或 `POST /api/agent/run`)

```
① 检索Agent      多路RAG → 置信度门控
    │             (has_valid_docs=false → 短路到neutral)
    ▼
② 推理Agent      LoRA + RAG → 交易意向JSON
    │             (不是下单指令 — 只是意向)
    ▼
③ 风控Agent      硬规则校验 → 允许/驳回 (一票否决权)
    │
    ▼
④ 最终决策        只有风控Agent通过才可执行
```

### 模式3: Dify Chatflow (3节点)

```
Start → Tool(runMultiAgent) → End
```

单次HTTP调用运行完整多Agent管道。

---

## 🧠 Agent能力

| 能力 | 实现 | 代码位置 |
|------|------|----------|
| **推理** | ReAct循环 — LLM在每次Action前输出Thought | `src/agent/core.py` |
| **规划** | LLM动态决定工具调用顺序 (行情→知识→生成→校验→回测→Walk-Forward→回答) | `src/agent/prompts.py` |
| **工具调用** | 8个注册工具, 结构化JSON分发 | `src/agent/tools.py` |
| **记忆管理** | 三层: 工作记忆(RAM) + 情景记忆(会话JSON) + 语义记忆(跨会话JSON) | `src/agent/memory.py` |
| **任务执行** | 真实回测、Walk-Forward分析、模拟交易 via FastAPI | `src/backtest/runner.py` |
| **RL奖励** | 8维reward [-1,+1], 评级A+到F, 三层反馈 (prompt/规则/DPO) | `src/agent/reward.py` |
| **风控** | 硬规则校验 + 一票否决权 — LLM永远没有最终决定权 | `src/agent/risk_agent.py` |
| **人格化** | 意图路由 (交易 vs 闲聊) + 小R人设 + 记忆 | `src/agent/personality.py` |

---

## 🧩 核心组件

| 组件 | 技术 | 用途 |
|------|------|------|
| Agent架构 | ReAct循环 + 3-Agent管道 | 推理、规划、工具调用、记忆、执行 |
| 基座模型 | Qwen2.5-7B-Instruct | 中文能力强, ROCm友好 |
| 微调 | LoRA (r=64) + DPO via PEFT/TRL | 交易知识 + 基于reward的偏好优化 |
| 推理 | vLLM (ROCm, V1引擎) | 高吞吐本地LLM服务 |
| Agent框架 | Gradio + httpx + FastAPI | 对话UI, LLM编排, 工具调用, HTTP API |
| 三层记忆 | WorkingMemory + EpisodicMemory + SemanticMemory | RAM → 会话JSON → 跨会话JSON |
| RL奖励 | 8维加权函数 | 收益、Alpha、Sharpe、Sortino、Calmar、回撤、连亏、稳健性 |
| RAG | 多路: 关键词 + BM25 + 稠密向量 + CrossEncoder | 置信度门控检索 (阈值0.45) |
| 策略DSL | YAML + JSON Schema | LLM友好的中间表示 |
| 回测引擎 | 自研 + CCXT + TA-Lib | 多仓位、滑点、Walk-Forward、15+指标 |
| 模拟交易 | Binance Testnet API | 带安全限额的免风险模拟 |
| Dify集成 | FastAPI + OpenAPI spec | `/api/agent/run` + `/api/agent/reward` 端点 |

---

## 🔧 环境要求

### 硬件

- AMD Radeon GPU (如 RX 7900 XTX, MI210), ROCm 6.2+
- 推荐 24 GB 以上显存

### 云环境

- **安睿云** AMD GPU实例
- JupyterLab终端访问
- VNC通过noVNC端口6080
- Python虚拟环境位于 `/opt/venv/`
- ROCm 6.2 + PyTorch 2.9.1

---

## 🚀 快速开始

### 一键安装

```bash
cd track2-agentic-ai
bash scripts/setup.sh
```

### 运行模式

```bash
# 默认: ReAct Agent循环
export AGENT_MODE=true
python src/chat_app.py

# 多Agent管道
export MULTI_AGENT_MODE=true
python src/chat_app.py

# 传统线性管道 (fallback)
export AGENT_MODE=false
python src/chat_app.py
```

### Dify集成

```bash
# 启动API服务
uvicorn src.api:app --host 0.0.0.0 --port 8080

# 调用多Agent管道
curl -X POST http://localhost:8080/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "BTC EMA crossover strategy, stop loss 3%"}'

# 计算RL奖励
curl -X POST http://localhost:8080/api/agent/reward \
  -H "Content-Type: application/json" \
  -d '{"metrics": {"total_return": 0.15, "sharpe_ratio": 1.8, ...}}'
```

### 端到端验证

```bash
bash scripts/verify_e2e.sh
```

---

## 📁 项目结构

```
track2-agentic-ai/
├── src/
│   ├── agent/                        # Agent系统 (13个模块)
│   │   ├── core.py                  # ReAct循环 + 意图路由
│   │   ├── memory.py                # 三层记忆 (工作/情景/语义)
│   │   ├── personality.py           # 意图分类 + 小R人设
│   │   ├── prompts.py               # ReAct系统提示 + DSL生成
│   │   ├── reward.py                # 8维RL奖励函数
│   │   ├── rl_feedback.py           # 三层RL反馈 (prompt/规则/DPO)
│   │   ├── tools.py                 # 8工具注册 + JSON解析
│   │   ├── protocol.py              # AgentMessage通信协议
│   │   ├── retrieval_agent.py       # 多路RAG检索Agent
│   │   ├── reasoning_agent.py       # LoRA推理Agent
│   │   ├── risk_agent.py            # 风控Agent (一票否决权)
│   │   └── orchestrator.py          # 多Agent管道编排器
│   ├── knowledge_base/
│   │   ├── multi_retriever.py       # 关键词+BM25+重排序+置信度门控
│   │   ├── chunker.py               # 量化文档分块 (512t, 表格感知)
│   │   ├── retriever.py             # 关键词检索器 (基础)
│   │   ├── knowledge_entries.py     # 31条知识条目
│   │   └── semantic.py              # 可选语义重排器
│   ├── dsl/                         # 策略DSL (schema, 校验, 转译)
│   ├── backtest/                    # 回测引擎
│   ├── tools/                       # 行情, 指标, 模拟交易
│   ├── llm/                         # LLM提示词
│   ├── api.py                       # 统一FastAPI服务 (含 /api/agent/run)
│   └── chat_app.py                  # Gradio对话UI (3种模式)
├── training/
│   ├── data/
│   │   ├── prepare_quant_lora_dataset.py  # 4类LoRA数据集 (2000条)
│   │   └── prepare_dsl_pairs.py            # 原始NL→DSL对
│   ├── scripts/
│   │   ├── train_qlora.py           # LoRA训练 (ROCm)
│   │   ├── train_dpo.py             # DPO训练 (TRL, ROCm)
│   │   └── prepare_dpo_data.py      # 从reward生成DPO偏好对
│   └── configs/
├── dify/
│   ├── tools/trading_api_openapi.yml  # 7个OpenAPI操作
│   └── workflows/SETUP_GUIDE.md      # 3节点Dify Chatflow指南
├── docs/
│   ├── technical_report.md          # 完整技术报告 (16个创新点)
│   ├── lora_training_spec.md        # LoRA训练规范 (4类样本, 防幻觉)
│   └── dsl_specification.md         # DSL正式规范
├── tests/
│   ├── test_agent.py                # 78个测试
│   ├── test_multi_agent.py          # 35个测试
│   ├── test_reward.py               # 24个测试
│   └── ...                          # 87个已有测试
├── scripts/
│   ├── setup.sh
│   └── verify_e2e.sh
├── requirements.txt
└── README.md
```

---

## 🎓 微调管道

### LoRA训练数据 (4类, 防幻觉)

| 类别 | 占比 | 目的 | 示例 |
|------|------|------|------|
| A: 结构化输出 | 60% | 强制JSON格式, 消除自由文本 | 指标条件 → 交易意向JSON |
| B: 推理链 | 25% | 交易决策的思维链 | 多步指标分析 → 意向 |
| C: 工具调用 | 10% | 正确的工具选择格式 | "查行情" → `{"tool": "get_market_data"}` |
| D: 边界拒答 | 5% | 防幻觉 — 信息不足时拒答 | 缺少参数 → `{"view": "neutral", "confidence": 0}` |

> 核心原则: LoRA学习**如何思考**, 不学习**具体知识** (事实由RAG维护)。

### DPO训练 (基于reward)

从RL反馈循环的reward排序策略对生成DPO训练数据：

```bash
# 从累积的reward生成DPO对
python training/scripts/prepare_dpo_data.py

# 在ROCm上DPO训练
python training/scripts/train_dpo.py --model /workspace/persistent/qwen-trader-merged
```

### LoRA配置

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen2.5-7B-Instruct |
| LoRA rank | 64 (SFT) / 16 (DPO) |
| LoRA alpha | 128 (SFT) / 32 (DPO) |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 学习率 | 2e-4 (SFT) / 5e-6 (DPO) |
| 精度 | bf16 |
| 轮次 | 3 (SFT) / 1 (DPO) |

---

## 📐 DSL规范

```yaml
strategy:
  name: "BTC_EMA_Breakout_Volume"
  market:
    exchange: binance
    pair: BTC/USDT
    timeframe: 1h
  indicators:
    - { name: ema_fast, type: EMA, params: { period: 20, field: close } }
    - { name: ema_slow, type: EMA, params: { period: 50, field: close } }
    - { name: rsi, type: RSI, params: { period: 14 } }
  entry:
    long: "ema_fast > ema_slow AND rsi < 70"
    short: null
  exit:
    long: "ema_fast < ema_slow"
    short: null
  risk:
    stop_loss: -0.03
    max_open_trades: 3
    stake_amount: 0.1
```

---

## 🧪 测试

| 测试文件 | 数量 | 覆盖范围 |
|----------|------|----------|
| `test_agent.py` | 78 | 三层记忆, 工具, 解析器, 意图分类, 人格化 |
| `test_multi_agent.py` | 35 | 协议, 检索Agent, 推理, 风控否决, BM25, 分块 |
| `test_reward.py` | 24 | reward计算, RL反馈, DPO对, 记忆整合 |
| `test_dsl_validator.py` | 10 | DSL schema校验 |
| `test_transpiler.py` | 10 | DSL → Freqtrade转译 |
| `test_expr_parser.py` | 12 | AST安全表达式求值 |
| `test_dsl_advanced.py` | 17 | 高级DSL (做空, 新指标, 嵌套表达式) |
| `test_nl_to_dsl_quality.py` | 26 | NL→DSL提取, 校验, 转译质量 |
| `test_rag_retrieval.py` | 10 | RAG关键词/别名匹配, 语义安全 |
| `test_cn_market.py` | 2 | 国内市场回测确定性 |
| `test_e2e.py` | 3+2 | 端到端管道 (2个async测试需pytest-asyncio) |
| **总计** | **224通过** | |

```bash
# 运行全部测试
python -m pytest tests/ -v

# 仅运行Agent测试
python -m pytest tests/test_agent.py tests/test_multi_agent.py tests/test_reward.py -v
```

---

## 📊 评审对标

| 维度 | 分值 | 对标说明 |
|------|------|----------|
| 功能完整性与应用价值 | 60 | ReAct Agent (8工具); 3-Agent管道 (检索→推理→风控); 三层记忆; RL奖励系统; 多路RAG+置信度门控; Dify集成; 意图路由+人格化; DPO训练; LoRA训练规范 (4类样本) |
| AMD Radeon GPU / ROCm优化 | 40 | vLLM推理; LoRA微调; DPO训练全部在ROCm上; 6.2×批处理加速 (32→202 tokens/s); RL奖励从AMD GPU上的回测计算; 全部本地推理 (无云API) |

---

## 📄 License

本项目为 AMD AI DevMaster 黑客松提交作品。
