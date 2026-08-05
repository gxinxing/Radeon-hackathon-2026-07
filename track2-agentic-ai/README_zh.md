# 🤖 基于 AMD Radeon GPU 的国内市场量化 Agent

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%207.2-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![vLLM](https://img.shields.io/badge/vLLM-ROCm-blue)](https://docs.vllm.ai/)
[![Qwen2.5](https://img.shields.io/badge/Qwen2.5-7B-6E49C8?logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Tests](https://img.shields.io/badge/Tests-282%20passed-brightgreen)](#-测试)
[![License](https://img.shields.io/badge/License-Hackathon-lightgrey)](#license)

[English](./README.md) | [中文](./README_zh.md)

> 在 AMD ROCm GPU 上构建的全链路国内市场量化 Agent：中文自然语言 → RAG 检索国内规则 →
> LLM 生成策略 DSL → 规范化校验 → 国内市场模拟回测（T+1、100 股整数手、禁止裸卖空、10% 涨跌停）→
> 风控报告 PASS/REVIEW/REJECT — 全部由 AMD Radeon GPU 上的 Qwen2.5-7B 微调模型驱动。

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

本项目构建了一个**国内市场自主量化交易Agent**：用户用中文自然语言描述策略需求，
Agent 自主完成规则检索、DSL 生成、规范化校验、模拟回测与风控决策，并在国内现货约束
（T+1、100 股整数手、禁裸卖空、涨跌停、佣金/印花税）下输出带 PASS/REVIEW/REJECT
结论的风险报告。

**核心亮点：**
- **全链路 Agent 管道** — 中文需求 → RAG 检索 → LLM 生成 DSL → 规范化 → 国内回测 → 风控报告
- **多Agent管道** — 检索Agent → 推理Agent → 风控Agent（一票否决权）；LLM 只产生交易意向，风控Agent决定是否执行
- **三层记忆架构** — 工作记忆（RAM）+ 情景记忆（会话JSON）+ 语义记忆（跨会话JSON）实现持久化学习
- **RL奖励系统** — 8维奖励函数 [-1, +1]，三层反馈：即时prompt注入、经验规则提取、偏好对生成
- **多路RAG检索** — 关键词 + BM25 + 稠密向量 + CrossEncoder重排序，置信度门控（得分<0.45强制neutral，防止幻觉）
- **Dify集成** — 6节点 Chatflow：输入 → RAG → LLM → 代码 → 回测 → 回答
- **Graph Engine 编排** — OpenAI 兼容本地编排层（:8083），Open WebUI 直接接入，剥离上游 tool_call 幻觉
- **100% AMD GPU** — vLLM推理 + FP16 LoRA 微调全部在 ROCm 上运行，无 NVIDIA CUDA 依赖

---

## 🏗 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     Open WebUI / 任意 OpenAI 客户端               │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│               Graph Engine (:8083, OpenAI 兼容)                   │
│                                                                    │
│  意图路由 (intent_router)                                          │
│    ├─ quant_strategy : LLM 生成 YAML DSL → 宽松校验 → mock 回测   │
│    ├─ quant_compute  : 本地 numpy 引擎 (VaR/因子IC/期权/组合优化)  │
│    └─ general        : LLM 直通 (禁用工具, 纯文本回答)             │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                   AMD ROCm GPU (gfx1100, ROCm 7.2.1)              │
│                                                                    │
│  ┌────────────────────────┐    ┌────────────────────────────────┐ │
│  │ vLLM (:8000)           │    │ FastAPI (:8080)                │ │
│  │ Qwen2.5-7B (已合并)    │    │ • /api/knowledge — RAG         │ │
│  │ FP16, ~8.2s/请求       │    │ • /api/cn/backtest/report      │ │
│  │ ~16 GB 显存            │    │ • 国内回测 (T+1/100股/禁做空)   │ │
│  └────────────────────────┘    └────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Dify 6节点 Chatflow: 输入→RAG→LLM→代码→回测→回答             │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 管道流程

### 模式1: 全链路 Agent 管道（主路径）

```
① 中文需求        例如 "沪深300，EMA20/50 金叉买入，止损5%，仓位30%"
    │
    ▼
② RAG 检索        多路检索国内市场规则 (T+1/100股/禁做空/涨跌停…)
    │
    ▼
③ LLM 生成        Qwen2.5-7B (LoRA) 生成策略 DSL (YAML)
    │
    ▼
④ 规范化+校验     修复 lot_size/constraints/exchange/short/risk 字段
    │
    ▼
⑤ 国内回测        确定性合成行情 + T+1/100股/涨跌停/佣金/印花税
    │
    ▼
⑥ 风控报告        PASS / REVIEW / REJECT + 收益/回撤/胜率指标
```

### 模式2: 多Agent管道 (`MULTI_AGENT_MODE=true`)

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

### 模式3: Dify Chatflow (6节点)

```
Start → RAG检索 → LLM生成 → 代码(校验/规范化) → 回测请求 → Answer
```

---

## 🧠 Agent能力

| 能力 | 实现 | 代码位置 |
|------|------|----------|
| **推理** | ReAct循环 — LLM在每次Action前输出Thought | `src/agent/core.py` |
| **规划** | LLM动态决定工具调用顺序 | `src/agent/prompts.py` |
| **工具调用** | 8个注册工具, 结构化JSON分发 | `src/agent/tools.py` |
| **记忆管理** | 三层: 工作记忆(RAM) + 情景记忆(会话JSON) + 语义记忆(跨会话JSON) | `src/agent/memory.py` |
| **任务执行** | 国内模拟回测 + 风控报告 via FastAPI | `src/backtest/cn_runner.py` |
| **RL奖励** | 8维reward [-1,+1], 评级A+到F, 三层反馈 | `src/agent/reward.py` |
| **风控** | 硬规则校验 + 一票否决权 — LLM永远没有最终决定权 | `src/agent/risk_agent.py` |
| **编排** | OpenAI 兼容 graph_engine, 意图路由三类问题 | `graph_engine.py` |

---

## 🧩 核心组件

| 组件 | 技术 | 用途 |
|------|------|------|
| Agent架构 | ReAct循环 + 3-Agent管道 | 推理、规划、工具调用、记忆、执行 |
| 基座模型 | Qwen2.5-7B-Instruct | 中文能力强, ROCm友好 |
| 微调 | FP16 LoRA (r=64, alpha=128) | 400 条国内 NL→DSL 训练对, loss=0.2848 |
| 推理 | vLLM (ROCm, 合并LoRA) | 高吞吐本地LLM服务 |
| 编排 | Graph Engine (:8083) | OpenAI 兼容层, 剥离上游 tool_call |
| 三层记忆 | WorkingMemory + EpisodicMemory + SemanticMemory | RAM → 会话JSON → 跨会话JSON |
| RL奖励 | 8维加权函数 | 收益、Alpha、Sharpe、Sortino、Calmar、回撤、连亏、稳健性 |
| RAG | 多路: 关键词 + BM25 + 稠密向量 + CrossEncoder | 置信度门控检索 (阈值0.45) |
| 策略DSL | YAML + JSON Schema | LLM友好的中间表示; CN DSL 含 constraints |
| 回测引擎 | `src/backtest/cn_runner.py` | T+1、100股、禁做空、涨跌停、佣金/印花税 |
| Dify集成 | 6节点 Chatflow + OpenAPI spec | `/api/knowledge` + `/api/cn/backtest/report` |

---

## 🔧 环境要求

### 硬件

- AMD Radeon GPU (gfx1100 或兼容), ROCm 7.2.1+
- 训练峰值显存 ~16.21 GB, 推荐 24 GB 以上

### 云环境

- **安睿云** AMD GPU实例 (gfx1100, ROCm 7.2.1)
- JupyterLab终端访问
- Python虚拟环境位于 `/opt/venv/`

---

## 🚀 快速开始

### 1. 确认 vLLM 推理服务 (:8000)

```bash
curl http://127.0.0.1:8000/v1/models
```

### 2. 启动后端 API (:8080)

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

### 3. 启动 Graph Engine (:8083, OpenAI 兼容编排层)

```bash
cd track2-agentic-ai
TRACK2_ROOT=$PWD nohup /opt/venv/bin/python -m uvicorn graph_engine:app \
  --host 0.0.0.0 --port 8083 > /tmp/graph_engine.log 2>&1 &
```

把 Open WebUI 的模型端点指向 `http://localhost:8083/v1` 即可对话。

### 4. 测试国内回测 API

```bash
curl -X POST http://127.0.0.1:8080/api/cn/backtest/report \
  -H "Content-Type: application/json" \
  -d '{"strategy":{"name":"CN_EMA","market":{"exchange":"cn_stock","instrument":"510300.SH","timeframe":"1d"},"indicators":[{"name":"ema_fast","type":"EMA","params":{"period":20,"field":"close"}},{"name":"ema_slow","type":"EMA","params":{"period":50,"field":"close"}}],"entry":{"long":"ema_fast > ema_slow","short":null},"exit":{"long":"ema_fast < ema_slow","short":null},"constraints":{"t_plus_one":true,"price_limit":0.1,"allow_short":false,"lot_size":100},"risk":{"stop_loss":-0.05,"max_position_pct":0.3,"max_drawdown":-0.15}}}'
```

### 5. 运行 24 例国内评估

```bash
/opt/venv/bin/python scripts/eval_cn_market_v2.py \
  --vllm-url http://127.0.0.1:8000/v1 \
  --model models/qwen-trader-merged \
  --output /persistent/track2/eval/cn_market_eval_final.json
```

### 6. 端到端验证与单测

```bash
# 单元测试 (离线, 无 GPU 要求)
python3 -m pytest tests/ --ignore=tests/test_e2e.py -q   # 282 passed

# E2E 验证脚本 (当前 CN 链路)
bash scripts/verify_e2e.sh

# 一键演示
bash scripts/run_demo.sh
```

---

## 📁 项目结构

```
track2-agentic-ai/
├── graph_engine.py               # OpenAI 兼容编排层 (:8083)
├── src/
│   ├── api.py                    # FastAPI 统一服务 (:8080)
│   ├── cn_pipeline.py            # 国内全链路管道
│   ├── agent/                    # Agent系统 (检索/推理/风控/记忆/RL)
│   ├── knowledge_base/           # RAG 知识库 (国内规则)
│   ├── dsl/                      # 策略DSL (schema/validator/canonicalizer)
│   ├── backtest/
│   │   ├── cn_runner.py          # 国内市场回测 (T+1/100股/禁做空)
│   │   └── server.py             # 回测微服务
│   ├── tools/                    # 工具 (行情/指标/模拟交易/外部)
│   ├── llm/                      # LLM 客户端与提示词
│   └── chat_app.py               # 旧版 Gradio 聊天UI (legacy)
├── training/
│   ├── data/generate_cn_market_pairs.py   # 国内 NL→DSL 训练对 (400条)
│   └── scripts/                  # train_qlora.py / merge_lora.py / serve_vllm.sh
├── dify/
│   ├── tools/trading_api_openapi.yml
│   └── workflows/                # 6节点 Chatflow SQL 补丁 + SETUP_GUIDE
├── docker/                       # docker-compose / Dockerfile (API + vLLM)
├── landing/index.html            # 单文件落地页
├── scripts/
│   ├── setup.sh / run_demo.sh / verify_e2e.sh
│   ├── eval_cn_market_v2.py      # 24例国内评估
│   └── remote_cn_pipeline.sh
├── docs/                         # 技术报告 / DSL规范 / 演示脚本
├── tests/                        # 282+ 测试
└── requirements.txt
```

---

## 🎓 微调管道

### 训练数据 (国内市场 400 条 NL→DSL 对)

| 类别 | 说明 |
|------|------|
| 结构化输出 | 中文策略需求 → 带 constraints 的 CN DSL |
| 国内市场约束 | T+1、100股、禁做空、涨跌停 全部体现在 DSL 中 |
| 边界拒答 | 信息不足时输出 neutral / 拒绝生成 |

> 核心原则: LoRA 学习**如何思考与生成合规 DSL**, 具体规则由 RAG 维护。

### FP16 LoRA 训练 (ROCm)

```bash
# 数据准备
python training/data/generate_cn_market_pairs.py

# 训练 (FP16 LoRA, r=64, alpha=128)
python training/scripts/train_qlora.py

# 合并 LoRA 并部署 vLLM
python training/scripts/merge_lora.py
bash training/scripts/serve_vllm.sh
```

### LoRA配置

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen2.5-7B-Instruct |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| 训练样本 | 400 条国内 NL→DSL 对 |
| 轮次 | 3 (39 步) |
| 最终 loss | 0.2848 (train_loss 运行平均) |
| Token accuracy | 98.1% |
| 训练时长 | 615 秒 (~10 分钟) |
| 峰值显存 | 16.21 GB |

> 注：旧版本曾使用 bitsandbytes QLoRA / DPO 训练，ROCm 环境不可用，最终提交采用 FP16 LoRA。

---

## 📐 DSL规范 (国内市场)

```yaml
strategy:
  name: "CN_EMA_Crossover"
  market:
    exchange: cn_stock
    instrument: 510300.SH      # 或 159915.SZ
    timeframe: 1d
  indicators:
    - { name: ema_fast, type: EMA, params: { period: 20, field: close } }
    - { name: ema_slow, type: EMA, params: { period: 50, field: close } }
  entry:
    long: "ema_fast > ema_slow"
    short: null
  exit:
    long: "ema_fast < ema_slow"
    short: null
  constraints:
    t_plus_one: true
    lot_size: 100
    allow_short: false
    price_limit: 0.1
  risk:
    stop_loss: -0.05
    max_position_pct: 0.3
    max_drawdown: -0.15
```

---

## 🧪 测试

| 测试文件 | 数量 | 覆盖范围 |
|----------|------|----------|
| `test_agent.py` | 78 | 三层记忆, 工具, 解析器, 意图分类, 人格化 |
| `test_multi_agent.py` | 35 | 协议, 检索Agent, 推理, 风控否决, BM25, 分块 |
| `test_external_tools.py` | 27 | 外部工具路由 |
| `test_nl_to_dsl_quality.py` | 26 | NL→DSL提取, 校验, 转译质量 |
| `test_reward.py` | 24 | reward计算, RL反馈, 记忆整合 |
| `test_dsl_advanced.py` | 20 | 高级DSL (做空, 新指标, 嵌套表达式) |
| `test_expr_parser.py` | 14 | AST安全表达式求值 |
| `test_compute_engine.py` | 12 | 本地 numpy 计算引擎 |
| `test_memory_extract.py` | 11 | 记忆抽取 |
| `test_dsl_validator.py` | 10 | DSL schema校验 |
| `test_transpiler.py` | 10 | DSL → Freqtrade转译 (legacy) |
| `test_rag_retrieval.py` | 10 | RAG关键词/别名匹配, 语义安全 |
| `test_e2e.py` | 5 | 端到端管道 (2个需运行服务, 否则跳过) |
| `test_cn_market.py` | 2 | 国内市场回测确定性 |
| `memory_consistency/` | 3 | 记忆一致性框架 |
| **总计** | **282通过** | 离线 `--ignore=tests/test_e2e.py` |

```bash
# 运行全部测试
python -m pytest tests/ --ignore=tests/test_e2e.py -q   # 282 passed

# 仅运行Agent测试
python -m pytest tests/test_agent.py tests/test_multi_agent.py tests/test_reward.py -v
```

---

## 📊 评审对标

| 维度 | 分值 | 对标说明 |
|------|------|----------|
| 功能完整性与应用价值 | 60 | ReAct Agent (8工具); 3-Agent管道 (检索→推理→风控); 三层记忆; RL奖励系统; 多路RAG+置信度门控; Dify 6节点; 意图路由; 国内市场约束回测; 24/24 评估通过 |
| AMD Radeon GPU / ROCm优化 | 40 | vLLM推理 (gfx1100); FP16 LoRA 微调 (r=64, 39步); 全部本地推理 (无云API); rocminfo/SHA256/训练日志/vLLM日志 AMD 证据 |

---

## 📄 License

本项目为 AMD AI DevMaster 黑客松提交作品。
