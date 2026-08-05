# Dify Workflow Setup Guide — 国内市场量化 Agent

完整设置 6 节点 Chatflow：用户输入 → RAG 检索（/api/knowledge）→ LLM 生成 CN DSL →
代码节点规范化校验 → 回测（/api/cn/backtest/report）→ 风险报告回答。

## 前置条件

1. **vLLM** 在 AMD GPU 上服务微调后的 Qwen2.5-7B，地址 `http://localhost:8000/v1`
   - 模型名：`models/qwen-trader-merged`
2. **后端 API**（FastAPI）在 `http://localhost:8080`
   - 端点：`/api/knowledge`、`/api/cn/backtest/report`、`/health`
3. **Dify** 通过 Docker Compose 运行
   - Dify 容器经 `host.docker.internal` 访问宿主机服务

## Docker 网络

```
┌──────────────────────────────────────────────────┐
│  Host (AMD GPU Instance)                         │
│                                                  │
│  vLLM :8000  ←─── host.docker.internal:8000      │
│  API  :8080  ←─── host.docker.internal:8080      │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │ Docker Network                            │    │
│  │  ┌──────────┐  ┌─────────┐  ┌────────┐ │    │
│  │  │ Dify API │  │ Dify Web │  │ Dify   │ │    │
│  │  │  :5001   │  │  :3000   │  │ Worker │ │    │
│  │  └──────────┘  └─────────┘  └────────┘ │    │
│  └─────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

### 快速启动 Dify

```bash
cd /workspace/persistent
git clone https://github.com/langgenius/dify.git
cd dify/docker
cp .env.example .env

# Linux 下让容器访问宿主机
echo 'DOCKER_HOST_NETWORK=dify-network' >> .env
# 并在 docker-compose 中为 api/worker 添加 extra_hosts:
#   extra_hosts: ["host.docker.internal:host-gateway"]

docker compose up -d
```

Dify Web: `http://localhost:3000`

## 第一步：配置模型供应商

1. 进入 **设置 → 模型供应商**
2. 安装 "OpenAI-API-compatible" 插件
3. 配置：
   - **API URL**: `http://host.docker.internal:8000/v1`
   - **API Key**: `EMPTY`（vLLM 无需鉴权）
   - **Model Name**: `models/qwen-trader-merged`
4. 保存并验证连接

## 第二步：导入工具（可选）

1. 进入 **工具 → 自定义 → 创建自定义工具**
2. 导入 `dify/tools/trading_api_openapi.yml`（legacy 工具，国内市场链路不依赖）
3. 国内市场 Chatflow 直接通过 HTTP 请求节点调用：
   - `GET http://host.docker.internal:8080/api/knowledge?query=...`
   - `POST http://host.docker.internal:8080/api/cn/backtest/report`

## 第三步：应用 CN 工作流（SQL 补丁）

已提交的 CN Chatflow 以 SQL 补丁形式提供（Dify 无 workflow JSON 导出）。按序在 Dify
PostgreSQL 上执行：

```bash
cd dify/workflows
for f in add_intent_routing.sql enable_agent_capabilities.sql \
         enable_multi_agent_architecture.sql fix_multi_agent_parser.sql \
         update_cn_market_workflow.sql fix_cn_code_node.sql fix_cn_dify_runtime.sql; do
  psql -h <dify-db-host> -U <user> -d <dify-db> -f "$f"
done
```

### 6 节点结构

| 节点 | 类型 | 说明 |
|------|------|------|
| Start | start | 接收用户中文策略需求 |
| RAG 检索 | http-request | `GET /api/knowledge` 检索国内市场规则（T+1/100股/禁做空/涨跌停） |
| LLM | llm | `models/qwen-trader-merged` 生成 CN DSL（JSON），强制 constraints |
| 代码 | code | JSON 提取 + 规范化：exchange=cn_stock、lot_size=100、allow_short=false、price_limit=0.1、负 stop_loss |
| 回测 | http-request | `POST /api/cn/backtest/report`（T+1/100股/涨跌停 模拟回测） |
| Answer | answer | 返回风险报告 + PASS/REVIEW/REJECT |

### CN DSL 输出约束（LLM 节点提示词）

```
只输出 JSON，不要 Markdown；顶层只能包含 strategy。
market 必须包含 exchange="cn_stock"、instrument 和 timeframe（如 510300.SH / 159915.SZ）。
constraints 必须包含 t_plus_one=true、price_limit=0.1、allow_short=false、lot_size=100。
risk 必须包含负数 stop_loss、max_position_pct 和负数 max_drawdown。
禁止生成数字货币、境外交易所、合约或永续内容。
```

## 第四步：测试

在 Dify 对话中提问：

> "沪深300，EMA20/50 金叉买入，止损5%，仓位30%"

预期流程：
1. RAG 节点检索国内规则
2. LLM 生成 CN DSL（AMD ROCm vLLM 推理）
3. 代码节点规范化 + 注入 constraints
4. 回测节点返回中文报告（合成行情、PASS/REVIEW/REJECT）
5. Answer 输出最终风险报告

## 环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| vLLM URL | `http://host.docker.internal:8000/v1` | vLLM API（宿主机） |
| API URL | `http://host.docker.internal:8080` | 后端 API（宿主机） |
| Model | `models/qwen-trader-merged` | vLLM 注册模型 ID |

## 故障排查

- **Dify 无法访问 vLLM**：确认 `host.docker.internal` 可解析；Linux 下为 Dify api/worker 容器添加 `extra_hosts: ["host.docker.internal:host-gateway"]`。
- **模型名不匹配**：vLLM 注册模型为 `models/qwen-trader-merged`（带路径前缀），Dify 中必须使用该精确名称。
- **回测报错**：确认 API 运行中（`curl http://localhost:8080/health`），且 DSL 满足 CN 约束（exchange=cn_stock、.SH/.SZ 代码、lot_size=100、allow_short=false、负 stop_loss）。
- **RAG 无结果**：确认 `/api/knowledge?query=T+1` 返回 `success:true`。

> 说明：旧版 SETUP_GUIDE 描述的 Binance Testnet / Freqtrade / paper-trade 分支属于 legacy
> Crypto 链路，已从当前工作流移除；本项目所有回测使用确定性合成国内行情，仅作系统闭环演示。
