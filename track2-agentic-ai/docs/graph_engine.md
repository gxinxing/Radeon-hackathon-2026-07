# Graph Engine — OpenAI 兼容本地编排层

把 Open WebUI（或任意 OpenAI 客户端）的请求通过一个 agentic 图编排：

```
client ──> graph_engine (:8083) ──> intent_router
                                    ├─ quant_strategy : LLM 生成 YAML DSL → 宽松校验 → mock 回测
                                    ├─ quant_compute  : 本地 numpy 引擎（VaR/因子IC/期权/组合优化）
                                    └─ general        : LLM 直通（system prompt 禁用工具）
```

## 为什么需要它

Open WebUI 直连 vLLM 时，模型可能幻觉输出 tool_call（如
`generate_quant_strategy_report not found`）。graph engine 把所有响应收敛为纯文本，
并在此拦截/剥离 tool_call，根治 "Tool not found" 类报错。

## 快速开始

```bash
# 启动（:8083）
cd track2-agentic-ai
TRACK2_ROOT=$PWD nohup /opt/venv/bin/python -m uvicorn graph_engine:app \
  --host 0.0.0.0 --port 8083 > /tmp/graph_engine.log 2>&1 &

# 把 Open WebUI 指向它（数据库 config 优先于环境变量）：
#   sqlite3 /var/lib/open-webui/webui.db \
#     "UPDATE config SET value='["http://localhost:8083/v1"]' WHERE key='openai.api_base_urls';"
#   然后重启 open-webui
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_URL` | `http://localhost:8000/v1` | 上游 vLLM 地址 |
| `MODEL_ID` | `models/qwen-trader-merged` | 暴露给客户端的模型名 |
| `GRAPH_PORT` | `8083` | 监听端口 |
| `TRACK2_ROOT` | `/workspace/radeon-repo/track2-agentic-ai` | 项目根（导入本地 compute/validator） |

## 验证

```bash
curl :8083/v1/models
curl :8083/health
# 策略类 / 计算类 / 日常类 三类问题全部走对应节点
```
