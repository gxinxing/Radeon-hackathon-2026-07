# Landing 页失效链接修复说明（交接给 Euclid）

> 本文件是 Track2 验收修复的交接说明。`landing/index.html` 由 Euclid 负责修改，
> 本仓库其余 agent 不直接改动该文件。

## 问题（验收报告 P0-2）

`landing/index.html` 第 585、621 行两处按钮指向失效链接：

```
https://minimize-orders-excel-saving.trycloudflare.com
```

- 实测：`curl` 返回 DNS 解析失败（exit 6），隧道已于 2026-08-02 终止且未重启。
- 影响：页面内 "Open Live Demo" / "Try the Live Demo" 按钮点不开，直接影响
  验收标准 A（"打开网站是正常的"）的线上演示部分。

## 建议修复

1. **首选**：将两处 `href` 改为可用的演示入口：
   - 若远端 Open WebUI / Graph Engine 重新暴露（新 cloudflared 隧道或公网地址），
     替换为新的可用 URL；
   - 否则改为指向仓库演示资料，例如
     `https://github.com/gxinxing/Radeon-hackathon-2026-07/blob/main/docs/track2_demo_script_cn.md`
     或 landing 内已有的技术报告链接，避免死链。
2. **次选**：若演示服务按文档可本地复现（vLLM :8000 → graph_engine :8083 → Open WebUI），
   在按钮下补充本地运行说明文字，并把按钮文案从 "Live Demo" 调整为
   "View Demo Script" / "Run Locally" 之类不承诺在线可点的措辞。
3. 修改后请用无头 Chrome 或 `curl -I` 复验两处链接返回 200。

## 相关文档

- `docs/graph_engine.md` — 本地 OpenAI 兼容编排层启动方式（:8083）
- `README.md` / `README_zh.md` — 快速开始（vLLM :8000 / API :8080 / graph_engine :8083）
