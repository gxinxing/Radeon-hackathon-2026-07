# Open WebUI Border Beam 注入方案

> 目标：给赛道二聊天窗口（Open Web UI）的**消息卡片 / 输入框**加上
> [beam.jakubantalik.com](https://beam.jakubantalik.com/) 那种「发光光束沿边框旋转扫过」的动效。
> 零依赖（纯 CSS + 原生 JS，无 WebGL / 无外部库），尊重明暗主题，
> 支持 `prefers-reduced-motion` 降级，滚出视口的卡片自动暂停动画（不影响滚动性能）。

---

## 1. 效果与原理

**效果**：聊天页每条消息的气泡边框上，有一束「彗星状」的光持续沿边框转圈扫过；
输入框边框同时有紫色（顺时针）与数据绿（逆时针）两束光交叉旋转。

**实现原理**（纯 CSS 即可完成，JS 只负责给卡片打标记）：

1. **裁出边框环**：给卡片加一个 `position:absolute; inset:0` 的覆盖层，`padding:1.5px`，
   再用两层 `linear-gradient(#fff)` 做 mask、`mask-composite: exclude`，
   把覆盖层裁成一条**贴着卡片圆角边框的 1.5px 圆环**（beam.jakubantalik.com 同款技巧）。
2. **环内放"光束"**：圆环背景用 `conic-gradient`（圆锥渐变）——
   绝大部分透明，只在某一段角度从主题色渐变到白色，形成带彗尾的光束。
3. **旋转**：`@keyframes` 让覆盖层持续 `transform: rotate(360deg)`。
   这是合成器动画（GPU 合成层），不触发布局/整页重绘，非常便宜。
   因为光束是"从圆心出发的角向楔形"，旋转时它正好贴着矩形边框走，经过圆角时会自然加速——
   这正是 border-beam 的标志性观感。
4. **降级与性能**：
   - `prefers-reduced-motion: reduce` → 动画停止，只剩静态微光，避免闪烁刺激；
   - 用 `IntersectionObserver` 在卡片滚出视口时给卡片加 `bb-paused`，
     CSS 里 `animation-play-state: paused` 暂停该卡片的动画（滚动长对话时零开销）。

---

## 2. 注入步骤（Open Web UI 管理后台）

1. 打开 Open Web UI，进入 **Settings（左下角头像）→ Interface（界面）**。
2. 在 **Custom CSS** 输入框粘贴下面的 **CSS 片段**。
3. 在 **Custom Code (JavaScript)** 输入框粘贴下面的 **JS 片段**（见第 4 节；
   如果你的版本没有 JS 输入框，见第 5.3 节的纯 CSS 降级方案）。
4. 点 **Save（保存）**，然后**强制刷新聊天页**（Cmd+Shift+R / Ctrl+Shift+R）。

> 说明：Open Web UI 把 Custom CSS 注入为全局 `<style>`、把 Custom Code 注入为页面级 JS。
> 两者在同一份配置里保存，刷新后即全局生效（对所有会话生效）。

---

## 3. CSS 片段（粘贴到 Settings → Interface → Custom CSS）

```css
/* ================================================================
   Border Beam for Open Web UI
   消息卡片 / 输入框边框旋转光束（beam.jakubantalik.com 风格）
   零依赖：conic-gradient + mask 圆环 + keyframes 旋转
   ================================================================ */

/* --- 被装饰的卡片必须是相对定位，才能承载绝对定位的光束 --- */
.bb-card { position: relative !important; }

/* --- 主光束：紫色（浅色主题自动变深紫）--- */
.bb-card::before,
.bb-card::after {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;              /* 跟随卡片圆角 */
  padding: 1.5px;                      /* 光束厚度 */
  pointer-events: none;                /* 不挡点击/选中 */
  z-index: 2;
  opacity: var(--bb-opacity, 0.55);
  will-change: transform;
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;         /* 只保留 1.5px 边框环 */
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask-composite: exclude;
}

/* 主束：逆时针留 300° 空档，末尾 60° 从主题色渐变到白色 = 彗尾 + 亮头 */
.bb-card::before {
  background: conic-gradient(from 0deg,
    transparent 0deg, transparent 300deg,
    var(--bb-c1, rgba(123, 57, 252, 0.38)) 324deg,
    var(--bb-c2, rgba(196, 181, 253, 0.95)) 348deg,
    #ffffff 359deg, #ffffff 360deg);
  animation: bb-spin var(--bb-dur, 7s) linear infinite;
}

/* 副束（输入框用 .bb-dual 才启用）：数据绿，反向、更慢 */
.bb-card.bb-dual::after {
  background: conic-gradient(from 0deg,
    transparent 0deg, transparent 330deg,
    var(--bb-c1b, rgba(46, 230, 168, 0.30)) 354deg,
    var(--bb-c2b, rgba(46, 230, 168, 0.90)) 358deg,
    #ffffff 359deg, #ffffff 360deg);
  animation: bb-spin-rev var(--bb-dur2, 11s) linear infinite;
}

/* --- 主题适配：默认深色；浅色主题换更深的紫/绿，保证浅底上可见 --- */
html[data-theme="dark"] .bb-card,
.dark .bb-card {
  --bb-c1: rgba(123, 57, 252, 0.40);
  --bb-c2: rgba(196, 181, 253, 0.95);
  --bb-c1b: rgba(46, 230, 168, 0.32);
  --bb-c2b: rgba(46, 230, 168, 0.92);
  --bb-opacity: 0.6;
}
html[data-theme="light"] .bb-card,
.light .bb-card {
  --bb-c1: rgba(109, 40, 217, 0.35);      /* 深紫 */
  --bb-c2: rgba(109, 40, 217, 0.90);
  --bb-c1b: rgba(4, 120, 87, 0.30);       /* 深绿 */
  --bb-c2b: rgba(4, 120, 87, 0.85);
  --bb-opacity: 0.55;
}

/* --- 输入框常驻双束；消息卡片 hover 时更亮 --- */
.bb-card.bb-dual { --bb-dur: 5s; --bb-dur2: 9s; --bb-opacity: 0.7; }
.bb-card:hover { --bb-opacity: 0.95; }

/* --- 滚出视口：JS 加 .bb-paused，暂停动画（省滚动性能）--- */
.bb-card.bb-paused::before,
.bb-card.bb-paused::after { animation-play-state: paused; }

@keyframes bb-spin     { to { transform: rotate(360deg); } }
@keyframes bb-spin-rev { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }

/* --- 系统「减弱动态效果」：光束静止为微光，避免闪烁 --- */
@media (prefers-reduced-motion: reduce) {
  .bb-card::before,
  .bb-card::after {
    animation: none !important;
    opacity: 0.22 !important;
  }
}
```

---

## 4. JS 片段（粘贴到 Settings → Interface → Custom Code (JavaScript)）

> 作用：找到每条消息的**气泡圆角容器**并打上 `bb-card` 标记（CSS 据此画光束）；
> 给输入框打 `bb-card bb-dual`（双光束）；用 `IntersectionObserver` 暂停视口外卡片的动画。
> 对 DOM 只做「加 class」这一种最小侵入，Open Web UI 重渲染/流式输出也不受影响。

```javascript
(function () {
  'use strict';
  if (window.__borderBeamInjected) return;
  window.__borderBeamInjected = true;

  var ROW_SEL = '[id^="message-"], .user-message';           // 消息行
  var CONTAINER_SEL = '#messages-container, .messages';       // 消息列表容器
  var INPUT_SEL = '#message-input-container, form.chat-input, .chat-input'; // 输入框

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      e.target.classList.toggle('bb-paused', !e.isIntersecting);
    });
  }, { rootMargin: '200px' }); // 提前 200px 开始/停止动画

  // 在消息行内找一个“看起来是气泡”的圆角大容器（避开按钮、代码块）
  function findBubble(row) {
    var nodes = row.querySelectorAll('[class*="rounded-"]');
    var best = null, bestArea = 0, r, area, el;
    for (var i = 0; i < nodes.length; i++) {
      el = nodes[i];
      if (el.closest('button, a, [role="button"], pre, code, .copy')) continue;
      r = el.getBoundingClientRect();
      if (r.width < 140 || r.height < 28) continue; // 跳过小按钮/药丸
      area = r.width * r.height;
      if (area > bestArea) { bestArea = area; best = el; }
    }
    return best; // 找不到就返回 null，交给调用方决定
  }

  function decorateRow(row) {
    if (row.classList.contains('bb-card')) return;
    var card = findBubble(row) || row; // 兜底：直接装饰整行
    if (card.classList.contains('bb-card')) return;
    card.classList.add('bb-card');
    io.observe(card);
  }

  function scan(container) {
    var rows = container.querySelectorAll(ROW_SEL);
    for (var i = 0; i < rows.length; i++) decorateRow(rows[i]);
  }

  function decorateInput() {
    var el = document.querySelector(INPUT_SEL);
    if (el && !el.classList.contains('bb-card')) {
      el.classList.add('bb-card', 'bb-dual');
      io.observe(el);
    }
  }

  function init() {
    decorateInput();
    var containers = document.querySelectorAll(CONTAINER_SEL);
    for (var i = 0; i < containers.length; i++) scan(containers[i]);
  }

  // 初始执行 + 监听流式输出 / 切换会话带来的新消息
  init();
  var timer = null;
  var mo = new MutationObserver(function () {
    clearTimeout(timer);
    timer = setTimeout(function () {
      var containers = document.querySelectorAll(CONTAINER_SEL);
      for (var i = 0; i < containers.length; i++) scan(containers[i]);
      decorateInput();
    }, 120); // 合并高频流式 mutation，避免频繁扫描
  });
  mo.observe(document.body, { childList: true, subtree: true });
})();
```

---

## 5. 验证

1. **基本效果**：保存并强制刷新后，聊天页任意用户/助手消息的气泡边框应有一束光
   持续沿边框扫过；输入框边框有两束光（紫 + 绿）反向旋转。3–5 秒内即可肉眼确认。
2. **流式输出**：向模型提问，观察回答流式渲染过程中，新出现的消息气泡自动带上光束
   （MutationObserver 自动打标，无需刷新）。
3. **明暗主题**：在 Settings → Interface 切换 Light / Dark（或 OLED Dark）主题，
   光束颜色应自动切换为深紫/深绿（浅色）或亮紫/亮绿（深色），浅色底上依然清晰。
4. **减弱动态效果**：系统开启「减弱动态效果」（macOS 系统设置 → 辅助功能 → 显示 →
   「减弱动态效果」；Windows：设置 → 辅助功能 → 动画效果），刷新后光束应静止为微光，无闪烁。
5. **滚动性能**：进入一个长对话，滚动页面——F12 → Performance 录制滚动过程，
   主线程不应有持续的动画帧占用；滚出视口的卡片动画被暂停（可临时在 DevTools 里
   给某条消息加/删 `bb-paused` 类对照观察）。
6. **无 JS 报错**：F12 → Console，应无红色报错（正常只有信息级日志）。

### 5.1 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 光束看不到 | 确认两个输入框都保存了；Open Web UI 缓存旧 JS，需强制刷新（Cmd+Shift+R）。 |
| 只有输入框有光、消息没有 | 该版本消息 DOM 结构变化（如 `[id^="message-"]` 被改），把新类名追加进 JS 里的 `ROW_SEL` 即可。 |
| 光束位置错位/太粗 | 调低 CSS 里 `padding`（1.5px → 1px），或降低 `--bb-opacity`。 |
| 浅色主题下太刺眼 | 调浅色主题分支里的 `--bb-opacity`（0.55 → 0.35）。 |
| 某些小气泡（如短回复）效果不明显 | 光束只显示在圆角边框上，气泡越小越不明显，属正常；可把 `--bb-dur` 调短（如 5s）让转速更快。 |

### 5.2 参数速查

- 转速：`--bb-dur`（主束，默认 7s）、`--bb-dur2`（副束，默认 11s），越小越快。
- 光束长度/颜色：conic-gradient 里的角度与 `--bb-c1 / --bb-c2`（主）、`--bb-c1b / --bb-c2b`（副）。
- 亮度：`--bb-opacity`（0–1，默认 0.55–0.7）。
- 粗细：`::before / ::after` 的 `padding`（默认 1.5px）。

### 5.3 没有 JS 输入框的老版本（纯 CSS 降级）

老版本 Open Web UI 的消息气泡带 `.chat-user` / `.chat-assistant` 类名，
把第 3 节 CSS 里所有 `.bb-card` 替换为 `.chat-user, .chat-assistant`、
`.bb-card.bb-dual` 替换为 `#chat-input` 即可（无视口暂停，其余效果一致）。
新版界面若找不到「Custom Code」输入框，说明版本较老，可改用此方案。

---

## 6. 与本仓库落地页的一致性

本效果与 `landing/index.html` 中 hero 策略控制台 / 浮动终端 / 卡片的光束边框同源同技术
（conic-gradient 光束 + mask 圆环 + keyframes 旋转，品牌色 `#7B39FC` 紫与 `#2EE6A8` 数据绿），
保证「网页 demo 与真实聊天界面」的视觉语言一致。
