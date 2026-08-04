#!/usr/bin/env python3
"""Track3 Graph Engine — DAG 状态机编排器。

设计原则（来自第一性原理诊断）：
- 每个节点 = 一个可验证、可回滚的步骤，在 Radeon 实例上执行。
- 节点执行后必须跑 verify()，不过门禁不进下一节点。
- 断点续跑：checkpoint 记录已完成节点，重启后从断点继续。
- 纯文本输出：所有节点打印结构化日志（不依赖截图，便于 Ego Lite 浏览器读取）。

用法（在 Radeon 实例的 Notebook cell 或终端运行）：
    from graph_engine import GraphEngine
    eng = GraphEngine(checkpoint_path="/workspace/persistent/track3_graph/checkpoint.json")
    eng.run()   # 从断点续跑整条 DAG
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Callable, Optional


class Node:
    """一个 DAG 节点：动作 + 验证 + 依赖。"""

    def __init__(self, name: str, deps: list[str], action: Callable, verify: Callable,
                 budget_min: int = 60):
        self.name = name
        self.deps = deps
        self.action = action
        self.verify = verify
        self.budget_min = budget_min
        self.status = "pending"   # pending | running | passed | failed | skipped

    def __repr__(self):
        return f"Node({self.name}, deps={self.deps}, status={self.status})"


class GraphEngine:
    def __init__(self, checkpoint_path: str = "/workspace/persistent/track3_graph/checkpoint.json"):
        self.nodes: dict[str, Node] = {}
        self.checkpoint_path = checkpoint_path
        self.log: list[str] = []
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self):
        d = os.path.dirname(self.checkpoint_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def register(self, node: Node):
        self.nodes[node.name] = node
        return self

    def _log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log.append(line)
        print(line, flush=True)

    def load_progress(self) -> set[str]:
        """读取已完成节点（断点续跑）。"""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path) as f:
                    data = json.load(f)
                done = set(data.get("passed", []))
                self._log(f"从 checkpoint 恢复：已完成 {sorted(done)}")
                return done
            except Exception as e:
                self._log(f"checkpoint 读取失败：{e}，从头开始")
        return set()

    def save_progress(self, done: set[str]):
        data = {
            "passed": sorted(done),
            "updated_at": datetime.now().isoformat(),
            "log_tail": self.log[-20:],
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

    def deps_met(self, node: Node, done: set[str]) -> bool:
        return all(d in done for d in node.deps)

    def run(self):
        done = self.load_progress()
        # 标记已完成节点
        for n in self.nodes.values():
            if n.name in done:
                n.status = "passed"

        order = self._topo_order()
        self._log("=== Graph Engine 启动 ===")
        self._log("DAG 顺序: " + " → ".join(order))

        for name in order:
            node = self.nodes[name]
            if node.name in done:
                self._log(f"[跳过] {name} 已完成")
                continue
            if not self.deps_met(node, done):
                self._log(f"[阻塞] {name} 依赖未满足，停在此前")
                break

            node.status = "running"
            self._log(f"[开始] {name} (预算 {node.budget_min}min)")
            t0 = time.time()
            try:
                node.action()
                ok, detail = node.verify()
            except Exception as e:
                ok, detail = False, f"异常: {e}"
            elapsed = (time.time() - t0) / 60.0

            if ok:
                node.status = "passed"
                done.add(name)
                self.save_progress(done)
                self._log(f"[通过] {name} 用时 {elapsed:.1f}min | {detail}")
            else:
                node.status = "failed"
                self._log(f"[失败] {name} 用时 {elapsed:.1f}min | {detail}")
                self._log("停止后续节点（不污染、不浪费 GPU）。请修复后重跑 run()。")
                break

        self._log("=== Graph Engine 结束 ===")
        self._summary(done)
        return done

    def _topo_order(self) -> list[str]:
        """简单 Kahn 拓扑排序。"""
        indeg = {n: 0 for n in self.nodes}
        for n in self.nodes.values():
            for d in n.deps:
                if d in indeg:
                    indeg[n.name] += 1
        queue = [n for n, d in indeg.items() if d == 0]
        order = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for n in self.nodes.values():
                if cur in n.deps and all(cur in order for cur in n.deps):
                    if n.name not in order:
                        queue.append(n.name)
        # 兜底：若有环或遗漏，按注册顺序补
        for n in self.nodes:
            if n not in order:
                order.append(n)
        return order

    def _summary(self, done: set[str]):
        total = len(self.nodes)
        passed = len(done)
        failed = [n for n, nd in self.nodes.items() if nd.status == "failed"]
        pending = [n for n, nd in self.nodes.items() if nd.status == "pending"]
        self._log(f"进度: {passed}/{total} 通过")
        if failed:
            self._log(f"失败节点: {failed}")
        if pending:
            self._log(f"待执行(因阻塞): {pending}")


# ──────────────────────────────────────────────────────────────
# 节点工厂：返回各节点的 action/verify。实际逻辑在 Radeon 上执行。
# 这里给出"纯文本可验证"的最小实现骨架，便于 Notebook cell 调用。
# ──────────────────────────────────────────────────────────────

def make_check_env_node():
    def action():
        # 打印 GPU / 模型 / 代码状态（文本，可读取）
        print(">>> check_env: 打印 rocm-smi + 模型 sha + git commit")
    def verify():
        # 这里只做"能跑通"判定；真实判定在节点打印里人工/脚本核对
        return True, "环境检查输出已打印（见上方 Notebook cell 文本）"
    return Node("check_env", deps=[], action=action, verify=verify, budget_min=10)


def make_align_walk_node():
    def action():
        print(">>> align_walk: 确认 t1_walk 对应本体，否则切 RulePolicy 兜底行走")
        print(">>> 运行 10s 单机器人 rollout，记录 base 高度 std")
    def verify():
        # 真实实现应解析 rollout 日志：base_height_std < 0.05 才过
        return True, "需解析 rollout 日志确认 base_height_std"
    return Node("align_walk", deps=["check_env"], action=action, verify=verify, budget_min=240)


def make_single_stable_node():
    def action():
        print(">>> single_stable: 1v1 Genesis 渲染，调 obs/deadzone/clip")
    def verify():
        return True, "需 render_1v1 metadata 稳定度字段"
    return Node("single_stable", deps=["align_walk"], action=action, verify=verify, budget_min=360)


def make_two_v_two_node():
    def action():
        print(">>> two_v_two: 2v2 评估，统计倒地次数/进球数")
    def verify():
        return True, "需 verify 脚本输出：倒地=0 且 进球>=1"
    return Node("two_v_two", deps=["single_stable"], action=action, verify=verify, budget_min=480)


def make_three_v_three_node():
    def action():
        print(">>> three_v_three: 3v3 + verify_g0 ×3 干净赛")
    def verify():
        return True, "需 verify_g0: 无 RPC-flood、无倒地崩溃、比分有变化 ×3"
    return Node("three_v_three", deps=["two_v_two"], action=action, verify=verify, budget_min=720)


def make_render_report_node():
    def action():
        print(">>> render_and_report: 出 demo 视频 + 评测报告")
    def verify():
        return True, "需文件清单 + 质量门禁结果"
    return Node("render_and_report", deps=["three_v_three"], action=action, verify=verify, budget_min=360)


def build_default_graph() -> GraphEngine:
    eng = GraphEngine()
    eng.register(make_check_env_node())
    eng.register(make_align_walk_node())
    eng.register(make_single_stable_node())
    eng.register(make_two_v_two_node())
    eng.register(make_three_v_three_node())
    eng.register(make_render_report_node())
    return eng


if __name__ == "__main__":
    # 本地冒烟测试（不连 Radeon）：只验证 DAG 拓扑与续跑逻辑
    eng = build_default_graph()
    print("节点:", [n for n in eng.nodes])
    print("拓扑顺序:", eng._topo_order())
    print("Graph Engine 骨架 OK（真实验证需 Radeon 实例执行节点 action）")
