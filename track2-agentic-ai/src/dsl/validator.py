"""DSL schema validator for trading strategy specifications.

Validates a parsed YAML strategy dict against the JSON Schema,
then performs semantic checks (indicator references, expression syntax).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_PATH = Path(__file__).parent / "schema.json"

# Operators and functions allowed in boolean expressions
_ALLOWED_EXPR_TOKENS = re.compile(
    r"^[\w\s\.\*\+\-\/\(\)<>=!&|,\d'\"]+$"
)
_INDICATOR_REF_RE = re.compile(r"[a-z][a-z0-9_]*")
_PY_KEYWORDS = {"and", "or", "not", "true", "false", "null", "none"}


def load_schema() -> dict[str, Any]:
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


def validate_dsl(strategy_dict: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a strategy DSL dict against schema and semantic rules.

    Returns (is_valid, error_messages).
    """
    errors: list[str] = []

    # --- Schema validation ---
    schema = load_schema()
    try:
        jsonschema.validate(instance=strategy_dict, schema=schema)
    except jsonschema.ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "root"
        errors.append(f"Schema error at '{path}': {e.message}")
        return False, errors

    strategy = strategy_dict["strategy"]

    # --- Semantic validation ---

    # Collect indicator names
    indicator_names = {ind["name"] for ind in strategy["indicators"]}
    # Built-in data columns always available
    builtin_cols = {"open", "high", "low", "close", "volume"}
    all_refs = indicator_names | builtin_cols

    # Check entry/exit expressions reference defined indicators
    for section in ("entry", "exit"):
        for direction in ("long", "short"):
            expr = strategy.get(section, {}).get(direction)
            if expr is None:
                continue
            refs = _extract_refs(expr)
            for ref in refs:
                if ref not in all_refs and ref not in _PY_KEYWORDS:
                    errors.append(
                        f"{section}.{direction} references undefined "
                        f"indicator: '{ref}'. "
                        f"Defined: {sorted(all_refs)}"
                    )

    # Check MACD has fast/slow/signal params
    for ind in strategy["indicators"]:
        if ind["type"] == "MACD":
            params = ind.get("params", {})
            if "fast_period" not in params or "slow_period" not in params:
                errors.append(
                    f"MACD indicator '{ind['name']}' requires "
                    "fast_period and slow_period"
                )

    # Check stop_loss is negative
    risk = strategy["risk"]
    if risk["stop_loss"] > 0:
        errors.append("risk.stop_loss must be negative (e.g. -0.03)")

    # Check trailing stop consistency
    if risk.get("trailing_stop") and risk.get("trailing_stop_positive", 0) <= 0:
        errors.append(
            "trailing_stop_positive must be > 0 when trailing_stop is enabled"
        )

    return len(errors) == 0, errors


def _extract_refs(expr: str) -> set[str]:
    """Extract identifier-like tokens from a boolean expression."""
    tokens = _INDICATOR_REF_RE.findall(expr)
    return {t for t in tokens if t not in _PY_KEYWORDS}
