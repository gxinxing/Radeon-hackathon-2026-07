"""DSL schema validator for trading strategy specifications.

Validates a parsed YAML strategy dict against the JSON Schema,
then performs semantic checks (indicator references, expression syntax).
Uses the AST-based expression parser for syntax validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from .expr_parser import validate_expression, get_expression_references

_SCHEMA_PATH = Path(__file__).parent / "schema.json"

# Indicators that produce multiple output columns (sub-fields)
_MULTI_COLUMN_INDICATORS: dict[str, list[str]] = {
    "MACD": ["_signal", "_hist"],
    "BollingerBands": ["_upper", "_middle", "_lower"],
    "Stochastic": ["_k", "_d"],
    "ICHIMOKU": ["_tenkan", "_kijun", "_spanA", "_spanB"],
}


def _expand_indicator_names(indicators: list[dict]) -> set[str]:
    """Expand indicator names to include sub-field columns.

    For example, a BollingerBands indicator named 'bb' produces:
    bb, bb_upper, bb_middle, bb_lower
    """
    names: set[str] = set()
    for ind in indicators:
        base_name = ind["name"]
        names.add(base_name)
        ind_type = ind["type"]
        for suffix in _MULTI_COLUMN_INDICATORS.get(ind_type, []):
            names.add(f"{base_name}{suffix}")
    return names


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

    # Collect indicator names (including multi-column sub-fields)
    indicator_names = _expand_indicator_names(strategy["indicators"])
    # Built-in data columns always available
    builtin_cols = {"open", "high", "low", "close", "volume"}
    all_refs = indicator_names | builtin_cols

    # Check entry/exit expressions reference defined indicators
    for section in ("entry", "exit"):
        for direction in ("long", "short"):
            expr = strategy.get(section, {}).get(direction)
            if expr is None:
                continue
            # Use AST-based parser for both syntax validation and reference extraction
            expr_errors = validate_expression(expr, indicator_names)
            for err in expr_errors:
                errors.append(f"{section}.{direction}: {err}")

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
