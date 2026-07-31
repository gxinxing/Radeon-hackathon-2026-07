"""Safe expression parser for DSL boolean expressions.

Uses Python's ast module to parse, validate, and evaluate DSL
entry/exit expressions against a strict whitelist of allowed
nodes. This replaces the previous eval()-based approach with
a direct AST evaluator — no eval() is used at all.

Supported DSL syntax:
    - Identifiers: indicator names (e.g. ema_fast, rsi)
    - Built-in columns: open, high, low, close, volume
    - Boolean operators: AND, OR, NOT (case-insensitive)
    - Comparison: >, <, >=, <=, ==, !=
    - Arithmetic: +, -, *, /
    - Numeric literals: 20, 0.5, 1.5e-3
    - Parentheses for grouping

The parser translates DSL expressions into Python AST, validates
all nodes against a whitelist, then evaluates directly by walking
the AST tree — no eval() or compile() is used.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

import numpy as np
import pandas as pd


# --- DSL → Python operator translation ---

def _translate_dsl_operators(expr: str) -> str:
    """Translate DSL boolean operators to Python equivalents.

    Keeps Python and/or/not keywords for proper operator precedence.
    Element-wise evaluation is handled by the AST evaluator.
    """
    result = expr
    result = re.sub(r'\bAND\b', 'and', result, flags=re.IGNORECASE)
    result = re.sub(r'\bOR\b', 'or', result, flags=re.IGNORECASE)
    result = re.sub(r'\bNOT\b', 'not ', result, flags=re.IGNORECASE)
    result = re.sub(r'\bnull\b', 'None', result, flags=re.IGNORECASE)
    return result


# --- AST node whitelist ---

_ALLOWED_NODES: set[type] = {
    # Expressions
    ast.Expression,
    ast.BoolOp,         # and / or
    ast.UnaryOp,        # not / - / +
    ast.BinOp,          # +, -, *, /
    ast.Compare,        # >, <, >=, <=, ==, !=
    # Operands
    ast.Name,           # variable names (indicator refs)
    ast.Constant,       # numeric literals (Python 3.8+)
    ast.Num,            # numeric literals (Python <3.8 compat)
    ast.Load,
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
    ast.And, ast.Or, ast.Not,
    ast.USub, ast.UAdd,  # unary - / +
}

# Built-in column names always available in expressions
_BUILTIN_COLUMNS = {"open", "high", "low", "close", "volume"}

# Python keywords that may appear in expressions
_PY_KEYWORDS = {"and", "or", "not", "True", "False", "None"}

# Operator mapping for direct AST evaluation (no eval needed)
_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

_CMP_OPS: dict[type, Any] = {
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def parse_expression(expr: str) -> ast.AST:
    """Parse a DSL expression into a validated AST.

    Translates DSL operators (AND→&, OR→|, NOT→~) and validates
    all nodes against a strict whitelist.

    Args:
        expr: DSL boolean expression string.

    Returns:
        Validated AST node.

    Raises:
        ValueError: If the expression contains disallowed syntax
            or references unknown identifiers.
    """
    py_expr = _translate_dsl_operators(expr)

    try:
        tree = ast.parse(py_expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Expression syntax error: {e.msg} (col {e.offset})") from e

    _validate_ast(tree)
    return tree


def _validate_ast(node: ast.AST) -> None:
    """Recursively validate AST nodes against the whitelist."""
    if type(node) not in _ALLOWED_NODES:
        raise ValueError(
            f"Disallowed syntax: {type(node).__name__} in expression"
        )

    # Validate Name nodes — must be a known identifier
    if isinstance(node, ast.Name):
        name = node.id
        if name not in _PY_KEYWORDS:
            # Names will be checked against available columns at eval time
            # but we can warn early about obviously wrong names
            pass

    # Recursively validate children
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


def get_expression_references(expr: str) -> set[str]:
    """Extract all identifier references from a DSL expression.

    Returns the set of variable names used in the expression,
    excluding Python keywords (and, or, not, True, False, None).
    """
    try:
        tree = parse_expression(expr)
    except ValueError:
        # Fallback: use regex extraction if AST parsing fails
        return _regex_extract_refs(expr)

    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id
            if name not in _PY_KEYWORDS and name not in _BUILTIN_COLUMNS:
                refs.add(name)
    return refs


def _regex_extract_refs(expr: str) -> set[str]:
    """Fallback regex-based reference extraction."""
    ident_re = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
    tokens = ident_re.findall(expr)
    return {t for t in tokens if t not in _PY_KEYWORDS and t not in _BUILTIN_COLUMNS}


def evaluate_expression(
    df: pd.DataFrame,
    expr: str,
    extra_columns: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """Parse and safely evaluate a DSL expression against a DataFrame.

    Uses direct AST evaluation — no eval() or compile() is used.
    The AST is walked node by node, applying operations via the
    operator module against the DataFrame namespace.

    Args:
        df: DataFrame with OHLCV + indicator columns.
        expr: DSL boolean expression string.
        extra_columns: Additional named columns to include in namespace.

    Returns:
        Boolean pandas Series. Returns all-False on error.
    """
    try:
        tree = parse_expression(expr)
    except (ValueError, SyntaxError, TypeError):
        return pd.Series(False, index=df.index)

    # Build namespace from DataFrame columns
    namespace: dict[str, Any] = {col: df[col] for col in df.columns}
    if extra_columns:
        namespace.update(extra_columns)

    try:
        result = _eval_ast_node(tree.body, namespace)
        if isinstance(result, pd.Series):
            return result.astype(bool).fillna(False)
        elif isinstance(result, (bool, np.bool_)):
            return pd.Series(bool(result), index=df.index)
        else:
            return pd.Series(False, index=df.index)
    except Exception:
        return pd.Series(False, index=df.index)


def _eval_ast_node(node: ast.AST, ns: dict[str, Any]) -> Any:
    """Recursively evaluate an AST node without eval().

    Only whitelisted node types are handled — anything else raises
    ValueError, which is caught by the caller.
    """
    # Literals
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):  # Python <3.8 compat
        return node.n

    # Variable references
    if isinstance(node, ast.Name):
        name = node.id
        if name in _PY_KEYWORDS:
            if name == "True":
                return True
            elif name == "False":
                return False
            elif name == "None":
                return None
        if name not in ns:
            raise ValueError(f"Undefined identifier: {name}")
        return ns[name]

    # Unary operations (-x, +x)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, ns)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.Not):
            return ~operand  # Element-wise NOT for pandas Series
        else:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    # Binary operations (+, -, *, /, %)
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, ns)
        right = _eval_ast_node(node.right, ns)
        op_func = _BIN_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
        return op_func(left, right)

    # Comparisons (>, <, >=, <=, ==, !=)
    if isinstance(node, ast.Compare):
        left = _eval_ast_node(node.left, ns)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_ast_node(comparator, ns)
            op_func = _CMP_OPS.get(type(op))
            if op_func is None:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            left = op_func(left, right)
        return left

    # Boolean operations (and → &, or → |) — element-wise for Series
    if isinstance(node, ast.BoolOp):
        values = [_eval_ast_node(v, ns) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for v in values[1:]:
                result = result & v  # Element-wise AND
            return result
        elif isinstance(node.op, ast.Or):
            result = values[0]
            for v in values[1:]:
                result = result | v  # Element-wise OR
            return result

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def validate_expression(expr: str, defined_indicators: set[str]) -> list[str]:
    """Validate a DSL expression: syntax + reference checks.

    Args:
        expr: DSL boolean expression string.
        defined_indicators: Set of defined indicator names.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    # Try to parse
    try:
        tree = parse_expression(expr)
    except ValueError as e:
        errors.append(f"Syntax error in expression '{expr}': {e}")
        return errors

    # Check all Name references are defined
    all_refs = _BUILTIN_COLUMNS | defined_indicators | _PY_KEYWORDS
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id
            if name not in all_refs:
                errors.append(
                    f"Expression references undefined identifier: '{name}'. "
                    f"Defined: {sorted(_BUILTIN_COLUMNS | defined_indicators)}"
                )

    return errors
