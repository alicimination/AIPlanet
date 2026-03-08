"""Safe-ish math execution helper using SymPy for symbolic tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


@dataclass
class MathToolResult:
    success: bool
    output: Any
    error: str = ""


def _normalize_expr(expr: str, variable: str) -> str:
    """Normalize common student shorthand (e.g., x2 -> x**2, ^ -> **)."""
    normalized = expr.replace("^", "**")
    normalized = re.sub(rf"\b{re.escape(variable)}(\d+)\b", rf"{variable}**\1", normalized)
    return normalized


def solve_expression(expression: str, variable: str = "x") -> MathToolResult:
    """Solve equations like x**2 - 5*x + 6 = 0, including shorthand like x2-5x+6=0."""
    try:
        sym_var = sp.symbols(variable)
        left, right = expression.split("=", maxsplit=1)
        left = _normalize_expr(left.strip(), variable)
        right = _normalize_expr(right.strip(), variable)

        transformations = standard_transformations + (implicit_multiplication_application,)
        local_dict = {variable: sym_var}

        left_expr = parse_expr(left, local_dict=local_dict, transformations=transformations)
        right_expr = parse_expr(right, local_dict=local_dict, transformations=transformations)
        eq = sp.Eq(left_expr, right_expr)
        roots = sp.solve(eq, sym_var)
        return MathToolResult(success=True, output=roots)
    except Exception as exc:
        return MathToolResult(success=False, output=None, error=str(exc))


def evaluate_expression(expression: str) -> MathToolResult:
    """Evaluate a SymPy-compatible expression."""
    try:
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(expression.replace("^", "**"), transformations=transformations)
        result = sp.simplify(expr)
        return MathToolResult(success=True, output=result)
    except Exception as exc:
        return MathToolResult(success=False, output=None, error=str(exc))


def get_allowed_functions() -> Dict[str, Any]:
    """Expose supported operations for UI/tooling transparency."""
    return {
        "solve_expression": "Solve equation in one variable",
        "evaluate_expression": "Simplify/evaluate symbolic expression",
    }
