"""Solver agent that combines RAG context + symbolic tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import re

from rag.retriever import RAGRetriever
from tools.python_math_tool import solve_expression, evaluate_expression


@dataclass
class SolverResult:
    plan: List[str]
    steps: List[str]
    final_answer: str
    retrieved_context: List[Dict]


class SolverAgent:
    """Produce plan and solution using RAG + symbolic helper."""

    def __init__(self):
        self.retriever = RAGRetriever()

    def _normalize_question(self, question: str) -> str:
        """Normalize common unicode math characters before parsing."""
        return (
            question.replace("−", "-")
            .replace("—", "-")
            .replace("–", "-")
            .replace("＝", "=")
            .replace("×", "*")
        )

    def _extract_equation(self, question: str) -> Optional[str]:
        """Extract a solvable equation from mixed natural-language prompts."""
        normalized = self._normalize_question(question)
        if "=" not in normalized:
            return None

        left, right = normalized.split("=", maxsplit=1)

        # Keep the likely math segment before '='.
        left = left.split("\n")[-1]
        if ":" in left:
            left = left.split(":")[-1]
        left = re.sub(r"[^A-Za-z0-9\s\*\+\-\^\(\)\./]", " ", left).strip()

        # Right side is usually compact; sanitize to math tokens.
        right = right.split("\n")[0]
        right = re.sub(r"[^A-Za-z0-9\s\*\+\-\^\(\)\./]", " ", right).strip()

        if not left or not right:
            return None
        if not re.search(r"[a-zA-Z]", left):
            return None

        return f"{left}={right}"

    def run(self, parsed_problem: Dict, strategy: str) -> SolverResult:
        question = parsed_problem["problem_text"]
        retrieved = self.retriever.retrieve(question, top_k=4)
        retrieved_ctx = [
            {"source": r.metadata.get("source", "unknown"), "score": r.score, "content": r.content}
            for r in retrieved
        ]

        plan = [
            f"Use strategy: {strategy}",
            "Retrieve relevant formulas and pitfalls",
            "Apply symbolic manipulation and compute answer",
        ]

        steps: List[str] = []
        answer = "Could not derive a final answer automatically."

        eq_expr = self._extract_equation(question)
        if eq_expr:
            res = solve_expression(eq_expr)
            if res.success:
                steps.append(f"Parsed equation: {eq_expr}")
                steps.append(f"Solved roots using SymPy: {res.output}")
                answer = str(res.output)
            else:
                steps.append(f"Equation parse failed: {res.error}")
        else:
            eval_match = re.search(r"simplify\s*:\s*(.+)$", question.lower())
            if eval_match:
                expr = self._normalize_question(eval_match.group(1))
                res = evaluate_expression(expr)
                if res.success:
                    steps.append(f"Simplified expression {expr}")
                    answer = str(res.output)

        if not steps:
            steps.append("Used retrieved math context to build a conceptual solution path.")
            answer = "Please review the explanation and verify with HITL for final confidence."

        return SolverResult(plan=plan, steps=steps, final_answer=answer, retrieved_context=retrieved_ctx)
