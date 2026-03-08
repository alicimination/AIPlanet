"""Solver agent that combines RAG context + symbolic tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
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

        eq_match = re.search(r"([\w\*\+\-\^\(\)\s/]+=[\w\*\+\-\^\(\)\s/]+)", question)
        if eq_match:
            expr = eq_match.group(1).replace("^", "**")
            res = solve_expression(expr)
            if res.success:
                steps.append(f"Parsed equation: {expr}")
                steps.append(f"Solved roots using SymPy: {res.output}")
                answer = str(res.output)
        else:
            eval_match = re.search(r"simplify\s*:\s*(.+)$", question.lower())
            if eval_match:
                expr = eval_match.group(1).replace("^", "**")
                res = evaluate_expression(expr)
                if res.success:
                    steps.append(f"Simplified expression {expr}")
                    answer = str(res.output)

        if not steps:
            steps.append("Used retrieved math context to build a conceptual solution path.")
            answer = "Please review the explanation and verify with HITL for final confidence."

        return SolverResult(plan=plan, steps=steps, final_answer=answer, retrieved_context=retrieved_ctx)
