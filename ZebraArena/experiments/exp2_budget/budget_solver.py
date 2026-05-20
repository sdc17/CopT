#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Budget-aware solver for Zebra puzzle.
Tracks budget but uses SOFT constraint - doesn't force stop, just tracks over/under budget.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from typing import Optional
from core.base_solver import BaseSolver, EnvType
from prompts.base_prompt import get_prompt_template


class BudgetSolver(BaseSolver):
    """
    Solver with soft budget constraint.
    Informs model of budget but allows exceeding it.
    Tracks budget_score = budget - actual_tool_calls (negative if over budget).
    """

    def __init__(
        self,
        env_type: EnvType,
        model_name: str,
        budget: int,
        hard_limit: bool = False,
        **kwargs
    ):
        """
        Initialize budget solver.

        Args:
            env_type: Environment type (NORMAL, ONLY_FACT, ONLY_RELATION)
            model_name: Name of the LLM model to use
            budget: Target budget for tool calls
            hard_limit: If True, force stop when budget exhausted. If False (default),
                       allow exceeding budget but track negative score.
            **kwargs: Additional arguments passed to BaseSolver
        """
        super().__init__(env_type=env_type, model_name=model_name, **kwargs)
        self.budget = budget
        self.remaining_budget = budget
        self.hard_limit = hard_limit

    def reset_metrics(self):
        """Reset per-puzzle metrics including budget."""
        super().reset_metrics()
        self.remaining_budget = self.budget

    def get_prompt_template(self) -> str:
        """Return prompt template based on environment type."""
        return get_prompt_template(self.env_type)

    def get_extra_prompt_sections(self) -> str:
        """Add budget constraint information to prompt."""
        constraint_type = "strict limit" if self.hard_limit else "target"
        return f"""## Budget Constraint
You have a budget of **{self.budget}** tool calls for this puzzle.
- Each query (fact or relation) costs 1 from your budget.
- Invalid queries that fail validation also cost 1.
- This is a {constraint_type} - plan your queries carefully.
- Current remaining budget will be shown after each query response.

**Strategy tips:**
- Start by reasoning about what information would be most valuable.
- Avoid redundant queries - track what you've already learned.
- Prioritize queries that can eliminate the most possibilities.
- If running low on budget, make your best guess based on available information."""

    def get_turn_context(self) -> str:
        """Show remaining budget in each turn."""
        if self.remaining_budget > 0:
            return f"[Budget: {self.remaining_budget}/{self.budget} queries remaining]"
        else:
            over = -self.remaining_budget
            return f"[Budget: OVER by {over} queries ({self.budget} budget)]"

    def on_after_query(self, query: dict, result: dict) -> None:
        """Decrement budget after each query (regardless of success)."""
        self.remaining_budget -= 1

    def should_stop_early(self) -> tuple[bool, Optional[str]]:
        """Stop only if hard_limit is True and budget is exhausted."""
        if self.hard_limit and self.remaining_budget <= 0:
            return True, f"Budget exhausted ({self.budget} queries used)"
        return False, None

    def get_budget_score(self) -> int:
        """
        Calculate budget score after puzzle completion.

        Returns:
            Positive if under budget, zero if exact, negative if over budget.
            Score = budget - actual_tool_calls = remaining_budget
        """
        return self.remaining_budget

    def get_extra_metrics(self) -> dict:
        """Return additional metrics for logging."""
        return {
            "budget": self.budget,
            "budget_score": self.get_budget_score(),
            "over_budget": self.remaining_budget < 0,
            "under_budget": self.remaining_budget > 0,
            "budget_utilization": (self.budget - self.remaining_budget) / self.budget if self.budget > 0 else 0,
        }
