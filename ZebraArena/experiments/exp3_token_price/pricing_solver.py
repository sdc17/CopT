#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-based pricing solver for Zebra puzzle.
Tracks virtual tool costs in tokens and shows cumulative usage.

Design:
- Tool price is shown in prompt (e.g., fact=500 tokens, relation=500 tokens)
- Goal: Minimize total token cost while solving accurately and reliably
- After each turn, show: [Token usage: XXX reasoning + YYY tools = ZZZ total]
- Track actual reasoning tokens + virtual tool tokens
"""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from core.base_solver import BaseSolver, EnvType
from prompts.base_prompt import get_prompt_template


# Default pricing configuration
DEFAULT_FACT_PRICE = 500
DEFAULT_RELATION_PRICE = 500


class PricingSolver(BaseSolver):
    """
    Solver with token-based pricing for tools.
    Tracks virtual tool costs and shows cumulative token usage.
    Goal is to minimize total cost while maintaining accuracy.
    """

    def __init__(
        self,
        env_type: EnvType,
        model_name: str,
        fact_price: int = DEFAULT_FACT_PRICE,
        relation_price: int = DEFAULT_RELATION_PRICE,
        **kwargs
    ):
        """
        Initialize pricing solver.

        Args:
            env_type: Environment type (NORMAL, ONLY_FACT, ONLY_RELATION)
            model_name: Name of the LLM model to use
            fact_price: Virtual token cost for fact queries
            relation_price: Virtual token cost for relation queries
            **kwargs: Additional arguments passed to BaseSolver
        """
        super().__init__(env_type=env_type, model_name=model_name, **kwargs)
        self.fact_price = fact_price
        self.relation_price = relation_price

        # Token tracking
        self.reasoning_tokens = 0  # Actual tokens from model output
        self.tool_tokens = 0  # Virtual tokens from tool usage
        self.fact_count = 0
        self.relation_count = 0

    def reset_metrics(self):
        """Reset per-puzzle metrics including token tracking."""
        super().reset_metrics()
        self.reasoning_tokens = 0
        self.tool_tokens = 0
        self.fact_count = 0
        self.relation_count = 0

    def get_prompt_template(self) -> str:
        """Return prompt template based on environment type."""
        return get_prompt_template(self.env_type)

    def get_extra_prompt_sections(self) -> str:
        """Add tool pricing information to prompt."""
        # Build tool pricing info based on env_type
        if self.env_type == EnvType.ONLY_FACT:
            tool_info = f"- Fact query: {self.fact_price} tokens"
        elif self.env_type == EnvType.ONLY_RELATION:
            tool_info = f"- Relation query: {self.relation_price} tokens"
        else:
            tool_info = f"""- Fact query: {self.fact_price} tokens
- Relation query: {self.relation_price} tokens"""

        return f"""## Tool Pricing & Optimization Goal
Each tool call has a token cost:
{tool_info}

**Your goal is to minimize total token cost while solving the puzzle accurately and reliably.**

After each response, you will see your cumulative cost:
[Token usage: XXXX reasoning + YYYY tools = ZZZZ total]

**Strategy guidelines:**
- Be efficient: Only query when necessary for deduction.
- Be strategic: Choose the query type that gives maximum information per token.
- Be accurate: Ensure your solution is correct - an incorrect solution wastes all spent tokens.
- Be reliable: When uncertain, query rather than guess incorrectly.

Balance efficiency with accuracy - the optimal strategy minimizes tokens while maintaining correctness."""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (chars / 4)."""
        if not text:
            return 0
        return len(text) // 4 + 1

    def get_total_tokens(self) -> int:
        """Get total token usage (reasoning + tools)."""
        return self.reasoning_tokens + self.tool_tokens

    def get_turn_context(self) -> str:
        """Show token usage in each turn."""
        total = self.get_total_tokens()
        return f"[Token usage: {self.reasoning_tokens} reasoning + {self.tool_tokens} tools = {total} total]"

    def on_after_query(self, query: dict, result: dict) -> None:
        """Add virtual token cost after each query."""
        query_type = query.get("type", "")

        if query_type == "fact":
            self.tool_tokens += self.fact_price
            self.fact_count += 1
        elif query_type == "relation":
            self.tool_tokens += self.relation_price
            self.relation_count += 1

    def on_llm_output(self, output: str) -> None:
        """
        Hook called after receiving LLM output.
        Updates reasoning token count.

        Args:
            output: The model's output text
        """
        self.reasoning_tokens += self._estimate_tokens(output)

    def get_extra_metrics(self) -> dict:
        """Return additional metrics for logging."""
        total = self.get_total_tokens()
        total_queries = self.fact_count + self.relation_count

        return {
            # Pricing config
            "fact_price": self.fact_price,
            "relation_price": self.relation_price,

            # Token tracking
            "reasoning_tokens": self.reasoning_tokens,
            "tool_tokens": self.tool_tokens,
            "total_tokens": total,

            # Query counts
            "fact_count": self.fact_count,
            "relation_count": self.relation_count,
            "total_queries": total_queries,

            # Derived metrics
            "fact_ratio": self.fact_count / total_queries if total_queries > 0 else 0,
            "avg_tokens_per_query": self.tool_tokens / total_queries if total_queries > 0 else 0,
            "reasoning_ratio": self.reasoning_tokens / total if total > 0 else 0,
        }
