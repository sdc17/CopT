#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base solver class for Zebra puzzle experiments.
"""

import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable
from jsonschema import validate, ValidationError

from .llm import llm_generate as default_llm_generate, normalize_llm_output
from .utils import safe_serialize


class EnvType(Enum):
    """Environment type determining which query types are allowed."""
    NORMAL = "normal"              # fact + relation
    ONLY_FACT = "only_fact"        # fact only
    ONLY_RELATION = "only_relation"  # relation only


class BaseSolver(ABC):
    """
    Abstract base class for Zebra puzzle solvers.

    Subclasses can override hooks to customize behavior for different
    experiment types (e.g., budget constraints, token costs).
    """

    def __init__(
        self,
        env_type: EnvType,
        model_name: str,
        llm_generate_fn: Optional[Callable] = None,
        max_input_tokens: int = 12000,
        keep_tail: int = 8,
    ):
        """
        Initialize the solver.

        Args:
            env_type: Type of environment (determines allowed query types)
            model_name: Name of the LLM model to use
            llm_generate_fn: Custom LLM generation function (optional)
            max_input_tokens: Maximum input tokens for context pruning
            keep_tail: Number of tail messages to keep during pruning
        """
        self.env_type = env_type
        self.model_name = model_name

        # Set up LLM generation function
        if llm_generate_fn is None:
            def _llm_wrapper(model: str, messages: list[dict]) -> str:
                return default_llm_generate(model, messages)
            self.llm_generate_fn = _llm_wrapper
        else:
            self.llm_generate_fn = llm_generate_fn

        self.max_input_tokens = max_input_tokens
        self.keep_tail = keep_tail

        # Metrics
        self.tool_call_num = 0
        self.success_tool_call_num = 0
        self.step = 0

    def reset_metrics(self):
        """Reset per-puzzle metrics."""
        self.tool_call_num = 0
        self.success_tool_call_num = 0
        self.step = 0

    # ==================== Abstract Methods ====================

    @abstractmethod
    def get_prompt_template(self) -> str:
        """
        Return the prompt template string.
        Must be implemented by subclasses.
        """
        pass

    # ==================== Hooks (Override in Subclasses) ====================

    def on_before_query(self, query: dict) -> dict:
        """
        Hook called before executing a query.
        Can modify the query or perform pre-processing.

        Args:
            query: The query dict to be executed

        Returns:
            Potentially modified query dict
        """
        return query

    def on_after_query(self, query: dict, result: dict) -> None:
        """
        Hook called after executing a query.
        Can update internal state based on query results.

        Args:
            query: The executed query
            result: The query result from environment
        """
        pass

    def should_stop_early(self) -> tuple[bool, Optional[str]]:
        """
        Hook to determine if solving should stop early.

        Returns:
            Tuple of (should_stop, reason_message)
        """
        return False, None

    def get_extra_prompt_sections(self) -> str:
        """
        Hook to add extra sections to the prompt.
        Override in subclasses for budget/cost info.

        Returns:
            Additional prompt text to append
        """
        return ""

    def get_turn_context(self) -> str:
        """
        Hook to add context to each turn's user message.
        Override in subclasses for dynamic info (remaining budget, etc.)

        Returns:
            Context string to prepend to environment responses
        """
        return ""

    def on_solution_found(self, solution: dict, is_correct: bool, result_detail: dict) -> None:
        """
        Hook called when a solution is found.

        Args:
            solution: The candidate solution
            is_correct: Whether solution matches ground truth
            result_detail: Detailed comparison result
        """
        pass

    def on_llm_output(self, output: str) -> None:
        """
        Hook called after receiving LLM output.
        Can be used to track token usage, etc.

        Args:
            output: The LLM output text
        """
        pass

    def get_extra_metrics(self) -> dict:
        """
        Hook to return additional metrics for logging.
        Override in subclasses to add custom metrics (e.g., budget tracking).

        Returns:
            Dictionary of additional metrics to save
        """
        return {}

    # ==================== Core Methods ====================

    def get_allowed_query_types(self) -> set:
        """Return set of allowed query types based on env_type."""
        if self.env_type == EnvType.ONLY_FACT:
            return {"fact"}
        elif self.env_type == EnvType.ONLY_RELATION:
            return {"relation"}
        return {"fact", "relation"}

    def is_valid_query_type(self, query: dict) -> bool:
        """Check if query type is allowed for this environment."""
        return query.get("type") in self.get_allowed_query_types()

    def build_system_prompt(
        self,
        gt: dict,
        houses: list[str],
        attributes: dict,
        domain: dict
    ) -> str:
        """
        Build the full system prompt.

        Args:
            gt: Ground truth solution dict with 'header' and 'rows'
            houses: List of house IDs
            attributes: Dict mapping attribute names to value lists
            domain: Same as attributes (for prompt display)

        Returns:
            Formatted system prompt string
        """
        N = len(houses)
        base_prompt = self.get_prompt_template()
        extra = self.get_extra_prompt_sections()

        prompt = base_prompt.format(
            houses=json.dumps(houses, ensure_ascii=False),
            attrs=json.dumps(list(attributes.keys()), ensure_ascii=False),
            domain=json.dumps(domain, ensure_ascii=False),
            header=json.dumps(gt["header"], ensure_ascii=False),
            N=N
        )

        if extra:
            prompt = prompt + "\n\n" + extra

        return prompt

    def is_query(self, obj: dict) -> bool:
        """Check if object is a valid query."""
        return isinstance(obj, dict) and self.is_valid_query_type(obj)

    def is_solution(self, obj: dict, attributes: dict, houses: list) -> bool:
        """Check if object looks like a solution."""
        return (
            isinstance(obj, dict)
            and isinstance(obj.get("rows"), list)
            and len(obj["rows"]) == len(houses)
        )

    def driver_loop(
        self,
        row: dict,
        log_dir: str,
        # Injected dependencies
        load_solution_fn: Callable,
        build_schemas_fn: Callable,
        canonicalize_query_fn: Callable,
        answer_query_fn: Callable,
        extract_query_dict_fn: Callable,
        extract_answer_dict_fn: Callable,
        check_final_solution_fn: Callable,
        solution_numpy_to_dict_fn: Callable,
        prune_messages_fn: Callable,
        rank: int = 0,
    ) -> Optional[dict]:
        """
        Main solving loop for a single puzzle.

        Args:
            row: Dataset row containing puzzle data
            log_dir: Directory to save results
            load_solution_fn: Function to load ground truth solution
            build_schemas_fn: Function to build JSON schemas
            canonicalize_query_fn: Function to canonicalize queries
            answer_query_fn: Function to execute queries
            extract_query_dict_fn: Function to extract query from LLM output
            extract_answer_dict_fn: Function to extract solution from LLM output
            check_final_solution_fn: Function to validate solution
            solution_numpy_to_dict_fn: Function to convert numpy solution
            prune_messages_fn: Function to prune message history
            rank: Process rank for logging

        Returns:
            Result dict if solution found, None otherwise
        """
        self.reset_metrics()

        puzzle_id = row["id"]
        out_path = os.path.join(log_dir, f"{puzzle_id}_response.json")

        if os.path.exists(out_path):
            print(f"[rank {rank}] Skip: responses already exist for problem {puzzle_id}")
            return None

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Load puzzle data
        gt_np = row["solution"]
        gt = solution_numpy_to_dict_fn(gt_np)
        puzzle = row["missing_puzzle"]
        houses, attributes, solution, attr_alias, value_alias = load_solution_fn(gt)

        # Build schemas based on env_type
        schemas = build_schemas_fn(self.env_type, houses, attributes)

        domain = {a: list(vs) for a, vs in attributes.items()}
        system_prompt = self.build_system_prompt(gt, houses, attributes, domain)

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": puzzle},
        ]
        response_log: list[dict] = []

        while True:
            self.step += 1

            # Check for early stopping
            should_stop, stop_reason = self.should_stop_early()
            if should_stop:
                print(f"[rank {rank}] Early stop: {stop_reason}")
                # Save partial results
                info = safe_serialize(row)
                info["response_log"] = response_log
                info["messages"] = messages  # Full conversation history sent to model
                info["early_stop"] = True
                info["early_stop_reason"] = stop_reason
                info["success_tool_call_num"] = self.success_tool_call_num
                info["tool_call_num"] = self.tool_call_num
                info["steps_num"] = self.step
                # Add extra metrics from subclass (e.g., budget tracking)
                info.update(self.get_extra_metrics())

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(safe_serialize(info), f, ensure_ascii=False, indent=2)
                return info

            # Prune messages
            messages = prune_messages_fn(messages, max_input_tokens=self.max_input_tokens, keep_tail=self.keep_tail)

            # Generate LLM response
            output_raw = self.llm_generate_fn(self.model_name, messages)
            output = normalize_llm_output(output_raw)

            # Call hook for output processing (e.g., token tracking)
            self.on_llm_output(output)

            response_log.append({
                "step": self.step,
                "role": "assistant",
                "content_raw": output_raw,
                "content": output
            })

            # Handle empty/malformed response
            if not output.strip():
                messages.append({
                    "role": "user",
                    "content": "Your last message was empty or malformed. Please re-send either a single <query> or a single <solution>."
                })
                continue

            # Try to extract solution first
            a = extract_answer_dict_fn(output)
            if a is not None and self.is_solution(a, attributes, houses):
                flag, result = check_final_solution_fn(a, gt)

                self.on_solution_found(a, flag, result)

                info = safe_serialize(row)
                info["response_log"] = response_log
                info["messages"] = messages  # Full conversation history sent to model
                info["answer"] = a
                info["gt"] = gt
                info["acc"] = flag
                info["result_detail"] = result
                info["success_tool_call_num"] = self.success_tool_call_num
                info["tool_call_num"] = self.tool_call_num
                info["steps_num"] = self.step
                # Add extra metrics from subclass (e.g., budget tracking)
                info.update(self.get_extra_metrics())

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(safe_serialize(info), f, ensure_ascii=False, indent=2)
                    print(f"All LLM outputs & query responses saved to: {out_path}")
                return info

            # Try to extract query
            q = extract_query_dict_fn(output)
            if q is not None and self.is_query(q):
                self.tool_call_num += 1

                # Apply pre-query hook
                q = self.on_before_query(q)

                # Canonicalize query
                canon, cerr = canonicalize_query_fn(q, houses, attributes)
                if cerr:
                    warn = f"canonicalize failed: {cerr}"
                    response_log.append({
                        "step": self.step,
                        "role": "env",
                        "query": json.dumps(safe_serialize(q), ensure_ascii=False),
                        "response": f"<query_response>{json.dumps({'ok': False, 'error': warn}, ensure_ascii=False)}</query_response>"
                    })
                    messages.append({"role": "assistant", "content": output})
                    messages.append({"role": "user", "content": f"The last JSON was invalid: {warn}. Retry."})
                    continue

                # Validate against schema
                query_type = canon["type"]
                if query_type not in schemas:
                    err = f"Query type '{query_type}' not allowed in {self.env_type.value} mode"
                    response_log.append({
                        "step": self.step,
                        "role": "env",
                        "query": json.dumps(safe_serialize(q), ensure_ascii=False),
                        "response": f"<query_response>{json.dumps({'ok': False, 'error': err}, ensure_ascii=False)}</query_response>"
                    })
                    messages.append({"role": "assistant", "content": output})
                    messages.append({"role": "user", "content": f"The last JSON was invalid: {err}. Retry."})
                    continue

                schema = schemas[query_type]
                try:
                    validate(instance=canon, schema=schema)
                except ValidationError as e:
                    err = f"Query validation failed: {e.message}"
                    response_log.append({
                        "step": self.step,
                        "role": "env",
                        "query": json.dumps(safe_serialize(q), ensure_ascii=False),
                        "response": f"<query_response>{json.dumps({'ok': False, 'error': err}, ensure_ascii=False)}</query_response>"
                    })
                    messages.append({"role": "assistant", "content": output})
                    messages.append({"role": "user", "content": f"The last JSON was invalid: {err}. Retry."})
                    continue

                # Execute query
                ans = answer_query_fn(canon, solution, attributes, houses, attr_alias, value_alias)

                if ans.get("ok"):
                    self.success_tool_call_num += 1

                # Apply post-query hook
                self.on_after_query(canon, ans)

                # Build response with optional turn context
                turn_context = self.get_turn_context()
                env_response = f"Environment answer: {json.dumps(ans, ensure_ascii=False)}"
                if turn_context:
                    env_response = f"{turn_context}\n{env_response}"

                response_log.append({
                    "step": self.step,
                    "role": "env",
                    "query": json.dumps(safe_serialize(q), ensure_ascii=False),
                    "query_response": f"<query_response>{json.dumps(ans, ensure_ascii=False)}</query_response>",
                    "env_response": env_response,  # Full response sent to model (includes turn_context)
                    "merged": json.dumps(ans, ensure_ascii=False)
                })

                messages.append({"role": "assistant", "content": f"<query>{json.dumps(canon, ensure_ascii=False)}</query>"})
                messages.append({"role": "user", "content": env_response})
                continue

            # Not a query or solution, treat as reasoning
            messages.append({"role": "assistant", "content": output})
