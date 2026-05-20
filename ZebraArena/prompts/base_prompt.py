#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for Zebra puzzle solving.
"""

import sys
from pathlib import Path
from enum import Enum

# Add zebrapuzzle root to path for imports
_ZEBRAPUZZLE_ROOT = Path(__file__).parent.parent
if str(_ZEBRAPUZZLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZEBRAPUZZLE_ROOT))


# Define EnvType locally to avoid circular imports
class EnvType(Enum):
    """Environment type determining which query types are allowed."""
    NORMAL = "normal"
    ONLY_FACT = "only_fact"
    ONLY_RELATION = "only_relation"


# ============================================================
# Shared prompt components
# ============================================================

_CAPABILITIES = """## Your capabilities
- You may reason step by step, but **all reasoning text must be wrapped inside <think> ... </think>**.
- If you cannot deduce the correct answer, you may issue a **query** to the environment to obtain information.
- Each query must be **pure JSON**, strictly following the query protocol.
- Each query must be wrapped inside <query> ... </query>.
- After receiving a response, integrate it into your reasoning and continue (still inside <think>).
- **Never** output both `<query>` and `<solution>` in the same message.
- If you are **not certain** the puzzle is uniquely solved, **do NOT output `<solution>`**. Output a `<query>` instead."""

_OUTPUT_RULES = """## Output rules
- Wrap all reasoning text in <think> ... </think>.
- Wrap every query JSON in <query> ... </query>.
- Wrap the final solution JSON in <solution> ... </solution>.
- Do not output explanations outside of these tags."""

_FINAL_REQUIREMENT = """## Final requirement
Once you have solved the puzzle, you must output the complete puzzle solution JSON, wrapped inside <solution> ... </solution>:

<solution>
{{
  "header": {header},
  "rows": [
    ["1", "<Name>", "<...>", "..."],
    ...
  ]
}}
</solution>

- The `header` must exactly match the attribute list above.
- The number of rows must equal {N}.
- Each row starts with "1","2",...,"{N}" and fills all attributes.
- Only use values from DOMAIN."""


# ============================================================
# Query protocol components
# ============================================================

_QUERY_COMMON = """### Common
- **Entity object** = {{ "attr":"<string>", "value":"<string>" }}
- **HOUSES** = {houses}
- **ATTRS** = {attrs}
- **DOMAIN** = {domain}
- `attr` / `value` strings will be normalized for case/whitespace at runtime."""

_FACT_QUERY_PROTOCOL = """### Fact queries (`type:"fact"`)
Use exactly this shape:
{{
  "type": "fact",
  "rel": "found_at",
  "house": "<one of HOUSES>",
  "attr": "<string>",
  "value": "<string>"
}}"""

_RELATION_QUERY_PROTOCOL = """### Relation queries (`type:"relation"`)

Allowed: "same_house", "not_at", "direct_left", "direct_right", "side_by_side", "left_of", "right_of", "one_between", "two_between"

{{
  "type": "relation",
  "rel":"<one of Allowed relations>",
  "lhs": {{ "attr":"<string>", "value":"<string>" }},
  "rhs": {{ "attr":"<string>", "value":"<string>" }}
}}"""


# ============================================================
# Full prompt templates
# ============================================================

NORMAL_PROMPT = f"""You are a reasoning agent solving a Zebra puzzle.

{_CAPABILITIES}

## Query protocol

{_QUERY_COMMON}
- `type` ∈ {{{{ "fact", "relation" }}}}

{_FACT_QUERY_PROTOCOL}

{_RELATION_QUERY_PROTOCOL}

{_OUTPUT_RULES}

{_FINAL_REQUIREMENT}""".strip()


ONLY_FACT_PROMPT = f"""You are a reasoning agent solving a Zebra puzzle.

{_CAPABILITIES}

## Query protocol

{_QUERY_COMMON}
- `type` ∈ {{{{ "fact" }}}}

{_FACT_QUERY_PROTOCOL}

{_OUTPUT_RULES}

{_FINAL_REQUIREMENT}""".strip()


ONLY_RELATION_PROMPT = f"""You are a reasoning agent solving a Zebra puzzle.

{_CAPABILITIES}

## Query protocol

{_QUERY_COMMON}
- `type` ∈ {{{{ "relation" }}}}

{_RELATION_QUERY_PROTOCOL}

{_OUTPUT_RULES}

{_FINAL_REQUIREMENT}""".strip()


def get_prompt_template(env_type) -> str:
    """
    Get prompt template for the given environment type.

    Args:
        env_type: Environment type (NORMAL, ONLY_FACT, ONLY_RELATION) - can be EnvType enum or string

    Returns:
        Prompt template string with placeholders for:
        - {houses}: JSON list of house IDs
        - {attrs}: JSON list of attribute names
        - {domain}: JSON dict of attribute -> values
        - {header}: JSON list of header columns
        - {N}: Number of houses
    """
    # Handle both EnvType enum and string values
    env_value = env_type.value if hasattr(env_type, 'value') else str(env_type)

    if env_value == "only_fact":
        return ONLY_FACT_PROMPT
    elif env_value == "only_relation":
        return ONLY_RELATION_PROMPT
    return NORMAL_PROMPT
