#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
from typing import Optional, Any, List

# ------------------ helpers ------------------

SMART_QUOTES_MAP = {
    "\u201c": '"', "\u201d": '"',  # “ ”
    "\u2018": "'", "\u2019": "'",  # ‘ ’
}

def _sanitize_json_text(s: str) -> str:
    """Replace smart quotes and trim whitespace."""
    for k, v in SMART_QUOTES_MAP.items():
        s = s.replace(k, v)
    return s.strip()

def _try_parse_json(s: str) -> Optional[Any]:
    try:
        return json.loads(_sanitize_json_text(s))
    except Exception:
        return None

def _extract_from_fenced(text: str) -> Optional[str]:
    """
    Look ONLY inside a text chunk for a fenced JSON block:
      - ```json ... ```
      - ``` ... ```
    Return inner string (without the fences), or None.
    """
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return None

def _extract_first_balanced_json(text: str) -> Optional[str]:
    """
    From a text chunk, return the FIRST balanced {...} region (simple stack).
    This is used only INSIDE <query>/<solution>/<answer> tags.
    """
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i+1]
    return None

def _extract_inner(text: str, tag_names: List[str]) -> Optional[str]:
    """
    Return inner content of the FIRST matching tag from tag_names.
    Example: tag_names=["query"] or ["answer","solution"]
    """
    for tag in tag_names:
        m = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1)
    return None

# ------------------ public API ------------------

def extract_query_dict(llm_output: str) -> Optional[dict]:
    """
    STRICT: Only extract queries from <query> ... </query>.
    - Inside the tag, try ```json ...``` or ``` ...``` fences first.
    - Otherwise, take the first balanced {...} block inside the tag.
    - Return dict if it parses AND looks like a { "type": "fact"|"relation", ... }.
    - Otherwise None.
    """
    inner = _extract_inner(llm_output, ["query"])
    if not inner:
        return None

    candidate = _extract_from_fenced(inner) or _extract_first_balanced_json(inner)
    if not candidate:
        return None

    obj = _try_parse_json(candidate)
    if not isinstance(obj, dict):
        return None

    # Accept either raw query or {"query": {...}}
    if "query" in obj and isinstance(obj["query"], dict):
        obj = obj["query"]

    if obj.get("type") in {"fact", "relation"}:
        return obj
    return None


def extract_answer_dict(llm_output: str) -> Optional[dict]:
    """
    STRICT: Only extract the final solution from <answer> ... </answer> OR <solution> ... </solution>.
    - Inside the tag, try fenced JSON first; else take first balanced {...}.
    - Must parse to a dict containing "header" and "rows".
    """
    inner = _extract_inner(llm_output, ["answer", "solution"])
    if not inner:
        return None

    candidate = _extract_from_fenced(inner) or _extract_first_balanced_json(inner)
    if not candidate:
        return None

    obj = _try_parse_json(candidate)
    if isinstance(obj, dict) and "header" in obj and "rows" in obj:
        return obj
    return None

# ------------------ demo ------------------
if __name__ == "__main__":
    demo = "<think><query>\n{\n  \"type\": \"fact\",\n  \"house\": \"h2\",\n  \"attr\": \"Pet\",\n  \"value\": \"\"\n}\n</query>"

    q = extract_query_dict(demo)
    a = extract_answer_dict(demo)
    print("QUERY:", json.dumps(q, ensure_ascii=False, indent=2))
    print("ANSWER:", json.dumps(a, ensure_ascii=False, indent=2))