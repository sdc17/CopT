#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM interaction utilities.
Supports Gemini API, Together AI API, and OpenAI API.
"""

import os
import re
from typing import Optional
from openai import OpenAI

# Try to import Together AI
try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Model name aliases for Together AI
# Maps short names to full provider/model paths
TOGETHER_ALIASES = {
    "Qwen3-235B": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "Llama-4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}

# OpenAI model patterns (for auto-detection)
OPENAI_MODEL_PATTERNS = ["gpt-4", "gpt-5", "o1", "o3", "o4"]


def _extract_content_from_choice(choice) -> Optional[str]:
    """Extract content from various response formats."""
    msg = getattr(choice, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if content:
            return content
        if isinstance(msg, dict) and "content" in msg:
            return msg["content"]
    text = getattr(choice, "text", None)
    if text:
        return text
    if isinstance(choice, dict):
        if "message" in choice and isinstance(choice["message"], dict):
            if "content" in choice["message"]:
                return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
    return None


def llm_generate_gemini(model: str, messages: list[dict], max_retries: int = 3) -> str:
    """
    Call LLM API via Gemini with retry logic.

    Args:
        model: Model name to use
        messages: List of message dicts with 'role' and 'content'
        max_retries: Number of retries on failure

    Returns:
        Generated content string, or empty string on failure
    """
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=GEMINI_OPENAI_BASE_URL,
    )
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
        except Exception as e:
            last_err = f"Chat API error: {e}"
            continue
        choices = getattr(resp, "choices", None) or (resp.get("choices") if isinstance(resp, dict) else None)
        if not choices:
            last_err = f"No choices in response: {resp}"
            continue
        content = _extract_content_from_choice(choices[0])
        if content:
            return content
        last_err = f"Unsupported response shape, first choice = {choices[0]!r}"
    return ""


def llm_generate_openai(model: str, messages: list[dict], max_retries: int = 3) -> str:
    """
    Call OpenAI API using the responses endpoint with retry logic.

    Args:
        model: Model name to use (e.g., "gpt-5.2", "gpt-4o", "o3")
        messages: List of message dicts with 'role' and 'content'
        max_retries: Number of retries on failure

    Returns:
        Generated content string, or empty string on failure

    Environment variables:
        - OPENAI_API_KEY: Required for OpenAI API
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    last_err = None

    # Convert messages to input string for responses API
    # The responses API uses a simpler input format
    input_text = "\n".join([
        f"{msg['role']}: {msg['content']}" for msg in messages
    ])

    for _ in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=input_text
            )
        except Exception as e:
            last_err = f"OpenAI API error: {e}"
            continue

        # Extract content from response
        try:
            output_text = getattr(response, 'output_text', None)
            if output_text:
                return output_text
            # Fallback: check for other response formats
            if hasattr(response, 'output'):
                return str(response.output)
            last_err = f"No output_text in response: {response}"
        except Exception as e:
            last_err = f"Error extracting content: {e}"
            continue

    print(f"OpenAI API failed after {max_retries} retries. Last error: {last_err}")
    return ""


def llm_generate_together(model: str, messages: list[dict], max_retries: int = 3) -> str:
    """
    Call LLM API via Together AI with retry logic.

    Args:
        model: Model name to use (e.g., "meta-llama/Llama-3.3-70B-Instruct")
        messages: List of message dicts with 'role' and 'content'
        max_retries: Number of retries on failure

    Returns:
        Generated content string, or empty string on failure

    Environment variables:
        - TOGETHER_API_KEY: Required for Together AI API
    """
    if not TOGETHER_AVAILABLE:
        raise ImportError("Together AI support not available. Please install: pip install together")

    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    last_err = None

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
        except Exception as e:
            last_err = f"Together API error: {e}"
            continue

        # Extract content from response
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                if content:
                    return content
            last_err = f"No content in response: {response}"
        except Exception as e:
            last_err = f"Error extracting content: {e}"
            continue

    print(f"Together API failed after {max_retries} retries. Last error: {last_err}")
    return ""


def llm_generate(model: str, messages: list[dict], max_retries: int = 3, use_together: bool = False, use_openai: bool = False) -> str:
    """
    Call LLM API with retry logic. Supports Gemini, Together AI, and OpenAI.

    Args:
        model: Model name to use (can use short alias or full path)
        messages: List of message dicts with 'role' and 'content'
        max_retries: Number of retries on failure
        use_together: If True, use Together AI API
        use_openai: If True, use OpenAI API

    Returns:
        Generated content string, or empty string on failure

    Environment variables:
        - LLM_API: Set to 'together', 'openai', or 'gemini' (default) to select API
        - GEMINI_API_KEY: Required for Gemini API
        - TOGETHER_API_KEY: Required for Together AI API
        - OPENAI_API_KEY: Required for OpenAI API

    Model name detection (case-insensitive):
        - Models containing "llama", "qwen", "mistral", "deepseek", or "mixtral"
          automatically use Together AI
        - Models starting with "gpt-4", "gpt-5", "o1", "o3", "o4"
          automatically use OpenAI API
        - Examples: "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2-7B", "gpt-5.2", etc.
        - Can be overridden with use_together, use_openai, or LLM_API environment variable

    Model name aliases:
        - Short names are automatically mapped to full paths via TOGETHER_ALIASES
        - Examples: "Llama-3.3-70B-Instruct-Turbo" -> "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    """
    # Apply model name alias if available
    model = TOGETHER_ALIASES.get(model, model)

    # Check environment variable to determine API
    api_env = os.getenv("LLM_API", "gemini").lower()

    # Auto-detect Together AI models by name pattern (case-insensitive)
    model_lower = model.lower()
    is_together_model = ("llama" in model_lower or "qwen" in model_lower or
                         "oss" in model_lower or "deepseek" in model_lower or
                         "mixtral" in model_lower)

    # Auto-detect OpenAI models by name pattern
    is_openai_model = any(model_lower.startswith(pattern) for pattern in OPENAI_MODEL_PATTERNS)

    if use_openai or api_env == "openai" or is_openai_model:
        return llm_generate_openai(model, messages, max_retries)
    elif use_together or api_env == "together" or is_together_model:
        if not TOGETHER_AVAILABLE:
            raise ImportError("Together AI support not available. Please install: pip install together")
        return llm_generate_together(model, messages, max_retries)
    else:
        return llm_generate_gemini(model, messages, max_retries)


def normalize_llm_output(raw: str) -> str:
    """
    Normalize messy LLM outputs with minimal, targeted fixes.

    Handles:
    - Malformed closing query tags
    - Unbalanced braces in query blocks
    - Stray single character replies
    """
    if not isinstance(raw, str):
        return ""
    s = raw

    # Fix common malformed closing query tag: "</query" -> "</query>"
    s = re.sub(r"</\s*query\s*$", "</query>", s, flags=re.IGNORECASE | re.MULTILINE)
    s = re.sub(r"</\s*query\s*(\n|\r|\s)*$", "</query>", s, flags=re.IGNORECASE)

    # If inside <query>...</query> braces are unbalanced, gently close with extra "}"
    def _fix_unbalanced_query(m):
        inner = m.group(1)
        opens = inner.count("{")
        closes = inner.count("}")
        if opens > closes:
            inner = inner + "}" * (opens - closes)
        return f"<query>{inner}</query>"

    s = re.sub(r"<\s*query\s*>(.*?)<\s*/\s*query\s*>", _fix_unbalanced_query,
               s, flags=re.IGNORECASE | re.DOTALL)

    # Drop a stray single ">" reply
    if s.strip() == ">":
        return ""

    return s
