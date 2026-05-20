import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


def _contains_percent_unit(text: str) -> bool:
    if text is None:
        return False
    text = str(text)
    return (
        r"\%" in text
        or "%" in text
        or re.search(r"(?i)\bpercent(?:age)?\b", text) is not None
    )


def _strip_percent_unit(text: str) -> str:
    if text is None:
        return ""

    cleaned = str(text).strip()
    cleaned = cleaned.replace(r"\%", "%")

    text_wrapper_match = re.fullmatch(r"\\text\{(.+)\}", cleaned)
    if text_wrapper_match:
        cleaned = text_wrapper_match.group(1).strip()

    cleaned = re.sub(r"(?i)\bpercentage\b", "", cleaned)
    cleaned = re.sub(r"(?i)\bpercent\b", "", cleaned)
    cleaned = cleaned.replace("%", "")
    return cleaned.strip()


def _default_math_answer_extraction_config():
    return [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                boxed="all",
                units=True,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        ),
        ExprExtractionConfig(),
    ]


def _extract_str_answer(s: str):
    if s is None:
        return ""
    text = s.strip()

    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx != -1:
        start = idx + len(marker)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        candidate = text[start : j - 1].strip()

        if candidate.startswith(r"\text{") and candidate.endswith("}"):
            inner = candidate[len(r"\text{") : -1].strip()
            candidate = inner

        return candidate

    m = re.search(r"[Aa]nswer(?: is|:)\s*([^\n\.]+)", text)
    if m:
        return m.group(1).strip()

    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and ln.strip() not in {"$", "$$", "\\[", "\\]"}
    ]
    if not lines:
        return ""
    cand = lines[-1]
    cand = cand.strip(" .\"'")
    return cand


def _try_percent_unit_fallback_verify(
    solution_str: str,
    ground_truth: str,
    gold_extraction_config,
    answer_extraction_config=None,
):
    if ground_truth is None:
        return False, None

    raw_answer = _extract_str_answer(solution_str)
    if not (_contains_percent_unit(raw_answer) or _contains_percent_unit(ground_truth)):
        return False, None

    normalized_gold = _strip_percent_unit(ground_truth)
    normalized_answer = _strip_percent_unit(raw_answer)
    if not normalized_gold or not normalized_answer:
        return False, None

    fallback_gold = parse(
        normalized_gold,
        extraction_config=gold_extraction_config,
    )
    fallback_answer = parse(
        normalized_answer,
        extraction_config=answer_extraction_config or _default_math_answer_extraction_config(),
        extraction_mode="first_match",
    )
    if len(fallback_gold) == 0 or len(fallback_answer) == 0:
        return False, None

    if verify(fallback_gold, fallback_answer):
        return True, str(fallback_answer)

    return False, None


def _extract_all_boxed_contents(text: str):
    if text is None:
        return []

    contents = []
    marker = r"\boxed{"
    start_idx = 0
    while True:
        idx = text.find(marker, start_idx)
        if idx == -1:
            break

        start = idx + len(marker)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1

        if depth == 0:
            contents.append(text[start : j - 1].strip())
            start_idx = j
        else:
            break

    return contents


def _normalize_multiple_choice_label(text: str, valid_choices="ABCD"):
    if text is None:
        return None

    cleaned = str(text).strip()
    text_wrapper_match = re.fullmatch(r"\\text\{(.+)\}", cleaned)
    if text_wrapper_match:
        cleaned = text_wrapper_match.group(1).strip()

    cleaned = cleaned.strip("$")
    cleaned = cleaned.strip("()[]{} .,:;\"'")
    cleaned = cleaned.strip("*")
    cleaned = cleaned.upper()

    if len(cleaned) == 1 and cleaned in valid_choices:
        return cleaned
    return None


def _extract_multiple_choice_label(text: str, valid_choices="ABCD"):
    if text is None:
        return None

    for boxed_content in reversed(_extract_all_boxed_contents(text)):
        label = _normalize_multiple_choice_label(boxed_content, valid_choices=valid_choices)
        if label is not None:
            return label

    matches = list(
        re.finditer(
            rf"(?i)(?:final answer|answer|option|choice)(?:\s+is|:)?\s*([{valid_choices}])\b",
            text,
        )
    )
    if matches:
        return matches[-1].group(1).upper()

    return None


def gsm8k_grader(solution_str: str, ground_truth: str) -> bool:
    gold = parse(
        ground_truth,
        extraction_config=[ExprExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"

    is_correct = verify(gold, answer)
    if is_correct:
        return True, str(answer)

    fallback_correct, fallback_answer = _try_percent_unit_fallback_verify(
        solution_str,
        ground_truth,
        gold_extraction_config=[ExprExtractionConfig()],
    )
    if fallback_correct:
        return True, fallback_answer

    return False, str(answer)


def math500_grader(solution_str: str, ground_truth: str) -> bool:
    if not ground_truth.startswith("$"):
        ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[LatexExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"

    is_correct = verify(gold, answer)
    if is_correct:
        return True, str(answer)

    fallback_correct, fallback_answer = _try_percent_unit_fallback_verify(
        solution_str,
        ground_truth,
        gold_extraction_config=[LatexExtractionConfig()],
    )
    if fallback_correct:
        return True, fallback_answer

    return False, str(answer)


def aime_grader(solution_str: str, ground_truth: str) -> bool:
    gold = parse(
        ground_truth,
        extraction_config=[ExprExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    return verify(gold, answer), str(answer)


def gpqa_grader(solution_str: str, ground_truth: str) -> bool:
    pred_choice = _extract_multiple_choice_label(solution_str, valid_choices="ABCD")
    gold_choice = _extract_multiple_choice_label(ground_truth, valid_choices="ABCD")
    if pred_choice is not None and gold_choice is not None:
        return pred_choice == gold_choice, pred_choice

    if not ground_truth.startswith("$"):
        ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[LatexExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    return verify(gold, answer), str(answer)


def answer_match(dataset_name, pred, gold):
    if dataset_name == "gsm8k":
        return gsm8k_grader(pred, gold)
    elif dataset_name == "math500":
        return math500_grader(pred, gold)
    elif "aime" in dataset_name:
        return aime_grader(pred, gold)
    elif dataset_name == "gpqa_diamond":
        return gpqa_grader(pred, gold)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def answer_extraction(pred):
    return gsm8k_grader(pred, None)
