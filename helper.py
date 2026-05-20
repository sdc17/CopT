import json
import os
import re
import sys
from datasets import Dataset


ZEBRA_ARENA_ENV_TYPES = {"normal", "only_fact", "only_relation"}


def load_default_effort(dataset_name):
    config_path = os.path.join("config", f"default_effort_{dataset_name}.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_dumps_field(value):
    return json.dumps(value, ensure_ascii=False)


def _json_loads_field(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _clean_thinking_text(text):
    return text.replace("<think>", "").replace("</think>", "").strip()


def _clean_answer_text(text, tokenizer=None):
    cleaned = text.strip()
    if tokenizer is not None:
        for token in [tokenizer.eos_token, tokenizer.pad_token]:
            if token:
                cleaned = cleaned.replace(token, "")
    for token in ["<|im_end|>", "<|endoftext|>"]:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def split_thinking_answer(output_ids, tokenizer, end_thinking_id=None):
    if isinstance(end_thinking_id, int) and end_thinking_id >= 0 and end_thinking_id in output_ids:
        index = len(output_ids) - output_ids[::-1].index(end_thinking_id)
        thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=False)
        answer = tokenizer.decode(output_ids[index:], skip_special_tokens=False)
        return _clean_thinking_text(thinking), _clean_answer_text(answer, tokenizer)

    raw = tokenizer.decode(output_ids, skip_special_tokens=False)
    marker = "</think>"
    if marker in raw:
        thinking, answer = raw.rsplit(marker, 1)
        return _clean_thinking_text(thinking), _clean_answer_text(answer, tokenizer)
    return "", _clean_answer_text(raw, tokenizer)


def trim_output_ids(output_ids, tokenizer):
    trimmed = list(output_ids)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return trimmed
    while trimmed and trimmed[-1] == pad_token_id:
        trimmed.pop()
    return trimmed


def _zebra_arena_root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ZebraArena")


def _ensure_zebra_arena_path():
    root = _zebra_arena_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _zebra_solution_to_dict(solution):
    _ensure_zebra_arena_path()
    from extract.grid_reward import solution_numpy_to_dict

    return solution_numpy_to_dict(solution)


def load_zebra_arena_dataset(data_dir, miss_num=1, space="Small"):
    import pandas as pd

    filename = f"filtered_puzzles_missing{miss_num}_{space}.parquet"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ZebraArena file not found: {path}")

    df = pd.read_parquet(path)
    records = []
    for row in df.to_dict(orient="records"):
        solution = _zebra_solution_to_dict(row["solution"])
        records.append({
            "id": row["id"],
            "puzzle": row.get("missing_puzzle", row.get("puzzle", "")),
            "solution": _json_dumps_field(solution),
            "missing_clue_number": int(row.get("missing_clue_number", 0)),
            "total_clue_number": int(row.get("total_clue_number", 0)),
            "space": str(row.get("space", space)),
            "size": str(row.get("size", "")),
        })
    if not records:
        raise ValueError(f"No ZebraArena records were loaded from {path}")
    return Dataset.from_list(records)


def _zebra_env_type_value(env_type):
    env_type = str(env_type or "normal").strip()
    if env_type not in ZEBRA_ARENA_ENV_TYPES:
        raise ValueError(f"Unsupported ZebraArena env_type: {env_type}")
    return env_type


def _zebra_load_components(gold, env_type):
    _ensure_zebra_arena_path()
    from env.response_server import load_solution
    from env.scheme import build_schemas

    env_type = _zebra_env_type_value(env_type)
    gt = gold["solution"]
    houses, attributes, solution, attr_alias, value_alias = load_solution(gt)
    schemas = build_schemas(env_type, houses, attributes)
    return gt, houses, attributes, solution, attr_alias, value_alias, schemas


def build_zebra_arena_messages(gold, env_type="normal"):
    _ensure_zebra_arena_path()
    from prompts.base_prompt import get_prompt_template

    gt, houses, attributes, *_ = _zebra_load_components(gold, env_type)
    template = get_prompt_template(_zebra_env_type_value(env_type))
    prompt = template.format(
        houses=json.dumps(houses, ensure_ascii=False),
        attrs=json.dumps(list(attributes.keys()), ensure_ascii=False),
        domain=json.dumps({a: list(vs) for a, vs in attributes.items()}, ensure_ascii=False),
        header=json.dumps(gt["header"], ensure_ascii=False),
        N=len(houses),
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": gold["puzzle"]},
    ]


def normalize_zebra_arena_output(output):
    _ensure_zebra_arena_path()
    from core.llm import normalize_llm_output

    return normalize_llm_output(output)


def prune_zebra_arena_messages(messages, max_input_tokens=12000, keep_tail=8):
    _ensure_zebra_arena_path()
    from utils.prune_messages import prune_messages
    from utils.prune_messages import messages_token_count

    def _drop_empty_assistant(items):
        return [
            m for m in items
            if not (m.get("role") == "assistant" and not str(m.get("content", "")).strip())
        ]

    messages = _drop_empty_assistant(messages)
    if messages_token_count(messages) <= max_input_tokens:
        return messages
    return _drop_empty_assistant(
        prune_messages(messages, max_input_tokens=max_input_tokens, keep_tail=keep_tail)
    )


def zebra_arena_history_content(pred, round_thinking="", action_text=""):
    pred = str(pred or "").strip()
    round_thinking = str(round_thinking or "").strip()
    action_text = str(action_text or "").strip()

    if round_thinking:
        if action_text:
            return f"<think>\n{round_thinking}\n</think>\n\n{action_text}"
        return f"<think>\n{round_thinking}\n</think>"

    tag_match = re.search(
        r"<\s*(query|solution|answer)\s*>.*?<\s*/\s*\1\s*>",
        pred,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tag_match:
        tag_pos = tag_match.start()
        reasoning = _clean_thinking_text(pred[:tag_pos])
        action = pred[tag_pos:].strip()
        if reasoning and action:
            return f"<think>\n{reasoning}\n</think>\n\n{action}"
        return action or pred

    if pred:
        if re.search(r"<\s*think\s*>", pred, flags=re.IGNORECASE):
            if re.search(r"<\s*/\s*think\s*>", pred, flags=re.IGNORECASE):
                return pred
            return f"{pred}\n</think>"
        return f"<think>\n{pred}\n</think>"
    return pred


def zebra_arena_open_think_history(text):
    text = str(text or "").strip()
    if not text:
        return text
    text = re.sub(r"<\s*/\s*think\s*>", "", text, flags=re.IGNORECASE).strip()
    if re.search(r"<\s*think\s*>", text, flags=re.IGNORECASE):
        return text
    return f"<think>\n{text}"


def is_qwen3_zebra_legacy_template(dataset_name, model_name):
    return (
        dataset_name == "zebra_arena"
        and "Qwen3" in model_name
        and "Qwen3.5" not in model_name
    )


def zebra_arena_query_history_content(output, canonical_query):
    text = str(output or "").strip()
    action_match = re.search(
        r"<\s*(query|solution|answer)\s*>.*?<\s*/\s*\1\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    reasoning_source = text[:action_match.start()] if action_match else text
    reasoning = _clean_thinking_text(reasoning_source)
    query_text = f"<query>{json.dumps(canonical_query, ensure_ascii=False)}</query>"
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{query_text}"
    return query_text


def _zebra_extract_query_dicts(output):
    _ensure_zebra_arena_path()
    from extract.extract_query import _extract_from_fenced, _extract_first_balanced_json, _try_parse_json

    text = str(output or "")
    candidates = []
    for match in re.finditer(r"<\s*query\s*>(.*?)(?:<\s*/\s*query\s*>|$)", text, flags=re.IGNORECASE | re.DOTALL):
        inner = match.group(1)
        candidate = _extract_from_fenced(inner) or _extract_first_balanced_json(inner)
        if not candidate:
            continue
        obj = _try_parse_json(candidate)
        if isinstance(obj, dict) and isinstance(obj.get("query"), dict):
            obj = obj["query"]
        if isinstance(obj, dict) and obj.get("type") in {"fact", "relation"}:
            candidates.append(obj)
    return candidates


def _zebra_value_attr_lookup(attributes):
    lookup = {}
    ambiguous = set()
    for attr, values in attributes.items():
        for value in values:
            key = str(value).strip().lower()
            if key in lookup and lookup[key][0] != attr:
                ambiguous.add(key)
            else:
                lookup[key] = (attr, value)
    for key in ambiguous:
        lookup.pop(key, None)
    return lookup


def _zebra_repair_entity(entity, attributes):
    if not isinstance(entity, dict):
        return entity
    repaired = dict(entity)
    attr_alias = {str(attr).strip().lower(): attr for attr in attributes}
    value_lookup = _zebra_value_attr_lookup(attributes)
    attr_raw = str(repaired.get("attr", "")).strip()
    value_raw = str(repaired.get("value", "")).strip()
    attr_key = attr_raw.lower()
    value_key = value_raw.lower()

    if attr_key not in attr_alias and attr_key in value_lookup:
        attr, value = value_lookup[attr_key]
        repaired["attr"] = attr
        repaired["value"] = value
        return repaired

    if attr_key in attr_alias:
        canon_attr = attr_alias[attr_key]
        canon_values = {str(v).strip().lower(): v for v in attributes.get(canon_attr, [])}
        if value_key not in canon_values and value_key in value_lookup:
            attr, value = value_lookup[value_key]
            repaired["attr"] = attr
            repaired["value"] = value
            return repaired

    return repaired


def _zebra_house_from_entity(entity, houses):
    if not isinstance(entity, dict):
        return None
    attr = str(entity.get("attr", "")).strip().lower()
    value = str(entity.get("value", "")).strip()
    if attr not in {"house", "houses", "position", "pos"}:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return None
    house = f"h{digits}"
    return house if house in houses else None


def _zebra_house_index(house):
    digits = "".join(ch for ch in str(house or "") if ch.isdigit())
    return int(digits) if digits else None


def _zebra_entity_house(solution, entity):
    attr = entity.get("attr")
    value = entity.get("value")
    for house, mapping in solution.items():
        if mapping.get(attr) == value:
            return house
    return None


def _zebra_answer_house_relation_query(query, houses, attributes, solution, canonicalize_query):
    rel = query.get("rel")
    if rel not in {"same_house", "not_at", "direct_left", "direct_right", "side_by_side", "left_of", "right_of", "one_between", "two_between"}:
        return None
    lhs = query.get("lhs")
    rhs = query.get("rhs")
    lhs_house = _zebra_house_from_entity(lhs, houses)
    rhs_house = _zebra_house_from_entity(rhs, houses)
    if bool(lhs_house) == bool(rhs_house):
        return None

    entity_side = rhs if lhs_house else lhs
    house_side = lhs_house or rhs_house
    entity_query = {"type": "relation", "rel": "same_house", "lhs": entity_side, "rhs": entity_side}
    entity_query["lhs"] = _zebra_repair_entity(entity_query["lhs"], attributes)
    canon_entity, cerr = canonicalize_query(entity_query, houses, attributes)
    if cerr:
        return None
    entity = canon_entity["lhs"]
    entity_house = _zebra_entity_house(solution, entity)
    entity_pos = _zebra_house_index(entity_house)
    house_pos = _zebra_house_index(house_side)
    if entity_pos is None or house_pos is None:
        answer = None
    elif rel == "same_house":
        answer = entity_house == house_side
    elif rel == "not_at":
        answer = entity_house != house_side
    elif rel == "direct_left":
        answer = (house_pos + 1 == entity_pos) if lhs_house else (entity_pos + 1 == house_pos)
    elif rel == "direct_right":
        answer = (entity_pos + 1 == house_pos) if lhs_house else (house_pos + 1 == entity_pos)
    elif rel == "side_by_side":
        answer = abs(entity_pos - house_pos) == 1
    elif rel == "left_of":
        answer = house_pos < entity_pos if lhs_house else entity_pos < house_pos
    elif rel == "right_of":
        answer = house_pos > entity_pos if lhs_house else entity_pos > house_pos
    elif rel == "one_between":
        answer = abs(entity_pos - house_pos) == 2
    else:
        answer = abs(entity_pos - house_pos) == 3

    canonical = {
        "type": "relation",
        "rel": rel,
        "lhs": {"attr": "House", "value": house_side} if lhs_house else entity,
        "rhs": entity if lhs_house else {"attr": "House", "value": house_side},
    }
    result = {**canonical, "ok": True, "answer": answer}
    return canonical, result


def _zebra_repair_query(query, houses, attributes):
    if not isinstance(query, dict):
        return query
    repaired = dict(query)
    if repaired.get("type") == "relation" and repaired.get("rel") == "found_at" and "house" in repaired:
        repaired["type"] = "fact"
    if repaired.get("type") == "fact":
        return repaired
    if repaired.get("type") == "relation":
        if isinstance(repaired.get("lhs"), dict):
            repaired["lhs"] = _zebra_repair_entity(repaired["lhs"], attributes)
        if isinstance(repaired.get("rhs"), dict):
            repaired["rhs"] = _zebra_repair_entity(repaired["rhs"], attributes)
    return repaired


def _zebra_env_response(result):
    if result.get("ok") and "answer" in result:
        truth_value = "TRUE" if result.get("answer") else "FALSE"
        return (
            f"Environment answer: {truth_value}. "
            "`ok` only means the query was valid; `answer` is the truth value. "
            f"Raw result: {json.dumps(result, ensure_ascii=False)}"
        )
    return f"Environment answer: {json.dumps(result, ensure_ascii=False)}"


def parse_zebra_arena_response(output, gold, env_type="normal"):
    _ensure_zebra_arena_path()
    from jsonschema import validate, ValidationError
    from env.scheme import canonicalize_query
    from env.response_server import answer_query
    from extract.extract_query import extract_answer_dict, extract_query_dict
    from extract.grid_reward import check_final_solution

    gt, houses, attributes, solution, attr_alias, value_alias, schemas = _zebra_load_components(gold, env_type)
    first_attr = next(iter(attributes), "Name")
    first_value = attributes[first_attr][0] if attributes.get(first_attr) else ""
    example_fact = {
        "type": "fact",
        "rel": "found_at",
        "house": houses[0] if houses else "h1",
        "attr": first_attr,
        "value": first_value,
    }
    example_message = (
        "Example valid fact query: "
        f"<query>{json.dumps(example_fact, ensure_ascii=False)}</query>. "
        "Use ATTRS as attr names and DOMAIN values as value strings."
    )

    answer = extract_answer_dict(output)
    if answer is not None and isinstance(answer.get("rows"), list) and len(answer["rows"]) == len(houses):
        is_correct, result_detail = check_final_solution(answer, gt)
        return {
            "type": "solution",
            "answer": answer,
            "correct": bool(is_correct),
            "prediction": result_detail,
        }

    query_candidates = _zebra_extract_query_dicts(output)
    query = query_candidates[0] if query_candidates else extract_query_dict(output)
    if query is None:
        if re.search(r"<\s*query\s*>", str(output or ""), flags=re.IGNORECASE):
            return {
                "type": "invalid",
                "message": (
                    "Your last <query> was invalid. The query content must be a single JSON object "
                    "following the fact or relation schema from the system prompt. Do not ask natural "
                    "language questions inside <query>. Retry with exactly one valid JSON <query> or "
                    f"a complete JSON <solution>. {example_message}"
                ),
            }
        if str(output or "").strip():
            return {"type": "reasoning"}
        return {
            "type": "invalid",
            "append_assistant": False,
            "message": "Your last message did not contain a valid <query> or <solution>. Please retry with exactly one valid tag.",
        }

    last_query = query
    last_error = None
    for raw_query in query_candidates or [query]:
        candidate = _zebra_repair_query(raw_query, houses, attributes)
        last_query = candidate
        query_type = candidate.get("type")
        if query_type not in schemas:
            last_error = f"Query type '{query_type}' is not allowed in {env_type} mode."
            continue

        canon, cerr = canonicalize_query(candidate, houses, attributes)
        if cerr:
            house_answer = _zebra_answer_house_relation_query(candidate, houses, attributes, solution, canonicalize_query)
            if house_answer is not None:
                canon, result = house_answer
                env_response = _zebra_env_response(result)
                return {
                    "type": "query",
                    "query": raw_query,
                    "canonical_query": canon,
                    "result": result,
                    "env_response": env_response,
                }
            last_error = f"canonicalize failed: {cerr}"
            continue

        try:
            validate(instance=canon, schema=schemas[canon["type"]])
        except ValidationError as e:
            last_error = f"Query validation failed: {e.message}"
            continue

        result = answer_query(canon, solution, attributes, houses, attr_alias, value_alias)
        env_response = _zebra_env_response(result)
        return {
            "type": "query",
            "query": raw_query,
            "canonical_query": canon,
            "result": result,
            "env_response": env_response,
        }

    return {
        "type": "invalid",
        "query": last_query,
        "message": f"The last JSON was invalid: {last_error}. Retry. {example_message}",
    }
