#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified response server for Zebra puzzle queries.
Supports both fact and relation queries.
"""

from typing import Optional, Dict, Tuple


def load_solution(obj):
    """
    Load ground truth solution from various formats.

    Args:
        obj: Solution object, either:
            - dict with 'header' and 'rows' keys
            - dict mapping house IDs to attribute dicts

    Returns:
        Tuple of (houses, attributes, solution, attr_alias, value_alias)
    """
    if isinstance(obj, dict) and "header" in obj and "rows" in obj:
        header = obj["header"]
        rows = obj["rows"]
        sol = {}
        for row in rows:
            h = f"h{row[0]}"
            sol[h] = {}
            for i, key in enumerate(header[1:], start=1):
                sol[h][key] = row[i]
    elif isinstance(obj, dict):
        sol = obj
    else:
        raise ValueError("Unsupported solution format.")

    attributes = {}
    for h, attrs in sol.items():
        for a, v in attrs.items():
            attributes.setdefault(a, set()).add(v)
    attributes = {a: sorted(list(vs)) for a, vs in attributes.items()}
    houses = sorted(sol.keys(), key=lambda x: _house_index(x))

    attr_alias, value_alias = build_canonical_maps(attributes)
    return houses, attributes, sol, attr_alias, value_alias


def _house_index(house_id: Optional[str]) -> Optional[int]:
    """Extract numeric index from house ID."""
    if not house_id:
        return None
    digits = "".join(ch for ch in house_id if ch.isdigit())
    return int(digits) if digits else None


def build_canonical_maps(attributes: Dict[str, list]):
    """
    Build case-insensitive lookup maps for attributes and values.

    Returns:
        Tuple of (attr_alias, value_alias) dicts
    """
    attr_alias = {}
    value_alias = {}
    for a, vals in attributes.items():
        attr_alias[a.strip().lower()] = a
        vmap = {}
        for v in vals:
            vmap[v.strip().lower()] = v
        value_alias[a] = vmap
    return attr_alias, value_alias


def normalize_attr_value(
    attributes: Dict[str, list],
    attr_alias: Dict[str, str],
    value_alias: Dict[str, Dict[str, str]],
    attr: str,
    value: str
) -> Tuple[str, str]:
    """Normalize attribute and value to canonical form."""
    a_key = attr.strip().lower()
    if a_key not in attr_alias:
        raise ValueError(f"unknown attr: {attr}")
    a_norm = attr_alias[a_key]

    v_key = value.strip().lower()
    vmap = value_alias.get(a_norm, {})
    if v_key not in vmap:
        raise ValueError(f"value '{value}' not in domain of '{a_norm}': {list(vmap.values())}")
    v_norm = vmap[v_key]
    return a_norm, v_norm


def normalize_entity(
    attributes: Dict[str, list],
    attr_alias: Dict[str, str],
    value_alias: Dict[str, Dict[str, str]],
    ent: Dict[str, str]
) -> Dict[str, str]:
    """Normalize entity dict to canonical form."""
    a, v = normalize_attr_value(attributes, attr_alias, value_alias, ent["attr"], ent["value"])
    return {"attr": a, "value": v}


def _house_of(
    solution: dict,
    attributes: Dict[str, list],
    attr_alias: Dict[str, str],
    value_alias: Dict[str, Dict[str, str]],
    ent: Dict[str, str]
) -> Optional[str]:
    """Find which house contains the given entity."""
    ent_norm = normalize_entity(attributes, attr_alias, value_alias, ent)
    a_norm, v_norm = ent_norm["attr"], ent_norm["value"]
    for h, mapping in solution.items():
        if mapping.get(a_norm) == v_norm:
            return h
    return None


def _pos(
    solution: dict,
    attributes: Dict[str, list],
    attr_alias: Dict[str, str],
    value_alias: Dict[str, Dict[str, str]],
    ent: Dict[str, str]
) -> Optional[int]:
    """Get position (index) of house containing the given entity."""
    h = _house_of(solution, attributes, attr_alias, value_alias, ent)
    return _house_index(h) if h else None


def eval_fact(
    solution: dict,
    attributes: Dict[str, list],
    houses: list,
    attr_alias: Dict[str, str],
    value_alias: Dict[str, Dict[str, str]],
    query: dict
) -> Optional[bool]:
    """
    Evaluate a fact query.

    Query format:
        {"type": "fact", "rel": "found_at", "house": "h1", "attr": "Color", "value": "red"}

    Returns:
        True/False indicating if the attribute-value is at the specified house
    """
    if query["rel"] != "found_at":
        raise ValueError(f"unsupported fact rel: {query['rel']}")

    house_id = query['house']
    if house_id not in houses:
        raise ValueError(f"unknown house: {house_id}")

    a_norm, v_norm = normalize_attr_value(
        attributes, attr_alias, value_alias,
        query["attr"], query["value"]
    )
    return solution.get(house_id, {}).get(a_norm) == v_norm


def eval_relation(
    solution: dict,
    attributes: Dict[str, list],
    houses: list,
    attr_alias: Dict[str, str],
    value_alias: Dict[str, Dict[str, str]],
    query: dict
) -> Optional[bool]:
    """
    Evaluate a relation query.

    Supported relations:
        - same_house: Both entities in same house
        - not_at: Entities in different houses
        - direct_left/direct_right: Adjacent positions
        - side_by_side: Adjacent (either direction)
        - left_of/right_of: Relative positioning
        - one_between: Exactly 1 house between
        - two_between: Exactly 2 houses between

    Returns:
        True/False indicating if the relation holds
    """
    rel = query["rel"]
    lhs, rhs = query.get("lhs"), query.get("rhs")

    if rel == "same_house":
        h1 = _house_of(solution, attributes, attr_alias, value_alias, lhs)
        h2 = _house_of(solution, attributes, attr_alias, value_alias, rhs)
        return (h1 is not None and h1 == h2)

    p1 = _pos(solution, attributes, attr_alias, value_alias, lhs) if lhs else None
    p2 = _pos(solution, attributes, attr_alias, value_alias, rhs) if rhs else None

    if p1 is None or p2 is None:
        return None

    if rel == "not_at":
        return p1 != p2
    if rel == "direct_left":
        return p1 + 1 == p2
    if rel == "direct_right":
        return p2 + 1 == p1
    if rel == "side_by_side":
        return abs(p1 - p2) == 1
    if rel == "left_of":
        return p1 < p2
    if rel == "right_of":
        return p1 > p2
    if rel == "one_between":
        return abs(p1 - p2) == 2
    if rel == "two_between":
        return abs(p1 - p2) == 3

    raise ValueError(f"unsupported rel: {rel}")


def answer_query(
    query: dict,
    solution: dict,
    attributes: Dict[str, list],
    houses: list,
    attr_alias: Dict[str, str],
    value_alias: Dict[str, Dict[str, str]]
) -> dict:
    """
    Execute a query and return the result.

    Args:
        query: Query dict with 'type' and other fields
        solution: Ground truth solution
        attributes: Attribute definitions
        houses: List of house IDs
        attr_alias: Attribute alias map
        value_alias: Value alias map

    Returns:
        Result dict with 'ok', 'answer' (if successful) or 'error' (if failed)
    """
    try:
        t = query["type"]
        if t == "fact":
            ans = eval_fact(solution, attributes, houses, attr_alias, value_alias, query)
            return {**query, "ok": True, "answer": ans}
        elif t == "relation":
            ans = eval_relation(solution, attributes, houses, attr_alias, value_alias, query)
            return {**query, "ok": True, "answer": ans}
        else:
            return {**query, "ok": False, "error": f"unknown query type {t}"}
    except Exception as e:
        return {**query, "ok": False, "error": str(e)}
