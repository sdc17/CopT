#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema definitions and query canonicalization for Zebra puzzle.
Supports both fact and relation queries with case/whitespace normalization.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from jsonschema import validate, ValidationError
from enum import Enum

# Add zebrapuzzle root to path for imports
_ZEBRAPUZZLE_ROOT = Path(__file__).parent.parent
import sys
if str(_ZEBRAPUZZLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZEBRAPUZZLE_ROOT))


# Define EnvType locally to avoid circular imports
class EnvType(Enum):
    """Environment type determining which query types are allowed."""
    NORMAL = "normal"
    ONLY_FACT = "only_fact"
    ONLY_RELATION = "only_relation"


# Allowed binary relations
REL_BINARY = {
    "same_house",
    "not_at",
    "direct_left", "direct_right",
    "side_by_side",
    "left_of", "right_of",
    "one_between", "two_between",
}


def build_alias_maps(attributes: dict):
    """
    Build case-insensitive alias maps for attributes and values.

    Returns:
        Tuple of (attr_alias, value_alias)
        - attr_alias: {lower_strip_attr -> canonical_attr}
        - value_alias: {canonical_attr: {lower_strip_value -> canonical_value}}
    """
    attr_alias = {}
    value_alias = {}
    for a, vs in attributes.items():
        attr_alias[a.strip().lower()] = a
        vmap = {}
        for v in vs:
            vmap[v.strip().lower()] = v
        value_alias[a] = vmap
    return attr_alias, value_alias


def normalize_house(house_raw: str, houses: list) -> str:
    """
    Normalize house ID with case/whitespace tolerance.

    Args:
        house_raw: Raw house string (e.g., " H3 ")
        houses: List of valid house IDs

    Returns:
        Normalized house ID (e.g., "h3")

    Raises:
        ValueError: If house is not in valid list
    """
    if not isinstance(house_raw, str):
        raise ValueError(f"house must be string, got {type(house_raw)}")
    h_norm = "h" + "".join(ch for ch in house_raw if ch.isdigit())
    if h_norm not in houses:
        raise ValueError(f"unknown house: {house_raw!r} (normalized to {h_norm!r}, not in {houses})")
    return h_norm


def normalize_entity(ent: dict, attr_alias: dict, value_alias: dict) -> dict:
    """
    Normalize entity {"attr":..., "value":...} to canonical form.

    Args:
        ent: Entity dict with 'attr' and 'value' keys
        attr_alias: Attribute alias map
        value_alias: Value alias map

    Returns:
        Normalized entity dict

    Raises:
        ValueError: If attr or value is invalid
    """
    if not isinstance(ent, dict):
        raise ValueError("entity must be an object")
    if "attr" not in ent or "value" not in ent:
        raise ValueError("entity must contain 'attr' and 'value'")

    a_key = str(ent["attr"]).strip().lower()
    if a_key not in attr_alias:
        raise ValueError(f"unknown attr: {ent['attr']!r}")

    a_canon = attr_alias[a_key]
    v_key = str(ent["value"]).strip().lower()
    vmap = value_alias.get(a_canon, {})
    if v_key not in vmap:
        raise ValueError(f"value {ent['value']!r} not in domain of {a_canon!r}: {list(vmap.values())}")
    v_canon = vmap[v_key]
    return {"attr": a_canon, "value": v_canon}


def canonicalize_query(query: dict, houses: list, attributes: dict):
    """
    Canonicalize a query with case/whitespace normalization.

    Args:
        query: Raw query dict
        houses: List of valid house IDs
        attributes: Attribute definitions

    Returns:
        Tuple of (canonicalized_query, error_string)
        - If success: (canon_query, None)
        - If failure: (None, error_message)
    """
    try:
        q = deepcopy(query)
        attr_alias, value_alias = build_alias_maps(attributes)

        if q.get("type") == "fact":
            rel = q.get("rel")
            if rel != "found_at":
                return None, f"unknown relation: {rel!r}"
            if "house" not in q or "attr" not in q or "value" not in q:
                return None, "fact query must contain 'house','attr','value'"
            q["house"] = normalize_house(q["house"], houses)
            ent = {"attr": q["attr"], "value": q["value"]}
            ent_n = normalize_entity(ent, attr_alias, value_alias)
            q["attr"], q["value"] = ent_n["attr"], ent_n["value"]
            return q, None

        elif q.get("type") == "relation":
            rel = q.get("rel")
            if rel not in REL_BINARY:
                return None, f"unknown relation: {rel!r}"
            if "lhs" not in q or "rhs" not in q:
                return None, f"relation {rel} requires 'lhs' and 'rhs'"
            q["lhs"] = normalize_entity(q["lhs"], attr_alias, value_alias)
            q["rhs"] = normalize_entity(q["rhs"], attr_alias, value_alias)
            return q, None

        else:
            return None, f"unknown query type: {q.get('type')!r}"

    except Exception as e:
        return None, str(e)


def _entity_schema_per_attr(attributes: dict):
    """Build entity schema with per-attribute value constraints."""
    return {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "attr": {"const": attr},
                    "value": {"type": "string", "enum": values}
                },
                "required": ["attr", "value"],
                "additionalProperties": False
            }
            for attr, values in attributes.items()
        ]
    }


def build_fact_schema(houses: list, attributes: dict) -> dict:
    """
    Build JSON schema for fact queries.

    Args:
        houses: List of valid house IDs
        attributes: Attribute definitions

    Returns:
        JSON schema dict
    """
    fact = {
        "type": "object",
        "properties": {
            "type": {"const": "fact"},
            "rel": {"const": "found_at"},
            "house": {"type": "string", "enum": houses},
            "attr": {"type": "string"},
            "value": {"type": "string"}
        },
        "required": ["type", "house", "attr", "value"],
        "additionalProperties": False,
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "attr": {"const": attr},
                    "value": {"type": "string", "enum": values}
                },
                "required": ["attr", "value"]
            }
            for attr, values in attributes.items()
        ]
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "FactQuery",
        "type": "object",
        "oneOf": [fact]
    }


def build_relation_schema(houses: list, attributes: dict) -> dict:
    """
    Build JSON schema for relation queries.

    Args:
        houses: List of valid house IDs
        attributes: Attribute definitions

    Returns:
        JSON schema dict
    """
    entity_def = _entity_schema_per_attr(attributes)

    rel_binary_names = [
        "left_of", "right_of", "direct_left", "direct_right",
        "side_by_side", "same_house", "not_at",
        "one_between", "two_between"
    ]

    rel_schemas = [
        {
            "type": "object",
            "properties": {
                "type": {"const": "relation"},
                "rel": {"const": r},
                "lhs": entity_def,
                "rhs": entity_def
            },
            "required": ["type", "rel", "lhs", "rhs"],
            "additionalProperties": False
        }
        for r in rel_binary_names
    ]

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "RelationQuery",
        "type": "object",
        "oneOf": rel_schemas
    }


def build_schemas(env_type, houses: list, attributes: dict) -> dict:
    """
    Build schemas based on environment type.

    Args:
        env_type: Environment type (NORMAL, ONLY_FACT, ONLY_RELATION) - can be EnvType enum or string
        houses: List of valid house IDs
        attributes: Attribute definitions

    Returns:
        Dict mapping query type to schema
    """
    # Handle both EnvType enum and string values
    env_value = env_type.value if hasattr(env_type, 'value') else str(env_type)

    schemas = {}
    if env_value in ("normal", "only_fact"):
        schemas["fact"] = build_fact_schema(houses, attributes)
    if env_value in ("normal", "only_relation"):
        schemas["relation"] = build_relation_schema(houses, attributes)
    return schemas
