#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utility functions.
"""

import numpy as np
from typing import Any


def safe_serialize(obj: Any) -> Any:
    """
    Recursively convert numpy arrays and other non-JSON-serializable
    objects to JSON-serializable format.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
