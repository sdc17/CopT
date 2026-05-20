#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic solver for Zebra puzzle - no special constraints.
This is the simplest solver for baseline evaluation.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from core.base_solver import BaseSolver, EnvType
from prompts.base_prompt import get_prompt_template


class BasicSolver(BaseSolver):
    """
    Basic solver without any special constraints.
    Used for baseline evaluation experiments.
    """

    def __init__(self, env_type: EnvType, model_name: str, **kwargs):
        """
        Initialize basic solver.

        Args:
            env_type: Environment type (NORMAL, ONLY_FACT, ONLY_RELATION)
            model_name: Name of the LLM model to use
            **kwargs: Additional arguments passed to BaseSolver
        """
        super().__init__(env_type=env_type, model_name=model_name, **kwargs)

    def get_prompt_template(self) -> str:
        """Return prompt template based on environment type."""
        return get_prompt_template(self.env_type)
