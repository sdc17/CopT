#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience script to run only_relation evaluation.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from zebrapuzzle.experiments.exp1_basic.run import main

if __name__ == "__main__":
    # Override default to only_relation
    sys.argv.extend(["--env_type", "only_relation"])
    main()
