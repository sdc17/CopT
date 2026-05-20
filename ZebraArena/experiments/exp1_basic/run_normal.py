#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience script to run normal (fact + relation) evaluation.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from run import main

if __name__ == "__main__":
    # Override default to normal
    sys.argv.extend(["--env_type", "normal"])
    main()
