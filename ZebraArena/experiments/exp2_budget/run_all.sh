#!/bin/bash
# Run all exp2_budget experiments
#
# Budget Table (based on filtered exp1_basic normal env data):
#   Small:  K*=1: (1,2,3), K*=2: (2,5,7), K*=3: (3,8,11)
#   Medium: K*=1: (1,2,3), K*=2: (2,6,8), K*=3: (3,11,14), K*=4: (4,15,19)
#   Format: (tight, normal, relaxed)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run.py"

# Configuration
MODEL="gemini-2.5-flash"
NUM_PROCESSES=32

# Environment types (only run normal for exp2, based on exp1 analysis)
ENV_TYPES=("normal")

# Budget levels
BUDGET_LEVELS=("tight" "normal" "relaxed")

# Size and missing number combinations (must match BUDGET_TABLE in run.py)
# Small: miss_num 1, 2, 3
# Medium: miss_num 1, 2, 3, 4
SMALL_MISS_NUMS=(1 2 3)
MEDIUM_MISS_NUMS=(1 2 3 4)

echo "=============================================="
echo "Running exp2_budget experiments"
echo "Model: ${MODEL}"
echo "Processes: ${NUM_PROCESSES}"
echo "=============================================="

for budget_level in "${BUDGET_LEVELS[@]}"; do
    echo ""
    echo "====== Budget Level: ${budget_level} ======"

    # # Small puzzles
    # for miss_num in "${SMALL_MISS_NUMS[@]}"; do
    #     for env_type in "${ENV_TYPES[@]}"; do
    #         echo ""
    #         echo ">>> Running: Small / miss${miss_num} / ${env_type} / ${budget_level}"
    #         python "${PYTHON_SCRIPT}" \
    #             --env_type "${env_type}" \
    #             --budget_level "${budget_level}" \
    #             --miss_num "${miss_num}" \
    #             --space "Small" \
    #             --model "${MODEL}" \
    #             --num_processes "${NUM_PROCESSES}"
    #     done
    # done

    # Medium puzzles
    for miss_num in "${MEDIUM_MISS_NUMS[@]}"; do
        for env_type in "${ENV_TYPES[@]}"; do
            echo ""
            echo ">>> Running: Medium / miss${miss_num} / ${env_type} / ${budget_level}"
            python "${PYTHON_SCRIPT}" \
                --env_type "${env_type}" \
                --budget_level "${budget_level}" \
                --miss_num "${miss_num}" \
                --space "Medium" \
                --model "${MODEL}" \
                --num_processes "${NUM_PROCESSES}"
        done
    done
done

echo ""
echo "=============================================="
echo "All exp2_budget experiments completed!"
echo "=============================================="
