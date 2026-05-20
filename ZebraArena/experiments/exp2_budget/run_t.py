#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run budget-constrained evaluation experiments for Zebra puzzle.

Budget Levels (based on K* = number of missing clues):
  Small:  tight=2×K*, normal=3×K*, relaxed=4×K*  (avg IR ≈ 2.5)
  Medium: tight=2×K*, normal=4×K*, relaxed=5×K*  (avg IR ≈ 3.6)
"""

import os
import sys
import argparse
import threading
import pandas as pd
from tqdm import tqdm

# Add parent paths for imports
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from core.base_solver import EnvType
from core.llm import llm_generate
from env.response_server import load_solution, answer_query
from env.scheme import build_schemas, canonicalize_query
from extract.extract_query import extract_query_dict, extract_answer_dict
from extract.grid_reward import check_final_solution, solution_numpy_to_dict
from utils.prune_messages import prune_messages
from experiments.exp2_budget.budget_solver import BudgetSolver


DATA_BATCH_SIZE = 1

# Budget lookup table: BUDGET_TABLE[model][space][miss_num] = (tight, normal, relaxed)
# Based on filtered exp1_basic normal env IR data:
#   tight: K* (theoretical optimal)
#   normal: round(avg IR)
#   relaxed: round(avg IR) + 1
BUDGET_TABLE = {
    "gemini-2.5-flash": {
        "Small": {
            # K*=1: IR=2.10 → tight=1, normal=2, relaxed=3
            1: (1, 2, 3),
            # K*=2: IR=2.51 → tight=2, normal=5, relaxed=7
            2: (2, 5, 7),
            # K*=3: IR=2.76 → tight=3, normal=8, relaxed=11
            3: (3, 8, 11),
        },
        "Medium": {
            # K*=1: IR=2.03 → tight=1, normal=2, relaxed=3
            1: (1, 2, 3),
            # K*=2: IR=3.05 → tight=2, normal=6, relaxed=8
            2: (2, 6, 8),
            # K*=3: IR=3.82 → tight=3, normal=11, relaxed=14
            3: (3, 11, 14),
            # K*=4: IR=3.84 → tight=4, normal=15, relaxed=19
            4: (4, 15, 19),
        },
    },
    "gemini-2.5-pro": {
        "Medium": {
            # K*=1: tight=1, normal=1, relaxed=2
            1: (1, 2, 3),
            # K*=2: tight=2, normal=4, relaxed=6
            2: (2, 4, 6),
            # K*=3: tight=3, normal=6, relaxed=9
            3: (3, 6, 9),
            # K*=4: tight=4, normal=8, relaxed=12
            4: (4, 8, 12),
        },
    },
    "Llama-3.3-70B": {
        "Medium": {
            # K*=1: tight=1, normal=1, relaxed=2
            1: (1, 5, 9),
            # K*=2: tight=2, normal=4, relaxed=6
            2: (2, 6, 10),
            # K*=3: tight=3, normal=6, relaxed=9
            3: (3, 7, 11),
            # K*=4: tight=4, normal=8, relaxed=12
            4: (4, 8, 12),
        },
    },
    "Qwen3-235B": {
        "Medium": {
            # K*=1: tight=1, normal=1, relaxed=2
            1: (1, 2, 4),
            # K*=2: tight=2, normal=4, relaxed=6
            2: (2, 5, 8),
            # K*=3: tight=3, normal=6, relaxed=9
            3: (3, 7, 11),
            # K*=4: tight=4, normal=8, relaxed=12
            4: (4, 9, 14),
        },
    },
}

LEVEL_INDEX = {"tight": 0, "normal": 1, "relaxed": 2}


def run_worker(
    rank: int,
    model_name: str,
    env_type: EnvType,
    budget: int,
    hard_limit: bool,
    batched_rows: list,
    log_dir: str,
):
    """Worker function to process puzzles with budget constraint."""
    solver = BudgetSolver(env_type=env_type, model_name=model_name, budget=budget, hard_limit=hard_limit)

    for row in tqdm(batched_rows, desc=str(rank), position=rank):
        row = row[0]
        print(f"[rank {rank}] Processing: {row['id']} (budget={budget})")

        solver.driver_loop(
            row=row,
            log_dir=log_dir,
            load_solution_fn=load_solution,
            build_schemas_fn=build_schemas,
            canonicalize_query_fn=canonicalize_query,
            answer_query_fn=answer_query,
            extract_query_dict_fn=extract_query_dict,
            extract_answer_dict_fn=extract_answer_dict,
            check_final_solution_fn=check_final_solution,
            solution_numpy_to_dict_fn=solution_numpy_to_dict,
            prune_messages_fn=prune_messages,
            rank=rank,
        )


def compute_budget(miss_num: int, budget_level: str, space: str, model: str) -> int:
    """
    Compute budget based on K* (miss_num), budget level, space size, and model.

    Args:
        miss_num: Number of missing clues (K*)
        budget_level: One of 'tight', 'normal', 'relaxed'
        space: Puzzle space size ('Small' or 'Medium')
        model: Model name (e.g., 'gemini-2.5-flash', 'gemini-2.5-pro')

    Returns:
        Budget value from lookup table
    """
    if model not in BUDGET_TABLE:
        raise ValueError(f"Unknown model: {model}. Available models: {list(BUDGET_TABLE.keys())}")
    if space not in BUDGET_TABLE[model]:
        raise ValueError(f"Unknown space: {space} for model {model}. Available: {list(BUDGET_TABLE[model].keys())}")
    if miss_num not in BUDGET_TABLE[model][space]:
        raise ValueError(f"Unknown miss_num: {miss_num} for model {model}, space {space}")

    level_idx = LEVEL_INDEX[budget_level]
    return BUDGET_TABLE[model][space][miss_num][level_idx]


def main():
    parser = argparse.ArgumentParser(description="Run budget-constrained Zebra puzzle evaluation")
    parser.add_argument("--env_type", type=str, default="normal",
                        choices=["normal", "only_fact", "only_relation"],
                        help="Environment type")

    # Budget can be specified either as explicit value or as level
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument("--budget", type=int, default=None,
                              help="Explicit tool call budget (overrides --budget_level)")
    budget_group.add_argument("--budget_level", type=str, default="normal",
                              choices=["tight", "normal", "relaxed"],
                              help="Budget level: tight=2×K*, normal=3×K*, relaxed=5×K*")

    parser.add_argument("--hard_limit", action="store_true",
                        help="If set, force stop when budget exhausted (default: soft limit)")
    parser.add_argument("--miss_num", type=int, default=1,
                        help="Number of missing clues (K*)")
    parser.add_argument("--space", type=str, default="Medium",
                        choices=["Small", "Medium"],
                        help="Puzzle space size")
    parser.add_argument("--model", type=str, default="Qwen3-235B",
                        help="Model name")
    parser.add_argument("--num_processes", type=int, default=32,
                        help="Number of parallel processes")
    parser.add_argument("--dataset_dir", type=str,
                        default="/atlas2/u/wanjiazh/reliable_tool/puzzle/dataset/filtered",
                        help="Dataset directory")
    parser.add_argument("--log_dir", type=str,
                        default="/atlas2/u/wanjiazh/reliable_tool/puzzle/zebrapuzzle/logs",
                        help="Log directory")

    args = parser.parse_args()

    env_type_map = {
        "normal": EnvType.NORMAL,
        "only_fact": EnvType.ONLY_FACT,
        "only_relation": EnvType.ONLY_RELATION,
    }
    env_type = env_type_map[args.env_type]

    # Determine budget
    if args.budget is not None:
        budget = args.budget
        budget_name = f"budget{budget}"
    else:
        budget = compute_budget(args.miss_num, args.budget_level, args.space, args.model)
        budget_name = f"{args.budget_level}"  # Use level name for clearer log paths

    print(f"Budget: {budget} (K*={args.miss_num}, level={args.budget_level if args.budget is None else 'explicit'})")
    print(f"Hard limit: {args.hard_limit}")

    dataset_path = os.path.join(
        args.dataset_dir,
        f"filtered_puzzles_missing{args.miss_num}_{args.space}.parquet"
    )
    df = pd.read_parquet(dataset_path)
    print(f"Loaded dataset {dataset_path}, total: {len(df)} puzzles")

    batched_problems = [
        df.iloc[i : i + DATA_BATCH_SIZE].to_dict(orient="records")
        for i in range(0, len(df), DATA_BATCH_SIZE)
    ]

    log_dir = os.path.join(
        args.log_dir,
        f"exp2_budget/{args.env_type}/{budget_name}/missing{args.miss_num}/{args.space}/{args.model}"
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"Results will be saved to: {log_dir}")

    # Run with threading (I/O-bound task, no need for multiprocessing)
    if args.num_processes == 1:
        run_worker(0, args.model, env_type, budget, args.hard_limit, batched_problems, log_dir)
    else:
        threads = []
        for i in range(args.num_processes):
            t = threading.Thread(
                target=run_worker,
                args=(i, args.model, env_type, budget, args.hard_limit, batched_problems[i :: args.num_processes], log_dir)
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    print("Done!")


if __name__ == "__main__":
    main()
