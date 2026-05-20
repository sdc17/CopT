#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run basic evaluation experiments for Zebra puzzle.
Supports all three environment types: normal, only_fact, only_relation.
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
from experiments.exp1_basic.basic_solver import BasicSolver


DATA_BATCH_SIZE = 1


def run_worker(
    rank: int,
    model_name: str,
    env_type: EnvType,
    batched_rows: list,
    log_dir: str,
):
    """
    Worker function to process puzzles.

    Args:
        rank: Process rank for progress display
        model_name: LLM model name
        env_type: Environment type
        batched_rows: List of puzzle batches to process
        log_dir: Directory to save results
    """
    solver = BasicSolver(env_type=env_type, model_name=model_name)

    for row in tqdm(batched_rows, desc=str(rank), position=rank):
        row = row[0]  # Unpack single-item batch
        print(f"[rank {rank}] Processing: {row['id']}")

        solver.driver_loop(
            row=row,
            log_dir=log_dir,
            # Inject dependencies
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


def main():
    parser = argparse.ArgumentParser(description="Run basic Zebra puzzle evaluation")
    parser.add_argument("--env_type", type=str, default="normal",
                        choices=["normal", "only_fact", "only_relation"],
                        help="Environment type")
    parser.add_argument("--miss_num", type=int, default=1,
                        help="Number of missing clues")
    parser.add_argument("--space", type=str, default="Small",
                        choices=["Small", "Medium", "Large"],
                        help="Puzzle space size")
    parser.add_argument("--model", type=str, default="gpt-oss-120b",
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

    # Map string to EnvType
    env_type_map = {
        "normal": EnvType.NORMAL,
        "only_fact": EnvType.ONLY_FACT,
        "only_relation": EnvType.ONLY_RELATION,
    }
    env_type = env_type_map[args.env_type]

    # Load dataset
    dataset_path = os.path.join(
        args.dataset_dir,
        f"filtered_puzzles_missing{args.miss_num}_{args.space}.parquet"
    )
    df = pd.read_parquet(dataset_path)
    print(f"Loaded dataset {dataset_path}, total: {len(df)} puzzles")

    # Batch puzzles
    batched_problems = [
        df.iloc[i : i + DATA_BATCH_SIZE].to_dict(orient="records")
        for i in range(0, len(df), DATA_BATCH_SIZE)
    ]

    # Setup log directory
    log_dir = os.path.join(
        args.log_dir,
        f"exp1_basic/{args.env_type}/missing{args.miss_num}/{args.space}/{args.model}"
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"Results will be saved to: {log_dir}")

    # Run with threading (I/O-bound task, no need for multiprocessing)
    if args.num_processes == 1:
        # Single thread - no threading overhead
        run_worker(0, args.model, env_type, batched_problems, log_dir)
    else:
        threads = []
        for i in range(args.num_processes):
            t = threading.Thread(
                target=run_worker,
                args=(i, args.model, env_type, batched_problems[i :: args.num_processes], log_dir)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    print("Done!")


if __name__ == "__main__":
    main()
