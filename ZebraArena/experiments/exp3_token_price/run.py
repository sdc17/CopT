#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run token-based pricing evaluation experiments for Zebra puzzle.

Pricing Design (model-specific to maintain ~20-25% tool cost ratio):
- gemini-2.5-flash: BASE=500, CHEAP=250, EXPENSIVE=1000 (avg 1733 tok/msg)
- gemini-2.5-pro:   BASE=800, CHEAP=400, EXPENSIVE=1600 (avg 2810 tok/msg)
- Qwen3-235B:       BASE=250, CHEAP=125, EXPENSIVE=500  (avg 838 tok/msg)

Goal: Minimize total token cost while solving accurately and reliably

Experiment Conditions:
  1. baseline:           fact=BASE, relation=BASE
  2. fact_cheap:         fact=CHEAP, relation=BASE
  3. fact_expensive:     fact=EXPENSIVE, relation=BASE
  4. relation_cheap:     fact=BASE, relation=CHEAP
  5. relation_expensive: fact=BASE, relation=EXPENSIVE
  6. both_cheap:         fact=CHEAP, relation=CHEAP
  7. both_expensive:     fact=EXPENSIVE, relation=EXPENSIVE
  8. fact_very_cheap:    fact=VERY_CHEAP, relation=VERY_EXPENSIVE
  9. fact_very_expensive: fact=VERY_EXPENSIVE, relation=VERY_CHEAP
  10. tool_free:         fact=0, relation=0
"""

import os
import sys
import argparse
import multiprocessing
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
from experiments.exp3_token_price.pricing_solver import PricingSolver


DATA_BATCH_SIZE = 1

# Model-specific pricing configuration
# Designed to keep tool cost at ~20-25% of total cost
# Based on avg tokens per message: flash=1733, pro=2810, qwen3=838
MODEL_PRICING = {
    "gemini-2.5-flash": {
        "BASE": 500,
        "CHEAP": 250,
        "EXPENSIVE": 1000,
        "VERY_CHEAP": 100,
        "VERY_EXPENSIVE": 2000,
    },
    "gemini-2.5-pro": {
        "BASE": 800,
        "CHEAP": 400,
        "EXPENSIVE": 1600,
        "VERY_CHEAP": 150,
        "VERY_EXPENSIVE": 3000,
    },
    "Qwen3-235B": {
        "BASE": 250,
        "CHEAP": 125,
        "EXPENSIVE": 500,
        "VERY_CHEAP": 50,
        "VERY_EXPENSIVE": 1000,
    },
}

# Default pricing (fallback for unknown models)
DEFAULT_PRICING = {
    "BASE": 500,
    "CHEAP": 250,
    "EXPENSIVE": 1000,
    "VERY_CHEAP": 100,
    "VERY_EXPENSIVE": 2000,
}


def get_model_pricing(model_name: str) -> dict:
    """Get pricing configuration for a specific model."""
    return MODEL_PRICING.get(model_name, DEFAULT_PRICING)


def get_pricing_conditions(model_name: str) -> dict:
    """Generate pricing conditions for a specific model."""
    p = get_model_pricing(model_name)
    return {
        "baseline": {
            "fact_price": p["BASE"],
            "relation_price": p["BASE"],
        },
        "fact_cheap": {
            "fact_price": p["CHEAP"],
            "relation_price": p["BASE"],
        },
        "fact_expensive": {
            "fact_price": p["EXPENSIVE"],
            "relation_price": p["BASE"],
        },
        "relation_cheap": {
            "fact_price": p["BASE"],
            "relation_price": p["CHEAP"],
        },
        "relation_expensive": {
            "fact_price": p["BASE"],
            "relation_price": p["EXPENSIVE"],
        },
        "both_cheap": {
            "fact_price": p["CHEAP"],
            "relation_price": p["CHEAP"],
        },
        "both_expensive": {
            "fact_price": p["EXPENSIVE"],
            "relation_price": p["EXPENSIVE"],
        },
        # Extreme pricing conditions - asymmetric to strongly influence query type choice
        "fact_very_cheap": {
            "fact_price": p["VERY_CHEAP"],
            "relation_price": p["VERY_EXPENSIVE"],
        },
        "fact_very_expensive": {
            "fact_price": p["VERY_EXPENSIVE"],
            "relation_price": p["VERY_CHEAP"],
        },
        "tool_free": {
            "fact_price": 0,
            "relation_price": 0,
        },
    }


# Condition names (for argparse choices)
CONDITION_NAMES = [
    "baseline", "fact_cheap", "fact_expensive",
    "relation_cheap", "relation_expensive",
    "both_cheap", "both_expensive",
    "fact_very_cheap", "fact_very_expensive", "tool_free"
]


def run_worker(
    rank: int,
    model_name: str,
    env_type: EnvType,
    fact_price: int,
    relation_price: int,
    batched_rows: list,
    log_dir: str,
):
    """Worker function to process puzzles with token-based pricing."""
    solver = PricingSolver(
        env_type=env_type,
        model_name=model_name,
        fact_price=fact_price,
        relation_price=relation_price,
    )

    for row in tqdm(batched_rows, desc=str(rank), position=rank):
        row = row[0]
        print(f"[rank {rank}] Processing: {row['id']} (fact={fact_price}, relation={relation_price})")

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


def main():
    parser = argparse.ArgumentParser(description="Run token-based pricing Zebra puzzle evaluation")
    parser.add_argument("--env_type", type=str, default="normal",
                        choices=["normal", "only_fact", "only_relation"],
                        help="Environment type")

    # Pricing can be specified as condition name or explicit values
    parser.add_argument("--condition", type=str, default="baseline",
                        choices=CONDITION_NAMES,
                        help="Pricing condition name")
    parser.add_argument("--fact_price", type=int, default=None,
                        help="Explicit fact query price (overrides condition)")
    parser.add_argument("--relation_price", type=int, default=None,
                        help="Explicit relation query price (overrides condition)")

    parser.add_argument("--miss_num", type=int, default=4,
                        help="Number of missing clues")
    parser.add_argument("--space", type=str, default="Medium",
                        choices=["Small", "Medium"],
                        help="Puzzle space size")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
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

    # Get model-specific pricing
    model_pricing = get_model_pricing(args.model)
    pricing_conditions = get_pricing_conditions(args.model)

    # Determine pricing
    if args.fact_price is not None or args.relation_price is not None:
        # Explicit pricing (use model's BASE as fallback)
        fact_price = args.fact_price if args.fact_price is not None else model_pricing["BASE"]
        relation_price = args.relation_price if args.relation_price is not None else model_pricing["BASE"]
        condition_name = f"fact{fact_price}_rel{relation_price}"
    else:
        # Use condition with model-specific pricing
        condition = pricing_conditions[args.condition]
        fact_price = condition["fact_price"]
        relation_price = condition["relation_price"]
        condition_name = args.condition

    print(f"Model: {args.model}")
    print(f"Model pricing config: BASE={model_pricing['BASE']}, CHEAP={model_pricing['CHEAP']}, EXPENSIVE={model_pricing['EXPENSIVE']}")

    print(f"Pricing: fact={fact_price}, relation={relation_price}")
    print(f"Condition: {condition_name}")
    print(f"Goal: Minimize total token cost while solving accurately")

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
        f"exp3_pricing/{args.env_type}/{condition_name}/missing{args.miss_num}/{args.space}/{args.model}"
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"Results will be saved to: {log_dir}")

    if args.num_processes == 1:
        run_worker(0, args.model, env_type, fact_price, relation_price, batched_problems, log_dir)
    else:
        processes = []
        for i in range(args.num_processes):
            p = multiprocessing.Process(
                target=run_worker,
                args=(i, args.model, env_type, fact_price, relation_price, batched_problems[i :: args.num_processes], log_dir)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print("Done!")


if __name__ == "__main__":
    main()
