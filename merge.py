import os
import glob
import json
import argparse


def main(args):
    
    model_name = args.model_name
    model_name = model_name.split("/")[-1]
    dataset_name = args.dataset_name
    method = args.method

    print("[Rank 0] All logs written, start merging...")
    all_details = []
    total_correct = 0
    total_samples = 0
    
    ### Token Length Stats 
    total_token_sum = 0
    correct_token_sum = 0
    wrong_token_sum = 0
    total_token_cnt = 0
    correct_token_cnt = 0
    wrong_token_cnt = 0

    log_pattern = f"logs/{model_name}_{dataset_name}_{method}_rank*.json"
    all_log_paths = list(glob.glob(log_pattern))
    if not all_log_paths:
        raise FileNotFoundError(
            "No rank logs found for merge pattern: "
            f"{log_pattern}"
        )
    for path in all_log_paths:
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
            total_correct += result["correct"]
            total_samples += result["total"]
            all_details.extend(result["details"])
            ### Token Length Stats 
            ls = result["length_stats"]
            total_token_sum += ls.get("avg_total_token_len", 0) * result["total"]
            total_token_cnt += result["total"]
            correct_token_sum += ls.get("correct_avg_total_token_len", 0) * result["correct"]
            correct_token_cnt += result["correct"]
            wrong_cnt = result["total"] - result["correct"]
            wrong_token_sum += ls.get("wrong_avg_total_token_len", 0) * wrong_cnt
            wrong_token_cnt += wrong_cnt

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    ### Token Length Stats 
    merged_length_stats = {
        "max_new_tokens": result["length_stats"]["max_new_tokens"],
        "avg_total_token_len": float(total_token_sum) / total_token_cnt if total_token_cnt else 0.0,
        "correct_avg_total_token_len": float(correct_token_sum) / correct_token_cnt if correct_token_cnt else 0.0,
        "wrong_avg_total_token_len": float(wrong_token_sum) / wrong_token_cnt if wrong_token_cnt else 0.0,
    }
    if dataset_name == "zebra_arena":
        merged_length_stats["zebra_arena_space"] = args.zebra_arena_space
        merged_length_stats["zebra_arena_env_type"] = args.zebra_arena_env_type
        merged_length_stats["zebra_arena_miss_num"] = args.zebra_arena_miss_num
    merged_result = {
        "accuracy": accuracy,
        "total": total_samples,
        "correct": total_correct,
        "length_stats": merged_length_stats,
        "details": all_details,
    }
    with open(f"logs/{model_name}_{dataset_name}_{method}_merged.json", "w", encoding="utf-8") as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=2)
    print(f"[Rank 0] Merged results saved. Accuracy: {accuracy:.2%}, Length: {merged_length_stats}")

    for path in all_log_paths:
        try:
            os.remove(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--dataset_name', type=str, default="gsm8k")
    parser.add_argument("--method", type=str, default="copt", choices=["copt", "cot", "cot_greedy"])
    parser.add_argument('--zebra_arena_space', type=str, default="Small", choices=["Small", "Medium", "Large"])
    parser.add_argument('--zebra_arena_env_type', type=str, default="normal", choices=["normal", "only_fact", "only_relation"])
    parser.add_argument('--zebra_arena_miss_num', type=int, default=1)
    args = parser.parse_args()
    main(args)
