import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import torch.distributed as dist
import argparse
import os
import re
from generation_utils import (
    set_seed,
    generate_cot,
    generate_copt_general,
)
from grader import answer_match
from helper import (
    load_default_effort,
    split_thinking_answer,
    trim_output_ids,
    _json_loads_field,
    load_zebra_arena_dataset,
    build_zebra_arena_messages,
    normalize_zebra_arena_output,
    prune_zebra_arena_messages,
    zebra_arena_history_content,
    zebra_arena_open_think_history,
    is_qwen3_zebra_legacy_template,
    parse_zebra_arena_response,
)


def main(args):
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    model_name = args.model_name
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    n_samples = args.n_samples
    method = args.method
    default_effort = load_default_effort(dataset_name)
    tau_a = args.tau_a if args.tau_a is not None else default_effort.get("tau_a", 0.0)
    tau_r = args.tau_r if args.tau_r is not None else default_effort.get("tau_r", 0.0)

    gen_kwargs = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
    }
    if dataset_name in {"gsm8k", "math500", "aime_2024", "aime_2025", "gpqa_diamond"}:
        gen_kwargs["task_type"] = "math"
    elif dataset_name == "zebra_arena":
        gen_kwargs["task_type"] = "coding"
    else:
        gen_kwargs["task_type"] = "default"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left' ###

    end_thinking_id = tokenizer.convert_tokens_to_ids("</think>")
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if not isinstance(end_thinking_id, int) or end_thinking_id < 0 or end_thinking_id == unk_token_id:
        end_thinking_id = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": local_rank}
    )
    
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    elif dataset_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif dataset_name == "aime_2024":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif dataset_name == "aime_2025":
        dataset = load_dataset("yentinglin/aime_2025", split="train")
    elif dataset_name == "gpqa_diamond":
        dataset = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    elif dataset_name == "zebra_arena":
        dataset = load_zebra_arena_dataset(args.zebra_arena_data_dir, args.zebra_arena_miss_num, args.zebra_arena_space)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))
    total_len = len(dataset)
    chunk_size = (total_len + world_size - 1) // world_size
    start = local_rank * chunk_size
    end = min(start + chunk_size, total_len)
    dataset = dataset.select(range(start, end))
    
    correct = 0
    total = 0
    details = []
    total_token_lens = []
    correct_token_lens = []
    wrong_token_lens = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        if args.dataset_name == "gsm8k":
            questions = batch["question"]
            golds = [str(a).split("####")[-1].strip() for a in batch["answer"]]
        elif args.dataset_name == "math500":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "aime_2024":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "aime_2025":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "gpqa_diamond":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["solution"]]
        elif args.dataset_name == "zebra_arena":
            questions = batch["puzzle"]
            golds = [
                {
                    "id": example_id,
                    "puzzle": puzzle,
                    "solution": _json_loads_field(solution),
                    "missing_clue_number": missing_clue_number,
                    "total_clue_number": total_clue_number,
                    "space": space,
                    "size": size,
                }
                for example_id, puzzle, solution, missing_clue_number, total_clue_number, space, size in zip(
                    batch["id"],
                    batch["puzzle"],
                    batch["solution"],
                    batch["missing_clue_number"],
                    batch["total_clue_number"],
                    batch["space"],
                    batch["size"],
                )
            ]

        if args.dataset_name == "zebra_arena":
            prompts = questions
        else:
            prompts = [
                f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                for q in questions
            ]

        if args.dataset_name != "zebra_arena":
            messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                for messages in messages_batch
            ]
        if args.dataset_name == "zebra_arena":
            zebra_states = []
            for gold, question in zip(golds, questions):
                zebra_states.append({
                    "gold": gold,
                    "question": question,
                    "messages": build_zebra_arena_messages(gold, args.zebra_arena_env_type),
                    "response_log": [],
                    "pred": "",
                    "thinking_content": "",
                    "answer_content": "",
                    "output_token_len": 0,
                    "tool_call_num": 0,
                    "success_tool_call_num": 0,
                    "seen_queries": {},
                    "steps_num": 0,
                    "is_correct": False,
                    "done": False,
                    "prediction": {
                        "ok": False,
                        "errors": [f"Reached max turns ({args.zebra_arena_max_turns}) without solution."],
                    },
                })

            zebra_turn_max_new_tokens = max(1, max_new_tokens // max(1, args.zebra_arena_max_turns))
            for turn_idx in range(1, args.zebra_arena_max_turns + 1):
                active_indices = [
                    state_idx
                    for state_idx, state in enumerate(zebra_states)
                    if not state["done"]
                ]
                if not active_indices:
                    break

                for state_idx in active_indices:
                    zebra_states[state_idx]["messages"] = prune_zebra_arena_messages(
                        zebra_states[state_idx]["messages"],
                        args.zebra_arena_max_input_tokens,
                        args.zebra_arena_keep_tail,
                    )

                if is_qwen3_zebra_legacy_template(dataset_name, model_name):
                    texts = []
                    for state_idx in active_indices:
                        text = tokenizer.apply_chat_template(
                            zebra_states[state_idx]["messages"],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        if not re.search(r"<\s*think\s*>\s*$", text.rstrip(), flags=re.IGNORECASE):
                            text = text + "<think>\n"
                        texts.append(text)
                else:
                    texts = [
                        tokenizer.apply_chat_template(
                            zebra_states[state_idx]["messages"],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        for state_idx in active_indices
                    ]

                zebra_inputs = tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                ).to(model.device)

                with torch.no_grad():
                    if method == "cot":
                        zebra_gen_kwargs = dict(gen_kwargs)
                        zebra_gen_kwargs["max_new_tokens"] = zebra_turn_max_new_tokens
                        zebra_generated_ids = generate_cot( # better memory efficiency 
                            model,
                            tokenizer,
                            **zebra_inputs,
                            **zebra_gen_kwargs,
                        )
                    elif method == "cot_greedy":
                        zebra_gen_kwargs = dict(gen_kwargs)
                        zebra_gen_kwargs["do_sample"] = False
                        zebra_gen_kwargs["max_new_tokens"] = zebra_turn_max_new_tokens
                        zebra_generated_ids = generate_cot( # better memory efficiency 
                            model,
                            tokenizer,
                            **zebra_inputs,
                            **zebra_gen_kwargs,
                        )
                    elif method == "copt":
                        zebra_gen_kwargs = dict(gen_kwargs)
                        zebra_gen_kwargs["max_new_tokens"] = zebra_turn_max_new_tokens
                        zebra_gen_kwargs["tau_a"] = tau_a
                        zebra_gen_kwargs["tau_r"] = tau_r
                        zebra_gen_kwargs["draft_max_new_tokens"] = 512
                        zebra_generated_ids = generate_copt_general(
                            model,
                            tokenizer,
                            **zebra_inputs,
                            **zebra_gen_kwargs,
                        )
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                zebra_prompt_len = zebra_inputs["input_ids"].shape[1]
                for active_pos, state_idx in enumerate(active_indices):
                    state = zebra_states[state_idx]
                    output_ids = trim_output_ids(
                        zebra_generated_ids[active_pos][zebra_prompt_len:].tolist(), tokenizer
                    )
                    state["output_token_len"] += len(output_ids)
                    round_thinking, round_answer = split_thinking_answer(output_ids, tokenizer, end_thinking_id)

                    raw_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                    pred_raw = normalize_zebra_arena_output(raw_output)
                    pred = pred_raw
                    round_thinking_for_log = round_thinking.strip()
                    if not round_thinking_for_log:
                        thinking_match = re.search(
                            r"<\s*think\s*>(.*?)(?:<\s*/\s*think\s*>|$)",
                            pred,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        if thinking_match:
                            round_thinking_for_log = thinking_match.group(1).strip()
                        elif re.search(r"<\s*/\s*think\s*>", pred, flags=re.IGNORECASE):
                            round_thinking_for_log = re.split(
                                r"<\s*/\s*think\s*>",
                                pred,
                                maxsplit=1,
                                flags=re.IGNORECASE,
                            )[0].strip()
                    if round_thinking_for_log:
                        state["thinking_content"] = (
                            state["thinking_content"] + "\n" + round_thinking_for_log
                        ).strip()
                    action_text_raw = normalize_zebra_arena_output(round_answer) if round_answer.strip() else pred_raw
                    action_text = action_text_raw
                    parse_text = action_text if action_text.strip() else pred
                    if parse_text == action_text and round_thinking_for_log:
                        history_content = zebra_arena_history_content(parse_text, round_thinking_for_log, parse_text)
                    else:
                        history_content = zebra_arena_history_content(parse_text)
                    state["pred"] = pred
                    state["steps_num"] = turn_idx
                    state["response_log"].append({
                        "step": turn_idx,
                        "role": "assistant",
                        "content": pred,
                        "thinking": round_thinking_for_log,
                        "answer": action_text,
                        "action_content": action_text,
                        "parse_content": parse_text,
                        "history_content": history_content,
                    })

                    parsed = parse_zebra_arena_response(parse_text, state["gold"], args.zebra_arena_env_type)
                    if parsed["type"] == "solution":
                        state["is_correct"] = parsed["correct"]
                        state["prediction"] = parsed["prediction"]
                        state["answer_content"] = json.dumps(parsed["answer"], ensure_ascii=False)
                        state["response_log"][-1]["history_content"] = ""
                        state["done"] = True
                        continue

                    if parsed["type"] == "query":
                        query_key = json.dumps(parsed["canonical_query"], ensure_ascii=False, sort_keys=True)
                        state["tool_call_num"] += 1
                        canonical_query_text = (
                            f"<query>{json.dumps(parsed['canonical_query'], ensure_ascii=False)}</query>"
                        )

                        query_history_content = canonical_query_text
                        state["response_log"][-1]["history_content"] = query_history_content
                        if query_key in state["seen_queries"]:
                            previous_response = state["seen_queries"][query_key]
                            repeat_response = (
                                "You already asked exactly this query. "
                                f"Previous {previous_response} "
                                "Use that result; ask a different valid <query> or provide a complete <solution>."
                            )
                            state["response_log"].append({
                                "step": turn_idx,
                                "role": "environment",
                                "query": parsed["query"],
                                "canonical_query": parsed["canonical_query"],
                                "repeated": True,
                                "content": repeat_response,
                            })
                            state["messages"].append({
                                "role": "assistant",
                                "content": query_history_content,
                            })
                            state["messages"].append({
                                "role": "user",
                                "content": repeat_response,
                            })
                            continue

                        if isinstance(parsed.get("result"), dict) and parsed["result"].get("ok"):
                            state["success_tool_call_num"] += 1
                        state["seen_queries"][query_key] = parsed["env_response"]
                        state["response_log"].append({
                            "step": turn_idx,
                            "role": "environment",
                            "query": parsed["query"],
                            "canonical_query": parsed["canonical_query"],
                            "result": parsed["result"],
                            "content": parsed["env_response"],
                        })
                        state["messages"].append({
                            "role": "assistant",
                            "content": query_history_content,
                        })
                        state["messages"].append({
                            "role": "user",
                            "content": parsed["env_response"],
                        })
                        continue

                    if parsed["type"] == "reasoning":
                        reasoning_history_content = zebra_arena_open_think_history(history_content or pred)
                        state["response_log"][-1]["history_content"] = reasoning_history_content
                        state["messages"].append({
                            "role": "assistant",
                            "content": reasoning_history_content,
                        })
                        continue

                    invalid_history_content = zebra_arena_open_think_history(history_content or pred)
                    state["response_log"].append({
                        "step": turn_idx,
                        "role": "environment",
                        "error": parsed["message"],
                        "history_content": invalid_history_content,
                    })
                    if invalid_history_content.strip():
                        state["messages"].append({"role": "assistant", "content": invalid_history_content})

            for state in zebra_states:
                correct += int(state["is_correct"])
                total += 1
                details.append({
                    "question": state["question"],
                    "gold": state["gold"],
                    "prediction": state["prediction"],
                    "correct": state["is_correct"],
                    "thinking": state["thinking_content"],
                    "answer_content": state["answer_content"],
                    "last_response": state["pred"],
                    "response_log": state["response_log"],
                    "messages": state["messages"],
                    "tool_call_num": state["tool_call_num"],
                    "success_tool_call_num": state["success_tool_call_num"],
                    "steps_num": state["steps_num"],
                    "env_type": args.zebra_arena_env_type,
                    "miss_num": args.zebra_arena_miss_num,
                    "per_turn_max_new_tokens": zebra_turn_max_new_tokens,
                })
                if total % 20 == 0:
                    print(f"Processed {total} examples, Accuracy: {correct/total:.2%}")
                total_token_lens.append(state["output_token_len"])
                if state["is_correct"]:
                    correct_token_lens.append(state["output_token_len"])
                else:
                    wrong_token_lens.append(state["output_token_len"])
            continue
        model_inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
    
        with torch.no_grad():
            if method == "cot":
                # generated_ids = model.generate(
                #     **model_inputs,
                #     **gen_kwargs,
                # )
                generated_ids = generate_cot( # better memory efficiency 
                    model,
                    tokenizer,
                    **model_inputs,
                    **gen_kwargs,
                )
            elif method == "cot_greedy":
                gen_kwargs["do_sample"] = False
                generated_ids = generate_cot( # better memory efficiency 
                    model,
                    tokenizer,
                    **model_inputs,
                    **gen_kwargs,
                )
            elif method == "copt":
                gen_kwargs["tau_a"] = tau_a
                gen_kwargs["tau_r"] = tau_r
                generated_ids = generate_copt_general(
                    model,
                    tokenizer,
                    **model_inputs,
                    **gen_kwargs,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
        
        prompt_len = model_inputs["input_ids"].shape[1]

        preds = [
            tokenizer.decode(generated_ids[idx][prompt_len:], skip_special_tokens=True)
            for idx in range(len(questions))
        ]

        for idx in range(len(questions)):
            gold = golds[idx]
            question = questions[idx]
            pred = preds[idx]
            output_ids = generated_ids[idx][prompt_len:].tolist()
            if isinstance(end_thinking_id, int) and end_thinking_id in output_ids:
                index = len(output_ids) - output_ids[::-1].index(end_thinking_id)
            else:
                index = 0
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            answer_content = pred[len(thinking_content):]
            is_correct, prediction = answer_match(dataset_name, answer_content, gold)
            correct += int(is_correct)
            total += 1
            details.append({
                "question": question,
                "gold": gold,
                "prediction": prediction,
                "correct": is_correct,
                "thinking": thinking_content,
                "answer_content": answer_content,
            })
            if total % 20 == 0:
                print(f"Processed {total} examples, Accuracy: {correct/total:.2%}")
                
            ### Token Length Stats
            output_token_ids = tokenizer.encode(pred, add_special_tokens=False)
            total_token_len = len(output_token_ids)
            total_token_lens.append(total_token_len)
            if is_correct:
                correct_token_lens.append(total_token_len)
            else:
                wrong_token_lens.append(total_token_len)

    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total:.2%}")
    
    ### Token Length Stats 
    avg = lambda l: float(sum(l)) / len(l) if l else 0.0
    length_stats = {
        "max_new_tokens": max_new_tokens,
        "avg_total_token_len": avg(total_token_lens),
        "correct_avg_total_token_len": avg(correct_token_lens),
        "wrong_avg_total_token_len": avg(wrong_token_lens),
    }
    if dataset_name == "zebra_arena":
        length_stats["per_turn_max_new_tokens"] = max(1, max_new_tokens // max(1, args.zebra_arena_max_turns))
        length_stats["zebra_arena_max_turns"] = args.zebra_arena_max_turns
        length_stats["zebra_arena_space"] = args.zebra_arena_space
        length_stats["zebra_arena_env_type"] = args.zebra_arena_env_type
        length_stats["zebra_arena_miss_num"] = args.zebra_arena_miss_num
    
    result = {
        "accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "length_stats": length_stats,
        "details": details
    }
    
    os.makedirs("logs", exist_ok=True)
    model_name = model_name.split("/")[-1]
    log_path = f"logs/{model_name}_{dataset_name}_{method}_rank{local_rank}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Rank {local_rank}] log written: {log_path}")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--dataset_name', type=str, default="gsm8k")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=32768)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument("--method", type=str, default="copt", choices=["copt", "cot", "cot_greedy"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tau_a', type=float, default=None) # CopT reasoning effort
    parser.add_argument('--tau_r', type=float, default=None) # CopT reasoning effort
    parser.add_argument('--zebra_arena_data_dir', type=str, default="ZebraArena/data")
    parser.add_argument('--zebra_arena_space', type=str, default="Small", choices=["Small", "Medium", "Large"])
    parser.add_argument('--zebra_arena_env_type', type=str, default="normal", choices=["normal", "only_fact", "only_relation"])
    parser.add_argument('--zebra_arena_miss_num', type=int, default=1)
    parser.add_argument('--zebra_arena_max_turns', type=int, default=16)
    parser.add_argument('--zebra_arena_max_input_tokens', type=int, default=12000)
    parser.add_argument('--zebra_arena_keep_tail', type=int, default=8)
    args = parser.parse_args()
    main(args)
