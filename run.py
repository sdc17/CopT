import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import torch.distributed as dist
import argparse
import os
from generation_utils import (
    set_seed,
    generate_cot,
    generate_copt_general
)
from grader import answer_match
from helper import load_default_effort


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
        prompts = [
            f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            for q in questions
        ]
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
    args = parser.parse_args()
    main(args)
