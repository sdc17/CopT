# Evaulate on General Reasoning

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py \
    --model_name Qwen/Qwen3-8B \
    --dataset_name gsm8k \
    --batch_size 256 \
    --method copt
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --method copt

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py \
    --model_name Qwen/Qwen3-8B \
    --dataset_name math500 \
    --batch_size 128 \
    --method copt 
python merge.py --model_name Qwen/Qwen3-8B --dataset_name math500 --method copt

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py \
    --model_name Qwen/Qwen3-8B \
    --dataset_name gpqa_diamond \
    --batch_size 32 \
    --method copt
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gpqa_diamond --method copt

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py \
    --model_name Qwen/Qwen3-8B \
    --dataset_name aime_2024 \
    --batch_size 30 \
    --max_new_tokens 38912 \
    --method copt
python merge.py --model_name Qwen/Qwen3-8B --dataset_name aime_2024 --method copt

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py \
    --model_name Qwen/Qwen3-8B \
    --dataset_name aime_2025 \
    --batch_size 30 \
    --max_new_tokens 38912 \
    --method copt
python merge.py --model_name Qwen/Qwen3-8B --dataset_name aime_2025 --method copt


# Evaulate on Agentic Reasoning

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run_agents.py \
    --model_name Qwen/Qwen3.5-35B-A3B \
    --dataset_name zebra_arena \
    --batch_size 16 \
    --method copt \
    --zebra_arena_space Small \
    --zebra_arena_max_turns 16
python merge.py --model_name Qwen/Qwen3.5-35B-A3B --dataset_name zebra_arena --method copt --zebra_arena_space Small

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run_agents.py \
    --model_name Qwen/Qwen3.5-35B-A3B \
    --dataset_name zebra_arena \
    --batch_size 8 \
    --method copt \
    --max_new_tokens 65536 \
    --zebra_arena_space Medium \
    --zebra_arena_max_turns 32
python merge.py --model_name Qwen/Qwen3.5-35B-A3B --dataset_name zebra_arena --method copt --zebra_arena_space Medium

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run_agents.py \
    --model_name Qwen/Qwen3.5-35B-A3B \
    --dataset_name zebra_arena \
    --batch_size 4 \
    --method copt \
    --max_new_tokens 98304 \
    --zebra_arena_space Large \
    --zebra_arena_max_turns 48
python merge.py --model_name Qwen/Qwen3.5-35B-A3B --dataset_name zebra_arena --method copt --zebra_arena_space Large
