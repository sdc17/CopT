<div align="center">
<h1>CopT: Contrastive On-Policy Thinking with Continuous Spaces for General and Agentic Reasoning</h1>
</div>

<p align="center">
    <a href="https://arxiv.org/pdf/2605.20075">
        <img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2605.20075-B31B1B?logo=arxiv" />
    </a>
    <a href="https://copt-web.github.io/">
        <img alt="Website" src="https://img.shields.io/badge/website-555555?logo=googlechrome" />
    </a><br>
</p>


## 👀 TL;DR
CopT is a reasoning pipeline with continuous-space verifiers, enabling LLMs to start with a draft answer and invoke on-policy thinking conditioned on it.

![CopT](assets/copt.png)

(a) Conceptual comparison between CoT thinking and CopT on-policy thinking. (b) CopT contrasts the output distributions under discrete and continuous inputs. (c) CopT improves peak accuracy, marked by *, across mathematics, coding, and agentic reasoning tasks and nearly halves token usage at matched accuracy.

## 🔍 Supported Models

* Qwen3 and Qwen3.5 model families, from 2B to 35B.

## ⚙️ Getting Started

### Clone the project
``` bash
git clone https://github.com/sdc17/CopT.git
cd CopT
```

### Environment setup
```bash
conda create -n copt python=3.12
conda activate copt
pip install -r requirements.txt
pip install transformers==5.7.0 # Only for Qwen3.5 support
```


## 📈 General Reasoning

```bash
# Evaluate on Math500, for example
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name math500 --batch_size 128 --method copt
python merge.py --model_name Qwen/Qwen3-8B --dataset_name math500 --method copt

# Reasoning effort control via tau_a and tau_r
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name math500 --batch_size 128 --method copt --tau_a 0.6 --tau_r 0.4
python merge.py --model_name Qwen/Qwen3-8B --dataset_name math500 --method copt
```
* Reasoning effort control
  * Decrease `--tau_a`: Increase reasoning effort by allowing fewer draft answers to be accepted directly.
  * Decrease `--tau_r`: Increase reasoning effort by making on-policy thinking rely less on draft answers.
  * Increase `--tau_a`: Decrease reasoning effort by allowing more draft answers to be accepted directly.
  * Increase `--tau_r`: Decrease reasoning effort by making on-policy thinking rely more on draft answers.
* Increase ``--nproc_per_node`` to enable faster evaluation on multiple GPUs
* Modify ``--model_name`` and ``--dataset_name`` for evaluation with different models and datasets
* Please check [run.sh](./run.sh) for more examples


## 📈 Agentic Reasoning

```bash

# Evaluate on ZebraArena as an example
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run_agents.py --model_name Qwen/Qwen3.5-35B-A3B --dataset_name zebra_arena --batch_size 16 --method copt --zebra_arena_space Small --zebra_arena_max_turns 16 
python merge.py --model_name Qwen/Qwen3.5-35B-A3B --dataset_name zebra_arena --method copt --zebra_arena_space Small

# Reasoning effort control via tau_a and tau_r
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run_agents.py --model_name Qwen/Qwen3.5-35B-A3B --dataset_name zebra_arena --batch_size 16 --method copt --zebra_arena_space Small --zebra_arena_max_turns 16 --tau_a 1.5 --tau_r 0
python merge.py --model_name Qwen/Qwen3.5-35B-A3B --dataset_name zebra_arena --method copt --zebra_arena_space Small
```

* Same reasoning effort control as general reasoning
* Download the dataset [here](https://drive.google.com/file/d/1GuLsT31rQgicmQJCS5QcnMUhiOkCU5xx/view?usp=sharing) or view the raw dataset [here](https://huggingface.co/datasets/WanjiaZhao/ZebraArena)
* Specify the dataset path with ``--zebra_arena_data_dir``
* We recommend using Qwen3.5 instead of Qwen3 for agentic tasks
* Increase ``--nproc_per_node`` to enable faster evaluation on multiple GPUs.
* Modify ``--model_name`` and ``--dataset_name`` for evaluation with different models and datasets
* Please check [run.sh](./run.sh) for more examples

## 💬 Acknowledgments

We thank the contributors of open-source projects [SwiReasoning](https://github.com/sdc17/SwiReasoning) and [ZebraArena](https://github.com/wanjiaZhao1203/ZebraArena).

## ✨ BibTeX

Please cite if you find our codebase helpful.

```bash
@misc{shi2026coptcontrastiveonpolicythinking,
      title={CopT: Contrastive On-Policy Thinking with Continuous Spaces for General and Agentic Reasoning}, 
      author={Dachuan Shi and Hanlin Zhu and Xiangchi Yuan and Wanjia Zhao and Kejing Xia and Wen Xiao and Wenke Lee},
      year={2026},
      eprint={2605.20075},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.20075}, 
}
```




