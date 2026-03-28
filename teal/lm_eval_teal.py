import argparse
import json
import os
import sys

from transformers import AutoConfig, AutoModelForCausalLM

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from teal.model import LlamaSparseConfig, LlamaSparseForCausalLM
from teal.model import MistralSparseConfig, MistralSparseForCausalLM
from utils.utils import get_sparse_model, get_tokenizer

AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--teal_path", type=str, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["arc_challenge", "mmlu", "openbookqa", "winogrande"],
    )
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--greedy_flag", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    from lm_eval.evaluator import evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict

    tokenizer = get_tokenizer(args.model_name)
    model = get_sparse_model(
        args.model_name,
        device="auto",
        histogram_path=os.path.join(args.teal_path, "histograms"),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    if args.greedy_flag:
        model.load_greedy_sparsities(os.path.join(args.teal_path, "lookup"), args.sparsity)
    else:
        model.set_uniform_sparsity(args.sparsity)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device="cuda",
        dtype=args.dtype,
    )

    task_dict = get_task_dict(args.tasks)
    result = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=args.limit,
        log_samples=False,
    )
    print(json.dumps(result.get("results", {}), indent=2))


if __name__ == "__main__":
    main()
