import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from teal.model import LlamaSparseConfig, LlamaSparseForCausalLM
from teal.model import MistralSparseConfig, MistralSparseForCausalLM
from utils.data import get_dataset
from utils.eval_ppl import eval_ppl
from utils.utils import get_sparse_model, get_tokenizer

AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)


def _register_projection_hooks(model):
    stats = defaultdict(lambda: {"zero": 0, "total": 0})
    handles = []

    for layer in model.model.layers:
        modules = {
            "q": layer.self_attn.sparse_fns["q"],
            "k": layer.self_attn.sparse_fns["k"],
            "v": layer.self_attn.sparse_fns["v"],
            "o": layer.self_attn.sparse_fns["o"],
            "up": layer.mlp.sparse_fns["up"],
            "gate": layer.mlp.sparse_fns["gate"],
            "down": layer.mlp.sparse_fns["down"],
        }

        for proj_name, module in modules.items():
            def _make_hook(name):
                def _hook(_module, _inputs, output):
                    if not torch.is_tensor(output):
                        return
                    y = output.detach()
                    stats[name]["zero"] += int((y == 0).sum().item())
                    stats[name]["total"] += int(y.numel())

                return _hook

            handles.append(module.register_forward_hook(_make_hook(proj_name)))

    return handles, stats


def _load_dataset(args):
    subset = None if args.subset in [None, "None", "none", ""] else args.subset
    size = None if args.size is not None and args.size <= 0 else args.size
    return get_dataset(
        args.dataset_name,
        subset=subset,
        split=args.split,
        size=size,
        start=args.start,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--teal_path", type=str, required=True)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--greedy_flag", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_name)
    model = get_sparse_model(
        args.model_name,
        device="auto",
        histogram_path=os.path.join(args.teal_path, "histograms"),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    # Dense PPL
    model.reset_sparsities()
    dense_dataset = _load_dataset(args)
    dense_ppl = eval_ppl(model, tokenizer, device=args.device, dataset=dense_dataset, debug=False)

    # Sparse PPL + projection sparsity
    if args.greedy_flag:
        model.load_greedy_sparsities(os.path.join(args.teal_path, "lookup"), args.sparsity)
    else:
        model.set_uniform_sparsity(args.sparsity)

    handles, stats = _register_projection_hooks(model)
    sparse_dataset = _load_dataset(args)
    sparse_ppl = eval_ppl(model, tokenizer, device=args.device, dataset=sparse_dataset, debug=False)

    for h in handles:
        h.remove()

    proj_zero_ratio = {}
    for name in ["q", "k", "v", "o", "up", "gate", "down"]:
        total = stats[name]["total"]
        zero = stats[name]["zero"]
        ratio = float(zero / total) if total > 0 else 0.0
        proj_zero_ratio[name] = ratio

    result = {
        "model_name": args.model_name,
        "dtype": args.dtype,
        "dataset": {
            "name": args.dataset_name,
            "subset": args.subset,
            "split": args.split,
            "size": args.size,
            "start": args.start,
        },
        "teal_code_default_sparsity_p": 0.5,
        "eval_sparsity_p": args.sparsity,
        "dense_ppl": dense_ppl,
        "sparse_ppl": sparse_ppl,
        "projection_zero_ratio": proj_zero_ratio,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
