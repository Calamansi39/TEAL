import sys,os
# sys.path.append('../')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# sys.path.append(os.path.join(parent_dir, 'utils'))

import torch
from tqdm import tqdm
import os
import argparse



if __name__ == "__main__":
    from utils.utils import get_tokenizer, get_sparse_model
    from utils.eval_ppl import eval_ppl

    from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
    from teal.model import MistralSparseForCausalLM, MistralSparseConfig

    from utils.data import get_dataset

    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("llama_sparse", LlamaSparseConfig)
    AutoConfig.register("mistral_sparse", MistralSparseConfig)

    AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
    AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf",help='Name of the model to use')
    parser.add_argument('--teal_path', type=str, required=True,help='Path to the teal input')
    parser.add_argument('--greedy_flag', action='store_true', help='Flag for greedy')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity level')
    parser.add_argument('--dtype', type=str, default='float16', help='Model dtype: float16/bfloat16/float32')
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2', help='Attention implementation')
    parser.add_argument('--dataset_name', type=str, default='tatsu-lab/alpaca', help='Dataset name for PPL')
    parser.add_argument('--subset', type=str, default=None, help='Dataset subset/config (e.g. wikitext-2-raw-v1)')
    parser.add_argument('--split', type=str, default='train', help='Dataset split')
    parser.add_argument('--size', type=int, default=250, help='Number of streaming samples; if <=0 use full split')
    parser.add_argument('--start', type=int, default=0, help='Streaming offset when size > 0')
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_name)
    subset = None if args.subset in [None, "None", "none", ""] else args.subset
    size = None if args.size is not None and args.size <= 0 else args.size
    model = get_sparse_model(
        args.model_name,
        device="auto",
        histogram_path=os.path.join(args.teal_path, "histograms"),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    def load_eval_dataset():
        return get_dataset(
            args.dataset_name,
            subset=subset,
            split=args.split,
            size=size,
            start=args.start,
        )


    print("Evaluating dense PPL")
    print("="*40)
    dense_ppl = eval_ppl(model, tokenizer, device="cuda", dataset=load_eval_dataset(), debug=False)
    print(f"PPL: {dense_ppl}")


    print("Evaluating sparse PPL at sparsity level: ", args.sparsity)
    print("="*40)
    if args.greedy_flag:
        print("Evaluating greedy PPL")
        greedy_path = os.path.join(args.teal_path, "lookup")
        model.load_greedy_sparsities(greedy_path, args.sparsity)
    else:
        print("Evaluating uniform PPL")
        model.set_uniform_sparsity(args.sparsity)

    sparse_ppl = eval_ppl(model, tokenizer, device="cuda", dataset=load_eval_dataset(), debug=False)
    print(f"PPL: {sparse_ppl}")

    print("="*40)
