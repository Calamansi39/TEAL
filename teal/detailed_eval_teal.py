import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except Exception as exc:
    raise ImportError("matplotlib is required for heatmap output.") from exc

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


PROJ_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
DEFAULT_HEATMAP_PROJS = {"q_proj", "k_proj", "v_proj", "up_proj", "gate_proj", "down_proj"}


def parse_layers(text: str):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def resolve_heatmap_projs(text: str):
    text = (text or "").strip().lower()
    if not text:
        return set(DEFAULT_HEATMAP_PROJS)
    out = set()
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "qkv":
            out.update({"q_proj", "k_proj", "v_proj"})
        elif part == "mlp":
            out.update({"up_proj", "gate_proj", "down_proj"})
        elif part == "attn":
            out.update({"q_proj", "k_proj", "v_proj", "o_proj"})
        elif part == "all":
            out.update(PROJ_ORDER)
        elif part in PROJ_ORDER:
            out.add(part)
        else:
            raise ValueError(f"Unknown projection token in --heatmap_projs: {part}")
    return out


def first_device(model):
    return next(model.parameters()).device


def load_eval_dataset(name, subset, split, size, start):
    subset = None if subset in [None, "None", "none", ""] else subset
    size = None if size is not None and size <= 0 else size
    return get_dataset(name, subset=subset, split=split, size=size, start=start)


def safe_eval_ppl(model, tokenizer, device, dataset):
    try:
        return eval_ppl(model, tokenizer, device=device, dataset=dataset, debug=False), True
    except RuntimeError as exc:
        # Happens when sequence is too short and no sliding-window steps are produced.
        if "non-empty TensorList" in str(exc):
            print(
                "[WARN] PPL skipped: dataset text is too short for current context/window settings. "
                "Set a larger --size to compute PPL."
            )
            return float("nan"), False
        raise


def run_single_forward_for_capture(model, tokenizer, dataset, max_length=2048):
    device = first_device(model)
    text = None
    for sample in dataset:
        t = sample.get("text", "")
        if isinstance(t, str) and t.strip():
            text = t
            break
    if text is None:
        return False

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attn = attn.to(device) if attn is not None else None
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attn)
    return True


def attach_input_hooks(
    model,
    target_layers_0idx,
    heatmap_seq_len,
    capture_projs,
    collect_stats=True,
    collect_capture=True,
    heatmap_d_start=0,
    heatmap_d_end=-1,
):
    n_layers = len(model.model.layers)
    stats = {k: [{"zero": 0, "total": 0} for _ in range(n_layers)] for k in PROJ_ORDER}
    captured = {}
    handles = []

    for i, layer in enumerate(model.model.layers):
        module_map = {
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
            "up_proj": layer.mlp.up_proj,
            "gate_proj": layer.mlp.gate_proj,
            "down_proj": layer.mlp.down_proj,
        }

        for proj_name, module in module_map.items():
            def _make_hook(layer_idx, p_name):
                def _hook(_module, inputs):
                    if not inputs:
                        return
                    x = inputs[0]
                    if not torch.is_tensor(x):
                        return
                    x_det = x.detach()
                    if collect_stats:
                        stats[p_name][layer_idx]["zero"] += int((x_det == 0).sum().item())
                        stats[p_name][layer_idx]["total"] += int(x_det.numel())

                    key = (layer_idx, p_name)
                    if (
                        collect_capture
                        and
                        p_name in capture_projs
                        and layer_idx in target_layers_0idx
                        and key not in captured
                        and x_det.dim() == 3
                        and x_det.size(0) >= 1
                    ):
                        x0 = x_det[0]
                        if heatmap_seq_len > 0:
                            x0 = x0[:heatmap_seq_len]
                        d0 = max(0, int(heatmap_d_start))
                        d1 = int(heatmap_d_end)
                        if d1 <= 0 or d1 > x0.size(1):
                            d1 = x0.size(1)
                        if d0 >= d1:
                            d0 = 0
                            d1 = x0.size(1)
                        x0 = x0[:, d0:d1]
                        captured[key] = x0.float().cpu()

                return _hook

            handles.append(module.register_forward_pre_hook(_make_hook(i, proj_name)))

    return handles, stats, captured


def summarize_stats(stats):
    per_layer = {}
    overall = {}
    for proj_name, entries in stats.items():
        layer_vals = []
        z_all = 0
        t_all = 0
        for e in entries:
            zero = e["zero"]
            total = e["total"]
            z_all += zero
            t_all += total
            layer_vals.append(float(zero / total) if total > 0 else 0.0)
        per_layer[proj_name] = layer_vals
        overall[proj_name] = float(z_all / t_all) if t_all > 0 else 0.0
    return per_layer, overall


def save_heatmaps(
    captured,
    output_dir,
    log_min_exp,
    square_s=0,
    square_d=64,
    dpi=300,
    max_side_inches=16.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for (layer_idx, proj_name), x in sorted(captured.items(), key=lambda kv: kv[0]):
        # x shape: [S, D], from original [B, S, D] with B=1
        x_abs = x.abs()
        maxv = float(x_abs.max().item())
        x_norm = x_abs / maxv if maxv > 0 else x_abs
        zero_mask = (x == 0).float()

        base = f"layer{layer_idx + 1:02d}_{proj_name}"
        heatmap_path = output_dir / f"{base}_norm01_heatmap.png"
        zero_path = output_dir / f"{base}_zero_mask.png"

        exp = int(log_min_exp) if log_min_exp is not None else 5
        if exp < 1:
            exp = 1
        vmin = 10.0 ** (-exp)
        x_plot = torch.clamp(x_norm, min=vmin, max=1.0)
        x_plot_sd = x_plot.numpy()
        zero_sd = zero_mask.numpy()
        # Plot as x=S, y=D by transposing from [S, D] -> [D, S].
        x_plot_ds = x_plot_sd.T
        zero_ds = zero_sd.T
        s_len, d_len = int(x.shape[0]), int(x.shape[1])
        max_side_inches = float(max_side_inches)
        if max_side_inches <= 0:
            max_side_inches = 16.0
        scale = max_side_inches / float(max(s_len, d_len, 1))
        fig_w = max(1.0, s_len * scale)
        fig_h = max(1.0, d_len * scale)

        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(
            x_plot_ds,
            aspect="equal",
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=1.0),
            extent=(0, s_len, 0, d_len),
            origin="lower",
        )
        cbar = plt.colorbar()
        ticks = [10.0 ** (-k) for k in range(exp, 0, -1)] + [1.0]
        cbar.set_ticks(ticks)
        plt.xlabel("S")
        plt.ylabel("D")
        plt.title(f"{base} normalized |x| in [0,1], log scale [1e-{exp}, 1] (x=S, y=D, B=1)")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=int(dpi))
        plt.close()

        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(
            zero_ds,
            aspect="equal",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            extent=(0, s_len, 0, d_len),
            origin="lower",
        )
        plt.colorbar()
        plt.xlabel("S")
        plt.ylabel("D")
        plt.title(f"{base} zero mask (X==0) (x=S, y=D)")
        plt.tight_layout()
        plt.savefig(zero_path, dpi=int(dpi))
        plt.close()

        item = {
            "layer_1idx": layer_idx + 1,
            "projection": proj_name,
            "shape_sd": [int(x.shape[0]), int(x.shape[1])],
            "norm01_heatmap": str(heatmap_path),
            "zero_mask_heatmap": str(zero_path),
        }

        sq_s = int(square_s) if square_s is not None else 0
        sq_d = int(square_d) if square_d is not None else 64
        if sq_s > 0 and sq_d > 0 and x_norm.size(0) >= sq_s and x_norm.size(1) >= sq_d:
            x_head = x_plot[:sq_s, :sq_d]
            x_tail = x_plot[-sq_s:, :sq_d]
            z_head = zero_mask[:sq_s, :sq_d]
            z_tail = zero_mask[-sq_s:, :sq_d]

            head_path = output_dir / f"{base}_head_s{sq_s}_d{sq_d}_norm01_heatmap.png"
            tail_path = output_dir / f"{base}_tail_s{sq_s}_d{sq_d}_norm01_heatmap.png"
            head_zero_path = output_dir / f"{base}_head_s{sq_s}_d{sq_d}_zero_mask.png"
            tail_zero_path = output_dir / f"{base}_tail_s{sq_s}_d{sq_d}_zero_mask.png"

            plt.figure(figsize=(6, 6))
            plt.imshow(
                x_head.numpy().T,
                aspect="equal",
                cmap="magma",
                norm=LogNorm(vmin=vmin, vmax=1.0),
                extent=(0, sq_s, 0, sq_d),
                origin="lower",
            )
            cbar = plt.colorbar()
            cbar.set_ticks(ticks)
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} head S[0:{sq_s}] D[0:{sq_d}] (x=S, y=D)")
            plt.tight_layout()
            plt.savefig(head_path, dpi=int(dpi))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(
                x_tail.numpy().T,
                aspect="equal",
                cmap="magma",
                norm=LogNorm(vmin=vmin, vmax=1.0),
                extent=(0, sq_s, 0, sq_d),
                origin="lower",
            )
            cbar = plt.colorbar()
            cbar.set_ticks(ticks)
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} tail S[-{sq_s}:] D[0:{sq_d}] (x=S, y=D)")
            plt.tight_layout()
            plt.savefig(tail_path, dpi=int(dpi))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(
                z_head.numpy().T,
                aspect="equal",
                cmap="gray_r",
                vmin=0,
                vmax=1,
                extent=(0, sq_s, 0, sq_d),
                origin="lower",
            )
            plt.colorbar()
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} head S[0:{sq_s}] zero mask (x=S, y=D)")
            plt.tight_layout()
            plt.savefig(head_zero_path, dpi=int(dpi))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(
                z_tail.numpy().T,
                aspect="equal",
                cmap="gray_r",
                vmin=0,
                vmax=1,
                extent=(0, sq_s, 0, sq_d),
                origin="lower",
            )
            plt.colorbar()
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} tail S[-{sq_s}:] zero mask (x=S, y=D)")
            plt.tight_layout()
            plt.savefig(tail_zero_path, dpi=int(dpi))
            plt.close()

            item.update(
                {
                    "square_shape_sd": [sq_s, sq_d],
                    "square_head_norm01_heatmap": str(head_path),
                    "square_tail_norm01_heatmap": str(tail_path),
                    "square_head_zero_mask": str(head_zero_path),
                    "square_tail_zero_mask": str(tail_zero_path),
                }
            )

        written.append(item)

    return written


def run_lm_eval(model, tokenizer, tasks, limit, batch_size, max_length, dtype):
    from lm_eval.evaluator import evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device="cuda",
        dtype=dtype,
    )
    task_dict = get_task_dict(tasks)
    result = evaluate(lm=lm, task_dict=task_dict, limit=limit, log_samples=False)
    return result.get("results", {})


def pick_metric(task_result, preferred):
    for k in preferred:
        if k in task_result:
            return float(task_result[k])
    return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--teal_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--greedy_flag", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--size", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks", nargs="+", default=["arc_challenge", "mmlu", "openbookqa", "winogrande"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_max_length", type=int, default=2048)
    parser.add_argument("--heatmap_layers", type=str, default="8,16,24")
    parser.add_argument(
        "--heatmap_layer_index_base",
        type=int,
        default=1,
        choices=[0, 1],
        help="Interpret heatmap layer ids as 0-based or 1-based.",
    )
    parser.add_argument("--heatmap_seq_len", type=int, default=512)
    parser.add_argument("--heatmap_d_start", type=int, default=0, help="Start channel index (inclusive).")
    parser.add_argument("--heatmap_d_end", type=int, default=-1, help="End channel index (exclusive); <=0 means full D.")
    parser.add_argument(
        "--heatmap_projs",
        type=str,
        default="qkv,mlp",
        help="Comma list: qkv, mlp, attn, all, or explicit proj names like q_proj,down_proj.",
    )
    parser.add_argument("--heatmap_dir", type=str, default=None)
    parser.add_argument(
        "--heatmap_stage",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="Which stage to capture heatmaps from. dense=before sparsification, sparse=after sparsification.",
    )
    parser.add_argument(
        "--heatmap_log_min_exp",
        type=int,
        default=5,
        help="Log-scale lower bound exponent E for heatmap color: [1e-E, 1].",
    )
    parser.add_argument(
        "--heatmap_square_s",
        type=int,
        default=0,
        help="If >0, also export square heatmaps for head/tail S windows with this size.",
    )
    parser.add_argument(
        "--heatmap_square_d",
        type=int,
        default=64,
        help="D width for square head/tail heatmaps.",
    )
    parser.add_argument("--heatmap_dpi", type=int, default=300, help="PNG dpi for heatmap rendering.")
    parser.add_argument(
        "--heatmap_max_side_inches",
        type=float,
        default=16.0,
        help="Maximum figure side in inches; other side scales by true S/D index ratio.",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_name)
    model = get_sparse_model(
        args.model_name,
        device="auto",
        histogram_path=os.path.join(args.teal_path, "histograms"),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    n_layers = len(model.model.layers)

    user_layers = parse_layers(args.heatmap_layers)
    capture_projs = resolve_heatmap_projs(args.heatmap_projs)
    if args.heatmap_layer_index_base == 0:
        target_layers_0idx = [x for x in user_layers if 0 <= x < n_layers]
    else:
        target_layers_0idx = [x - 1 for x in user_layers if 1 <= x <= n_layers]
    captured = {}

    model.reset_sparsities()
    dense_dataset = load_eval_dataset(args.dataset_name, args.subset, args.split, args.size, args.start)

    dense_handles = []
    if args.heatmap_stage == "dense":
        dense_handles, _, captured = attach_input_hooks(
            model,
            set(target_layers_0idx),
            args.heatmap_seq_len,
            capture_projs,
            collect_stats=False,
            collect_capture=True,
            heatmap_d_start=args.heatmap_d_start,
            heatmap_d_end=args.heatmap_d_end,
        )
    dense_ppl, dense_ok = safe_eval_ppl(model, tokenizer, args.device, dense_dataset)
    for h in dense_handles:
        h.remove()
    if args.heatmap_stage == "dense" and not dense_ok and len(captured) == 0:
        fallback_dataset = load_eval_dataset(args.dataset_name, args.subset, args.split, args.size, args.start)
        run_single_forward_for_capture(model, tokenizer, fallback_dataset, max_length=args.eval_max_length)

    if args.greedy_flag:
        model.load_greedy_sparsities(os.path.join(args.teal_path, "lookup"), args.sparsity)
    else:
        model.set_uniform_sparsity(args.sparsity)

    sparse_handles = []
    if args.heatmap_stage == "sparse":
        sparse_handles, hook_stats, captured = attach_input_hooks(
            model,
            set(target_layers_0idx),
            args.heatmap_seq_len,
            capture_projs,
            collect_stats=True,
            collect_capture=True,
            heatmap_d_start=args.heatmap_d_start,
            heatmap_d_end=args.heatmap_d_end,
        )
    else:
        sparse_handles, hook_stats, _ = attach_input_hooks(
            model,
            set(),
            args.heatmap_seq_len,
            capture_projs,
            collect_stats=True,
            collect_capture=False,
            heatmap_d_start=args.heatmap_d_start,
            heatmap_d_end=args.heatmap_d_end,
        )

    sparse_dataset = load_eval_dataset(args.dataset_name, args.subset, args.split, args.size, args.start)
    sparse_ppl, sparse_ppl_ok = safe_eval_ppl(model, tokenizer, args.device, sparse_dataset)
    if not sparse_ppl_ok and len(captured) == 0:
        # Ensure heatmaps can still be produced for very small dataset sizes.
        fallback_dataset = load_eval_dataset(args.dataset_name, args.subset, args.split, args.size, args.start)
        run_single_forward_for_capture(model, tokenizer, fallback_dataset, max_length=args.eval_max_length)

    for h in sparse_handles:
        h.remove()

    per_layer_ratio, overall_ratio = summarize_stats(hook_stats)
    heatmap_dir = args.heatmap_dir or os.path.join(args.teal_path, "heatmaps_teal")
    heatmaps = save_heatmaps(
        captured,
        heatmap_dir,
        args.heatmap_log_min_exp,
        square_s=args.heatmap_square_s,
        square_d=args.heatmap_square_d,
        dpi=args.heatmap_dpi,
        max_side_inches=args.heatmap_max_side_inches,
    )

    eval_results = run_lm_eval(
        model,
        tokenizer,
        args.tasks,
        args.limit,
        args.eval_batch_size,
        args.eval_max_length,
        args.dtype,
    )

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
        "x_zero_ratio_before_xw": {
            "per_layer": per_layer_ratio,
            "overall": overall_ratio,
            "note": "Ratios are measured on linear inputs X before XW over sparse PPL pass.",
        },
        "heatmap_stage": args.heatmap_stage,
        "heatmap_layer_index_base": int(args.heatmap_layer_index_base),
        "heatmap_log_scale": {
            "vmax": 1.0,
            "vmin": 10.0 ** (-int(args.heatmap_log_min_exp)),
        },
        "heatmap_square_window": {
            "enabled": bool(int(args.heatmap_square_s) > 0),
            "s": int(args.heatmap_square_s),
            "d": int(args.heatmap_square_d),
        },
        "heatmap_render": {
            "dpi": int(args.heatmap_dpi),
            "max_side_inches": float(args.heatmap_max_side_inches),
        },
        "heatmap_projs": sorted(list(capture_projs)),
        "heatmap_targets_user_layers_1idx": user_layers,
        "heatmap_targets_resolved_layers_0idx": target_layers_0idx,
        "heatmap_d_range": [int(args.heatmap_d_start), int(args.heatmap_d_end)],
        "heatmaps": heatmaps,
        "benchmark": {
            "arc_challenge": pick_metric(eval_results.get("arc_challenge", {}), ["acc_norm,none", "acc,none"]),
            "mmlu": pick_metric(eval_results.get("mmlu", {}), ["acc,none"]),
            "openbookqa": pick_metric(eval_results.get("openbookqa", {}), ["acc_norm,none", "acc,none"]),
            "winogrande": pick_metric(eval_results.get("winogrande", {}), ["acc,none"]),
        },
        "raw_eval_results": eval_results,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
