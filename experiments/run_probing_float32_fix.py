#!/usr/bin/env python3
"""
Fix: Re-compute CoT perturbation metrics with float32 to resolve nan issue.
Only computes perturbation/cross_sim/hs_norm for models that have nan values.
Saves updated metrics alongside existing probing results.
"""
import json, gc, os, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
CKPT_DIR = BASE / "results_v2" / "probing"
DEVICE = "cuda:0"  # Use cuda:0 since CUDA_VISIBLE_DEVICES handles mapping

MODELS = {
    "Qwen2-1.5B-Inst": "/path/to/models/Qwen2-1.5B-Instruct",
    "Qwen2.5-3B-Inst": "/path/to/models/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Inst": "/path/to/models/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Base": "/path/to/models/Qwen2.5-3B",
    "Qwen2.5-7B-Base": "/path/to/models/Qwen2.5-7B",
}

# Questions
import random
SEED = 42
N_QUESTIONS = 50

with open(BASE / "math_real_200.json") as f:
    data = json.load(f)
questions = data[:N_QUESTIONS]


def collect_hs_float32(model, tok, questions, prompt_type="baseline", use_chat=True):
    """Collect hidden states with float32 to avoid inf."""
    model.eval()
    activations = {}
    hooks = []
    n_layers = model.config.num_hidden_layers

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0][0, -1, :].detach().float()  # Force float32
        else:
            hs = output[0, -1, :].detach().float()
        # Clip inf/nan
        hs = torch.clamp(hs, -1e6, 1e6)
        activations['last_layer'] = hs.cpu().numpy().copy()

    h = model.model.layers[n_layers - 1].register_forward_hook(hook_fn)
    hooks.append(h)

    all_hs = []
    import torch
    for q_data in questions:
        q = q_data["query"]
        if prompt_type == "cot":
            q_modified = q + "\nLet's think step by step."
        else:
            q_modified = q

        if use_chat:
            messages = [{"role": "user", "content": q_modified}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = q_modified

        inp = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inp, max_new_tokens=256,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )

        hs = activations.get('last_layer')
        if hs is not None:
            all_hs.append(hs)
        else:
            all_hs.append(np.zeros(model.config.hidden_size, dtype=np.float32))

    for h in hooks:
        h.remove()

    return np.array(all_hs, dtype=np.float32)


import torch

def compute_perturbation_metrics(base_hs, cot_hs):
    """Compute perturbation, cross-similarity, intra-similarity, norms."""
    # Verify no inf/nan
    assert np.all(np.isfinite(base_hs)), "base_hs has inf/nan"
    assert np.all(np.isfinite(cot_hs)), "cot_hs has inf/nan"

    # Perturbation: 1 - cosine_sim
    perturbations = []
    for i in range(len(base_hs)):
        b, c = base_hs[i], cot_hs[i]
        cos = np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c) + 1e-10)
        perturbations.append(1 - cos)

    # Cross-condition similarity
    cross = [np.dot(base_hs[i], cot_hs[i]) / (np.linalg.norm(base_hs[i]) * np.linalg.norm(cot_hs[i]) + 1e-10)
             for i in range(len(base_hs))]

    # Intra-condition similarity
    sim_b = cosine_similarity(base_hs)
    np.fill_diagonal(sim_b, np.nan)
    sim_c = cosine_similarity(cot_hs)
    np.fill_diagonal(sim_c, np.nan)

    # Norms
    norms_b = np.linalg.norm(base_hs, axis=1)
    norms_c = np.linalg.norm(cot_hs, axis=1)

    return {
        "mean_cot_perturbation": float(np.mean(perturbations)),
        "std_cot_perturbation": float(np.std(perturbations)),
        "mean_cross_sim": float(np.mean(cross)),
        "mean_intra_sim_base": float(np.nanmean(sim_b)),
        "mean_intra_sim_cot": float(np.nanmean(sim_c)),
        "mean_hs_norm_base": float(np.mean(norms_b)),
        "mean_hs_norm_cot": float(np.mean(norms_c)),
    }


def main():
    print(f"Float32 Perturbation Fix | {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for model_name, model_path in MODELS.items():
        ckpt_file = CKPT_DIR / f"{model_name}_probing.json"
        if not ckpt_file.exists():
            print(f"  SKIP {model_name} (no existing results)", flush=True)
            continue

        d = json.load(open(ckpt_file))
        m = d["metrics"]

        # Check if perturbation is already valid
        if m.get("mean_cot_perturbation") is not None and not np.isnan(m.get("mean_cot_perturbation", float('nan'))):
            print(f"  SKIP {model_name} (perturbation already valid: {m['mean_cot_perturbation']:.4f})", flush=True)
            continue

        print(f"\n  Processing {model_name}...", flush=True)
        use_chat = "Inst" in model_name or "Inst" in model_name

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Load with float32 explicitly
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float32,  # KEY FIX: use float32
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        model.eval()
        print(f"    Loaded (float32): {model.config.num_hidden_layers} layers", flush=True)

        # Collect base hidden states
        print(f"    Collecting base HS...", flush=True)
        base_hs = collect_hs_float32(model, tok, questions, "baseline", use_chat)
        print(f"      base_hs: shape={base_hs.shape}, finite={np.all(np.isfinite(base_hs))}, "
              f"norm_range=[{np.linalg.norm(base_hs, axis=1).min():.1f}, {np.linalg.norm(base_hs, axis=1).max():.1f}]",
              flush=True)

        # Collect CoT hidden states
        print(f"    Collecting CoT HS...", flush=True)
        cot_hs = collect_hs_float32(model, tok, questions, "cot", use_chat)
        print(f"      cot_hs: shape={cot_hs.shape}, finite={np.all(np.isfinite(cot_hs))}, "
              f"norm_range=[{np.linalg.norm(cot_hs, axis=1).min():.1f}, {np.linalg.norm(cot_hs, axis=1).max():.1f}]",
              flush=True)

        # Compute perturbation metrics
        perturb_metrics = compute_perturbation_metrics(base_hs, cot_hs)
        print(f"    Perturbation: {perturb_metrics['mean_cot_perturbation']:.4f}", flush=True)
        print(f"    Cross-sim:    {perturb_metrics['mean_cross_sim']:.4f}", flush=True)
        print(f"    HS norm base: {perturb_metrics['mean_hs_norm_base']:.1f}", flush=True)
        print(f"    HS norm cot:  {perturb_metrics['mean_hs_norm_cot']:.1f}", flush=True)

        # Update existing results
        m.update(perturb_metrics)
        with open(ckpt_file, 'w') as f:
            json.dump(d, f, indent=2)
        print(f"    Saved updated metrics to {ckpt_file}", flush=True)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nDone: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
