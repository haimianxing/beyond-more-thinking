#!/usr/bin/env python3
"""
Logit Lens via TransformerLens: Layer-Level Interpretability
=============================================================
Uses TransformerLens 3.0 for clean logit lens extraction.
Projects every layer's residual stream through the unembed matrix
to reveal at which layer the model "commits" to an answer.

Hypotheses:
  H1: 3B-Instruct commits to answers at earlier layers than 1.5B
  H2: CoT disrupts the answer trajectory for 3B-Instruct
  H3: 3B-Base does NOT show early commitment
  H4: Questions where CoT hurts show the model "had the right answer"
      at intermediate layers but was redirected

Models: Qwen2-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-3B-Base
Dataset: MATH (50 questions)
"""
import sys, os, json, gc, random, re, warnings
import torch, numpy as np
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
SEED = 42
N_QUESTIONS = 50

MODELS = {
    "Qwen2-1.5B-Inst": "/path/to/models/Qwen2-1.5B-Instruct",
    "Qwen2.5-3B-Inst": "/path/to/models/Qwen2.5-3B-Instruct",
    "Qwen2.5-3B-Base": "/path/to/models/Qwen2.5-3B",
}

DATA_FILE = BASE / "math_real_200.json"
CKPT_DIR = BASE / "results_v2" / "logit_lens"


def extract_ans(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums: return nums[-1]
    for pat in [
        r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+)([^\n.]+)',
        r'answer[:\s]+([^\n.]+)',
    ]:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            nums = re.findall(r'-?\d+\.?\d*', matches[-1].group(1))
            if nums: return nums[-1]
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line: continue
        nums = re.findall(r'-?\d+\.?\d*', line)
        if nums: return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]


def check(p, g):
    p, g = p.strip().replace(',','').replace(' ',''), str(g).strip().replace(',','').replace(' ','')
    if p == g: return True
    try: return abs(float(p)-float(g)) < 1e-6
    except: return p.lower() == g.lower()


def run_logit_lens_tl(model_name, model_path, questions, prompt_type="baseline"):
    """
    Use TransformerLens to run logit lens on all layers.
    """
    from transformer_lens import HookedTransformer

    print(f"  Loading {model_name} via TransformerLens...", flush=True)

    # Load with TL's from_pretrained (handles dtype, norm, unembed automatically)
    tl_model = HookedTransformer.from_pretrained_no_processing(
        model_path,
        dtype=torch.float16,
        device="cuda:0",
    )
    tl_model.eval()
    n_layers = tl_model.cfg.n_layers
    d_vocab = tl_model.cfg.d_vocab
    print(f"  Loaded: {n_layers} layers, d_vocab={d_vocab}", flush=True)

    use_chat = "Inst" in model_name or model_name.endswith("Instruct")

    results = []

    for qi, q_data in enumerate(questions):
        q = q_data["query"]
        gt = str(q_data.get("ground_truth", q_data.get("answer", "")))
        gt_clean = gt.strip().replace(',', '').replace(' ', '')

        # Build prompt
        content = q + "\nLet's think step by step." if prompt_type == "cot" else q
        if use_chat:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            messages = [{"role": "user", "content": content}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            del tok
        else:
            prompt = content

        # Tokenize with TL's tokenizer
        tokens = tl_model.to_tokens(prompt)
        seq_len = tokens.shape[1]

        # Run with cache to get all activations
        with torch.no_grad():
            logits, cache = tl_model.run_with_cache(tokens)

        # Logit Lens: project each layer's residual stream through unembed
        layer_predictions = []

        for layer in range(n_layers):
            # Get residual stream at this layer (last token position)
            # TL caches activations by name
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]  # (d_model,)

            # Project through LayerNorm + Unembed (TL handles this)
            # TL's built-in logit lens: apply ln_final + W_U
            resid_normed = tl_model.ln_final(resid.unsqueeze(0)).squeeze(0)
            layer_logits = resid_normed @ tl_model.W_U  # (d_vocab,)
            layer_probs = torch.softmax(layer_logits.float(), dim=-1)

            top_prob, top_id = layer_probs.max(dim=-1)
            top_token = tl_model.to_string(top_id.item())

            # Get GT answer probability
            gt_token_ids = tl_model.to_tokens(gt_clean, prepend_bos=False).squeeze()
            if gt_token_ids.dim() == 0:
                gt_token_ids = gt_token_ids.unsqueeze(0)
            gt_prob = float(layer_probs[gt_token_ids[0]]) if len(gt_token_ids) >= 1 else 0.0

            # Entropy
            entropy = -torch.sum(layer_probs * torch.log(layer_probs + 1e-10)).item()

            layer_predictions.append({
                "layer": layer,
                "top_token": top_token,
                "top_prob": float(top_prob),
                "gt_prob": gt_prob,
                "entropy": float(entropy),
            })

        results.append({
            "q": qi,
            "gt": gt_clean,
            "layer_data": layer_predictions,
            "n_layers": n_layers,
        })

        if (qi+1) % 10 == 0:
            avg_gt = np.mean([max(ld["gt_prob"] for ld in r["layer_data"]) for r in results])
            print(f"    [{model_name}/{prompt_type}] {qi+1}/{len(questions)} "
                  f"avg_max_gt_prob={avg_gt:.4f}", flush=True)

    del tl_model; torch.cuda.empty_cache(); gc.collect()
    return results


def analyze_results(results, model_name, prompt_type):
    """Extract key metrics from logit lens results."""
    n_layers = results[0]["n_layers"]

    emergence_layers = []
    max_gt_probs = []

    for r in results:
        emergence = n_layers
        max_prob = 0
        for ld in r["layer_data"]:
            if ld["gt_prob"] > max_prob:
                max_prob = ld["gt_prob"]
            if ld["gt_prob"] > 0.05 and emergence == n_layers:
                emergence = ld["layer"]
        emergence_layers.append(emergence)
        max_gt_probs.append(max_prob)

    # Average trajectory
    layer_gt = np.zeros(n_layers)
    layer_top = np.zeros(n_layers)
    layer_ent = np.zeros(n_layers)

    for r in results:
        for ld in r["layer_data"]:
            layer_gt[ld["layer"]] += ld["gt_prob"]
            layer_top[ld["layer"]] += ld["top_prob"]
            layer_ent[ld["layer"]] += ld["entropy"]

    layer_gt /= len(results)
    layer_top /= len(results)
    layer_ent /= len(results)

    return {
        "model": model_name,
        "prompt_type": prompt_type,
        "n_layers": n_layers,
        "n_questions": len(results),
        "mean_emergence_layer": float(np.mean(emergence_layers)),
        "std_emergence_layer": float(np.std(emergence_layers)),
        "peak_gt_layer": int(np.argmax(layer_gt)),
        "peak_gt_prob": float(np.max(layer_gt)),
        "mean_max_gt_prob": float(np.mean(max_gt_probs)),
        "layer_gt_probs": layer_gt.tolist(),
        "layer_top_probs": layer_top.tolist(),
        "layer_entropies": layer_ent.tolist(),
    }


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Logit Lens (TransformerLens) | {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    with open(DATA_FILE) as f:
        data = json.load(f)
    questions = data[:N_QUESTIONS]
    print(f"Using {len(questions)} MATH questions", flush=True)

    all_results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"Model: {model_name}", flush=True)
        print(f"{'='*70}", flush=True)

        for prompt_type in ["baseline", "cot"]:
            key = f"{model_name}_{prompt_type}"
            ckpt = CKPT_DIR / f"{key}.json"
            if ckpt.exists():
                print(f"  SKIP {key} (cached)", flush=True)
                all_results[key] = json.load(open(ckpt))
                continue

            print(f"\n  Running {key}...", flush=True)
            lens_results = run_logit_lens_tl(model_name, model_path, questions, prompt_type)
            metrics = analyze_results(lens_results, model_name, prompt_type)

            save_data = {"metrics": metrics}
            with open(ckpt, 'w') as f:
                json.dump(save_data, f, indent=2)
            all_results[key] = save_data

            print(f"  => {key}: emergence={metrics['mean_emergence_layer']:.1f}/{metrics['n_layers']}, "
                  f"peak_gt={metrics['peak_gt_prob']:.4f} @L{metrics['peak_gt_layer']}", flush=True)

    # === Summary ===
    print(f"\n{'='*80}", flush=True)
    print("LOGIT LENS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<22} {'Type':>6} {'Emerg L':>8} {'Peak GT':>8} {'Peak L':>7} {'Max GT':>8}", flush=True)
    print("-" * 62, flush=True)

    for key, data in sorted(all_results.items()):
        m = data["metrics"]
        typ = "CoT" if "cot" in key else "Base"
        print(f"{m['model']:<22} {typ:>6} {m['mean_emergence_layer']:>8.1f} "
              f"{m['peak_gt_prob']:>8.4f} {m['peak_gt_layer']:>7d} {m['mean_max_gt_prob']:>8.4f}", flush=True)

    # === Hypothesis Testing ===
    print(f"\n[H1] 3B-Inst earlier emergence than 1.5B-Inst:", flush=True)
    i15 = all_results.get("Qwen2-1.5B-Inst_baseline", {}).get("metrics", {})
    i3 = all_results.get("Qwen2.5-3B-Inst_baseline", {}).get("metrics", {})
    if i15 and i3:
        print(f"  1.5B: emergence at {i15['mean_emergence_layer']:.1f}/{i15['n_layers']}", flush=True)
        print(f"  3B:   emergence at {i3['mean_emergence_layer']:.1f}/{i3['n_layers']}", flush=True)

    print(f"\n[H2] CoT disrupts 3B-Inst:", flush=True)
    i3c = all_results.get("Qwen2.5-3B-Inst_cot", {}).get("metrics", {})
    if i3 and i3c:
        print(f"  Baseline peak GT: {i3['peak_gt_prob']:.4f}", flush=True)
        print(f"  CoT peak GT:      {i3c['peak_gt_prob']:.4f}", flush=True)
        print(f"  Delta: {i3c['peak_gt_prob']-i3['peak_gt_prob']:+.4f}", flush=True)

    print(f"\n[H3] Base vs Instruct (3B):", flush=True)
    b3 = all_results.get("Qwen2.5-3B-Base_baseline", {}).get("metrics", {})
    if i3 and b3:
        print(f"  Base peak GT:  {b3['peak_gt_prob']:.4f}", flush=True)
        print(f"  Inst peak GT:  {i3['peak_gt_prob']:.4f}", flush=True)

    print(f"\nDone: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
