#!/usr/bin/env python3
"""
Logit Lens Experiment v2: Richer Analysis for Strategy Reversal
================================================================
Key improvement over v1: At the input position, the model predicts
its FIRST generated token (usually text like "To", "Let"), not the
answer digit. So raw GT probability is ~0 and meaningless.

This version adds:
  1. Answer digit RANK across layers (more informative than probability)
  2. Top-5 tokens per layer saved for qualitative analysis
  3. Digit mass (total probability on all digit tokens)
  4. Entropy trajectory comparison between models
  5. "Confidence layer" - where top-1 probability first exceeds threshold

Hypotheses:
  H1: 3B-Inst shows earlier confidence commitment (entropy drops faster)
  H2: CoT increases entropy for 3B-Inst (disrupts internalized reasoning)
  H3: 3B-Base has higher entropy throughout (no structured output pattern)
  H4: Answer digit rank drops (improves) earlier for 3B-Inst vs 1.5B

Models: Qwen2-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-3B-Base
Dataset: MATH (50 questions)
"""
import sys, os, json, gc, random, re, warnings
import torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
SEED = 42
N_QUESTIONS = 50
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES handles mapping

MODELS = {
    "Qwen2-1.5B-Inst": ("/path/to/models/Qwen2-1.5B-Instruct", True),
    "Qwen2.5-3B-Inst": ("/path/to/models/Qwen2.5-3B-Instruct", True),
    "Qwen2.5-3B-Base": ("/path/to/models/Qwen2.5-3B", False),
}

DATA_FILE = BASE / "math_real_200.json"
CKPT_DIR = BASE / "results_v2" / "logit_lens_v2"


def extract_ans(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums: return nums[-1]
    for pat in [
        r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+|so the answer is|final answer[:\s]+)([^\n.]+)',
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
        m = re.search(r'(?:is|=|equals?)\s*\$?(-?\d+\.?\d*)', line, re.IGNORECASE)
        if m: return m.group(1)
        nums = re.findall(r'-?\d+\.?\d*', line)
        if nums: return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]


def check(p, g):
    p, g = p.strip().replace(',','').replace(' ',''), str(g).strip().replace(',','').replace(' ','')
    if p == g: return True
    try: return abs(float(p)-float(g)) < 1e-6
    except: return p.lower() == g.lower()


def get_answer_first_token_id(tok, answer_str):
    """Get the first token ID for a numerical answer."""
    tokens = tok.encode(answer_str, add_special_tokens=False)
    if len(tokens) == 0:
        return None
    return tokens[0]


def find_digit_token_ids(tok):
    """Find all token IDs that correspond to digit characters (0-9, minus)."""
    digit_ids = set()
    for ch in '0123456789-':
        ids = tok.encode(ch, add_special_tokens=False)
        for tid in ids:
            digit_ids.add(tid)
    # Also check multi-char digit tokens
    for num in range(100):
        ids = tok.encode(str(num), add_special_tokens=False)
        # Only add single-token encodings
        if len(ids) == 1:
            digit_ids.add(ids[0])
    return digit_ids


def run_logit_lens(model, tok, questions, model_name, use_chat, prompt_type="baseline"):
    """
    Run Logit Lens with richer metrics: rank, digit mass, top-5 tokens.
    """
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_vocab = len(tok)

    # Pre-compute digit token IDs
    digit_ids = find_digit_token_ids(tok)

    # Collect hidden states from ALL layers via forward hooks
    all_layer_hs = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0][0, -1, :].detach()  # keep original dtype
            else:
                hs = output[0, -1, :].detach()
            all_layer_hs[layer_idx] = hs.cpu()
        return hook_fn

    for i in range(n_layers):
        h = model.model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    embed_hs = {}
    def embed_hook(module, input, output):
        o = output[0] if isinstance(output, tuple) else output
        if o.dim() == 3:
            embed_hs['embed'] = o[0, -1, :].detach().cpu()
        elif o.dim() == 2:
            embed_hs['embed'] = o[-1, :].detach().cpu()

    h_embed = model.model.embed_tokens.register_forward_hook(embed_hook)
    hooks.append(h_embed)

    results = []

    for qi, q_data in enumerate(questions):
        q = q_data["query"]
        gt = str(q_data.get("ground_truth", q_data.get("answer", "")))
        gt_clean = gt.strip().replace(',', '').replace(' ', '')

        # Get GT answer first token
        gt_tid = get_answer_first_token_id(tok, gt_clean)

        # Build prompt
        content = q + "\nLet's think step by step." if prompt_type == "cot" else q
        if use_chat:
            messages = [{"role": "user", "content": content}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = content

        inp = tok(prompt, return_tensors="pt").to(model.device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inp, output_hidden_states=False)

        lm_head = model.lm_head
        final_norm = model.model.norm

        layer_data = []

        # Process embedding + each transformer layer
        layer_sources = []
        if 'embed' in embed_hs:
            layer_sources.append(("embed", embed_hs['embed']))
        for li in range(n_layers):
            if li in all_layer_hs:
                layer_sources.append((li, all_layer_hs[li]))

        for layer_name, hs_cpu in layer_sources:
            hs = hs_cpu.to(model.device)
            hs_normed = final_norm(hs.unsqueeze(0)).squeeze(0)
            logits = lm_head(hs_normed)
            probs = torch.softmax(logits.float(), dim=-1)

            # Top-1 token
            top_prob, top_id = probs.max(dim=-1)
            top_token = tok.decode([top_id.item()])

            # GT answer first token probability and RANK
            gt_prob = 0.0
            gt_rank = d_vocab  # default: worst possible rank
            if gt_tid is not None:
                gt_prob = probs[gt_tid].item()
                gt_rank = (probs >= gt_prob).sum().item()

            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            # Digit mass: total probability on digit tokens
            digit_mass = sum(probs[tid].item() for tid in digit_ids if tid < d_vocab)

            # Top-5 tokens
            top5_probs, top5_ids = probs.topk(5)
            top5_tokens = [tok.decode([t.item()]) for t in top5_ids]

            layer_data.append({
                "layer": layer_name,
                "top_token": top_token,
                "top_prob": float(top_prob),
                "gt_prob": float(gt_prob),
                "gt_rank": int(gt_rank),
                "entropy": float(entropy),
                "digit_mass": float(digit_mass),
                "top5_tokens": top5_tokens,
                "top5_probs": [float(p) for p in top5_probs],
            })

        # Clear for next question
        all_layer_hs.clear()
        embed_hs.clear()

        results.append({
            "q": qi,
            "gt": gt_clean,
            "layer_data": layer_data,
            "n_layers": n_layers,
        })

        if (qi+1) % 10 == 0:
            # Show sample layer info for debugging
            last_ld = layer_data[-1]  # last transformer layer
            print(f"    [{model_name}/{prompt_type}] {qi+1}/{len(questions)} "
                  f"L{last_ld['layer']}: top='{last_ld['top_token']}' ({last_ld['top_prob']:.3f}) "
                  f"gt_rank={last_ld['gt_rank']} digit_mass={last_ld['digit_mass']:.4f}", flush=True)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    return results


def analyze_lens_results(results, model_name, prompt_type):
    """Analyze Logit Lens results with rank-based and entropy-based metrics."""
    n_layers = results[0]["n_layers"] if results else 0

    # Per-question metrics
    max_gt_ranks = []
    min_gt_ranks = []  # best (lowest) rank achieved
    best_rank_layers = []

    for r in results:
        ranks = [ld["gt_rank"] for ld in r["layer_data"] if isinstance(ld["layer"], int)]
        if ranks:
            min_gt_ranks.append(min(ranks))
            max_gt_ranks.append(max(ranks))
            best_rank_layers.append(ranks.index(min(ranks)))
        else:
            min_gt_ranks.append(n_layers)
            max_gt_ranks.append(n_layers)
            best_rank_layers.append(n_layers)

    # Average trajectories across layers
    layer_gt_probs = np.zeros(n_layers)
    layer_gt_ranks = np.zeros(n_layers)
    layer_top_probs = np.zeros(n_layers)
    layer_entropies = np.zeros(n_layers)
    layer_digit_mass = np.zeros(n_layers)
    n_valid = np.zeros(n_layers)

    for r in results:
        for ld in r["layer_data"]:
            if isinstance(ld["layer"], int):
                li = ld["layer"]
                layer_gt_probs[li] += ld["gt_prob"]
                layer_gt_ranks[li] += ld["gt_rank"]
                layer_top_probs[li] += ld["top_prob"]
                layer_entropies[li] += ld["entropy"]
                layer_digit_mass[li] += ld["digit_mass"]
                n_valid[li] += 1

    mask = n_valid > 0
    layer_gt_probs[mask] /= n_valid[mask]
    layer_gt_ranks[mask] /= n_valid[mask]
    layer_top_probs[mask] /= n_valid[mask]
    layer_entropies[mask] /= n_valid[mask]
    layer_digit_mass[mask] /= n_valid[mask]

    # Confidence layer: first layer where avg top_prob > 0.5
    confidence_layer = n_layers
    for i, p in enumerate(layer_top_probs):
        if p > 0.5:
            confidence_layer = i
            break

    # Entropy midpoint layer: first layer where entropy < half of max entropy
    max_entropy = float(layer_entropies.max()) if layer_entropies.max() > 0 else 1
    entropy_midpoint = n_layers
    for i, e in enumerate(layer_entropies):
        if e < max_entropy / 2:
            entropy_midpoint = i
            break

    # Sample top-5 tokens from first question at key layers for qualitative analysis
    sample_top5 = {}
    if results:
        ld = results[0]["layer_data"]
        key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        for ld_entry in ld:
            if isinstance(ld_entry["layer"], int) and ld_entry["layer"] in key_layers:
                sample_top5[f"L{ld_entry['layer']}"] = list(zip(
                    ld_entry["top5_tokens"], ld_entry["top5_probs"]
                ))

    metrics = {
        "model": model_name,
        "prompt_type": prompt_type,
        "n_layers": n_layers,
        "n_questions": len(results),
        "mean_best_gt_rank": float(np.mean(min_gt_ranks)),
        "std_best_gt_rank": float(np.std(min_gt_ranks)),
        "mean_best_rank_layer": float(np.mean(best_rank_layers)),
        "confidence_layer": confidence_layer,
        "entropy_midpoint_layer": entropy_midpoint,
        "max_entropy": float(max_entropy),
        "final_entropy": float(layer_entropies[-1]) if n_layers > 0 else 0,
        "mean_max_gt_prob": float(np.mean(
            [max(ld["gt_prob"] for ld in r["layer_data"] if isinstance(ld["layer"], int))
             for r in results]
        )),
        "layer_gt_probs": layer_gt_probs.tolist(),
        "layer_gt_ranks": layer_gt_ranks.tolist(),
        "layer_top_probs": layer_top_probs.tolist(),
        "layer_entropies": layer_entropies.tolist(),
        "layer_digit_mass": layer_digit_mass.tolist(),
        "sample_top5_q0": sample_top5,
    }

    return metrics


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Logit Lens v2 Experiment | {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    with open(DATA_FILE) as f:
        data = json.load(f)
    questions = data[:N_QUESTIONS]
    print(f"Using {len(questions)} MATH questions", flush=True)

    all_results = {}

    for model_name, (model_path, use_chat) in MODELS.items():
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

            print(f"\n  Loading model for {key}...", flush=True)
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tok.pad_token is None: tok.pad_token = tok.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.float16, low_cpu_mem_usage=True,
            ).to(DEVICE)
            model.eval()
            print(f"  Loaded: {model.config.num_hidden_layers} layers, d={model.config.hidden_size}", flush=True)

            print(f"  Running Logit Lens v2 ({prompt_type})...", flush=True)
            lens_results = run_logit_lens(model, tok, questions, model_name, use_chat, prompt_type)

            metrics = analyze_lens_results(lens_results, model_name, prompt_type)

            # Save per-question layer data for first 5 questions (for detailed analysis)
            sample_questions = []
            for r in lens_results[:5]:
                sample_questions.append({
                    "q": r["q"],
                    "gt": r["gt"],
                    "layer_data": [{
                        "layer": ld["layer"],
                        "top_token": ld["top_token"],
                        "top_prob": ld["top_prob"],
                        "gt_prob": ld["gt_prob"],
                        "gt_rank": ld["gt_rank"],
                        "entropy": ld["entropy"],
                        "digit_mass": ld["digit_mass"],
                        "top5_tokens": ld["top5_tokens"],
                        "top5_probs": ld["top5_probs"],
                    } for ld in r["layer_data"]],
                })

            save_data = {
                "model": model_name,
                "prompt_type": prompt_type,
                "metrics": metrics,
                "sample_questions": sample_questions,
            }

            with open(ckpt, 'w') as f:
                json.dump(save_data, f, indent=2)
            all_results[key] = save_data

            print(f"\n  => {key}:", flush=True)
            print(f"     Confidence layer (top_prob>0.5): {metrics['confidence_layer']} / {metrics['n_layers']}", flush=True)
            print(f"     Entropy midpoint:                {metrics['entropy_midpoint_layer']} / {metrics['n_layers']}", flush=True)
            print(f"     Final entropy:                   {metrics['final_entropy']:.2f}", flush=True)
            print(f"     Mean best GT rank:               {metrics['mean_best_gt_rank']:.0f}", flush=True)
            print(f"     Mean best rank layer:            {metrics['mean_best_rank_layer']:.1f}", flush=True)
            print(f"     Mean max GT prob:                {metrics['mean_max_gt_prob']:.6f}", flush=True)
            print(f"     Sample top-5 (Q0):", flush=True)
            for layer_key, t5 in sorted(metrics["sample_top5_q0"].items()):
                tokens_str = ", ".join(f"'{t}'({p:.3f})" for t, p in t5)
                print(f"       {layer_key}: {tokens_str}", flush=True)

            del model; torch.cuda.empty_cache(); gc.collect()

    # === Summary Table ===
    print(f"\n{'='*80}", flush=True)
    print("LOGIT LENS v2 SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<22} {'Type':>6} {'Conf L':>7} {'Entr Mid':>9} {'Final E':>8} {'Best Rank':>10} {'Rank L':>7}", flush=True)
    print("-" * 72, flush=True)

    for key, data in sorted(all_results.items()):
        m = data["metrics"]
        typ = "CoT" if "cot" in key else "Base"
        print(f"{m['model']:<22} {typ:>6} {m['confidence_layer']:>7d} "
              f"{m['entropy_midpoint_layer']:>9d} {m['final_entropy']:>8.2f} "
              f"{m['mean_best_gt_rank']:>10.0f} {m['mean_best_rank_layer']:>7.1f}", flush=True)

    # === Hypothesis Testing ===
    print(f"\n{'='*80}", flush=True)
    print("HYPOTHESIS TESTING", flush=True)
    print(f"{'='*80}", flush=True)

    i15 = all_results.get("Qwen2-1.5B-Inst_baseline", {}).get("metrics", {})
    i3 = all_results.get("Qwen2.5-3B-Inst_baseline", {}).get("metrics", {})
    b3 = all_results.get("Qwen2.5-3B-Base_baseline", {}).get("metrics", {})
    i3c = all_results.get("Qwen2.5-3B-Inst_cot", {}).get("metrics", {})

    if i15 and i3:
        print(f"\n[H1] Confidence Commitment:", flush=True)
        print(f"  1.5B-Inst: confidence at layer {i15['confidence_layer']}/{i15['n_layers']} "
              f"(entropy mid={i15['entropy_midpoint_layer']})", flush=True)
        print(f"  3B-Inst:   confidence at layer {i3['confidence_layer']}/{i3['n_layers']} "
              f"(entropy mid={i3['entropy_midpoint_layer']})", flush=True)
        # Relative positions
        r15 = i15['confidence_layer'] / max(i15['n_layers'], 1)
        r3 = i3['confidence_layer'] / max(i3['n_layers'], 1)
        print(f"  Relative: 1.5B={r15:.2f}, 3B={r3:.2f}", flush=True)
        print(f"  Prediction: 3B commits earlier → internalized reasoning", flush=True)

    if i3 and i3c:
        print(f"\n[H2] CoT Disruption on 3B-Inst:", flush=True)
        print(f"  Baseline: confidence={i3['confidence_layer']}/{i3['n_layers']}, "
              f"final_entropy={i3['final_entropy']:.2f}", flush=True)
        print(f"  CoT:      confidence={i3c['confidence_layer']}/{i3c['n_layers']}, "
              f"final_entropy={i3c['final_entropy']:.2f}", flush=True)
        print(f"  Entropy delta: {i3c['final_entropy'] - i3['final_entropy']:+.2f}", flush=True)
        print(f"  Prediction: CoT increases entropy (disrupts confidence)", flush=True)

    if b3 and i3:
        print(f"\n[H3] Base vs Instruct (3B):", flush=True)
        print(f"  Base:  confidence={b3['confidence_layer']}/{b3['n_layers']}, "
              f"final_entropy={b3['final_entropy']:.2f}, digit_mass_last={b3['layer_digit_mass'][-1]:.4f}", flush=True)
        print(f"  Inst:  confidence={i3['confidence_layer']}/{i3['n_layers']}, "
              f"final_entropy={i3['final_entropy']:.2f}, digit_mass_last={i3['layer_digit_mass'][-1]:.4f}", flush=True)
        print(f"  Prediction: Inst has lower entropy, higher confidence (structured output)", flush=True)

    # Digit mass trajectory comparison
    print(f"\n[Digit Mass Trajectory] (average probability on digit tokens across layers):", flush=True)
    for key in sorted(all_results.keys()):
        m = all_results[key]["metrics"]
        dm = m["layer_digit_mass"]
        n = len(dm)
        # Show at 25%, 50%, 75%, 100% of layers
        pts = [n//4, n//2, 3*n//4, n-1]
        vals = [dm[p] for p in pts]
        typ = "CoT" if "cot" in key else "Base"
        print(f"  {m['model']:<22} {typ:>4}: L{pts[0]}={vals[0]:.4f} L{pts[1]}={vals[1]:.4f} "
              f"L{pts[2]}={vals[2]:.4f} L{pts[3]}={vals[3]:.4f}", flush=True)

    print(f"\nDone: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
