#!/usr/bin/env python3
"""
Extended Mid-Layer Probing (N=200)
===================================
Addresses SAC CW3 — "probing N=50 is underpowered".
Extends from 50 to 200 MATH questions for tighter PCA CI.

Produces: Mid-layer metrics for 3B-Instruct, 1.5B-Instruct, 3B-Base
          at 200 questions (vs original 50).

Usage: CUDA_VISIBLE_DEVICES=6 python3 -u run_extended_probing.py
"""
import sys, os, json, gc, random, re, warnings
import torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
SEED = 42
N_QUESTIONS = 200  # Extended from 50
DEVICE = "cuda:0"

MODELS = {
    "Qwen2.5-3B-Inst": ("/path/to/models/Qwen2.5-3B-Instruct", True),
    "Qwen2.5-1.5B-Inst": ("/path/to/models/Qwen/Qwen2.5-1.5B-Instruct", True),
    "Qwen2.5-3B-Base": ("/path/to/models/Qwen2.5-3B", False),
}

DATA_FILE = BASE / "math_real_200.json"
CKPT_DIR = BASE / "results_v2" / "extended_probing"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def extract_ans(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums: return nums[-1]
    for pat in [r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+)([^\n.]+)',
                r'answer[:\s]+([^\n.]+)']:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            nums = re.findall(r'-?\d+\.?\d*', matches[-1].group(1))
            if nums: return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]


def check(p, g):
    p, g = p.strip().replace(',','').replace(' ',''), str(g).strip().replace(',','').replace(' ','')
    if p == g: return True
    try: return abs(float(p)-float(g)) < 1e-6
    except: return p.lower() == g.lower()


def build_prompt(q, tok, use_chat, prompt_type):
    content = q + "\nLet's think step by step." if prompt_type == "cot" else q
    if use_chat:
        messages = [{"role": "user", "content": content}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def run_probing(model, tok, questions, model_name, use_chat):
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    lm_head = model.lm_head
    final_norm = model.model.norm

    results_per_condition = {}

    for prompt_type in ["baseline", "cot"]:
        print(f"    Running {prompt_type} (N={len(questions)})...", flush=True)
        all_hs_per_layer = {i: [] for i in range(n_layers)}

        for qi, q_data in enumerate(questions):
            q = q_data["query"]

            prompt = build_prompt(q, tok, use_chat, prompt_type)
            inp = tok(prompt, return_tensors="pt").to(DEVICE)

            layer_hs = {}
            def make_single_hook(li):
                def fn(m, inp_arg, out):
                    o = out[0] if isinstance(out, tuple) else out
                    layer_hs[li] = o[0, -1, :].detach().cpu().float()
                return fn

            temp_hooks = []
            for i in range(n_layers):
                temp_hooks.append(
                    model.model.layers[i].register_forward_hook(make_single_hook(i))
                )

            with torch.no_grad():
                model(**inp, output_hidden_states=False)

            for h in temp_hooks:
                h.remove()

            for i in range(n_layers):
                all_hs_per_layer[i].append(layer_hs[i])

            if (qi+1) % 50 == 0:
                print(f"      {prompt_type}: {qi+1}/{len(questions)} done", flush=True)

        for i in range(n_layers):
            all_hs_per_layer[i] = torch.stack(all_hs_per_layer[i])

        results_per_condition[prompt_type] = all_hs_per_layer

    # Analysis per layer
    print(f"    Computing metrics per layer (N={len(questions)})...", flush=True)
    layer_metrics = []

    for li in range(n_layers):
        base_hs = results_per_condition["baseline"][li].numpy()
        cot_hs = results_per_condition["cot"][li].numpy()

        # PCA top-1 variance
        pca_base = PCA(n_components=1).fit(base_hs)
        pca_cot = PCA(n_components=1).fit(cot_hs)

        # Cosine distance
        cos_dists = []
        for i in range(len(base_hs)):
            b, c = base_hs[i], cot_hs[i]
            bn, cn = np.linalg.norm(b), np.linalg.norm(c)
            if bn > 1e-8 and cn > 1e-8:
                cos_dists.append(1 - np.dot(b, c) / (bn * cn))

        # Logit Lens entropy
        entropies_base, entropies_cot = [], []
        for qi in range(len(base_hs)):
            # Baseline entropy
            hs_b = results_per_condition["baseline"][li][qi].to(model.device).half()
            hs_normed_b = final_norm(hs_b.unsqueeze(0)).squeeze(0)
            logits_b = lm_head(hs_normed_b)
            probs_b = torch.softmax(logits_b.float(), dim=-1)
            entropies_base.append(-torch.sum(probs_b * torch.log(probs_b + 1e-10)).item())

            # CoT entropy
            hs_c = results_per_condition["cot"][li][qi].to(model.device).half()
            hs_normed_c = final_norm(hs_c.unsqueeze(0)).squeeze(0)
            logits_c = lm_head(hs_normed_c)
            probs_c = torch.softmax(logits_c.float(), dim=-1)
            entropies_cot.append(-torch.sum(probs_c * torch.log(probs_c + 1e-10)).item())

        layer_metrics.append({
            "layer": li,
            "pca1_base": float(pca_base.explained_variance_ratio_[0]),
            "pca1_cot": float(pca_cot.explained_variance_ratio_[0]),
            "cos_dist": float(np.mean(cos_dists)) if cos_dists else 0.0,
            "entropy_base": float(np.mean(entropies_base)),
            "entropy_cot": float(np.mean(entropies_cot)),
            "entropy_delta": float(np.mean(entropies_cot)) - float(np.mean(entropies_base)),
        })

    return {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_questions": len(questions),
        "layer_metrics": layer_metrics,
    }


def main():
    print(f"Extended Mid-Layer Probing (N={N_QUESTIONS}) | {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    with open(DATA_FILE) as f:
        data = json.load(f)
    questions = data[:N_QUESTIONS]
    print(f"Using {len(questions)} MATH questions (extended from 50)", flush=True)

    all_results = {}

    for model_name, (model_path, use_chat) in MODELS.items():
        ckpt = CKPT_DIR / f"{model_name}_midlayer_N{N_QUESTIONS}.json"
        if ckpt.exists():
            print(f"\n  SKIP {model_name} (cached)", flush=True)
            all_results[model_name] = json.load(open(ckpt))
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model_name}", flush=True)
        print(f"{'='*60}", flush=True)

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(DEVICE)
        model.eval()
        print(f"  Loaded: {model.config.num_hidden_layers} layers, d={model.config.hidden_size}", flush=True)

        result = run_probing(model, tok, questions, model_name, use_chat)

        with open(ckpt, 'w') as f:
            json.dump(result, f, indent=2)
        all_results[model_name] = result

        # Print summary
        lm = result["layer_metrics"]
        print(f"\n  Summary for {model_name}:", flush=True)
        print(f"  {'Layer':>5} {'PCA1-B':>8} {'PCA1-C':>8} {'CosD':>7} {'Ent-B':>7} {'Ent-C':>7} {'Ent-Δ':>6}", flush=True)
        for m in lm:
            if m["layer"] % 6 == 0 or m["layer"] == result["n_layers"]-1:
                print(f"  {m['layer']:>5} {m['pca1_base']:>8.3f} {m['pca1_cot']:>8.3f} "
                      f"{m['cos_dist']:>7.3f} {m['entropy_base']:>7.2f} {m['entropy_cot']:>7.2f} "
                      f"{m['entropy_delta']:>+6.3f}", flush=True)

        del model; torch.cuda.empty_cache(); gc.collect()

    # Compare with original N=50 results
    print(f"\n{'='*80}", flush=True)
    print(f"COMPARISON: N=50 vs N={N_QUESTIONS}", flush=True)
    print(f"{'='*80}", flush=True)

    orig_dir = BASE / "results_v2" / "midlayer_probing"
    for mn, res in all_results.items():
        lm = res["layer_metrics"]
        final = lm[-1]
        print(f"\n  {mn} (N={res['n_questions']}):", flush=True)
        print(f"    Final PCA1: base={final['pca1_base']:.3f}, cot={final['pca1_cot']:.3f}", flush=True)
        print(f"    Final entropy: base={final['entropy_base']:.2f}, cot={final['entropy_cot']:.2f}, Δ={final['entropy_delta']:+.3f}", flush=True)

        # Compare with N=50
        orig_file = orig_dir / f"{mn}_midlayer.json"
        if orig_file.exists():
            orig = json.load(open(orig_file))
            orig_final = orig["layer_metrics"][-1]
            print(f"    vs N=50: PCA1 base={orig_final['pca1_base']:.3f}→{final['pca1_base']:.3f}, "
                  f"entropy Δ={orig_final['entropy_delta']:+.3f}→{final['entropy_delta']:+.3f}", flush=True)

    print(f"\nDone: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
