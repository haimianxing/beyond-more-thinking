#!/usr/bin/env python3
"""
Gemma-2-9B-IT Mid-Layer Probing Experiment
============================================
Purpose: Test the commitment zone prediction from Section 8.1.
If Gemma-2-9B lacks a concentrated commitment zone (or has one not
disrupted by CoT), the framework's prediction (a) is confirmed.

Based on: run_midlayer_probing.py (same protocol for cross-model comparison)
"""
import sys, os, json, gc, random, re, warnings
import torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
SEED = 42
N_QUESTIONS = 50
DEVICE = "cuda:0"

GEMMA_MODEL_PATH = "/path/to/models/gemma-2-9b-it"
QWEN25_3B_PATH = "/path/to/models/Qwen2.5-3B-Instruct"  # reference comparison

DATA_FILE = BASE / "math_real_200.json"
CKPT_DIR = BASE / "results_v2" / "gemma_probing"


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


def build_prompt(q, tok, use_chat, prompt_type):
    content = q + "\nLet's think step by step." if prompt_type == "cot" else q
    if use_chat:
        messages = [{"role": "user", "content": content}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def run_midlayer_probing(model, tok, questions, model_name, use_chat):
    """Extract hidden states from ALL layers under baseline and CoT conditions."""
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    lm_head = model.lm_head
    final_norm = model.model.norm

    results_per_condition = {}

    for prompt_type in ["baseline", "cot"]:
        print(f"    Running {prompt_type}...", flush=True)

        all_hs_per_layer = {i: [] for i in range(n_layers)}

        for qi, q_data in enumerate(questions):
            q = q_data["query"]

            prompt = build_prompt(q, tok, use_chat, prompt_type)
            inp = tok(prompt, return_tensors="pt").to(DEVICE)

            layer_hs = {}
            def make_single_hook(li):
                def fn(m, inp, out):
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

            if (qi+1) % 10 == 0:
                print(f"      {prompt_type}: {qi+1}/{len(questions)} done", flush=True)

        for i in range(n_layers):
            all_hs_per_layer[i] = torch.stack(all_hs_per_layer[i])

        results_per_condition[prompt_type] = all_hs_per_layer

    # === Analysis per layer ===
    print(f"    Computing metrics per layer...", flush=True)
    layer_metrics = []

    for li in range(n_layers):
        base_hs = results_per_condition["baseline"][li].numpy()
        cot_hs = results_per_condition["cot"][li].numpy()

        pca_base = PCA(n_components=1)
        pca_base.fit(base_hs)
        base_top1_var = float(pca_base.explained_variance_ratio_[0])

        pca_cot = PCA(n_components=1)
        pca_cot.fit(cot_hs)
        cot_top1_var = float(pca_cot.explained_variance_ratio_[0])

        cos_dists = []
        for i in range(len(base_hs)):
            b = base_hs[i]
            c = cot_hs[i]
            b_norm = np.linalg.norm(b)
            c_norm = np.linalg.norm(c)
            if b_norm > 1e-8 and c_norm > 1e-8:
                cos_sim = np.dot(b, c) / (b_norm * c_norm)
                cos_dists.append(1 - cos_sim)
        mean_cos_dist = float(np.mean(cos_dists)) if cos_dists else 0.0

        entropies = []
        for qi in range(len(base_hs)):
            hs = results_per_condition["baseline"][li][qi].to(model.device)
            hs = hs.half()
            hs_normed = final_norm(hs.unsqueeze(0)).squeeze(0)
            logits = lm_head(hs_normed)
            probs = torch.softmax(logits.float(), dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies.append(ent)
        mean_entropy_base = float(np.mean(entropies))

        entropies_cot = []
        for qi in range(len(cot_hs)):
            hs = results_per_condition["cot"][li][qi].to(model.device)
            hs = hs.half()
            hs_normed = final_norm(hs.unsqueeze(0)).squeeze(0)
            logits = lm_head(hs_normed)
            probs = torch.softmax(logits.float(), dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies_cot.append(ent)
        mean_entropy_cot = float(np.mean(entropies_cot))

        layer_metrics.append({
            "layer": li,
            "pca1_base": base_top1_var,
            "pca1_cot": cot_top1_var,
            "cos_dist": mean_cos_dist,
            "entropy_base": mean_entropy_base,
            "entropy_cot": mean_entropy_cot,
            "entropy_delta": mean_entropy_cot - mean_entropy_base,
        })

    return {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_questions": len(questions),
        "layer_metrics": layer_metrics,
    }


def load_reference_qwen25_3b():
    """Load existing Qwen2.5-3B-Inst mid-layer probing results for comparison."""
    ref_path = BASE / "results_v2" / "midlayer_probing" / "Qwen2.5-3B-Inst_midlayer.json"
    if ref_path.exists():
        return json.load(open(ref_path))
    print("  [WARN] No Qwen2.5-3B-Inst reference found, skipping comparison")
    return None


def analyze_commitment_zone(gemma_result, qwen_result=None):
    """Analyze whether Gemma shows a commitment zone like Qwen2.5-3B."""
    gemma_lm = gemma_result["layer_metrics"]
    gemma_n = gemma_result["n_layers"]

    print(f"\n{'='*70}")
    print("COMMITMENT ZONE ANALYSIS: Gemma-2-9B-IT vs Qwen2.5-3B-Inst")
    print(f"{'='*70}")

    # Find where entropy drops below 2.0 for Gemma
    gemma_emergence = gemma_n
    for m in gemma_lm:
        if m["entropy_base"] < 2.0:
            gemma_emergence = m["layer"]
            break

    gemma_final = gemma_lm[-1]
    print(f"\n  Gemma-2-9B-IT ({gemma_n} layers):")
    print(f"    Entropy < 2.0 at layer: {gemma_emergence}/{gemma_n} ({gemma_emergence/gemma_n*100:.0f}%)")
    print(f"    Final entropy: base={gemma_final['entropy_base']:.2f}, cot={gemma_final['entropy_cot']:.2f}")
    print(f"    Final entropy delta (CoT effect): {gemma_final['entropy_delta']:+.3f}")
    print(f"    Final PCA1: base={gemma_final['pca1_base']:.3f}, cot={gemma_final['pca1_cot']:.3f}")

    if qwen_result:
        qwen_lm = qwen_result["layer_metrics"]
        qwen_n = qwen_result["n_layers"]
        qwen_emergence = qwen_n
        for m in qwen_lm:
            if m["entropy_base"] < 2.0:
                qwen_emergence = m["layer"]
                break
        qwen_final = qwen_lm[-1]
        print(f"\n  Qwen2.5-3B-Inst ({qwen_n} layers) [REFERENCE]:")
        print(f"    Entropy < 2.0 at layer: {qwen_emergence}/{qwen_n} ({qwen_emergence/qwen_n*100:.0f}%)")
        print(f"    Final entropy: base={qwen_final['entropy_base']:.2f}, cot={qwen_final['entropy_cot']:.2f}")
        print(f"    Final entropy delta (CoT effect): {qwen_final['entropy_delta']:+.3f}")

        print(f"\n  COMPARISON:")
        print(f"    Gemma commitment zone depth: {gemma_emergence/gemma_n*100:.0f}%")
        print(f"    Qwen  commitment zone depth: {qwen_emergence/qwen_n*100:.0f}%")
        print(f"    Gemma final entropy: {gemma_final['entropy_base']:.2f}")
        print(f"    Qwen  final entropy: {qwen_final['entropy_base']:.2f}")

        if gemma_final['entropy_base'] > 1.5:
            print(f"\n  >> RESULT: Gemma-2-9B does NOT develop a concentrated commitment zone")
            print(f"     (final entropy {gemma_final['entropy_base']:.2f} >> Qwen2.5-3B's {qwen_final['entropy_base']:.2f})")
            print(f"  >> PREDICTION (a) CONFIRMED: models with positive CoT effects lack commitment zone")
        else:
            print(f"\n  >> RESULT: Gemma-2-9B DOES develop a concentrated commitment zone")
            print(f"     (final entropy {gemma_final['entropy_base']:.2f})")
            if gemma_final['entropy_delta'] < 0.1:
                print(f"     But CoT does NOT disrupt it (entropy delta {gemma_final['entropy_delta']:+.3f})")
                print(f"  >> PREDICTION (a) CONFIRMED (alternative): commitment zone exists but not disrupted")
            else:
                print(f"     And CoT DOES disrupt it (entropy delta {gemma_final['entropy_delta']:+.3f})")
                print(f"  >> FRAMEWORK NEEDS REVISION: Gemma has zone + CoT disrupts, yet CoT helps accuracy")
    else:
        if gemma_final['entropy_base'] > 1.5:
            print(f"\n  >> RESULT: Gemma-2-9B does NOT develop a concentrated commitment zone")
            print(f"     Final entropy {gemma_final['entropy_base']:.2f} is HIGH (not concentrated)")
            print(f"  >> PREDICTION (a) CONFIRMED")
        else:
            print(f"\n  >> RESULT: Gemma-2-9B DOES develop a concentrated commitment zone")
            print(f"     Final entropy {gemma_final['entropy_base']:.2f}")


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    import time
    print(f"Gemma-2-9B-IT Mid-Layer Probing | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    with open(DATA_FILE) as f:
        data = json.load(f)
    questions = data[:N_QUESTIONS]
    print(f"Using {len(questions)} MATH questions", flush=True)

    # Load reference Qwen2.5-3B results
    qwen_ref = load_reference_qwen25_3b()

    # === Run Gemma-2-9B-IT probing ===
    gemma_ckpt = CKPT_DIR / "Gemma-2-9B-IT_midlayer.json"
    if gemma_ckpt.exists():
        print(f"\n  SKIP Gemma-2-9B-IT (cached)", flush=True)
        gemma_result = json.load(open(gemma_ckpt))
    else:
        print(f"\n{'='*60}", flush=True)
        print(f"Model: Gemma-2-9B-IT", flush=True)
        print(f"{'='*60}", flush=True)

        tok = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL_PATH, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(DEVICE)
        model.eval()
        print(f"  Loaded: {model.config.num_hidden_layers} layers, d={model.config.hidden_size}", flush=True)

        gemma_result = run_midlayer_probing(model, tok, questions, "Gemma-2-9B-IT", use_chat=True)

        with open(gemma_ckpt, 'w') as f:
            json.dump(gemma_result, f, indent=2)

        del model; torch.cuda.empty_cache(); gc.collect()

    # Print Gemma summary
    lm = gemma_result["layer_metrics"]
    n = gemma_result["n_layers"]
    print(f"\n  Gemma-2-9B-IT Layer Summary ({n} layers):", flush=True)
    print(f"  {'Layer':>5} {'PCA1-Base':>10} {'PCA1-CoT':>10} {'CosDist':>8} {'Ent-Base':>9} {'Ent-CoT':>9} {'Ent-Δ':>7}", flush=True)
    for m in lm:
        if m["layer"] % 6 == 0 or m["layer"] == n-1:
            print(f"  {m['layer']:>5} {m['pca1_base']:>10.3f} {m['pca1_cot']:>10.3f} "
                  f"{m['cos_dist']:>8.3f} {m['entropy_base']:>9.2f} {m['entropy_cot']:>9.2f} "
                  f"{m['entropy_delta']:>+7.3f}", flush=True)

    # === Commitment Zone Analysis ===
    analyze_commitment_zone(gemma_result, qwen_ref)

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
