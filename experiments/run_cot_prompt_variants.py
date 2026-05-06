#!/usr/bin/env python3
"""
CoT Prompt Variants Experiment
===============================
Addresses SAC P1-A: "Single CoT prompt" criticism.
Tests whether the CoT penalty is specific to "Let's think step by step"
by running two additional prompts on Qwen2.5-3B-Instruct/MATH.

Prompt variants:
  1. fewshot_cot: 3-shot CoT with math exemplars
  2. structured:  "Solve step by step. Show your work. Put the final answer in \\boxed{}."
  3. original:   "Let's think step by step." (baseline for comparison, already exists)

Baseline comparison: Base-512 (no CoT prompt, 512 tokens)

Usage: CUDA_VISIBLE_DEVICES=1 python3 -u run_cot_prompt_variants.py
"""
import sys, os, json, time, gc, random, re, warnings
import torch, numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
SEEDS = [42]
N_QUESTIONS = 200
DEVICE = "cuda:0"

MODEL_NAME = "Qwen2.5-3B"
MODEL_PATH = "/path/to/models/Qwen2.5-3B-Instruct"

DATA_FILE = BASE / "math_real_200.json"
CKPT_DIR = BASE / "results_v2" / "cot_variants"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Few-shot CoT exemplars (MATH-level algebra)
FEWSHOT_EXEMPLARS = """Problem: Find the value of $x$ such that $2x + 5 = 13$.
Solution: Subtract 5 from both sides: $2x = 8$. Divide by 2: $x = 4$.
Answer: $\\boxed{4}$

Problem: If $a^2 - 3a + 2 = 0$, find all values of $a$.
Solution: Factor: $(a-1)(a-2) = 0$. So $a = 1$ or $a = 2$.
Answer: $\\boxed{1, 2}$

Problem: What is the sum of the first 10 positive integers?
Solution: Using the formula $S = n(n+1)/2$: $S = 10(11)/2 = 55$.
Answer: $\\boxed{55}$

"""

PROMPT_VARIANTS = {
    "fewshot_cot": {
        "prefix": FEWSHOT_EXEMPLARS,
        "suffix": "\nShow your reasoning step by step, then give the final answer in \\boxed{}.",
    },
    "structured": {
        "prefix": "",
        "suffix": "\nSolve step by step. Show your work clearly. Put the final answer in \\boxed{}.",
    },
    "original_cot": {
        "prefix": "",
        "suffix": "\nLet's think step by step.",
    },
}


def extract_ans(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums:
            return nums[-1]
    for pat in [r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+)([^\n.]+)',
                r'answer[:\s]+([^\n.]+)']:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            nums = re.findall(r'-?\d+\.?\d*', matches[-1].group(1))
            if nums:
                return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]


def check(p, g):
    p = p.strip().replace(',', '').replace(' ', '')
    g = str(g).strip().replace(',', '').replace(' ', '')
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return p.lower() == g.lower()


def main():
    print(f"CoT Prompt Variants | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Model: {MODEL_NAME} | Device: {DEVICE}", flush=True)

    with open(DATA_FILE) as f:
        questions = json.load(f)[:N_QUESTIONS]
    print(f"Loaded {len(questions)} MATH questions", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"  {MODEL_NAME} loaded ({model.config.num_hidden_layers} layers)", flush=True)

    # First, run baseline (no CoT) for comparison
    print(f"\n{'='*60}", flush=True)
    print("Running BASELINE (no CoT, 512 tokens)", flush=True)
    print(f"{'='*60}", flush=True)

    baseline_results = {}
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        ckpt = CKPT_DIR / f"{MODEL_NAME}_MATH_baseline_512_s{seed}.json"
        if ckpt.exists():
            print(f"  SKIP baseline s{seed} (cached)", flush=True)
            baseline_results[seed] = json.load(open(ckpt))
            continue

        results = []
        for i in range(N_QUESTIONS):
            q = questions[i]["query"]
            gt = str(questions[i].get("ground_truth", questions[i].get("answer", "")))

            messages = [{"role": "user", "content": q}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=512, do_sample=False,
                                     pad_token_id=tok.eos_token_id)

            gen_text = tok.decode(out[0], skip_special_tokens=True)
            if prompt in gen_text:
                gen_text = gen_text[len(prompt):]

            gen_tok = out.shape[1] - inp["input_ids"].shape[1]
            truncated = gen_tok >= 507

            ans = extract_ans(gen_text)
            ok = check(ans, gt)

            results.append({
                "q": i, "ok": ok, "tok": gen_tok,
                "truncated": truncated, "ans": ans, "gt": gt,
            })

            if (i + 1) % 50 == 0:
                acc = sum(r["ok"] for r in results) / len(results) * 100
                print(f"    baseline/s{seed}: {i+1}/{N_QUESTIONS} acc={acc:.1f}%", flush=True)

        data = {"metadata": {"model": MODEL_NAME, "method": "baseline_512",
                             "seed": seed, "n": N_QUESTIONS},
                "results": results}
        with open(ckpt, 'w') as f:
            json.dump(data, f)

        acc = sum(r["ok"] for r in results) / len(results) * 100
        baseline_results[seed] = data
        print(f"  Baseline s{seed}: acc={acc:.1f}%", flush=True)

    # Run each CoT prompt variant
    variant_results = {}

    for vname, vconfig in PROMPT_VARIANTS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Running variant: {vname}", flush=True)
        print(f"{'='*60}", flush=True)

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            ckpt = CKPT_DIR / f"{MODEL_NAME}_MATH_{vname}_s{seed}.json"
            if ckpt.exists():
                print(f"  SKIP {vname} s{seed} (cached)", flush=True)
                variant_results.setdefault(vname, {})[seed] = json.load(open(ckpt))
                continue

            results = []
            for i in range(N_QUESTIONS):
                q = questions[i]["query"]
                gt = str(questions[i].get("ground_truth", questions[i].get("answer", "")))

                # Build prompt with variant
                content = vconfig["prefix"] + q + vconfig["suffix"]
                messages = [{"role": "user", "content": content}]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inp = tok(prompt, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=512, do_sample=False,
                                         pad_token_id=tok.eos_token_id)

                gen_text = tok.decode(out[0], skip_special_tokens=True)
                if prompt in gen_text:
                    gen_text = gen_text[len(prompt):]

                gen_tok = out.shape[1] - inp["input_ids"].shape[1]
                truncated = gen_tok >= 507

                ans = extract_ans(gen_text)
                ok = check(ans, gt)

                results.append({
                    "q": i, "ok": ok, "tok": gen_tok,
                    "truncated": truncated, "ans": ans, "gt": gt,
                })

                if (i + 1) % 50 == 0:
                    acc = sum(r["ok"] for r in results) / len(results) * 100
                    print(f"    {vname}/s{seed}: {i+1}/{N_QUESTIONS} acc={acc:.1f}%", flush=True)

            data = {"metadata": {"model": MODEL_NAME, "method": vname,
                                 "seed": seed, "n": N_QUESTIONS},
                    "results": results}
            with open(ckpt, 'w') as f:
                json.dump(data, f)

            acc = sum(r["ok"] for r in results) / len(results) * 100
            variant_results.setdefault(vname, {})[seed] = data
            print(f"  {vname} s{seed}: acc={acc:.1f}%", flush=True)

    # === Compute McNemar tests ===
    print(f"\n{'='*80}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    from scipy.stats import chi2

    # Get baseline accuracy
    base_res = baseline_results[SEEDS[0]]["results"]
    base_acc = sum(r["ok"] for r in base_res) / len(base_res) * 100
    base_ok = set(i for i, r in enumerate(base_res) if r["ok"])
    print(f"\nBaseline (no CoT): {base_acc:.1f}%", flush=True)

    for vname in PROMPT_VARIANTS:
        if vname not in variant_results:
            continue
        vres = variant_results[vname][SEEDS[0]]["results"]
        v_acc = sum(r["ok"] for r in vres) / len(vres) * 100
        v_ok = set(i for i, r in enumerate(vres) if r["ok"])

        # McNemar test
        n_base_only = len(base_ok - v_ok)  # baseline correct, variant wrong
        n_var_only = len(v_ok - base_ok)   # variant correct, baseline wrong
        n_both = len(base_ok & v_ok)
        n_neither = len(set(range(len(vres))) - base_ok - v_ok)

        b = max(n_base_only, 0.5)
        c = max(n_var_only, 0.5)
        mcnemar_chi2 = (abs(b - c) - 0.5)**2 / (b + c)
        mcnemar_p = 1 - chi2.cdf(mcnemar_chi2, 1)

        cot_delta = v_acc - base_acc

        print(f"\n  {vname}:", flush=True)
        print(f"    Accuracy: {v_acc:.1f}% (CoT Δ = {cot_delta:+.1f}pp)", flush=True)
        print(f"    McNemar: base_only={n_base_only}, var_only={n_var_only}, "
              f"both={n_both}, neither={n_neither}", flush=True)
        print(f"    McNemar χ²={mcnemar_chi2:.2f}, p={mcnemar_p:.3f}", flush=True)
        print(f"    Significant at p<0.05: {'YES' if mcnemar_p < 0.05 else 'NO'}", flush=True)

        # Bootstrap 95% CI
        diffs = []
        for _ in range(10000):
            idx = np.random.choice(len(vres), len(vres), replace=True)
            b_acc = np.mean([1 if i in base_ok else 0 for i in idx])
            v_acc_b = np.mean([1 if i in v_ok else 0 for i in idx])
            diffs.append((v_acc_b - b_acc) * 100)
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        print(f"    95% CI: [{ci_lo:.1f}, {ci_hi:+.1f}]", flush=True)

        # Avg tokens
        avg_tok = np.mean([r["tok"] for r in vres])
        trunc_rate = sum(r["truncated"] for r in vres) / len(vres) * 100
        print(f"    Avg tokens: {avg_tok:.0f}, Truncation: {trunc_rate:.0f}%", flush=True)

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
