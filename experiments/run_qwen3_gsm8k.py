#!/usr/bin/env python3
"""Run controlled decomposition on Qwen3-8B for GSM8K (cross-domain validation).

Critical: Qwen3-8B showed POSITIVE CoT effect on MATH (+5.5pp).
We need to verify if this holds on GSM8K.
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

SEED = 42
N = 200
BASE = Path(__file__).parent
MODEL_PATH = "/path/to/models/Qwen3-8B"
OUT = BASE / "results_qwen3"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

def extract_ans(text):
    # GSM8K answers are typically numbers after #### or at end
    # Try boxed first
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums: return nums[-1]
    # Try #### format
    hash_ans = re.findall(r'####\s*(-?\d+\.?\d*)', text)
    if hash_ans: return hash_ans[-1].strip()
    # Try "the answer is"
    for pat in [r'(?:therefore|thus|the answer is)[:\s]+([^\n.]+)', r'answer[:\s]+([^\n.]+)']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            nums = re.findall(r'-?\d+\.?\d*', m.group(1))
            if nums: return nums[-1]
    # Last number in text
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]

def check(p, g):
    p, g = p.strip().replace(',','').replace(' ','').rstrip('.'), str(g).strip().replace(',','').replace(' ','').rstrip('.')
    if p == g: return True
    try: return abs(float(p)-float(g)) < 1e-6
    except: return p.lower() == g.lower()

def run_condition(model, tok, questions, device, max_tok=256, use_cot=False):
    results = []
    for i, q in enumerate(questions):
        if use_cot:
            prompt = f"Question: {q['query']}\nLet's think step by step.\nAnswer:"
        else:
            prompt = f"Question: {q['query']}\nAnswer:"

        inp = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False,
                                 pad_token_id=tok.eos_token_id or tok.eos_token)
        full = tok.decode(out[0], skip_special_tokens=True)
        ans = extract_ans(full)
        gen_tok = out.shape[1] - inp["input_ids"].shape[1]
        correct = check(ans, q['ground_truth'])
        truncated = (gen_tok >= max_tok - 2)
        results.append({
            'idx': i, 'pred': ans, 'gt': q['ground_truth'],
            'correct': correct, 'tokens': gen_tok, 'truncated': truncated
        })
        if (i+1) % 50 == 0:
            acc = sum(r['correct'] for r in results) / len(results) * 100
            trunc = sum(r['truncated'] for r in results) / len(results) * 100
            print(f"  [{i+1}/{N}] acc={acc:.1f}% trunc={trunc:.1f}%", flush=True)
    return results

def main():
    print(f"Qwen3-8B GSM8K Controlled Decomposition | {time.strftime('%Y-%m-%d %H:%M')}", flush=True)

    # Load GSM8K dataset
    with open("/path/to/data/gsm8k_real_200.json") as f:
        alldata = json.load(f)
    random.seed(SEED)
    questions = random.sample(alldata, min(N, len(alldata)))
    print(f"Dataset: GSM8K, {len(questions)} questions", flush=True)

    # Load model
    device = "cuda:0"
    print(f"Loading Qwen3-8B...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()
    print(f"Model loaded on {device}", flush=True)

    conditions = [
        ("Base-256", 256, False),
        ("Base-512", 512, False),
        ("CoT-512", 512, True),
    ]

    all_results = {}
    for cond_name, max_tok, use_cot in conditions:
        print(f"\n--- Running {cond_name} (max_tok={max_tok}, cot={use_cot}) ---", flush=True)
        results = run_condition(model, tok, questions, device, max_tok, use_cot)
        acc = sum(r['correct'] for r in results) / len(results) * 100
        trunc = sum(r['truncated'] for r in results) / len(results) * 100
        avg_tok = sum(r['tokens'] for r in results) / len(results)
        print(f"  RESULT: {cond_name} => acc={acc:.1f}% trunc={trunc:.1f}% avg_tok={avg_tok:.0f}", flush=True)
        all_results[cond_name] = {
            'accuracy': acc, 'truncation_rate': trunc, 'avg_tokens': avg_tok,
            'details': results
        }

        # Save checkpoint
        outf = OUT / f"Qwen3-8B_GSM8K_controlled.json"
        with open(outf, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved to {outf}", flush=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Qwen3-8B on GSM8K ({N} questions):")
    for cond, data in all_results.items():
        print(f"  {cond}: {data['accuracy']:.1f}% acc, {data['truncation_rate']:.1f}% trunc, {data['avg_tokens']:.0f} avg tok")

    if 'Base-256' in all_results and 'Base-512' in all_results:
        token_delta = all_results['Base-512']['accuracy'] - all_results['Base-256']['accuracy']
        print(f"  Token Δ: {token_delta:+.1f}pp")
    if 'Base-512' in all_results and 'CoT-512' in all_results:
        cot_delta = all_results['CoT-512']['accuracy'] - all_results['Base-512']['accuracy']
        print(f"  CoT Δ: {cot_delta:+.1f}pp")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
