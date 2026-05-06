#!/usr/bin/env python3
"""Quick verification: baseline with 512 tokens vs 256 tokens.

Goal: Test if the "compute threshold effect" is real or an artifact of truncation.

If baseline_512 approx= cot_long_512, then CoT prompt doesn't add value -- it's all about token budget.
If baseline_512 << cot_long_512, then CoT prompting genuinely helps beyond just token budget.

Usage:
    python3 run_baseline_512_verification.py \
        --model_paths '{"Qwen2-1.5B": "/path/to/Qwen2-1.5B-Instruct", "Qwen2.5-3B": "/path/to/Qwen2.5-3B-Instruct", "Qwen2.5-7B": "/path/to/Qwen2.5-7B-Instruct"}' \
        --data_file /path/to/math_real_200.json \
        --output_dir ./results_v2 \
        --gpu 2
"""
import sys, os, json, time, re, gc, random, argparse
from pathlib import Path
from collections import Counter
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = Path(__file__).parent
SEEDS = [42, 123, 456]
N_SAMPLES = 200

def extract_ans(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums: return nums[-1]
    for pat in [
        r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+|so the answer is|final answer[:\s]+|in total[,:\s]+|in all[,:\s]+)([^\n.]+)',
        r'answer[:\s]+([^\n.]+)',
    ]:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            m = matches[-1]
            nums = re.findall(r'-?\d+\.?\d*', m.group(1))
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

def build_prompt(tok, q, cot=False):
    content = q
    if cot:
        content = f"{q}\nLet's think step by step."
    if hasattr(tok, 'apply_chat_template'):
        messages = [{"role": "user", "content": content}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if cot: return f"Question: {q}\nLet's think step by step.\nAnswer:"
        return f"Question: {q}\nAnswer:"

def main():
    parser = argparse.ArgumentParser(description="Baseline-512 Verification")
    parser.add_argument("--model_paths", required=True,
                        help="JSON dict mapping model names to paths, e.g. '{\"Qwen2.5-3B\": \"/path/to/model\"}'")
    parser.add_argument("--data_file", required=True, help="Path to MATH questions JSON")
    parser.add_argument("--output_dir", required=True, help="Directory for results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    MODELS = json.loads(args.model_paths)
    CKPT = Path(args.output_dir)
    CKPT.mkdir(parents=True, exist_ok=True)

    print(f"Baseline-512 Verification | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    results_all = {}

    for mname, mpath in MODELS.items():
        print(f"\n{'='*60}\nLoading {mname}...\n{'='*60}", flush=True)

        tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            mpath, trust_remote_code=True, dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        print(f"{mname} loaded", flush=True)

        with open(args.data_file) as f:
            alldata = json.load(f)
        data = alldata[:N_SAMPLES]

        for method_name, use_cot, max_tok in [
            ("baseline_256", False, 256),
            ("baseline_512", False, 512),
            ("cot_512", True, 512),
        ]:
            for seed in SEEDS:
                random.seed(seed); np.random.seed(seed)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

                key = f"{mname}_MATH_{method_name}_s{seed}"
                label = f"{mname[:6]}|MATH|{method_name}|s{seed}"
                results = []

                for i in range(N_SAMPLES):
                    q = data[i]["query"]
                    gt = str(data[i].get("ground_truth", data[i].get("answer", "")))

                    prompt = build_prompt(tok, q, cot=use_cot)
                    inp = tok(prompt, return_tensors="pt").to(device)

                    try:
                        t0 = time.time()
                        with torch.no_grad():
                            out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False, pad_token_id=tok.eos_token_id)
                        lat = time.time() - t0
                        ans = extract_ans(tok.decode(out[0], skip_special_tokens=True))
                        gen_tok = out.shape[1] - inp["input_ids"].shape[1]
                        truncated = gen_tok >= max_tok - 5
                        results.append({
                            "q": i, "ok": check(ans, gt),
                            "tok": gen_tok, "lat": round(lat, 4),
                            "truncated": truncated
                        })
                    except Exception as e:
                        results.append({"q": i, "ok": False, "tok": 0, "lat": 0, "err": str(e), "truncated": False})

                    if (i+1) % 100 == 0 or (i+1) == N_SAMPLES:
                        acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
                        trunc_rate = sum(1 for r in results if r.get("truncated")) / len(results) * 100
                        print(f"  {label} [{i+1}/{N_SAMPLES}] acc={acc:.1f}% trunc={trunc_rate:.0f}%", flush=True)

                # Save
                out_file = CKPT / f"{key}_verify.json"
                with open(out_file, 'w') as f:
                    json.dump({"metadata": {"model": mname, "ds": "MATH", "method": method_name, "seed": seed, "done": True}, "results": results}, f)

                acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
                trunc = sum(1 for r in results if r.get("truncated")) / len(results) * 100
                avg_tok = sum(r["tok"] for r in results) / len(results)
                results_all[key] = {"acc": round(acc, 1), "trunc": round(trunc, 0), "avg_tok": round(avg_tok, 0)}
                print(f"  => {key}: acc={acc:.1f}% trunc={trunc:.0f}% avg_tok={avg_tok:.0f}", flush=True)

        del model; torch.cuda.empty_cache(); gc.collect()
        print(f"{mname} freed", flush=True)

    # Print comparison table
    print(f"\n{'='*80}")
    print("VERIFICATION RESULTS: baseline_256 vs baseline_512 vs cot_512")
    print(f"{'='*80}")
    print(f"\n{'Model':<14} {'Method':<16} {'Acc%':>6} {'Trunc%':>7} {'AvgTok':>7}")
    print("-"*55)
    for key in sorted(results_all):
        r = results_all[key]
        parts = key.rsplit('_', 1)
        base = parts[0]
        print(f"{base:<24} {r['acc']:>5.1f}% {r['trunc']:>6.0f}% {r['avg_tok']:>7.0f}")

    # Per-model comparison
    print(f"\n{'='*80}")
    print("KEY COMPARISON (averaged across 3 seeds)")
    print(f"{'='*80}")
    for model in MODELS:
        prefix = f"{model}_MATH"
        b256 = [v for k, v in results_all.items() if k.startswith(f"{prefix}_baseline_256")]
        b512 = [v for k, v in results_all.items() if k.startswith(f"{prefix}_baseline_512")]
        c512 = [v for k, v in results_all.items() if k.startswith(f"{prefix}_cot_512")]

        if b256 and b512 and c512:
            b256a = np.mean([x['acc'] for x in b256])
            b512a = np.mean([x['acc'] for x in b512])
            c512a = np.mean([x['acc'] for x in c512])
            b256t = np.mean([x['trunc'] for x in b256])
            b512t = np.mean([x['trunc'] for x in b512])

            print(f"\n{model}/MATH:")
            print(f"  baseline_256: {b256a:.1f}% (trunc={b256t:.0f}%)")
            print(f"  baseline_512: {b512a:.1f}% (trunc={b512t:.0f}%)")
            print(f"  cot_512:      {c512a:.1f}%")
            print(f"  delta(base_512 - base_256) = {b512a-b256a:+.1f}%  (token budget effect)")
            print(f"  delta(cot_512 - base_512)  = {c512a-b512a:+.1f}%  (CoT prompt effect)")
            print(f"  delta(cot_512 - base_256)  = {c512a-b256a:+.1f}%  (total effect)")

            if b512a > b256a + 5 and c512a > b512a + 2:
                print(f"  => BOTH token budget AND CoT matter")
            elif b512a > b256a + 5 and abs(c512a - b512a) < 2:
                print(f"  => Token budget is the key factor, CoT adds little")
            elif abs(b512a - b256a) < 3 and c512a > b256a + 5:
                print(f"  => CoT prompt is the key factor, token budget adds little")
            else:
                print(f"  => Mixed effects")

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

if __name__ == "__main__":
    main()
