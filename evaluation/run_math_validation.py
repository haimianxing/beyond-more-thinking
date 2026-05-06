#!/usr/bin/env python3
"""
EXPERIMENT A: MATH-500 Cross-Dataset Validation
Tests: Does Reasoning Non-Convergence hold on MATH dataset?
Models: 0.5B, 3B, 7B on MATH (harder than GSM8K)
Methods: baseline, cot_t0 (deterministic), cot_t0_512 (longer context)
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
N_SAMPLES = 200
BASE = Path(__file__).parent
CKPT = BASE / "results_math_v2"
CKPT.mkdir(exist_ok=True)
DATA_FILE = BASE / "math_real_200.json"

MODELS = {
    "Qwen2.5-0.5B": "/path/to/models/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-7B":   "/path/to/models/Qwen2.5-7B-Instruct",
}

METHODS = ["baseline", "cot_t0_256", "cot_t0_512"]

def extract_ans(text):
    if "####" in text:
        nums = re.findall(r'-?\d+\.?\d*', text.split("####")[-1])
        return nums[-1] if nums else text.strip()[-50:]
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        return nums[-1] if nums else boxed[-1]
    for pat in [r'(?:therefore|thus|the answer is)[:\s]+([^\n.]+)',
                r'answer[:\s]+([^\n.]+)', r'=\s*([+-]?\d+\.?\d*)\s*$']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            nums = re.findall(r'-?\d+\.?\d*', m.group(1) if 'answer' in pat.lower() or 'thus' in pat.lower() else m.group(0))
            if nums: return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]

def check(p, g):
    p, g = p.strip().replace(',','').replace(' ',''), str(g).strip().replace(',','').replace(' ','')
    if p == g: return True
    try: return abs(float(p)-float(g)) < 1e-6
    except: return p.lower() == g.lower()

def make_input(tok, question, use_cot=False):
    if use_cot:
        content = f"{question}\n\nPlease think step by step, then give your final numerical answer after ####."
    else:
        content = f"{question}\n\nGive ONLY the final numerical answer. Do NOT show any work or reasoning."
    messages = [{"role": "user", "content": content}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")

def main():
    device = "cuda:0"
    print(f"MATH Cross-Dataset Validation | {datetime.now()}", flush=True)

    with open(DATA_FILE) as f:
        alldata = json.load(f)
    print(f"Loaded {len(alldata)} MATH problems", flush=True)

    for mname, mpath in MODELS.items():
        print(f"\n{'='*60}\nLoading {mname}...\n{'='*60}", flush=True)
        tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            mpath, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        print(f"{mname} loaded", flush=True)

        for method in METHODS:
            ckpt_file = CKPT / f"{mname}_MATH_{method}_s{SEED}.json"
            if ckpt_file.exists():
                with open(ckpt_file) as f:
                    existing = json.load(f)
                if existing.get("metadata", {}).get("done"):
                    acc = sum(1 for r in existing["results"] if r.get("ok")) / len(existing["results"]) * 100
                    print(f"  SKIP {mname}/{method} (acc={acc:.1f}%)", flush=True)
                    continue
                results = existing.get("results", [])
                start = len(results)
            else:
                results = []
                start = 0

            if start >= N_SAMPLES: continue

            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            data = alldata[:N_SAMPLES]
            label = f"{mname[:8]}|{method}"

            for i in range(start, N_SAMPLES):
                q = data[i]["query"]
                gt = str(data[i].get("ground_truth", ""))

                try:
                    if method == "baseline":
                        inp = make_input(tok, q, use_cot=False)
                        inp = inp.to(device)
                        t0 = time.time()
                        with torch.no_grad():
                            out = model.generate(**inp, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
                        lat = time.time() - t0
                        response = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                        tokn = out.shape[1] - inp["input_ids"].shape[1]
                    elif method == "cot_t0_256":
                        inp = make_input(tok, q, use_cot=True)
                        inp = inp.to(device)
                        t0 = time.time()
                        with torch.no_grad():
                            out = model.generate(**inp, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
                        lat = time.time() - t0
                        response = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                        tokn = out.shape[1] - inp["input_ids"].shape[1]
                    elif method == "cot_t0_512":
                        inp = make_input(tok, q, use_cot=True)
                        inp = inp.to(device)
                        t0 = time.time()
                        with torch.no_grad():
                            out = model.generate(**inp, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
                        lat = time.time() - t0
                        response = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                        tokn = out.shape[1] - inp["input_ids"].shape[1]

                    ans = extract_ans(response)
                    ok = check(ans, gt)
                    results.append({"q": i, "ok": ok, "tok": tokn, "lat": round(lat, 4),
                                   "pred": ans, "gt": gt})
                except Exception as e:
                    results.append({"q": i, "ok": False, "tok": 0, "lat": 0,
                                   "pred": "", "gt": gt, "err": str(e)})

                if (i+1) % 20 == 0 or (i+1) == N_SAMPLES:
                    done = (i+1) == N_SAMPLES
                    with open(ckpt_file, 'w') as f:
                        json.dump({"metadata": {"model": mname, "ds": "MATH",
                                               "method": method, "seed": SEED,
                                               "done": done, "n": i+1}, "results": results}, f)
                    acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
                    print(f"  {label} [{i+1}/{N_SAMPLES}] acc={acc:.1f}%", flush=True)

        del model; torch.cuda.empty_cache(); gc.collect()
        print(f"{mname} freed", flush=True)

    print(f"\nDone: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
