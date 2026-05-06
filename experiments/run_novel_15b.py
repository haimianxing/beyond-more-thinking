#!/usr/bin/env python3
"""
Qwen2.5-1.5B (base, non-Instruct) on GPU 3
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
from collections import Counter
from datetime import datetime
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
N_SAMPLES = 200
BASE = Path(__file__).parent
CKPT = BASE / "results_novel"
CKPT.mkdir(exist_ok=True)
DATA_FILE = BASE / "gsm8k_real_200.json"

MODELS = {
    "Qwen2.5-1.5B": "/path/to/models/Qwen2.5-1.5B",
}

METHODS = ["baseline", "cot_short", "cot_short_t03", "cot_short_t07", "bon4_vote"]

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

def gen_baseline(model, tok, q, device):
    inp = tok(f"Question: {q}\nAnswer:", return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    return extract_ans(tok.decode(out[0], skip_special_tokens=True)), out.shape[1]-inp["input_ids"].shape[1], lat

def gen_cot_det(model, tok, q, device, max_tok=256):
    inp = tok(f"Question: {q}\nLet's think step by step.\nAnswer:", return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    return extract_ans(tok.decode(out[0], skip_special_tokens=True)), out.shape[1]-inp["input_ids"].shape[1], lat

def gen_cot_stochastic(model, tok, q, device, max_tok=256, temperature=0.3):
    inp = tok(f"Question: {q}\nLet's think step by step.\nAnswer:", return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, do_sample=True,
                            temperature=temperature, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    return extract_ans(tok.decode(out[0], skip_special_tokens=True)), out.shape[1]-inp["input_ids"].shape[1], lat

def gen_bon(model, tok, q, device, n=4):
    inp = tok(f"Question: {q}\nLet's think step by step.\nAnswer:", return_tensors="pt").to(device)
    answers = []
    total_tok = 0
    t0 = time.time()
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=True,
                               temperature=0.8, pad_token_id=tok.eos_token_id)
        answers.append(extract_ans(tok.decode(out[0], skip_special_tokens=True)))
        total_tok += out.shape[1] - inp["input_ids"].shape[1]
    lat = time.time() - t0
    counts = Counter(answers)
    best = counts.most_common(1)[0][0]
    return best, total_tok, lat

def main():
    device = "cuda:0"
    print(f"[1.5B] Novel Experiment | {datetime.now()}", flush=True)

    with open(DATA_FILE) as f:
        alldata = json.load(f)
    print(f"Loaded {len(alldata)} problems", flush=True)

    for mname, mpath in MODELS.items():
        print(f"\nLoading {mname} from {mpath}...", flush=True)
        tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            mpath, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        print(f"{mname} loaded", flush=True)

        for method in METHODS:
            ckpt_file = CKPT / f"{mname}_GSM8K_{method}_s{SEED}.json"
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

            if start >= N_SAMPLES:
                continue

            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)

            data = alldata[:N_SAMPLES]
            label = f"{mname[:8]}|{method[:12]}"

            for i in range(start, N_SAMPLES):
                q = data[i]["query"]
                gt = str(data[i].get("ground_truth", ""))

                try:
                    if method == "baseline":
                        ans, tokn, lat = gen_baseline(model, tok, q, device)
                    elif method == "cot_short":
                        ans, tokn, lat = gen_cot_det(model, tok, q, device, 256)
                    elif method == "cot_short_t03":
                        ans, tokn, lat = gen_cot_stochastic(model, tok, q, device, 256, 0.3)
                    elif method == "cot_short_t07":
                        ans, tokn, lat = gen_cot_stochastic(model, tok, q, device, 256, 0.7)
                    elif method == "bon4_vote":
                        ans, tokn, lat = gen_bon(model, tok, q, device, 4)

                    ok = check(ans, gt)
                    results.append({"q": i, "ok": ok, "tok": tokn, "lat": round(lat, 4),
                                   "pred": ans, "gt": gt})
                except Exception as e:
                    results.append({"q": i, "ok": False, "tok": 0, "lat": 0,
                                   "pred": "", "gt": gt, "err": str(e)})

                if (i+1) % 20 == 0 or (i+1) == N_SAMPLES:
                    done = (i+1) == N_SAMPLES
                    with open(ckpt_file, 'w') as f:
                        json.dump({"metadata": {"model": mname, "ds": "GSM8K",
                                               "method": method, "seed": SEED,
                                               "done": done, "n": i+1},
                                  "results": results}, f)
                    acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
                    print(f"  {label} [{i+1}/{N_SAMPLES}] acc={acc:.1f}%", flush=True)

        del model; torch.cuda.empty_cache(); gc.collect()
        print(f"{mname} freed", flush=True)

    print(f"\n[1.5B] Done: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
