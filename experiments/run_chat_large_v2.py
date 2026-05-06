#!/usr/bin/env python3
"""
TTS Novel Experiment v2: CORRECT Chat Template + Temperature Sweep
==================================================================
Key fix: Use Qwen2.5's official chat template instead of raw prompts.
Same-family Instruct models only: 0.5B, 3B, 7B (no 1.5B base).
Tests: baseline(no-CoT), cot(T=0), cot(T=0.3), cot(T=0.7), bon4(T=0.8)
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
CKPT = BASE / "results_novel_v2"
CKPT.mkdir(exist_ok=True)
DATA_FILE = BASE / "gsm8k_real_200.json"

MODELS = {
    "Qwen2.5-7B":   "/path/to/models/Qwen2.5-7B-Instruct",
}

METHODS = ["baseline", "cot_t0", "cot_t03", "cot_t05", "cot_t07", "bon4"]

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
    """Use Qwen2.5's official chat template"""
    if use_cot:
        content = f"{question}\n\nPlease think step by step, then give your final numerical answer after ####."
    else:
        content = f"{question}\n\nGive ONLY the final numerical answer. Do NOT show any work or reasoning."
    messages = [{"role": "user", "content": content}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt")
    return inp

def generate(model, tok, inp, device, max_tok=256, do_sample=False, temperature=1.0):
    inp = inp.to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_tok,
            do_sample=do_sample, temperature=temperature,
            pad_token_id=tok.eos_token_id
        )
    lat = time.time() - t0
    new_tokens = out.shape[1] - inp["input_ids"].shape[1]
    response = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return response, new_tokens, lat

def main():
    device = "cuda:0"
    print(f"TTS Novel v2 (Chat Template) | {datetime.now()}", flush=True)
    print(f"Models: {list(MODELS.keys())}", flush=True)
    print(f"Methods: {METHODS}", flush=True)

    with open(DATA_FILE) as f:
        alldata = json.load(f)
    print(f"Loaded {len(alldata)} problems", flush=True)

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
            label = f"{mname[:8]}|{method[:8]}"

            for i in range(start, N_SAMPLES):
                q = data[i]["query"]
                gt = str(data[i].get("ground_truth", ""))

                try:
                    if method == "baseline":
                        inp = make_input(tok, q, use_cot=False)
                        response, tokn, lat = generate(model, tok, inp, device, max_tok=64, do_sample=False)
                    elif method.startswith("cot_"):
                        temp_map = {"cot_t0": 0.0, "cot_t03": 0.3, "cot_t05": 0.5, "cot_t07": 0.7}
                        temperature = temp_map[method]
                        do_sample = temperature > 0
                        inp = make_input(tok, q, use_cot=True)
                        response, tokn, lat = generate(model, tok, inp, device, max_tok=256,
                                                       do_sample=do_sample, temperature=temperature)
                    elif method == "bon4":
                        inp = make_input(tok, q, use_cot=True)
                        answers = []
                        total_tok = 0
                        t0 = time.time()
                        for _ in range(4):
                            resp, t, _ = generate(model, tok, inp, device, max_tok=256,
                                                 do_sample=True, temperature=0.8)
                            answers.append(extract_ans(resp))
                            total_tok += t
                        lat = time.time() - t0
                        counts = Counter(answers)
                        ans = counts.most_common(1)[0][0]
                        tokn = total_tok
                        response = ans  # for logging

                    if method != "bon4":
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
                        json.dump({"metadata": {"model": mname, "ds": "GSM8K",
                                               "method": method, "seed": SEED,
                                               "done": done, "n": i+1,
                                               "chat_template": True},
                                  "results": results}, f)
                    acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
                    print(f"  {label} [{i+1}/{N_SAMPLES}] acc={acc:.1f}%", flush=True)

        del model; torch.cuda.empty_cache(); gc.collect()
        print(f"{mname} freed", flush=True)

    print(f"\nDone: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
