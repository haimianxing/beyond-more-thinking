#!/usr/bin/env python3
"""
PRIORITY EXPERIMENTS: Fill Innovation 3 model scale gap
========================================================
Run Qwen2.5-1.5B on MATH to add a 3rd scale point for Phase Transition.
Also run Qwen2.5-3B on MATH for a 4th scale point.
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

MODELS = {
    "Qwen2.5-1.5B": "/path/to/models/Qwen2.5-1.5B",
    "Qwen2.5-3B":   "/path/to/models/Qwen2.5-3B-Instruct",
}

METHODS = {
    "baseline":     {"use_cot": False, "max_tokens": 64,  "temp": 0.0},
    "cot_t0_256":   {"use_cot": True,  "max_tokens": 256, "temp": 0.0},
    "cot_t0_512":   {"use_cot": True,  "max_tokens": 512, "temp": 0.0},
}

def extract_ans(text, ds="MATH"):
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
    device = "cuda:0"  # Will map to CUDA_VISIBLE_DEVICES
    print(f"Scale Gap Fill Experiments | {datetime.now()}", flush=True)

    with open(BASE / "math_real_200.json") as f:
        alldata = json.load(f)

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Loading {model_name}")
        print(f"{'='*60}", flush=True)

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        print(f"{model_name} loaded", flush=True)

        for method_name, cfg in METHODS.items():
            ckpt_file = CKPT / f"{model_name}_MATH_{method_name}_s{SEED}.json"
            if ckpt_file.exists():
                with open(ckpt_file) as f:
                    existing = json.load(f)
                if existing.get("metadata", {}).get("done"):
                    acc = sum(1 for r in existing["results"] if r.get("ok")) / len(existing["results"]) * 100
                    print(f"  SKIP {model_name}/{method_name} (acc={acc:.1f}%)", flush=True)
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

            for i in range(start, N_SAMPLES):
                q = data[i]["query"]
                gt = str(data[i].get("ground_truth", ""))

                try:
                    inp = make_input(tok, q, use_cot=cfg["use_cot"])
                    inp = inp.to(device)
                    t0 = time.time()
                    with torch.no_grad():
                        out = model.generate(
                            **inp,
                            max_new_tokens=cfg["max_tokens"],
                            do_sample=cfg["temp"] > 0,
                            temperature=cfg["temp"] if cfg["temp"] > 0 else None,
                            pad_token_id=tok.eos_token_id
                        )
                    lat = time.time() - t0
                    response = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                    tokn = out.shape[1] - inp["input_ids"].shape[1]

                    ans = extract_ans(response, "MATH")
                    ok = check(ans, gt)
                    results.append({"q": i, "ok": ok, "tok": tokn, "lat": round(lat, 4),
                                   "pred": ans, "gt": gt})
                except Exception as e:
                    results.append({"q": i, "ok": False, "tok": 0, "lat": 0,
                                   "pred": "", "gt": gt, "err": str(e)})

                if (i+1) % 20 == 0 or (i+1) == N_SAMPLES:
                    done = (i+1) == N_SAMPLES
                    with open(ckpt_file, 'w') as f:
                        json.dump({"metadata": {"model": model_name, "ds": "MATH",
                                               "method": method_name, "seed": SEED,
                                               "done": done, "n": i+1,
                                               "chat_template": True},
                                  "results": results}, f)
                    acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
                    print(f"  {model_name}|{method_name} [{i+1}/{N_SAMPLES}] acc={acc:.1f}%", flush=True)

        del model; torch.cuda.empty_cache(); gc.collect()
        print(f"{model_name} freed", flush=True)

    print(f"\nScale gap experiments done: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
