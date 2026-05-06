#!/usr/bin/env python3
"""
Multi-seed validation: Qwen2.5-7B MATH + LLaMA-3-8B GSM8K
Addresses CW2 (single seed) from SAC review.
Seeds: 123, 456, 789
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

N_SAMPLES = 200
BASE = Path(__file__).parent

def extract_ans(text, ds="GSM8K"):
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

EXPERIMENTS = [
    # Qwen2.5-7B MATH multi-seed
    {"model_name": "Qwen2.5-7B", "model_path": "/path/to/models/Qwen2.5-7B-Instruct",
     "data_file": "math_real_200.json", "ckpt_dir": "results_math_v2", "ds_name": "MATH",
     "seeds": [123, 456, 789],
     "methods": {
         "baseline":     {"use_cot": False, "max_tokens": 64,  "temp": 0.0},
         "cot_t0_256":   {"use_cot": True,  "max_tokens": 256, "temp": 0.0},
         "cot_t0_512":   {"use_cot": True,  "max_tokens": 512, "temp": 0.0},
     }},
    # LLaMA-3-8B GSM8K multi-seed (just 1 extra seed for now)
    {"model_name": "Llama-3-8B", "model_path": "/path/to/models/Llama-3-8B-Instruct",
     "data_file": "gsm8k_real_200.json", "ckpt_dir": "results_llama_v2", "ds_name": "GSM8K",
     "seeds": [123],
     "methods": {
         "baseline":     {"use_cot": False, "max_tokens": 64,  "temp": 0.0},
         "cot_t0":       {"use_cot": True,  "max_tokens": 256, "temp": 0.0},
     }},
]

def main():
    device = "cuda:0"  # Will map to CUDA_VISIBLE_DEVICES
    print(f"Multi-Seed Validation | {datetime.now()}", flush=True)

    for exp in EXPERIMENTS:
        model_name = exp["model_name"]
        model_path = exp["model_path"]
        data_file = BASE / exp["data_file"]
        ckpt_dir = BASE / exp["ckpt_dir"]
        ds_name = exp["ds_name"]

        print(f"\n{'='*60}")
        print(f"Loading {model_name} from {model_path}")
        print(f"{'='*60}", flush=True)

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        print(f"{model_name} loaded ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)", flush=True)

        with open(data_file) as f:
            alldata = json.load(f)

        for seed in exp["seeds"]:
            for method_name, cfg in exp["methods"].items():
                ckpt_file = ckpt_dir / f"{model_name}_{ds_name}_{method_name}_s{seed}.json"
                if ckpt_file.exists():
                    with open(ckpt_file) as f:
                        existing = json.load(f)
                    if existing.get("metadata", {}).get("done"):
                        acc = sum(1 for r in existing["results"] if r.get("ok")) / len(existing["results"]) * 100
                        print(f"  SKIP {model_name}/{method_name}/s{seed} (acc={acc:.1f}%)", flush=True)
                        continue
                    results = existing.get("results", [])
                    start = len(results)
                else:
                    results = []
                    start = 0

                if start >= N_SAMPLES:
                    continue

                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                data = alldata[:N_SAMPLES]
                label = f"{model_name[:12]}|{method_name}|s{seed}"

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

                        ans = extract_ans(response, ds_name)
                        ok = check(ans, gt)
                        results.append({"q": i, "ok": ok, "tok": tokn, "lat": round(lat, 4),
                                       "pred": ans, "gt": gt})
                    except Exception as e:
                        results.append({"q": i, "ok": False, "tok": 0, "lat": 0,
                                       "pred": "", "gt": gt, "err": str(e)})

                    if (i+1) % 20 == 0 or (i+1) == N_SAMPLES:
                        done = (i+1) == N_SAMPLES
                        with open(ckpt_file, 'w') as f:
                            json.dump({"metadata": {"model": model_name, "ds": ds_name,
                                                   "method": method_name, "seed": seed,
                                                   "done": done, "n": i+1}, "results": results}, f)
                        acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
                        print(f"  {label} [{i+1}/{N_SAMPLES}] acc={acc:.1f}%", flush=True)

        del model; torch.cuda.empty_cache(); gc.collect()
        print(f"{model_name} freed", flush=True)

    print(f"\nMulti-seed done: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
