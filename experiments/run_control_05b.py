#!/usr/bin/env python3
"""
Control experiment: Qwen2.5-0.5B-Instruct (same family as 3B/7B)
Purpose: Distinguish scale effect from training difference (Qwen2 vs Qwen2.5)
If 0.5B shows CoT helpful → scale effect confirmed
If 0.5B shows CoT harmful → training difference confound
"""
import sys, os, json, time, re, gc, random
import torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = Path(__file__).parent
SEEDS = [42, 123, 456]
N_SAMPLES = 200
DEVICE = "cuda:0"

MODEL_PATH = "/path/to/models/Qwen2.5-0.5B-Instruct"
DATA_FILE = BASE / "math_real_200.json"

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
    messages = [{"role": "user", "content": content}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_method(model, tok, q, device, method, max_tok):
    if method == "baseline":
        prompt = build_prompt(tok, q, cot=False)
    else:  # cot_512
        prompt = build_prompt(tok, q, cot=True)
    
    inp = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    ans = extract_ans(tok.decode(out[0], skip_special_tokens=True))
    gen_tok = out.shape[1] - inp["input_ids"].shape[1]
    truncated = gen_tok >= max_tok - 5
    return ans, gen_tok, lat, truncated

def main():
    print(f"Control Experiment: Qwen2.5-0.5B-Instruct | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    with open(DATA_FILE) as f:
        data = json.load(f)
    questions = data[:N_SAMPLES]
    
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"Qwen2.5-0.5B loaded on {DEVICE}", flush=True)
    
    CKPT = BASE / "results_v2"
    
    for method_name, use_cot, max_tok in [
        ("baseline_256", False, 256),
        ("baseline_512", False, 512),
        ("cot_512", True, 512),
    ]:
        for seed in SEEDS:
            ckpt_file = CKPT / f"Qwen2.5-0.5B_MATH_{method_name}_s{seed}_verify.json"
            if ckpt_file.exists():
                d = json.load(open(ckpt_file))
                if d.get("metadata", {}).get("done"):
                    print(f"  SKIP Qwen2.5-0.5B/{method_name}/s{seed}", flush=True)
                    continue
            
            random.seed(seed); np.random.seed(seed)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            
            results = []
            for i in range(N_SAMPLES):
                q = questions[i]["query"]
                gt = str(questions[i].get("ground_truth", questions[i].get("answer", "")))
                
                try:
                    ans, tokn, lat, trunc = run_method(model, tok, q, DEVICE, method_name, max_tok)
                    results.append({"q": i, "ok": check(ans, gt), "tok": tokn, "lat": round(lat, 4), "truncated": trunc})
                except Exception as e:
                    results.append({"q": i, "ok": False, "tok": 0, "lat": 0, "err": str(e), "truncated": False})
                
                if (i+1) % 100 == 0 or (i+1) == N_SAMPLES:
                    acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
                    trunc_rate = sum(1 for r in results if r.get("truncated")) / len(results) * 100
                    print(f"  0.5B/{method_name}/s{seed} [{i+1}/{N_SAMPLES}] acc={acc:.1f}% trunc={trunc_rate:.0f}%", flush=True)
            
            with open(ckpt_file, 'w') as f:
                json.dump({"metadata": {"model": "Qwen2.5-0.5B", "ds": "MATH", "method": method_name, "seed": seed, "done": True}, "results": results}, f)
            
            acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
            trunc = sum(1 for r in results if r.get("truncated")) / len(results) * 100
            print(f"  => Qwen2.5-0.5B/{method_name}/s{seed}: acc={acc:.1f}% trunc={trunc:.0f}%", flush=True)
    
    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"Qwen2.5-0.5B control experiment done: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

if __name__ == "__main__":
    main()
