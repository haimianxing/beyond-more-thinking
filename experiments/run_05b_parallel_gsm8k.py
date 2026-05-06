#!/usr/bin/env python3
"""Parallel 0.5B GSM8K experiment runner. Usage: CUDA_VISIBLE_DEVICES=X python3 run_05b_parallel_gsm8k.py METHOD"""
import os, sys, json, time, gc, random, re, warnings
import torch, numpy as np
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

METHOD = sys.argv[1]  # bon8_vote, entropy_stop, adaptive
SEEDS = [42, 123, 456]
N = 200
DEVICE = "cuda:0"

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_PATH = "/path/to/models/Qwen2.5-0.5B-Instruct"
DATA_FILE = Path(__file__).parent / "gsm8k_real_200.json"
CKPT_DIR = Path(__file__).parent / "results_v2"

MAX_TOKENS_SHORT = 256
MAX_TOKENS_LONG = 512
TEMPERATURE = 0.7
TOP_P = 0.9

def extract_ans(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums: return nums[-1]
    for pat in [
        r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+|so the answer is|final answer[:\s]+)([^\n.]+)',
        r'answer[:\s]+([^\n.]+)',
    ]:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            nums = re.findall(r'-?\d+\.?\d*', matches[-1].group(1))
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

def build_prompt(tok, q, cot=True):
    content = f"{q}\nLet's think step by step." if cot else q
    messages = [{"role": "user", "content": content}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def gen_bon(model, tok, q, device, n=8):
    prompt = build_prompt(tok, q, cot=True)
    inp = tok(prompt, return_tensors="pt").to(device)
    answers = []
    total_tok = 0
    t0 = time.time()
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=MAX_TOKENS_SHORT, do_sample=True,
                                 temperature=TEMPERATURE, top_p=TOP_P, pad_token_id=tok.eos_token_id)
        gen_text = tok.decode(out[0], skip_special_tokens=True)
        if prompt in gen_text: gen_text = gen_text[len(prompt):]
        answers.append(extract_ans(gen_text))
        total_tok += out.shape[1] - inp["input_ids"].shape[1]
    lat = time.time() - t0
    counts = Counter(answers)
    best = counts.most_common(1)[0][0]
    return best, total_tok, lat

def gen_entropy_stop(model, tok, q, device, max_tok=512):
    prompt = build_prompt(tok, q, cot=True)
    inp = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        f = model(**inp, use_cache=True)
    kv = f.past_key_values
    logits = f.logits[0,-1,:].float()
    gen = 0
    hc = 0
    for s in range(max_tok):
        probs = torch.softmax(logits/0.7, dim=-1)
        top1 = probs.max().item()
        tid = torch.multinomial(probs, 1).item()
        gen += 1
        if gen > 50 and top1 > 0.6:
            hc += 1
            if hc >= 3: break
        else: hc = 0
        with torch.no_grad():
            o = model(input_ids=torch.tensor([[tid]]).to(device), past_key_values=kv, use_cache=True)
            kv = o.past_key_values
            logits = o.logits[0,-1,:].float()
    lat = time.time() - t0
    text_so_far = tok.decode([tid])
    return "unknown", gen, lat

def difficulty(q):
    wc = len(q.split())
    c = bool(re.search(r'\\frac|/\d+', q)) + bool(re.search(r'\\sqrt|sqrt', q)) + bool(re.search(r'\^', q))
    return "hard" if wc > 100 or c >= 2 else "medium" if wc > 50 or c >= 1 else "easy"

def gen_adaptive(model, tok, q, device):
    d = difficulty(q)
    budget = {"easy": 128, "medium": 256, "hard": 512}[d]
    prompt = build_prompt(tok, q, cot=True)
    inp = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=budget, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    gen_text = tok.decode(out[0], skip_special_tokens=True)
    if prompt in gen_text: gen_text = gen_text[len(prompt):]
    ans = extract_ans(gen_text)
    return ans, out.shape[1]-inp["input_ids"].shape[1], lat

def main():
    print(f"[{METHOD}] GSM8K parallel | {time.strftime('%H:%M:%S')}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"[{METHOD}] Model loaded", flush=True)

    with open(DATA_FILE) as f:
        alldata = json.load(f)[:N]

    for seed in SEEDS:
        ckpt_file = CKPT_DIR / f"{MODEL_NAME}_GSM8K_{METHOD}_s{seed}.json"
        if ckpt_file.exists():
            d = json.load(open(ckpt_file))
            if d.get("metadata", {}).get("done"):
                print(f"[{METHOD}] SKIP s{seed}", flush=True)
                continue

        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        results = []
        for i, q_data in enumerate(alldata):
            q = q_data["query"]
            gt = str(q_data.get("ground_truth", q_data.get("answer", "")))
            try:
                if METHOD == "bon8_vote":
                    ans, tokn, lat = gen_bon(model, tok, q, DEVICE, 8)
                elif METHOD == "bon4_vote":
                    ans, tokn, lat = gen_bon(model, tok, q, DEVICE, 4)
                elif METHOD == "entropy_stop":
                    ans, tokn, lat = gen_entropy_stop(model, tok, q, DEVICE)
                elif METHOD == "adaptive":
                    ans, tokn, lat = gen_adaptive(model, tok, q, DEVICE)
                ok = check(ans, gt)
                results.append({"q": i, "ok": ok, "ans": ans, "tok": tokn, "lat": round(lat, 4)})
            except Exception as e:
                results.append({"q": i, "ok": False, "tok": 0, "lat": 0, "err": str(e)})

            if (i+1) % 50 == 0:
                acc = sum(r.get("ok", False) for r in results) / len(results) * 100
                print(f"[{METHOD}] s{seed} [{i+1}/{N}] acc={acc:.1f}%", flush=True)

        acc = sum(r.get("ok", False) for r in results) / len(results) * 100
        with open(ckpt_file, 'w') as f:
            json.dump({"metadata": {"model": MODEL_NAME, "ds": "GSM8K", "method": METHOD,
                       "seed": seed, "done": True, "n": N}, "results": results}, f)
        print(f"[{METHOD}] s{seed} DONE acc={acc:.1f}%", flush=True)

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"[{METHOD}] ALL DONE | {time.strftime('%H:%M:%S')}", flush=True)

if __name__ == "__main__":
    main()
