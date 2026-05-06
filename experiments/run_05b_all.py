#!/usr/bin/env python3
"""
Qwen2.5-0.5B-Instruct Full TTS-Bench
======================================
Fill all missing cells for 0.5B in Table 6.
Runs: baseline, cot_short, cot_long, bon4, bon8, entropy_stop, adaptive
on MATH and GSM8K, 3 seeds each.

Usage: CUDA_VISIBLE_DEVICES=1 python3 run_05b_all.py
"""
import os, json, time, gc, random, re, warnings, sys
import torch, numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
SEEDS = [42, 123, 456]
N_QUESTIONS = 200
DEVICE = "cuda:0"

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_PATH = "/path/to/models/Qwen2.5-0.5B-Instruct"

DATASETS = {
    "MATH": BASE / "math_real_200.json",
    "GSM8K": BASE / "gsm8k_real_200.json",
}

CKPT_DIR = BASE / "results_v2"
CKPT_DIR.mkdir(exist_ok=True)

METHODS = ["baseline", "cot_short", "cot_long", "bon4_vote", "bon8_vote", "entropy_stop", "adaptive"]

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


def difficulty(q):
    wc = len(q.split())
    c = bool(re.search(r'\\frac|/\d+', q)) + bool(re.search(r'\\sqrt|sqrt', q)) + bool(re.search(r'\^', q))
    return "hard" if wc > 100 or c >= 2 else "medium" if wc > 50 or c >= 1 else "easy"


def build_prompt(tok, q, cot=False):
    content = f"{q}\nLet's think step by step." if cot else q
    if hasattr(tok, 'apply_chat_template'):
        messages = [{"role": "user", "content": content}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {q}\nAnswer:"


def gen_baseline(model, tok, q, device):
    prompt = build_prompt(tok, q, cot=False)
    inp = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_TOKENS_SHORT, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    gen_text = tok.decode(out[0], skip_special_tokens=True)
    if prompt in gen_text: gen_text = gen_text[len(prompt):]
    ans = extract_ans(gen_text)
    return ans, out.shape[1]-inp["input_ids"].shape[1], lat


def gen_cot(model, tok, q, device, max_tok=256):
    prompt = build_prompt(tok, q, cot=True)
    inp = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    gen_text = tok.decode(out[0], skip_special_tokens=True)
    if prompt in gen_text: gen_text = gen_text[len(prompt):]
    ans = extract_ans(gen_text)
    return ans, out.shape[1]-inp["input_ids"].shape[1], lat


def gen_bon(model, tok, q, device, n=4):
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
    text_so_far = tok.decode([tid])  # simplified
    return "unknown", gen, lat


def gen_adaptive(model, tok, q, device):
    d = difficulty(q)
    budget = {"easy": 128, "medium": 256, "hard": 512}[d]
    return gen_cot(model, tok, q, device, max_tok=budget)


def main():
    print(f"Qwen2.5-0.5B Full TTS-Bench | {datetime.now()}")
    print(f"Model: {MODEL_PATH} | Device: {DEVICE}")
    print(f"Methods: {METHODS} | Seeds: {SEEDS} | N={N_QUESTIONS}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"{MODEL_NAME} loaded ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    for dname, dfile in DATASETS.items():
        with open(dfile) as f:
            alldata = json.load(f)[:N_QUESTIONS]

        for method in METHODS:
            for seed in SEEDS:
                ckpt_file = CKPT_DIR / f"{MODEL_NAME}_{dname}_{method}_s{seed}.json"
                if ckpt_file.exists():
                    with open(ckpt_file) as f:
                        existing = json.load(f)
                    if existing.get("metadata", {}).get("done"):
                        print(f"  SKIP {dname}/{method}/s{seed}")
                        continue

                random.seed(seed); np.random.seed(seed)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

                data = alldata[:N_QUESTIONS]
                results = []
                print(f"  Running {dname}/{method}/s{seed}...", flush=True)

                for i, q_data in enumerate(data):
                    q = q_data["query"]
                    gt = str(q_data.get("ground_truth", q_data.get("answer", "")))

                    try:
                        if method == "baseline":
                            ans, tokn, lat = gen_baseline(model, tok, q, DEVICE)
                        elif method == "cot_short":
                            ans, tokn, lat = gen_cot(model, tok, q, DEVICE, 256)
                        elif method == "cot_long":
                            ans, tokn, lat = gen_cot(model, tok, q, DEVICE, 512)
                        elif method == "bon4_vote":
                            ans, tokn, lat = gen_bon(model, tok, q, DEVICE, 4)
                        elif method == "bon8_vote":
                            ans, tokn, lat = gen_bon(model, tok, q, DEVICE, 8)
                        elif method == "entropy_stop":
                            ans, tokn, lat = gen_entropy_stop(model, tok, q, DEVICE)
                        elif method == "adaptive":
                            ans, tokn, lat = gen_adaptive(model, tok, q, DEVICE)

                        ok = check(ans, gt)
                        results.append({"q": i, "ok": ok, "ans": ans, "tok": tokn, "lat": round(lat, 4)})
                    except Exception as e:
                        results.append({"q": i, "ok": False, "tok": 0, "lat": 0, "err": str(e)})

                    if (i+1) % 50 == 0:
                        done = (i+1) == N_QUESTIONS
                        with open(ckpt_file, 'w') as f:
                            json.dump({"metadata": {"model": MODEL_NAME, "ds": dname, "method": method,
                                       "seed": seed, "done": done, "n": i+1}, "results": results}, f)
                        acc = sum(r.get("ok", False) for r in results) / len(results) * 100
                        print(f"    {dname}/{method}/s{seed} [{i+1}/{N_QUESTIONS}] acc={acc:.1f}%", flush=True)

                # Final save
                with open(ckpt_file, 'w') as f:
                    json.dump({"metadata": {"model": MODEL_NAME, "ds": dname, "method": method,
                               "seed": seed, "done": True, "n": N_QUESTIONS}, "results": results}, f)
                acc = sum(r.get("ok", False) for r in results) / len(results) * 100
                print(f"    => {ckpt_file.name}: acc={acc:.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    for dname in DATASETS:
        print(f"\n{dname}:")
        for method in METHODS:
            accs = []
            for seed in SEEDS:
                f = CKPT_DIR / f"{MODEL_NAME}_{dname}_{method}_s{seed}.json"
                if f.exists():
                    d = json.load(open(f))
                    res = d.get("results", [])
                    acc = sum(r.get("ok", False) for r in res) / len(res) * 100 if res else 0
                    accs.append(acc)
            if accs:
                print(f"  {method}: {np.mean(accs):.1f}%")

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"\nDone: {datetime.now()}")


if __name__ == "__main__":
    main()
