#!/usr/bin/env python3
"""Run a single method+seed combination. Usage: CUDA_VISIBLE_DEVICES=X python3 run_05b_single_seed.py METHOD SEED"""
import os, sys, json, time, gc, random, re, warnings
import torch, numpy as np
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

METHOD = sys.argv[1]
SEED = int(sys.argv[2])
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

def main():
    print(f"[{METHOD}] GSM8K seed={SEED} | {time.strftime('%H:%M:%S')}", flush=True)

    ckpt_file = CKPT_DIR / f"{MODEL_NAME}_GSM8K_{METHOD}_s{SEED}.json"
    if ckpt_file.exists():
        d = json.load(open(ckpt_file))
        if d.get("metadata", {}).get("done"):
            print(f"[{METHOD}] SKIP s{SEED} (already done)", flush=True)
            return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"[{METHOD}] Model loaded for s{SEED}", flush=True)

    with open(DATA_FILE) as f:
        alldata = json.load(f)[:N]

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    results = []
    for i, q_data in enumerate(alldata):
        q = q_data["query"]
        gt = str(q_data.get("ground_truth", q_data.get("answer", "")))
        try:
            if METHOD in ("bon8_vote", "bon4_vote"):
                n = 8 if "8" in METHOD else 4
                ans, tokn, lat = gen_bon(model, tok, q, DEVICE, n)
            else:
                raise ValueError(f"Unsupported method: {METHOD}")
            ok = check(ans, gt)
            results.append({"q": i, "ok": ok, "ans": ans, "tok": tokn, "lat": round(lat, 4)})
        except Exception as e:
            results.append({"q": i, "ok": False, "tok": 0, "lat": 0, "err": str(e)})

        if (i+1) % 50 == 0:
            acc = sum(r.get("ok", False) for r in results) / len(results) * 100
            print(f"[{METHOD}] s{SEED} [{i+1}/{N}] acc={acc:.1f}%", flush=True)

    acc = sum(r.get("ok", False) for r in results) / len(results) * 100
    with open(ckpt_file, 'w') as f:
        json.dump({"metadata": {"model": MODEL_NAME, "ds": "GSM8K", "method": METHOD,
                   "seed": SEED, "done": True, "n": N}, "results": results}, f)
    print(f"[{METHOD}] s{SEED} DONE acc={acc:.1f}% | {time.strftime('%H:%M:%S')}", flush=True)

    del model; torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
