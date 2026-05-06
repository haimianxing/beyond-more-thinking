#!/usr/bin/env python3
"""Run Qwen2-7B-Instruct controlled decomposition for a single seed.
Usage: CUDA_VISIBLE_DEVICES=X python3 run_qwen2_7b_seeds.py SEED

Runs: base_256, base_512, cot_512 on both MATH and GSM8K
"""
import os, sys, json, time, gc, re, warnings, random
import torch, numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

SEED = int(sys.argv[1])
DEVICE = "cuda:0"

MODEL_NAME = "Qwen2-7B-Instruct"
MODEL_PATH = "/path/to/models/Qwen2-7B-Instruct"
N = 200

BASE = Path(__file__).parent
DATA_FILES = {
    "MATH": BASE / "math_real_200.json",
    "GSM8K": BASE / "gsm8k_real_200.json",
}
CKPT_DIR = BASE / "results_v2" / "qwen2_7b_controlled"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


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


def run_condition(model, tok, questions, ds, condition, seed):
    """Run one condition (base_256, base_512, cot_512) and save results."""
    ckpt_file = CKPT_DIR / f"{ds}_{condition}_s{seed}.json"
    if ckpt_file.exists():
        d = json.load(open(ckpt_file))
        if d.get("metadata", {}).get("done"):
            print(f"  SKIP {ds}/{condition}/s{seed}", flush=True)
            return

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    max_tok = 256 if "256" in condition else 512
    cot = condition.startswith("cot")

    results = []
    t0 = time.time()
    for i, q_data in enumerate(questions):
        q = q_data["query"]
        gt = str(q_data.get("ground_truth", q_data.get("answer", "")))

        content = f"{q}\nLet's think step by step." if cot else q
        messages = [{"role": "user", "content": content}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen_text = tok.decode(out[0], skip_special_tokens=True)
        if prompt in gen_text: gen_text = gen_text[len(prompt):]
        ans = extract_ans(gen_text)
        gen_tok = out.shape[1] - inp["input_ids"].shape[1]

        # Compute truncation
        truncated = gen_tok >= max_tok - 5

        ok = check(ans, gt)
        results.append({"q": i, "ok": ok, "ans": ans, "gen_tok": gen_tok, "truncated": truncated})

        if (i+1) % 50 == 0:
            acc = sum(r["ok"] for r in results) / len(results) * 100
            trunc_rate = sum(r["truncated"] for r in results) / len(results) * 100
            print(f"  {ds}/{condition}/s{seed} [{i+1}/{N}] acc={acc:.1f}% trunc={trunc_rate:.0f}%", flush=True)

    lat = time.time() - t0
    acc = sum(r["ok"] for r in results) / len(results) * 100
    trunc_rate = sum(r["truncated"] for r in results) / len(results) * 100

    with open(ckpt_file, 'w') as f:
        json.dump({"metadata": {"model": MODEL_NAME, "ds": ds, "condition": condition,
                   "seed": seed, "done": True, "n": N, "acc": round(acc, 1),
                   "trunc_rate": round(trunc_rate, 1)}, "results": results}, f)
    print(f"  {ds}/{condition}/s{seed} DONE acc={acc:.1f}% trunc={trunc_rate:.0f}%", flush=True)


def main():
    print(f"[Qwen2-7B] seed={SEED} | {time.strftime('%H:%M:%S')}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"[Qwen2-7B] Model loaded for s{SEED}", flush=True)

    for ds_name, data_file in DATA_FILES.items():
        with open(data_file) as f:
            questions = json.load(f)[:N]

        for condition in ["base_256", "base_512", "cot_512"]:
            run_condition(model, tok, questions, ds_name, condition, SEED)

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"[Qwen2-7B] s{SEED} ALL DONE | {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
