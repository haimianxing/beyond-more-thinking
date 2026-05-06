#!/usr/bin/env python3
"""Extended N=321 Evaluation on Qwen2.5-3B-Instruct/MATH.

Run Base-256, Base-512, CoT-512, Format-Only on the merged 321-question MATH dataset.
This increases statistical power from 0.42 (N=200) to ~0.55 for the -7.5pp CoT effect.

Usage:
    python3 -u run_extended_n321.py \
        --model_path /path/to/Qwen2.5-3B-Instruct \
        --data_file /path/to/math_merged_all.json \
        --output_dir ./results_extended_n321 \
        --gpu 5
"""
import sys, os, json, time, gc, re, warnings, argparse
import torch, numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
N_QUESTIONS = 321

# Four conditions to test
CONDITIONS = {
    "base_256": {"suffix": "", "max_tokens": 256},
    "base_512": {"suffix": "", "max_tokens": 512},
    "cot_512": {"suffix": "\nLet's think step by step.", "max_tokens": 512},
    "format_only_512": {"suffix": "\nSolve this. Put answer in \\boxed{}.", "max_tokens": 512},
}


def extract_ans(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums: return nums[-1]
    for pat in [r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+)([^\n.]+)',
                r'answer[:\s]+([^\n.]+)']:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            nums = re.findall(r'-?\d+\.?\d*', matches[-1].group(1))
            if nums: return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]


def check(p, g):
    p = p.strip().replace(',', '').replace(' ', '')
    g = str(g).strip().replace(',', '').replace(' ', '')
    if p == g: return True
    try: return abs(float(p) - float(g)) < 1e-6
    except: return p.lower() == g.lower()


def main():
    parser = argparse.ArgumentParser(description="Extended N=321 Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--data_file", required=True, help="Path to merged MATH questions JSON")
    parser.add_argument("--output_dir", required=True, help="Directory for results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    DEVICE = "cuda:0"

    MODEL_PATH = args.model_path
    DATA_FILE = Path(args.data_file)
    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Extended N=321 Evaluation | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    with open(DATA_FILE) as f:
        questions = json.load(f)[:N_QUESTIONS]
    print(f"Loaded {len(questions)} MATH questions (merged)", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}", flush=True)

    all_results = {}

    for cname, cconfig in CONDITIONS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Running: {cname} (max_tokens={cconfig['max_tokens']})", flush=True)
        print(f"{'='*60}", flush=True)

        ckpt = OUT / f"qwen25_3b_MATH_{cname}_n321.json"
        if ckpt.exists():
            print(f"  SKIP {cname} (cached)", flush=True)
            all_results[cname] = json.load(open(ckpt))
            continue

        results = []
        for i in range(len(questions)):
            q = questions[i]["query"]
            gt = str(questions[i].get("ground_truth", questions[i].get("answer", "")))

            content = q + cconfig["suffix"]
            messages = [{"role": "user", "content": content}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=cconfig["max_tokens"],
                                     do_sample=False, pad_token_id=tok.eos_token_id)

            gen_text = tok.decode(out[0], skip_special_tokens=True)
            if prompt in gen_text:
                gen_text = gen_text[len(prompt):]

            gen_tok = out.shape[1] - inp["input_ids"].shape[1]
            truncated = gen_tok >= cconfig["max_tokens"] - 5

            ans = extract_ans(gen_text)
            ok = check(ans, gt)

            results.append({
                "q": i, "ok": ok, "tok": gen_tok,
                "truncated": truncated, "ans": ans, "gt": gt,
            })

            if (i + 1) % 50 == 0:
                acc = sum(r["ok"] for r in results) / len(results) * 100
                print(f"    {cname}: {i+1}/{len(questions)} acc={acc:.1f}%", flush=True)

        data = {"metadata": {"model": "Qwen2.5-3B-Instruct", "method": cname,
                             "n": len(questions), "max_tokens": cconfig["max_tokens"]},
                "results": results}
        with open(ckpt, 'w') as f:
            json.dump(data, f)

        acc = sum(r["ok"] for r in results) / len(results) * 100
        avg_tok = np.mean([r["tok"] for r in results])
        trunc_rate = sum(r["truncated"] for r in results) / len(results) * 100
        all_results[cname] = data
        print(f"  {cname}: acc={acc:.1f}% avg_tok={avg_tok:.0f} trunc={trunc_rate:.0f}%", flush=True)

    # === Summary ===
    print(f"\n{'='*80}", flush=True)
    print("EXTENDED N=321 SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    for cname, data in all_results.items():
        res = data["results"]
        acc = sum(r["ok"] for r in res) / len(res) * 100
        avg_tok = np.mean([r["tok"] for r in res])
        trunc = sum(r["truncated"] for r in res) / len(res) * 100
        print(f"  {cname:25s}: acc={acc:.1f}% avg_tok={avg_tok:.0f} trunc={trunc:.0f}%", flush=True)

    # Compute deltas
    if "base_512" in all_results and "cot_512" in all_results:
        base_acc = sum(r["ok"] for r in all_results["base_512"]["results"]) / len(all_results["base_512"]["results"]) * 100
        cot_acc = sum(r["ok"] for r in all_results["cot_512"]["results"]) / len(all_results["cot_512"]["results"]) * 100
        print(f"\n  CoT Effect (N=321): {cot_acc:.1f}% - {base_acc:.1f}% = {cot_acc-base_acc:+.1f}pp", flush=True)

    if "base_512" in all_results and "format_only_512" in all_results:
        base_acc = sum(r["ok"] for r in all_results["base_512"]["results"]) / len(all_results["base_512"]["results"]) * 100
        fmt_acc = sum(r["ok"] for r in all_results["format_only_512"]["results"]) / len(all_results["format_only_512"]["results"]) * 100
        print(f"  Format Effect (N=321): {fmt_acc:.1f}% - {base_acc:.1f}% = {fmt_acc-base_acc:+.1f}pp", flush=True)

    if "base_256" in all_results and "base_512" in all_results:
        b256 = sum(r["ok"] for r in all_results["base_256"]["results"]) / len(all_results["base_256"]["results"]) * 100
        b512 = sum(r["ok"] for r in all_results["base_512"]["results"]) / len(all_results["base_512"]["results"]) * 100
        print(f"  Token Effect (N=321): {b512:.1f}% - {b256:.1f}% = {b512-b256:+.1f}pp", flush=True)

    # McNemar tests
    from scipy import stats
    for pair_name, (c1_name, c2_name) in [("CoT Effect", ("base_512", "cot_512")),
                                            ("Format Effect", ("base_512", "format_only_512")),
                                            ("Token Effect", ("base_256", "base_512"))]:
        if c1_name in all_results and c2_name in all_results:
            r1 = all_results[c1_name]["results"]
            r2 = all_results[c2_name]["results"]
            n = min(len(r1), len(r2))
            b = sum(1 for i in range(n) if not r1[i]["ok"] and r2[i]["ok"])
            c = sum(1 for i in range(n) if r1[i]["ok"] and not r2[i]["ok"])
            if b + c > 0:
                chi2 = (abs(b - c) - 0.5) ** 2 / (b + c)
                p_val = 1 - stats.chi2.cdf(chi2, 1)
                print(f"  McNemar {pair_name}: b={b}, c={c}, chi2={chi2:.2f}, p={p_val:.4f}", flush=True)

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
