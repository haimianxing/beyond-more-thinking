#!/usr/bin/env python3
"""
LOGPROB CONFIDENCE COMPARISON for Innovation 2
================================================
Compare token-length AUC vs logprob-based AUC to show that
token-length is competitive without needing logprob access.

This addresses CW5 from the SAC review.
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
N_SAMPLES = 200
BASE = Path(__file__).parent

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

def make_input(tok, question):
    content = f"{question}\n\nPlease think step by step, then give your final numerical answer after ####."
    messages = [{"role": "user", "content": content}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")

def main():
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES maps to physical GPU
    print(f"Logprob Confidence Comparison | {datetime.now()}", flush=True)

    model_path = "/path/to/models/Qwen2.5-7B-Instruct"
    data_file = BASE / "math_real_200.json"
    ckpt_dir = BASE / "results_logprob"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"Loading Qwen2.5-7B from {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    with open(data_file) as f:
        alldata = json.load(f)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data = alldata[:N_SAMPLES]

    ckpt_file = ckpt_dir / "logprob_confidence_math_7b.json"
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            existing = json.load(f)
        if existing.get("metadata", {}).get("done"):
            print("Already done, loading results...", flush=True)
            results = existing["results"]
        else:
            results = existing.get("results", [])
            start = len(results)
    else:
        results = []
        start = 0

    if start < N_SAMPLES:
        for i in range(start, N_SAMPLES):
            q = data[i]["query"]
            gt = str(data[i].get("ground_truth", ""))

            try:
                inp = make_input(tok, q)
                inp = inp.to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inp,
                        max_new_tokens=256,
                        do_sample=False,
                        temperature=None,
                        pad_token_id=tok.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                # Get generated token count
                gen_tokens = out.sequences.shape[1] - inp["input_ids"].shape[1]
                response = tok.decode(out.sequences[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

                # Extract answer
                ans = extract_ans(response, "MATH")
                ok = check(ans, gt)

                # Compute logprob-based confidence
                # Method 1: Mean logprob of generated tokens
                if len(out.scores) > 0:
                    scores_tensor = torch.stack(out.scores, dim=0)  # [seq_len, vocab]
                    log_probs = []
                    for t_idx, score in enumerate(out.scores):
                        if t_idx < gen_tokens and t_idx < out.sequences.shape[1] - inp["input_ids"].shape[1]:
                            token_id = out.sequences[0][inp["input_ids"].shape[1] + t_idx]
                            log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                            log_probs.append(log_prob)

                    mean_logprob = np.mean(log_probs) if log_probs else 0
                    min_logprob = np.min(log_probs) if log_probs else 0
                    # PPL-like: exp(-mean_logprob)
                    perplexity = np.exp(-mean_logprob) if mean_logprob != 0 else float('inf')
                    # Last token logprob (answer confidence)
                    last_logprob = log_probs[-1] if log_probs else 0
                else:
                    mean_logprob = 0
                    min_logprob = 0
                    perplexity = float('inf')
                    last_logprob = 0

                results.append({
                    "q": i, "ok": ok, "tok": gen_tokens,
                    "pred": ans, "gt": gt,
                    "mean_logprob": mean_logprob,
                    "min_logprob": min_logprob,
                    "perplexity": perplexity,
                    "last_logprob": last_logprob,
                })

            except Exception as e:
                results.append({
                    "q": i, "ok": False, "tok": 0,
                    "pred": "", "gt": gt, "err": str(e),
                    "mean_logprob": 0, "min_logprob": 0,
                    "perplexity": float('inf'), "last_logprob": 0,
                })

            if (i+1) % 20 == 0 or (i+1) == N_SAMPLES:
                done = (i+1) == N_SAMPLES
                with open(ckpt_file, 'w') as f:
                    json.dump({"metadata": {"model": "Qwen2.5-7B", "ds": "MATH",
                                           "method": "logprob_confidence", "seed": SEED,
                                           "done": done, "n": i+1}, "results": results}, f)
                acc = sum(1 for r in results if r.get("ok")) / len(results) * 100
                print(f"  logprob [{i+1}/{N_SAMPLES}] acc={acc:.1f}%", flush=True)

    # Analyze results
    from sklearn.metrics import roc_auc_score
    from scipy import stats as sp_stats

    labels = [0 if r['ok'] else 1 for r in results]
    tokens = [r['tok'] for r in results]
    mean_lps = [r['mean_logprob'] for r in results]
    last_lps = [r['last_logprob'] for r in results]
    ppls = [min(r['perplexity'], 1000) for r in results]  # cap at 1000

    print(f"\n{'='*60}")
    print(f"LOGPROB vs TOKEN-LENGTH CONFIDENCE COMPARISON")
    print(f"{'='*60}")

    if len(set(labels)) > 1:
        # Token-length AUC
        auc_tok = roc_auc_score(labels, tokens)
        r_tok, p_tok = sp_stats.pointbiserialr([1-r['ok'] for r in results], tokens)

        # Mean logprob AUC (lower logprob = more likely wrong)
        auc_lp = roc_auc_score(labels, [-lp for lp in mean_lps])  # negate so higher = more likely wrong
        r_lp, p_lp = sp_stats.pointbiserialr([1-r['ok'] for r in results], mean_lps)

        # Last token logprob AUC
        auc_last = roc_auc_score(labels, [-lp for lp in last_lps])

        # Perplexity AUC
        auc_ppl = roc_auc_score(labels, ppls)

        print(f"""
  Method                    AUC     Comparison
  ─────────────────────────────────────────────────
  Token-length (free)       {auc_tok:.3f}   ← Innovation 2
  Mean logprob              {auc_lp:.3f}   (requires logprob access)
  Last token logprob        {auc_last:.3f}   (requires logprob access)
  Perplexity                {auc_ppl:.3f}   (requires logprob access)

  Token-length AUC / Mean logprob AUC = {auc_tok/auc_lp:.2f}x
  Token-length captures {auc_tok/auc_lp*100:.1f}% of logprob signal

  Point-biserial:
    Token-length: r={r_tok:.3f}, p={p_tok:.2e}
    Mean logprob:  r={r_lp:.3f}, p={p_lp:.2e}
""")

    # Cleanup
    del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
