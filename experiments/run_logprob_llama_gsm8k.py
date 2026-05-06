#!/usr/bin/env python3
"""
LOGPROB CONFIDENCE — LLaMA-3-8B GSM8K (cross-family validation for Innovation 2)
Validates token > logprob finding on a DIFFERENT model family (LLaMA).
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
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

def make_input(tok, question):
    content = f"{question}\n\nPlease think step by step, then give your final numerical answer after ####."
    messages = [{"role": "user", "content": content}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")

def main():
    device = "cuda:0"
    print(f"Logprob LLaMA-3-8B GSM8K | {datetime.now()}", flush=True)

    model_path = "/path/to/models/Llama-3-8B-Instruct"
    data_file = BASE / "gsm8k_real_200.json"
    ckpt_dir = BASE / "results_logprob"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"Loading LLaMA-3-8B from {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"Model loaded, {sum(p.numel() for p in model.parameters())/1e9:.2f}B params", flush=True)

    with open(data_file) as f:
        alldata = json.load(f)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data = alldata[:N_SAMPLES]

    ckpt_file = ckpt_dir / "logprob_confidence_gsm8k_llama.json"
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

                gen_tokens = out.sequences.shape[1] - inp["input_ids"].shape[1]
                response = tok.decode(out.sequences[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                ans = extract_ans(response, "GSM8K")
                ok = check(ans, gt)

                if len(out.scores) > 0:
                    log_probs = []
                    for t_idx, score in enumerate(out.scores):
                        if t_idx < gen_tokens:
                            token_id = out.sequences[0][inp["input_ids"].shape[1] + t_idx]
                            log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                            log_probs.append(log_prob)
                    mean_logprob = np.mean(log_probs) if log_probs else 0
                    min_logprob = np.min(log_probs) if log_probs else 0
                    perplexity = np.exp(-mean_logprob) if mean_logprob != 0 else float('inf')
                    last_logprob = log_probs[-1] if log_probs else 0
                else:
                    mean_logprob = min_logprob = perplexity = last_logprob = 0

                results.append({
                    "q": i, "ok": ok, "tok": gen_tokens,
                    "pred": ans, "gt": gt,
                    "mean_logprob": mean_logprob, "min_logprob": min_logprob,
                    "perplexity": perplexity, "last_logprob": last_logprob,
                })

            except Exception as e:
                results.append({
                    "q": i, "ok": False, "tok": 0, "pred": "", "gt": gt, "err": str(e),
                    "mean_logprob": 0, "min_logprob": 0, "perplexity": float('inf'), "last_logprob": 0,
                })

            if (i+1) % 20 == 0 or (i+1) == N_SAMPLES:
                done = (i+1) == N_SAMPLES
                with open(ckpt_file, 'w') as f:
                    json.dump({"metadata": {"model": "LLaMA-3-8B", "ds": "GSM8K",
                                           "method": "logprob_confidence", "seed": SEED,
                                           "done": done, "n": i+1}, "results": results}, f)
                acc = sum(1 for r in results if r.get('ok')) / len(results) * 100
                print(f"  [{i+1}/{N_SAMPLES}] acc={acc:.1f}%", flush=True)

    # Analysis
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pointbiserialr, mannwhitneyu, spearmanr

    labels = [0 if r['ok'] else 1 for r in results]
    tokens = [r['tok'] for r in results]
    mean_lps = [r['mean_logprob'] for r in results]
    last_lps = [r['last_logprob'] for r in results]

    print(f"\n{'='*60}")
    print(f"TOKEN vs LOGPROB (LLaMA-3-8B GSM8K@256)")
    print(f"{'='*60}")

    if len(set(labels)) > 1:
        auc_tok = roc_auc_score(labels, tokens)
        r_tok, p_tok = pointbiserialr([1-r['ok'] for r in results], tokens)
        u_tok, p_mw_tok = mannwhitneyu([r['tok'] for r in results if not r['ok']],
                                        [r['tok'] for r in results if r['ok']], alternative='greater')

        auc_lp = roc_auc_score(labels, [-lp for lp in mean_lps])
        r_lp, p_lp = pointbiserialr([1-r['ok'] for r in results], mean_lps)
        u_lp, p_mw_lp = mannwhitneyu([-r['mean_logprob'] for r in results if not r['ok']],
                                      [-r['mean_logprob'] for r in results if r['ok']], alternative='greater')

        auc_last = roc_auc_score(labels, [-lp for lp in last_lps])
        r_sp, p_sp = spearmanr(tokens, mean_lps)

        tok_c = [r['tok'] for r in results if r['ok']]
        tok_w = [r['tok'] for r in results if not r['ok']]
        lp_c = [r['mean_logprob'] for r in results if r['ok']]
        lp_w = [r['mean_logprob'] for r in results if not r['ok']]

        def cohens_d(a, b):
            ps = np.sqrt((np.var(a) + np.var(b)) / 2)
            return (np.mean(b) - np.mean(a)) / ps if ps > 0 else 0

        print(f"""
  Metric              AUC     Cohen's d  Mann-Whitney p
  ──────────────────────────────────────────────────────────────
  Token-length        {auc_tok:.3f}   {cohens_d(tok_c, tok_w):+.2f}       {p_mw_tok:.2e}
  Mean logprob        {auc_lp:.3f}   {cohens_d(lp_c, lp_w):+.2f}       {p_mw_lp:.2e}
  Last token logprob  {auc_last:.3f}

  Token/Logprob AUC ratio: {auc_tok/max(auc_lp,0.01):.2f}x
  Signal independence (Spearman r): {r_sp:.3f}
""")

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"Done: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
