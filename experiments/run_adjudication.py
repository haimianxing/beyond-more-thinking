#!/usr/bin/env python3
"""Adjudication Experiment -- regenerate raw outputs for semantic analysis.

Only runs 50 questions (sample) for baseline and format_only on Qwen2.5-3B.
Saves full raw outputs for manual adjudication.

Usage:
    python3 -u run_adjudication.py \
        --model_path /path/to/Qwen2.5-3B-Instruct \
        --data_file /path/to/math_real_200.json \
        --output_dir ./results_adjudication \
        --base_results /path/to/qwen25_3b_MATH_baseline_s42.json \
        --fmt_results /path/to/qwen25_3b_MATH_format_only_s42.json \
        --gpu 4
"""
import json, re, random, os, sys, time, gc, argparse
import torch, numpy as np

def main():
    parser = argparse.ArgumentParser(description="Adjudication Experiment")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--data_file", required=True, help="Path to MATH questions JSON")
    parser.add_argument("--output_dir", required=True, help="Directory for results")
    parser.add_argument("--base_results", required=True, help="Path to baseline results JSON")
    parser.add_argument("--fmt_results", required=True, help="Path to format_only results JSON")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    DEVICE = "cuda:0"

    MODEL_PATH = args.model_path
    DATA_FILE = args.data_file
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    N_SAMPLE = 50
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load questions
    with open(DATA_FILE) as f:
        questions = json.load(f)

    # Load existing results to find discordant pairs
    base_results = json.load(open(args.base_results))
    fmt_results = json.load(open(args.fmt_results))

    # Find discordant: format correct but baseline wrong
    discordant = []
    both_wrong = []
    for i in range(len(base_results['results'])):
        b = base_results['results'][i]
        f = fmt_results['results'][i]
        if f['ok'] and not b['ok']:
            discordant.append(i)
        elif not f['ok'] and not b['ok']:
            both_wrong.append(i)

    # Also sample some where both correct and both wrong for balance
    both_correct = [i for i in range(len(base_results['results']))
                    if base_results['results'][i]['ok'] and fmt_results['results'][i]['ok']]

    print(f"Discordant (fmt ok, base wrong): {len(discordant)}")
    print(f"Both wrong: {len(both_wrong)}")
    print(f"Both correct: {len(both_correct)}")

    # Prioritize discordant cases
    sample_indices = []
    sample_indices.extend(discordant)
    random.shuffle(both_wrong)
    sample_indices.extend(both_wrong[:min(15, len(both_wrong))])
    random.shuffle(both_correct)
    sample_indices.extend(both_correct[:min(10, len(both_correct))])
    sample_indices = sorted(set(sample_indices))[:N_SAMPLE]

    print(f"\nSampled {len(sample_indices)} questions for adjudication")
    print(f"  Discordant: {len([i for i in sample_indices if i in discordant])}")
    print(f"  Both wrong: {len([i for i in sample_indices if i in both_wrong and i not in discordant])}")
    print(f"  Both correct: {len([i for i in sample_indices if i in both_correct])}")

    # Define prompts
    BASELINE_SUFFIX = ""
    FORMAT_SUFFIX = "\nSolve this. Put answer in \\boxed{}."

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("\nLoading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print("Model loaded.")

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

    def gt_in_output(text, gt):
        """Check if GT number appears anywhere in output text."""
        gt_clean = str(gt).strip().replace(',', '').replace(' ', '')
        gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
        if not gt_nums:
            return False, None
        target = gt_nums[-1]
        output_nums = re.findall(r'-?\d+\.?\d*', text)
        for num in output_nums:
            try:
                if abs(float(num) - float(target)) < 1e-6:
                    return True, num
            except:
                pass
        return False, None

    # Run inference
    results = {}
    for cond_name, suffix in [("baseline", BASELINE_SUFFIX), ("format_only", FORMAT_SUFFIX)]:
        print(f"\n{'='*60}")
        print(f"Running: {cond_name} on {len(sample_indices)} questions")
        print(f"{'='*60}")

        cond_results = []
        for idx, qi in enumerate(sample_indices):
            q = questions[qi]
            gt = str(q.get('ground_truth', q.get('answer', '')))
            query = q['query']

            content = query + suffix
            messages = [{"role": "user", "content": content}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=512, do_sample=False,
                                     pad_token_id=tok.eos_token_id)

            gen_text = tok.decode(out[0], skip_special_tokens=True)
            if prompt in gen_text:
                gen_text = gen_text[len(prompt):]

            gen_tok = out.shape[1] - inp["input_ids"].shape[1]

            ans = extract_ans(gen_text)
            regex_ok = check(ans, gt)
            semantic_ok, semantic_val = gt_in_output(gen_text, gt)

            r = {
                "q_idx": qi,
                "gt": gt,
                "gen_text": gen_text,
                "gen_tok": gen_tok,
                "regex_ans": ans,
                "regex_ok": regex_ok,
                "semantic_ok": semantic_ok,
                "semantic_val": semantic_val,
                "extraction_artifact": semantic_ok and not regex_ok,
            }
            cond_results.append(r)

            if (idx + 1) % 10 == 0:
                print(f"  {idx+1}/{len(sample_indices)} done")

        results[cond_name] = cond_results

        # Quick stats
        regex_acc = sum(r['regex_ok'] for r in cond_results) / len(cond_results) * 100
        semantic_acc = sum(r['semantic_ok'] for r in cond_results) / len(cond_results) * 100
        extraction_artifacts = sum(r['extraction_artifact'] for r in cond_results)
        print(f"\n  Regex accuracy: {regex_acc:.1f}%")
        print(f"  Semantic accuracy: {semantic_acc:.1f}%")
        print(f"  Extraction artifacts (semantic ok, regex wrong): {extraction_artifacts}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ========================================
    # Comparative Analysis
    # ========================================
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*60}")

    base = results['baseline']
    fmt = results['format_only']

    base_by_q = {r['q_idx']: r for r in base}
    fmt_by_q = {r['q_idx']: r for r in fmt}

    both_regex_correct = 0
    fmt_only_regex = 0
    base_only_regex = 0
    both_regex_wrong = 0

    both_semantic_correct = 0
    fmt_only_semantic = 0
    base_only_semantic = 0

    extraction_gain_fmt = 0
    extraction_gain_base = 0

    for qi in sample_indices:
        b = base_by_q[qi]
        f = fmt_by_q[qi]

        if b['regex_ok'] and f['regex_ok']: both_regex_correct += 1
        elif f['regex_ok'] and not b['regex_ok']: fmt_only_regex += 1
        elif b['regex_ok'] and not f['regex_ok']: base_only_regex += 1
        else: both_regex_wrong += 1

        if b['semantic_ok'] and f['semantic_ok']: both_semantic_correct += 1
        elif f['semantic_ok'] and not b['semantic_ok']: fmt_only_semantic += 1
        elif b['semantic_ok'] and not f['semantic_ok']: base_only_semantic += 1

        if f['regex_ok'] and not b['regex_ok']:
            if b['semantic_ok']:
                extraction_gain_base += 1

    print(f"\nRegex accuracy (N={len(sample_indices)}):")
    print(f"  Baseline:    {sum(r['regex_ok'] for r in base)}/{len(base)} = {sum(r['regex_ok'] for r in base)/len(base)*100:.1f}%")
    print(f"  Format-only: {sum(r['regex_ok'] for r in fmt)}/{len(fmt)} = {sum(r['regex_ok'] for r in fmt)/len(fmt)*100:.1f}%")
    print(f"  Delta (regex): {(sum(r['regex_ok'] for r in fmt) - sum(r['regex_ok'] for r in base))/len(base)*100:+.1f}pp")

    print(f"\nSemantic accuracy (GT number present in output):")
    print(f"  Baseline:    {sum(r['semantic_ok'] for r in base)}/{len(base)} = {sum(r['semantic_ok'] for r in base)/len(base)*100:.1f}%")
    print(f"  Format-only: {sum(r['semantic_ok'] for r in fmt)}/{len(fmt)} = {sum(r['semantic_ok'] for r in fmt)/len(fmt)*100:.1f}%")
    print(f"  Delta (semantic): {(sum(r['semantic_ok'] for r in fmt) - sum(r['semantic_ok'] for r in base))/len(base)*100:+.1f}pp")

    print(f"\nBreakdown:")
    print(f"  Both regex correct: {both_regex_correct}")
    print(f"  Format-only regex correct: {fmt_only_regex}")
    print(f"  Baseline-only regex correct: {base_only_regex}")
    print(f"  Both regex wrong: {both_regex_wrong}")

    print(f"\nExtraction artifact analysis:")
    print(f"  Baseline has GT in output but regex missed: {extraction_gain_base}/{fmt_only_regex} fmt-only-correct cases")
    print(f"  (If high: format improvement is partly extraction)")
    print(f"  (If low: format genuinely improves reasoning)")

    # Show examples of fmt_only_regex cases
    print(f"\n{'='*60}")
    print("DETAILED: Format-only correct, Baseline wrong cases")
    print(f"{'='*60}")
    for qi in sample_indices:
        b = base_by_q[qi]
        f = fmt_by_q[qi]
        if f['regex_ok'] and not b['regex_ok']:
            print(f"\nQ#{qi} | GT={b['gt']}")
            print(f"  BASE regex: {b['regex_ans']} (ok={b['regex_ok']}, semantic={b['semantic_ok']})")
            print(f"  FMT  regex: {f['regex_ans']} (ok={f['regex_ok']}, semantic={f['semantic_ok']})")
            if b['semantic_ok']:
                print(f"  ** BASELINE EXTRACTION ARTIFACT: GT {b['gt']} was in output but not extracted")
            print(f"  BASE output (last 200 chars): ...{b['gen_text'][-200:]}")
            print(f"  FMT  output (last 200 chars): ...{f['gen_text'][-200:]}")

    # Save full results
    with open(os.path.join(OUT_DIR, 'adjudication_results.json'), 'w') as f:
        json.dump({
            "sample_indices": sample_indices,
            "n_discordant": len([i for i in sample_indices if i in discordant]),
            "baseline": base,
            "format_only": fmt,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUT_DIR}/adjudication_results.json")

    # ========================================
    # KEY CONCLUSION
    # ========================================
    regex_delta = (sum(r['regex_ok'] for r in fmt) - sum(r['regex_ok'] for r in base))
    semantic_delta = (sum(r['semantic_ok'] for r in fmt) - sum(r['semantic_ok'] for r in base))
    base_extraction_fail = sum(1 for r in base if r['semantic_ok'] and not r['regex_ok'])
    fmt_extraction_fail = sum(1 for r in fmt if r['semantic_ok'] and not r['regex_ok'])

    print(f"\n{'='*60}")
    print("KEY CONCLUSION")
    print(f"{'='*60}")
    print(f"Regex delta:    {regex_delta:+d} questions ({regex_delta/len(base)*100:+.1f}pp)")
    print(f"Semantic delta: {semantic_delta:+d} questions ({semantic_delta/len(base)*100:+.1f}pp)")
    print(f"\nBaseline extraction failures (GT in output, regex missed): {base_extraction_fail}")
    print(f"Format extraction failures: {fmt_extraction_fail}")
    print(f"\nIf semantic_delta approx= regex_delta -> format genuinely improves reasoning")
    print(f"If semantic_delta < regex_delta -> part of gain is extraction artifact")
    print(f"If semantic_delta approx= 0 -> entire gain is extraction artifact (bad)")


if __name__ == "__main__":
    main()
