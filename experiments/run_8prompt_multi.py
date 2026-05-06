#!/usr/bin/env python3
"""8-Prompt Decomposition — Multi-Model vLLM/HF Script.

Usage:
    python3 -u run_8prompt_multi.py \
        --model_path /path/to/model \
        --model_name "LLaMA-3-8B" \
        --model_tag "llama" \
        --data_file /path/to/math_real_200.json \
        --output_dir ./results_prompt_8way_llama \
        --engine hf \
        --gpu 4

    # With anti-format condition:
    python3 -u run_8prompt_multi.py \
        --model_path /path/to/model \
        --model_name "Qwen3-4B" \
        --model_tag "qwen3_4b" \
        --data_file /path/to/math_real_200.json \
        --output_dir ./results_prompt_8way_qwen3 \
        --engine hf \
        --gpu 6 \
        --anti-format
"""
import sys, os, json, time, gc, random, re, warnings, argparse
import torch, numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
N_QUESTIONS = 200
SEEDS = [42]

FEWSHOT_EXEMPLARS = """Problem: Find the value of $x$ such that $2x + 5 = 13$.
Solution: Subtract 5 from both sides: $2x = 8$. Divide by 2: $x = 4$.
Answer: $\\boxed{4}$

Problem: If $a^2 - 3a + 2 = 0$, find all values of $a$.
Solution: Factor: $(a-1)(a-2) = 0$. So $a = 1$ or $a = 2$.
Answer: $\\boxed{1, 2}$

Problem: What is the sum of the first 10 positive integers?
Solution: Using the formula $S = n(n+1)/2$: $S = 10(11)/2 = 55$.
Answer: $\\boxed{55}$

"""

PROMPTS = {
    "baseline": {"prefix": "", "suffix": ""},
    "original_cot": {"prefix": "", "suffix": "\nLet's think step by step."},
    "fewshot_cot": {"prefix": FEWSHOT_EXEMPLARS, "suffix": "\nShow your reasoning step by step, then give the final answer in \\boxed{}."},
    "structured": {"prefix": "", "suffix": "\nSolve step by step. Show your work clearly. Put the final answer in \\boxed{}."},
    "neutral_len": {"prefix": "", "suffix": "\nPlease solve this problem."},
    "verbose_neutral": {"prefix": "", "suffix": "\nPlease carefully consider this problem and provide your answer."},
    "format_only": {"prefix": "", "suffix": "\nSolve this. Put answer in \\boxed{}."},
    "cot_only": {"prefix": "", "suffix": "\nThink through this step by step."},
    "anti_format": {"prefix": "", "suffix": "\nDo not use any special formatting, lists, or structured output. Just write your answer plainly."},
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


def run_hf(model_path, model_name, model_tag, prompts, data_file, out_dir, device="cuda:0"):
    """Run using HuggingFace transformers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    with open(data_file) as f:
        questions = json.load(f)[:N_QUESTIONS]

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"Model loaded: {model_name} on {device}", flush=True)

    all_results = {}
    tag = model_tag.lower().replace("-", "").replace(".", "").replace(" ", "")

    for pname, pconfig in prompts.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Running: {pname}", flush=True)

        for seed in SEEDS:
            ckpt = out_dir / f"{tag}_MATH_{pname}_s{seed}.json"
            if ckpt.exists():
                print(f"  SKIP {pname} s{seed} (cached)", flush=True)
                all_results[pname] = json.load(open(ckpt))
                continue

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            results = []
            for i in range(N_QUESTIONS):
                q = questions[i]["query"]
                gt = str(questions[i].get("ground_truth", questions[i].get("answer", "")))

                content = pconfig["prefix"] + q + pconfig["suffix"]
                messages = [{"role": "user", "content": content}]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inp = tok(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=512, do_sample=False,
                                         pad_token_id=tok.eos_token_id)

                gen_text = tok.decode(out[0], skip_special_tokens=True)
                if prompt in gen_text:
                    gen_text = gen_text[len(prompt):]

                gen_tok = out.shape[1] - inp["input_ids"].shape[1]
                truncated = gen_tok >= 507

                ans = extract_ans(gen_text)
                ok = check(ans, gt)

                results.append({"q": i, "ok": ok, "tok": gen_tok,
                                "truncated": truncated, "ans": ans, "gt": gt})

                if (i + 1) % 50 == 0:
                    acc = sum(r["ok"] for r in results) / len(results) * 100
                    print(f"    {pname}: {i+1}/{N_QUESTIONS} acc={acc:.1f}%", flush=True)

            data = {"metadata": {"model": model_name, "method": pname,
                                 "seed": seed, "n": N_QUESTIONS},
                    "results": results}
            with open(ckpt, 'w') as f:
                json.dump(data, f)

            acc = sum(r["ok"] for r in results) / len(results) * 100
            all_results[pname] = data
            print(f"  {pname}: acc={acc:.1f}%", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return all_results


def run_vllm(model_path, model_name, model_tag, prompts, data_file, out_dir):
    """Run using vLLM for faster batched inference."""
    from vllm import LLM, SamplingParams

    with open(data_file) as f:
        questions = json.load(f)[:N_QUESTIONS]

    llm = LLM(model=model_path, tensor_parallel_size=1,
              gpu_memory_utilization=0.9, max_model_len=1024,
              trust_remote_code=True, dtype="float16")
    print(f"vLLM loaded: {model_name}", flush=True)

    sampling = SamplingParams(temperature=0, max_tokens=512)
    all_results = {}
    tag = model_tag.lower().replace("-", "").replace(".", "").replace(" ", "")

    for pname, pconfig in prompts.items():
        ckpt = out_dir / f"{tag}_MATH_{pname}_s{42}.json"
        if ckpt.exists():
            print(f"  SKIP {pname} (cached)", flush=True)
            all_results[pname] = json.load(open(ckpt))
            continue

        print(f"\nRunning: {pname}", flush=True)
        # Build all prompts at once
        batch_prompts = []
        for i in range(N_QUESTIONS):
            q = questions[i]["query"]
            content = pconfig["prefix"] + q + pconfig["suffix"]
            messages = [{"role": "user", "content": content}]
            prompt_text = llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(prompt_text)

        # Batch generate
        outputs = llm.generate(batch_prompts, sampling)

        results = []
        for i, output in enumerate(outputs):
            gt = str(questions[i].get("ground_truth", questions[i].get("answer", "")))
            gen_text = output.outputs[0].text
            gen_tok = len(output.outputs[0].token_ids)
            truncated = gen_tok >= 507

            ans = extract_ans(gen_text)
            ok = check(ans, gt)
            results.append({"q": i, "ok": ok, "tok": gen_tok,
                            "truncated": truncated, "ans": ans, "gt": gt})

        data = {"metadata": {"model": model_name, "method": pname,
                             "seed": 42, "n": N_QUESTIONS},
                "results": results}
        with open(ckpt, 'w') as f:
            json.dump(data, f)

        acc = sum(r["ok"] for r in results) / len(results) * 100
        all_results[pname] = data
        print(f"  {pname}: acc={acc:.1f}%", flush=True)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="8-Prompt Decomposition Multi-Model")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--model_name", required=True, help="Human-readable model name")
    parser.add_argument("--model_tag", required=True, help="Short tag for output files")
    parser.add_argument("--data_file", required=True, help="Path to MATH questions JSON")
    parser.add_argument("--output_dir", required=True, help="Directory for results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--engine", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--anti_format", action="store_true", help="Only run anti-format condition")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select prompts
    if args.anti_format:
        prompts = {k: v for k, v in PROMPTS.items() if k == "anti_format"}
        print(f"Anti-format only mode", flush=True)
    else:
        prompts = PROMPTS

    print(f"8-Prompt Decomposition | {args.model_name} | {args.engine} | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    if args.engine == "vllm":
        all_results = run_vllm(args.model_path, args.model_name, args.model_tag,
                               prompts, args.data_file, out_dir)
    else:
        all_results = run_hf(args.model_path, args.model_name, args.model_tag,
                             prompts, args.data_file, out_dir)

    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY: {args.model_name}", flush=True)
    print(f"{'='*80}", flush=True)

    base_acc = None
    if "baseline" in all_results:
        base_res = all_results["baseline"]["results"]
        base_acc = sum(r["ok"] for r in base_res) / len(base_res) * 100
        print(f"\nBaseline: {base_acc:.1f}%", flush=True)

    for pname, data in all_results.items():
        if pname == "baseline": continue
        res = data["results"]
        acc = sum(r["ok"] for r in res) / len(res) * 100
        delta = acc - base_acc if base_acc else 0
        avg_tok = np.mean([r["tok"] for r in res])
        print(f"  {pname:20s}: acc={acc:.1f}% (delta={delta:+.1f}pp) avg_tok={avg_tok:.0f}", flush=True)

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
