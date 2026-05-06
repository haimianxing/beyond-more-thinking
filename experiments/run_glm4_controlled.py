#!/usr/bin/env python3
"""Run controlled decomposition on GLM-4-9B (base) as 5th model family.

GLM-4 uses a different architecture (GLM, not decoder-only transformer),
providing cross-family validation. Using base model since Chat version
has transformers compatibility issues.
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Monkey-patch for GLM-4 compatibility with newer transformers
if not hasattr(nn.Module, 'all_tied_weights_keys'):
    @property
    def _all_tied_weights_keys(self):
        val = getattr(self, '_tied_weights_keys', None)
        return val if val is not None else {}
    nn.Module.all_tied_weights_keys = _all_tied_weights_keys

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

SEED = 42
N = 200
BASE = Path(__file__).parent
MODEL_PATH = "/path/to/models/glm-4-9b"
OUT = BASE / "results_glm4"
OUT.mkdir(parents=True, exist_ok=True)

def extract_ans(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums: return nums[-1]
    for pat in [r'(?:therefore|thus|the answer is)[:\s]+([^\n.]+)', r'answer[:\s]+([^\n.]+)']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            nums = re.findall(r'-?\d+\.?\d*', m.group(1))
            if nums: return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]

def check(p, g):
    p, g = p.strip().replace(',','').replace(' ',''), str(g).strip().replace(',','').replace(' ','')
    if p == g: return True
    try: return abs(float(p)-float(g)) < 1e-6
    except: return p.lower() == g.lower()

def run_condition(model, tok, questions, device, max_tok=256, use_cot=False):
    results = []
    for i, q in enumerate(questions):
        if use_cot:
            prompt = f"Question: {q['query']}\nLet's think step by step.\nAnswer:"
        else:
            prompt = f"Question: {q['query']}\nAnswer:"

        ids = tok.encode(prompt)
        inp = {
            'input_ids': torch.tensor([ids]).to(device),
            'attention_mask': torch.ones(1, len(ids), dtype=torch.long).to(device)
        }
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False,
                                 pad_token_id=tok.eos_token_id or tok.eos_token)
        full = tok.decode(out[0], skip_special_tokens=True)
        ans = extract_ans(full)
        gen_tok = out.shape[1] - len(ids)
        correct = check(ans, q['ground_truth'])
        truncated = (gen_tok >= max_tok - 2)
        results.append({
            'idx': i, 'pred': ans, 'gt': q['ground_truth'],
            'correct': correct, 'tokens': gen_tok, 'truncated': truncated
        })
        if (i+1) % 50 == 0:
            acc = sum(r['correct'] for r in results) / len(results) * 100
            trunc = sum(r['truncated'] for r in results) / len(results) * 100
            print(f"  [{i+1}/{N}] acc={acc:.1f}% trunc={trunc:.1f}%", flush=True)
    return results

def main():
    print(f"GLM-4-9B Controlled Decomposition | {time.strftime('%Y-%m-%d %H:%M')}", flush=True)

    # Load MATH dataset
    with open("/path/to/data/math_real_200.json") as f:
        alldata = json.load(f)
    random.seed(SEED)
    questions = random.sample(alldata, min(N, len(alldata)))
    print(f"Dataset: MATH, {len(questions)} questions", flush=True)

    # Load model
    device = "cuda:0"
    print(f"Loading GLM-4-9B...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Patch tokenizer _pad to accept padding_side kwarg (transformers compat)
    _orig_pad = tok._pad.__func__ if hasattr(tok._pad, '__func__') else tok._pad
    def _patched_pad(self, *args, **kwargs):
        kwargs.pop('padding_side', None)
        return _orig_pad(self, *args, **kwargs)
    import types
    tok._pad = types.MethodType(_patched_pad, tok)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Model init needs config.max_length; generation needs separate config
    if not hasattr(config, 'max_length'):
        config.max_length = getattr(config, 'seq_length', 2048)
    # Add num_hidden_layers for StaticCache compat (ChatGLM uses num_layers)
    if not hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = getattr(config, 'num_layers', 40)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        config=config,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()
    # Remove max_length from model config so generation doesn't complain
    if hasattr(model.config, 'max_length'):
        delattr(model.config, 'max_length')
    # Ensure num_hidden_layers is set on model config too
    if not hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = getattr(model.config, 'num_layers', 40)
    # Set generation config separately; disable static cache
    model.generation_config.max_length = 4096
    model.generation_config.cache_implementation = None
    print(f"Model loaded on {device}", flush=True)

    conditions = [
        ("Base-256", 256, False),
        ("Base-512", 512, False),
        ("CoT-512", 512, True),
    ]

    all_results = {}
    for cond_name, max_tok, use_cot in conditions:
        print(f"\n--- Running {cond_name} (max_tok={max_tok}, cot={use_cot}) ---", flush=True)
        results = run_condition(model, tok, questions, device, max_tok, use_cot)
        acc = sum(r['correct'] for r in results) / len(results) * 100
        trunc = sum(r['truncated'] for r in results) / len(results) * 100
        avg_tok = sum(r['tokens'] for r in results) / len(results)
        print(f"  RESULT: {cond_name} => acc={acc:.1f}% trunc={trunc:.1f}% avg_tok={avg_tok:.0f}", flush=True)
        all_results[cond_name] = {
            'accuracy': acc, 'truncation_rate': trunc, 'avg_tokens': avg_tok,
            'details': results
        }

        # Save checkpoint
        outf = OUT / f"GLM-4-9B_MATH_controlled.json"
        with open(outf, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved to {outf}", flush=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"GLM-4-9B on MATH ({N} questions):")
    for cond, data in all_results.items():
        print(f"  {cond}: {data['accuracy']:.1f}% acc, {data['truncation_rate']:.1f}% trunc, {data['avg_tokens']:.0f} avg tok")

    if 'Base-256' in all_results and 'Base-512' in all_results:
        token_delta = all_results['Base-512']['accuracy'] - all_results['Base-256']['accuracy']
        print(f"  Token Δ: {token_delta:+.1f}pp")
    if 'Base-512' in all_results and 'CoT-512' in all_results:
        cot_delta = all_results['CoT-512']['accuracy'] - all_results['Base-512']['accuracy']
        print(f"  CoT Δ: {cot_delta:+.1f}pp")
    print(f"{'='*60}")

if __name__ == "__main__":
    import random
    main()
