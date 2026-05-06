#!/usr/bin/env python3
"""Shared inference utilities for transformers and vLLM backends."""

import torch
import gc


def load_model_transformers(model_path, device="cuda:0", dtype=torch.float16):
    """Load model and tokenizer using HuggingFace transformers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    return model, tok


def generate_single(model, tok, query, suffix="", max_new_tokens=512,
                    device="cuda:0", do_sample=False, temperature=0.7):
    """Run single-question inference with chat template.

    Returns (generated_text, num_generated_tokens).
    """
    content = query + suffix
    messages = [{"role": "user", "content": content}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, temperature=temperature if do_sample else 1.0,
                             pad_token_id=tok.eos_token_id)

    gen_text = tok.decode(out[0], skip_special_tokens=True)
    if prompt in gen_text:
        gen_text = gen_text[len(prompt):]

    gen_tok = out.shape[1] - inp["input_ids"].shape[1]
    return gen_text, gen_tok


def cleanup_model(model):
    """Delete model and free GPU memory."""
    del model
    torch.cuda.empty_cache()
    gc.collect()
