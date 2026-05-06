# Beyond More Thinking: When Format Instruction Dominates CoT Effects in Specific Model Families

Anonymous submission to NeurIPS 2026.

## Overview

This repository contains code for reproducing all experiments from the paper.
The paper identifies "strategic overthinking" in test-time compute scaling across
multiple model families and scales.

Key findings:
1. **Truncation Paradox**: Standard 256-token budgets hide 32-37pp of accuracy
2. **CoT Reversal**: Generic CoT hurts Qwen2.5/LLaMA families (-4.5 to -7.5pp)
3. **Format Dominance**: Format instruction, not reasoning scaffolding, drives
   prompt gains for negative-CoT families

## Setup

```bash
pip install -r requirements.txt
```

## Model Access

Download models from HuggingFace:

| Model | Link |
|-------|------|
| Qwen2.5-0.5B/1.5B/3B/7B-Instruct | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct (and other sizes) |
| Qwen2.5-0.5B/1.5B/3B/7B-Base | https://huggingface.co/Qwen/Qwen2.5-3B (and other sizes) |
| Qwen2-1.5B/7B-Instruct | https://huggingface.co/Qwen/Qwen2-1.5B-Instruct |
| Qwen3-4B/8B | https://huggingface.co/Qwen/Qwen3-4B |
| Llama-3-8B-Instruct | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct |
| Gemma-2-9B-IT | https://huggingface.co/google/gemma-2-9b-it |
| Gemma-2-2B-IT | https://huggingface.co/google/gemma-2-2b-it |

## Data

| Dataset | Source |
|---------|--------|
| MATH | https://github.com/hendrycks/math |
| GSM8K | https://github.com/openai/grade-school-math |
| MMLU | https://github.com/hendrycks/test |

Evaluation subsets (200 questions each) are included in `results/` directories.

## Repository Structure

```
src/              — Shared utilities (evaluation, prompts, inference)
experiments/      — Experiment runner scripts
analysis/         — Statistical analysis and results aggregation
evaluation/       — SAC review and answer validation
data/             — Data preparation scripts (MMLU subset)
configs/          — Path configuration template
results/          — Pre-computed experiment results
```

## Running Experiments

Each experiment script accepts command-line arguments for paths:

```bash
# Example: 8-prompt decomposition
python experiments/run_8prompt_decomposition.py \
    --model_path /path/to/Qwen2.5-3B-Instruct \
    --data_file /path/to/math_real_200.json \
    --output_dir ./results/prompt_8way \
    --gpu 0

# Example: Format-only cross-family
python experiments/run_format_only_cross_family.py \
    --model_path /path/to/Qwen2.5-3B-Instruct \
    --data_file /path/to/math_real_200.json \
    --output_dir ./results/format_cross_family \
    --gpu 0
```

Run `python <script> --help` for full argument list.

## Experiment-to-Paper Mapping

| Script | Paper Section | Key Result |
|--------|---------------|------------|
| `run_8prompt_decomposition.py` | §4.2 Format vs Reasoning | 24pp prompt range, format dominance |
| `run_8prompt_multi.py` | §4.2 Multi-model | 6-model cross-family validation |
| `run_adjudication.py` | §4.2 Extraction Artifact | 63% extraction vs 37% genuine |
| `run_format_only_cross_family.py` | §5.3 Cross-Family | Family-dependent format effects |
| `run_format_base_vllm.py` | §5.4 Base Models | Instruction-tuning specificity |
| `run_extended_n321.py` | §5.5 Extended Validation | N=321 power analysis |
| `run_baseline_512_verification.py` | §3 Truncation Paradox | 256 vs 512 token budgets |
| `run_probing.py` | §5.7 Mechanistic | Representational disruption |
| `run_cross_family_1024.py` | §5.6 Cross-Family | Extended token budget validation |

## Pre-computed Results

The `results_*/` directories contain all experiment outputs (JSON format) for
verification. Each JSON file includes per-question predictions, correctness
flags, and token counts.

## Citation

```bibtex
@inproceedings{anonymous2026beyond,
  title={Beyond More Thinking: When Format Instruction Dominates CoT Effects
         in Specific Model Families},
  author={Anonymous Authors},
  booktitle={NeurIPS 2026},
  year={2026}
}
```
