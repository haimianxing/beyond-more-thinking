#!/usr/bin/env python3
"""8-prompt templates for controlled decomposition experiments.

Each prompt decomposes the CoT effect into orthogonal factors:
  - Format instruction: "Put answer in \boxed{}"
  - Reasoning scaffolding: "Think step by step"
  - Length control: neutral instructions of varying verbosity
  - Few-shot exemplars: 3-shot math CoT
"""

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
    "baseline": {
        "prefix": "",
        "suffix": "",
    },
    "original_cot": {
        "prefix": "",
        "suffix": "\nLet's think step by step.",
    },
    "fewshot_cot": {
        "prefix": FEWSHOT_EXEMPLARS,
        "suffix": "\nShow your reasoning step by step, then give the final answer in \\boxed{}.",
    },
    "structured": {
        "prefix": "",
        "suffix": "\nSolve step by step. Show your work clearly. Put the final answer in \\boxed{}.",
    },
    "neutral_len": {
        "prefix": "",
        "suffix": "\nPlease solve this problem.",
    },
    "verbose_neutral": {
        "prefix": "",
        "suffix": "\nPlease carefully consider this problem and provide your answer.",
    },
    "format_only": {
        "prefix": "",
        "suffix": "\nSolve this. Put answer in \\boxed{}.",
    },
    "cot_only": {
        "prefix": "",
        "suffix": "\nThink through this step by step.",
    },
}

# Anti-format control prompt (used in Qwen3 experiments)
ANTI_FORMAT = {
    "prefix": "",
    "suffix": "\nDo NOT use \\boxed{} or any special formatting. Just write the final number plainly.",
}
