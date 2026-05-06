#!/usr/bin/env python3
"""Prepare MMLU STEM subset for cross-domain TTS evaluation.

Downloads MMLU from HuggingFace and creates 200-question STEM subset.
"""
import json, random
from pathlib import Path

BASE = Path(__file__).parent

# STEM subjects that require reasoning
SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_physics",
    "computer_security", "conceptual_physics", "electrical_engineering",
    "elementary_mathematics", "high_school_chemistry", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
    "medical_genetics", "virology"
]

def load_mmlu():
    """Load MMLU from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        return ds
    except Exception as e:
        print(f"Error loading from HF: {e}")
        return None

def main():
    print("Loading MMLU dataset...", flush=True)
    ds = load_mmlu()
    if ds is None:
        print("FAILED to load MMLU")
        return

    print(f"MMLU loaded: {len(ds)} total questions", flush=True)

    # Filter STEM subjects
    stem_data = []
    for item in ds:
        if item.get('subject', '').lower() in SUBJECTS:
            stem_data.append(item)

    print(f"STEM questions: {len(stem_data)}", flush=True)

    # Format for TTS evaluation
    # MMLU has: question, choices (A/B/C/D), answer (A/B/C/D)
    formatted = []
    for item in stem_data:
        choices = item.get('choices', [])
        letters = ['A', 'B', 'C', 'D']
        choice_str = ' '.join([f"({l}) {c}" for l, c in zip(letters, choices)])
        formatted.append({
            'query': f"{item['question']}\n{choice_str}",
            'ground_truth': item['answer'],
            'subject': item.get('subject', 'unknown'),
        })

    # Sample 200 questions
    random.seed(42)
    if len(formatted) >= 200:
        sample = random.sample(formatted, 200)
    else:
        sample = formatted

    outf = BASE / "mmlu_stem_200.json"
    with open(outf, 'w') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sample)} questions to {outf}", flush=True)

    # Subject distribution
    from collections import Counter
    subj_counts = Counter(q['subject'] for q in sample)
    print("Subject distribution:")
    for s, c in subj_counts.most_common(10):
        print(f"  {s}: {c}")

if __name__ == "__main__":
    main()
