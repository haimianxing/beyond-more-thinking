#!/usr/bin/env python3
"""Prepare MMLU STEM subset using direct download from HuggingFace files.

Fallback approach: download MMLU test data directly instead of using datasets library.
"""
import json, random, csv, os
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

def load_mmlu_from_datasets():
    """Try loading MMLU using datasets library with different configs."""
    try:
        from datasets import load_dataset
        # Try cais/mmlu first
        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        return ds
    except Exception as e:
        print(f"cais/mmlu failed: {e}")

    try:
        from datasets import load_dataset
        # Try lukaemon/mmlu
        ds = load_dataset("lukaemon/mmlu", "all", split="test", trust_remote_code=True)
        return ds
    except Exception as e:
        print(f"lukaemon/mmlu failed: {e}")

    return None

def load_mmlu_from_csv():
    """Load MMLU from local CSV files if available."""
    # Check common locations
    possible_paths = [
        Path("/path/to/datasets/mmlu"),
        Path("/path/to/datasets/mmlu"),
        Path.home() / ".cache/huggingface/datasets/cais___mmlu",
    ]
    for p in possible_paths:
        if p.exists():
            print(f"Found MMLU at {p}")
            return p
    return None

def create_mmlu_from_hf_api():
    """Download MMLU test data directly via HuggingFace API."""
    import urllib.request

    all_data = []
    for subject in SUBJECTS:
        url = f"https://huggingface.co/datasets/cais/mmlu/resolve/main/data/test-00000-of-00001.parquet"
        # Try direct parquet download
        try:
            import pandas as pd
            local_path = BASE / f"mmlu_{subject}.parquet"
            if not local_path.exists():
                # Use the datasets streaming approach
                from datasets import load_dataset
                ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True, streaming=True)
                rows = []
                for item in ds:
                    if item.get('subject', '').lower() == subject:
                        rows.append(item)
                all_data.extend(rows)
                continue
        except Exception as e:
            print(f"  Streaming {subject} failed: {e}")

    return all_data if all_data else None

def main():
    print("Loading MMLU dataset...", flush=True)

    # Method 1: Try datasets library
    ds = load_mmlu_from_datasets()
    if ds is not None:
        print(f"MMLU loaded via datasets: {len(ds)} total questions", flush=True)

        stem_data = []
        for item in ds:
            if item.get('subject', '').lower() in SUBJECTS:
                stem_data.append(item)

        print(f"STEM questions: {len(stem_data)}", flush=True)

        # Format
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

        random.seed(42)
        if len(formatted) >= 200:
            sample = random.sample(formatted, 200)
        else:
            sample = formatted

        outf = BASE / "mmlu_stem_200.json"
        with open(outf, 'w') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(sample)} questions to {outf}", flush=True)

        from collections import Counter
        subj_counts = Counter(q['subject'] for q in sample)
        print("Subject distribution:")
        for s, c in subj_counts.most_common(10):
            print(f"  {s}: {c}")
        return

    # Method 2: Try streaming per subject
    print("Trying streaming per subject...", flush=True)
    try:
        from datasets import load_dataset
        all_data = []
        for subject in SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test",
                                  trust_remote_code=True, download_mode="force_redownload")
                for item in ds:
                    all_data.append(item)
                print(f"  {subject}: {len(ds)} questions", flush=True)
            except Exception as e:
                print(f"  {subject}: FAILED - {e}", flush=True)

        if all_data:
            print(f"Total STEM: {len(all_data)}", flush=True)
            formatted = []
            for item in all_data:
                choices = item.get('choices', [])
                letters = ['A', 'B', 'C', 'D']
                choice_str = ' '.join([f"({l}) {c}" for l, c in zip(letters, choices)])
                formatted.append({
                    'query': f"{item['question']}\n{choice_str}",
                    'ground_truth': item['answer'],
                    'subject': item.get('subject', 'unknown'),
                })
            random.seed(42)
            sample = random.sample(formatted, min(200, len(formatted)))

            outf = BASE / "mmlu_stem_200.json"
            with open(outf, 'w') as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(sample)} questions to {outf}", flush=True)
            return
    except Exception as e:
        print(f"Streaming approach failed: {e}", flush=True)

    print("FAILED: Could not load MMLU via any method")
    print("Will try to use GSM8K as fallback non-math dataset...")

    # Fallback: Use GSM8K which is arithmetic (different from MATH algebra)
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        print(f"GSM8K loaded: {len(ds)} questions", flush=True)

        formatted = []
        for item in ds:
            # GSM8K has answer after ####
            ans = item['answer'].split('####')[-1].strip()
            formatted.append({
                'query': item['question'],
                'ground_truth': ans,
                'subject': 'gsm8k_arithmetic',
            })

        random.seed(42)
        sample = random.sample(formatted, min(200, len(formatted)))

        outf = BASE / "gsm8k_200.json"
        with open(outf, 'w') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(sample)} GSM8K questions to {outf}", flush=True)
    except Exception as e:
        print(f"GSM8K also failed: {e}")

if __name__ == "__main__":
    main()
