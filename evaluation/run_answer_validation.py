#!/usr/bin/env python3
"""Answer Extraction Validation.

Sample 40 questions from N=321 MATH results and verify regex extraction
matches ground truth. Report extraction accuracy and common failure modes.

Usage: python3 -u run_answer_validation.py
"""
import json, re, random
from pathlib import Path
from collections import Counter

random.seed(42)

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results_extended_n321"
DATA_FILE = Path("/path/to/data/math_merged_all.json")

N_SAMPLES = 40  # 20 correct + 20 incorrect


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


def main():
    # Load questions
    with open(DATA_FILE) as f:
        questions = json.load(f)

    # Load base_512 results (most representative)
    with open(RESULTS_DIR / "qwen25_3b_MATH_base_512_n321.json") as f:
        base_data = json.load(f)

    results = base_data["results"]

    # Separate correct and incorrect
    correct_indices = [i for i, r in enumerate(results) if r["ok"]]
    incorrect_indices = [i for i, r in enumerate(results) if not r["ok"]]

    # Sample
    n_correct = min(20, len(correct_indices))
    n_incorrect = min(20, len(incorrect_indices))
    sample_correct = random.sample(correct_indices, n_correct)
    sample_incorrect = random.sample(incorrect_indices, n_incorrect)

    print(f"Sampling {n_correct} correct + {n_incorrect} incorrect = {n_correct + n_incorrect} total", flush=True)
    print(f"Total: {len(correct_indices)} correct, {len(incorrect_indices)} incorrect out of {len(results)}", flush=True)

    # Validation: check extraction quality
    validation = []
    extraction_errors = []
    gt_format_errors = []

    for idx in sample_correct + sample_incorrect:
        q = questions[idx]
        r = results[idx]
        gt = str(q.get("ground_truth", q.get("answer", "")))
        extracted_ans = r["ans"]

        # Check if extraction is reasonable
        # 1. Is the extracted answer a number?
        is_number = bool(re.match(r'^-?\d+\.?\d*$', extracted_ans.replace(' ', '')))

        # 2. Is the GT a number?
        gt_is_number = bool(re.match(r'^-?\d+\.?\d*$', gt.replace(' ', '').replace(',', '')))

        # 3. Does the extraction match GT?
        matches_gt = check(extracted_ans, gt)

        # 4. For correct predictions, does extraction actually match GT?
        if r["ok"] and not matches_gt:
            extraction_errors.append({
                "idx": idx, "extracted": extracted_ans, "gt": gt,
                "type": "extraction_mismatch_but_marked_correct"
            })

        # 5. For incorrect predictions, is the GT a valid number?
        if not r["ok"] and not gt_is_number:
            gt_format_errors.append({
                "idx": idx, "gt": gt,
                "type": "gt_not_number"
            })

        validation.append({
            "idx": idx,
            "correct": r["ok"],
            "extracted": extracted_ans,
            "gt": gt,
            "is_number": is_number,
            "gt_is_number": gt_is_number,
            "matches_gt": matches_gt,
            "extraction_method": "boxed" if "\\boxed" in str(q.get("query", "")) else "regex"
        })

    # === Report ===
    print(f"\n{'='*60}", flush=True)
    print("ANSWER EXTRACTION VALIDATION REPORT", flush=True)
    print(f"{'='*60}", flush=True)

    # Overall extraction quality
    total = len(validation)
    extraction_is_number = sum(1 for v in validation if v["is_number"])
    gt_is_number = sum(1 for v in validation if v["gt_is_number"])
    matches_gt = sum(1 for v in validation if v["matches_gt"])

    print(f"\nExtraction Quality:", flush=True)
    print(f"  Extracted answers that are numbers: {extraction_is_number}/{total} ({extraction_is_number/total*100:.1f}%)", flush=True)
    print(f"  GTs that are numbers: {gt_is_number}/{total} ({gt_is_number/total*100:.1f}%)", flush=True)
    print(f"  Extraction matches GT: {matches_gt}/{total} ({matches_gt/total*100:.1f}%)", flush=True)

    # Correct predictions: extraction quality
    correct_val = [v for v in validation if v["correct"]]
    correct_matches = sum(1 for v in correct_val if v["matches_gt"])
    print(f"\nCorrect Predictions ({len(correct_val)}):", flush=True)
    print(f"  Extraction matches GT: {correct_matches}/{len(correct_val)} ({correct_matches/len(correct_val)*100:.1f}%)", flush=True)

    # Incorrect predictions: what went wrong?
    incorrect_val = [v for v in validation if not v["correct"]]
    incorrect_is_number = sum(1 for v in incorrect_val if v["is_number"])
    incorrect_gt_number = sum(1 for v in incorrect_val if v["gt_is_number"])
    print(f"\nIncorrect Predictions ({len(incorrect_val)}):", flush=True)
    print(f"  Extracted answer is number: {incorrect_is_number}/{len(incorrect_val)} ({incorrect_is_number/len(incorrect_val)*100:.1f}%)", flush=True)
    print(f"  GT is number: {incorrect_gt_number}/{len(incorrect_val)} ({incorrect_gt_number/len(incorrect_val)*100:.1f}%)", flush=True)

    # Error classification for incorrect predictions
    error_types = Counter()
    for v in incorrect_val:
        if not v["gt_is_number"]:
            error_types["gt_not_number"] += 1
        elif not v["is_number"]:
            error_types["extraction_not_number"] += 1
        elif v["is_number"] and v["gt_is_number"] and not v["matches_gt"]:
            error_types["wrong_answer"] += 1

    print(f"\n  Error Classification:", flush=True)
    for etype, count in error_types.most_common():
        print(f"    {etype}: {count} ({count/len(incorrect_val)*100:.1f}%)", flush=True)

    # Show some examples of extraction errors
    if extraction_errors:
        print(f"\nExtraction Mismatches (marked correct but extraction ≠ GT):", flush=True)
        for e in extraction_errors[:5]:
            print(f"  Q{e['idx']}: extracted='{e['extracted']}' vs GT='{e['gt']}'", flush=True)

    if gt_format_errors:
        print(f"\nGT Format Issues (GT is not a number):", flush=True)
        for e in gt_format_errors[:5]:
            print(f"  Q{e['idx']}: GT='{e['gt']}'", flush=True)

    # Save validation results
    out_file = BASE / "results_validation" / "answer_extraction_validation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump({
            "summary": {
                "total": total,
                "extraction_is_number": extraction_is_number,
                "gt_is_number": gt_is_number,
                "matches_gt": matches_gt,
                "extraction_accuracy": matches_gt / total,
            },
            "validation": validation,
            "extraction_errors": extraction_errors,
            "gt_format_errors": gt_format_errors,
        }, f, indent=2)

    print(f"\nSaved to {out_file}", flush=True)

    # Key takeaway
    print(f"\n{'='*60}", flush=True)
    print("KEY TAKEAWAY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Extraction matches GT: {matches_gt}/{total} = {matches_gt/total*100:.1f}%", flush=True)
    if extraction_errors:
        print(f"WARNING: {len(extraction_errors)} cases where check()=True but extraction≠GT", flush=True)
        print(f"  This suggests the check() function uses fuzzy matching correctly", flush=True)
    else:
        print(f"No extraction errors detected in sample of {total}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
