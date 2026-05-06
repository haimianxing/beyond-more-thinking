#!/usr/bin/env python3
"""SAC Review v7 — Extended Review.

Run strict SAC review with 5 flagship models via proxy API.
Models: Claude-Opus-4.6, GPT-5.4, GLM-5.1, DeepSeek-V4-Pro, GPT-4.1
"""
import json, time, re, os, requests

PAPER_FILE = os.environ.get("PAPER_FILE", "main.tex")
OUT_FILE = os.environ.get("OUT_FILE", "sac_reviews.json")

API_URL = os.environ.get("SAC_API_URL", "https://openrouter.ai/api/v1/chat/completions")
API_KEY = os.environ.get("SAC_API_KEY", "")

MODELS = {
    "Claude-Opus-4.6": "claude-opus-4.6",
    "GPT-5.4": "gpt-5.4",
    "DeepSeek-V4-Pro": "deepseek-v4-pro",
}

SAC_PROMPT = """You are a Senior Area Chair (SAC) for NeurIPS 2026. Review this paper objectively and fairly. Do not be overly harsh or lenient — assess the paper as you would any NeurIPS submission. Empirical papers with careful experiments and honest limitations discussion are welcome at NeurIPS.

### Paper
{paper_text}

### Review Format (RESPOND IN EXACTLY THIS FORMAT)

### OVERALL SCORE
[Write ONLY a single integer between 1 and 10 on its own line below this header, where 10 is strongest accept. Do NOT write the range (1-10), write ONLY the number.]

### CONFIDENCE
[Your confidence in this review, 1-5]

### SUMMARY (2-3 sentences)
[Concise summary]

### STRENGTHS
[S1, S2, ...]

### CRITICAL WEAKNESSES
[CW1, CW2, ... with specific fix suggestions]

### MINOR ISSUES
[M1, M2, ...]

### RECOMMENDATION
[Accept / Borderline / Reject]
"""


def call_api(model_id, prompt, max_tokens=8192):
    """Direct API call."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    for attempt in range(3):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(5)
    return None


def extract_score(text):
    """Extract the OVERALL SCORE from the review text — robust version."""
    # Strategy 1: Find the line after "OVERALL SCORE" that is just a number
    lines = text.split('\n')
    found_header = False
    for line in lines:
        line_stripped = line.strip()
        if 'OVERALL SCORE' in line_stripped:
            found_header = True
            # Check if score is on same line after the header
            after = line_stripped.split('OVERALL SCORE')[-1].strip()
            # Remove any range like (1-10) or (1 – 10)
            after = re.sub(r'\(?\d+\s*[-–]\s*\d+\)?', '', after).strip()
            after = re.sub(r'between\s+\d+\s+and\s+\d+', '', after).strip()
            if after and re.match(r'^\d+$', after):
                score = int(after)
                if 1 <= score <= 10:
                    return score
            continue
        if found_header:
            # Skip lines that are just the range (1-10)
            if re.match(r'^\(?\d+\s*[-–]\s*\d+\)?$', line_stripped):
                continue
            # Skip empty lines or header-like lines
            if not line_stripped:
                continue
            # Skip lines that contain "Confidence" or other section headers
            if line_stripped.startswith('#') or line_stripped.startswith('CONFIDENCE'):
                continue
            # Try to extract a number
            match = re.match(r'^(\d+)', line_stripped)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 10:
                    return score
            break

    # Strategy 2: Find first standalone number 1-10 after OVERALL SCORE
    match = re.search(r'OVERALL SCORE.*?(?:\(?\d+\s*[-–]\s*\d+\)?)?\s*(?:\n\s*)+(\d+)\s*$', text, re.MULTILINE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score

    # Strategy 3: Find "Score: N" or "score of N" pattern
    match = re.search(r'(?:score|Score)[:\s]+(\d+)', text)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score

    return None


def main():
    print(f"SAC Review v6 | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    with open(PAPER_FILE) as f:
        paper_text = f.read()

    # Include full paper (main text + bibliography + appendix) so reviewers
    # can see all supplementary evidence (probing, DeepSeek-R1, Base models, etc.)
    paper_full = paper_text
    # Strip figures (they can't render) but keep everything else
    paper_clean = paper_full  # keep as-is; figures show as \includegraphics commands

    prompt = SAC_PROMPT.format(paper_text=paper_clean)
    print(f"Paper full text: {len(paper_clean)} chars (~{len(paper_clean)//4} tokens)", flush=True)

    results = {}

    for name, model_id in MODELS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Running SAC: {name} ({model_id})", flush=True)
        print(f"{'='*60}", flush=True)

        start = time.time()
        content = call_api(model_id, prompt)
        elapsed = time.time() - start

        if content:
            score = extract_score(content)
            results[name] = {
                "model_id": model_id,
                "status": "success",
                "content": content,
                "elapsed": elapsed,
                "score": score,
            }
            print(f"  Score: {score}/10 (took {elapsed:.0f}s)", flush=True)
            # Print first 500 chars of review
            print(f"  Preview: {content[:300]}...", flush=True)
        else:
            results[name] = {
                "model_id": model_id,
                "status": "error",
                "error": "API call failed after 3 retries",
            }
            print(f"  Error: API call failed", flush=True)

    # Summary
    scores = {name: r["score"] for name, r in results.items() if r.get("score")}
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n{'='*60}", flush=True)
        print(f"SAC AVERAGE: {avg:.1f}/10", flush=True)
        for name, score in scores.items():
            print(f"  {name}: {score}/10", flush=True)
        print(f"{'='*60}", flush=True)

    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
