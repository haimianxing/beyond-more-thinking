#!/usr/bin/env python3
"""
Novelty Search V3: Fixed arXiv HTTPS + rate-limited Semantic Scholar.
Uses curl + Python only (NO MCP webReader).
"""
import subprocess, json, re, time, sys
from pathlib import Path
from xml.etree import ElementTree
from collections import defaultdict

OUTPUT = Path(__file__).parent / "novelty_search_results_v3.json"

def run_curl(url, timeout=30):
    try:
        result = subprocess.run(
            ["curl", "-ksL", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout+5
        )
        return result.stdout
    except:
        return ""

def search_arxiv(query, max_results=20):
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query={encoded}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    text = run_curl(url)
    if not text or "<entry>" not in text:
        return []
    papers = []
    try:
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ElementTree.fromstring(text)
        for entry in root.findall('atom:entry', ns):
            title_el = entry.find('atom:title', ns)
            if title_el is None: continue
            title = re.sub(r'\s+', ' ', title_el.text.strip())
            summary_el = entry.find('atom:summary', ns)
            summary = re.sub(r'\s+', ' ', summary_el.text.strip())[:500] if summary_el is not None else ""
            published_el = entry.find('atom:published', ns)
            published = published_el.text[:10] if published_el is not None else ""
            link_el = entry.find('atom:id', ns)
            link = link_el.text if link_el is not None else ""
            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns) if a.find('atom:name', ns) is not None]
            papers.append({
                "title": title,
                "authors": ", ".join(authors[:3]) + ("..." if len(authors) > 3 else ""),
                "date": published,
                "arxiv": link,
                "abstract": summary,
            })
    except Exception as e:
        print(f"    Parse error: {e}", flush=True)
    return papers

def search_ss(query, limit=20, year="2023-2026"):
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&year={year}&fields=title,authors,year,abstract,url,externalIds"
    text = run_curl(url)
    if not text:
        return []
    try:
        data = json.loads(text)
        papers = []
        for p in data.get("data", []):
            authors = [a.get("name", "") for a in (p.get("authors") or [])[:3]]
            arxiv_id = (p.get("externalIds") or {}).get("ArXiv", "")
            papers.append({
                "title": p.get("title", ""),
                "authors": ", ".join(authors),
                "date": str(p.get("year", "")),
                "arxiv": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else p.get("url", ""),
                "abstract": (p.get("abstract") or "")[:500],
            })
        return papers
    except:
        return []

def main():
    seen = set()
    all_results = defaultdict(list)

    # arXiv queries - broader, HTTPS
    arxiv_queries = {
        "cot_reversal": [
            'all:"chain-of-thought" AND all:negative',
            'all:"chain of thought" AND all:hurts',
            'all:"CoT" AND all:"performance" AND all:"degradation"',
            'ti:"chain of thought" AND all:scale',
            'all:"let\'s think step by step" AND all:negative',
            'all:"reasoning" AND all:"hurts" AND all:"performance"',
            'ti:reasoning AND all:harmful AND all:language',
        ],
        "truncation_paradox": [
            'all:truncation AND all:token AND all:LLM',
            'all:"token budget" AND all:benchmark',
            'all:truncation AND all:evaluation AND all:generation',
            'all:"generation length" AND all:accuracy AND all:model',
        ],
        "bon_inversion": [
            'all:"best-of-n" AND all:scaling',
            'all:"majority voting" AND all:language AND all:model',
            'all:"rejection sampling" AND all:inference',
            'all:"repeated sampling" AND all:LLM',
            'all:"best-of-n" AND all:reasoning',
        ],
        "commitment_zone": [
            'all:"logit lens" AND all:reasoning',
            'all:probing AND all:"instruction tuning" AND all:layer',
            'all:"hidden state" AND all:reasoning AND all:LLM',
            'all:"representation" AND all:trajectory AND all:reasoning',
            'all:"mid-layer" AND all:probing',
        ],
        "prompt_alignment": [
            'all:"prompt" AND all:"chain of thought" AND all:"effect"',
            'all:"prompt" AND all:"sensitivity" AND all:reasoning',
            'all:"structured" AND all:"prompt" AND all:reasoning',
        ],
        "test_time_compute": [
            'all:"test-time compute"',
            'all:"test time compute" AND all:scaling',
            'all:overthinking AND all:reasoning AND all:model',
            'all:"inference time" AND all:compute AND all:optimal',
            'all:"test-time" AND all:scaling AND all:strategy',
            'all:"compute-optimal" AND all:inference',
        ],
        "cot_scale_interaction": [
            'all:"chain of thought" AND all:"model size"',
            'all:CoT AND all:"model scale" AND all:effect',
            'all:"instruction tuning" AND all:reasoning AND all:performance',
            'all:"base model" AND all:"instruct" AND all:reasoning',
        ],
    }

    print("=== arXiv Search ===", flush=True)
    for cat, qs in arxiv_queries.items():
        for q in qs:
            papers = search_arxiv(q, max_results=15)
            count = 0
            for p in papers:
                tkey = re.sub(r'\s+', '', p["title"].lower())[:80]
                if tkey not in seen and len(tkey) > 10:
                    seen.add(tkey)
                    p["source"] = "arxiv"
                    p["category"] = cat
                    all_results[cat].append(p)
                    count += 1
            if count > 0:
                print(f"  {cat}: +{count} (total {len(all_results[cat])})", flush=True)
            time.sleep(0.3)

    # Semantic Scholar - with 3s delays to avoid 429
    ss_queries = [
        ("cot_reversal", "chain of thought hurts larger models negative effect"),
        ("cot_reversal", "CoT reasoning degradation instruction tuned"),
        ("cot_reversal", "when does chain of thought fail reasoning"),
        ("truncation_paradox", "truncation bias evaluation language models"),
        ("truncation_paradox", "token budget reasoning evaluation"),
        ("bon_inversion", "best of N sampling scaling law"),
        ("bon_inversion", "majority voting effect model size"),
        ("commitment_zone", "mid layer probing reasoning language model"),
        ("commitment_zone", "logit lens instruction tuning"),
        ("prompt_alignment", "prompt sensitivity chain of thought"),
        ("prompt_alignment", "structured reasoning prompt scale effect"),
        ("test_time_compute", "test time compute scaling optimal"),
        ("test_time_compute", "overthinking reasoning language models"),
        ("test_time_compute", "compute optimal inference strategy"),
        ("test_time_compute", "inference time compute scaling model size"),
        ("cot_scale_interaction", "chain of thought model size interaction"),
        ("cot_scale_interaction", "instruction tuning reasoning performance scale"),
    ]

    print("\n=== Semantic Scholar Search ===", flush=True)
    for cat, q in ss_queries:
        papers = search_ss(q, limit=15)
        count = 0
        for p in papers:
            tkey = re.sub(r'\s+', '', p["title"].lower())[:80]
            if tkey not in seen and len(tkey) > 10:
                seen.add(tkey)
                p["source"] = "semantic_scholar"
                p["category"] = cat
                all_results[cat].append(p)
                count += 1
        if count > 0:
            print(f"  {cat}: +{count} (total {len(all_results[cat])})", flush=True)
        time.sleep(3)  # avoid 429

    # Summary
    total = sum(len(v) for v in all_results.values())
    print(f"\n{'='*60}", flush=True)
    for cat in ["cot_reversal", "truncation_paradox", "bon_inversion", "commitment_zone", "prompt_alignment", "test_time_compute", "cot_scale_interaction"]:
        print(f"  {cat}: {len(all_results.get(cat, []))} papers", flush=True)
    print(f"  TOTAL: {total} unique papers", flush=True)
    print(f"{'='*60}", flush=True)

    with open(OUTPUT, 'w') as f:
        json.dump(dict(all_results), f, indent=2, ensure_ascii=False)
    print(f"Saved to: {OUTPUT}", flush=True)

if __name__ == "__main__":
    main()
