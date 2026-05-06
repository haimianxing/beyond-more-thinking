#!/usr/bin/env python3
"""
Novelty Search: Search arXiv + Semantic Scholar for papers potentially overlapping
with "Beyond More Thinking" core contributions.
Uses curl + Python only (NO MCP webReader).
"""
import subprocess, json, re, time, sys
from pathlib import Path
from xml.etree import ElementTree

OUTPUT = Path(__file__).parent / "novelty_search_results.json"

# Our 6 core contributions as search queries
SEARCH_QUERIES = {
    "cot_reversal": {
        "description": "CoT hurts larger models / CoT reversal",
        "arxiv_queries": [
            'all:"chain of thought" AND all:hurts AND all:larger',
            'all:"chain of thought" AND all:harmful AND all:model',
            'all:CoT AND all:negative AND all:scale',
            'all:"chain of thought" AND all:"performance degradation"',
            'all:reasoning AND all:hurts AND all:performance',
        ],
        "ss_queries": [
            "chain of thought prompting harmful larger models",
            "CoT negative effect model scale",
            "reasoning degradation instruction tuned models",
        ]
    },
    "truncation_paradox": {
        "description": "Token budget evaluation bias / truncation hides accuracy",
        "arxiv_queries": [
            'all:truncation AND all:token AND all:budget AND all:evaluation',
            'all:truncation AND all:accuracy AND all:benchmark',
            'all:"token budget" AND all:scaling AND all:test',
            'all:incomplete AND all:generation AND all:evaluation',
            'all:truncation AND all:"hidden accuracy"',
        ],
        "ss_queries": [
            "truncation token budget evaluation bias LLM",
            "incomplete generation evaluation benchmark",
        ]
    },
    "bon_inversion": {
        "description": "Best-of-N / majority voting hurts larger models",
        "arxiv_queries": [
            'all:"best of N" AND all:scaling AND all:voting',
            'all:majority AND all:voting AND all:hurts',
            'all:"best of N" AND all:inversion',
            'all:sampling AND all:strategy AND all:model AND all:size',
            'all:rejection AND all:sampling AND all:scale',
        ],
        "ss_queries": [
            "best of N sampling scaling law",
            "majority voting model size dependent",
        ]
    },
    "factorial_decomposition": {
        "description": "Separating token budget effect from CoT prompt effect",
        "arxiv_queries": [
            'all:"token budget" AND all:"chain of thought" AND all:decomposition',
            'all:confound AND all:CoT AND all:token',
            'all:factorial AND all:test AND all:time AND all:compute',
            'all:"controlled experiment" AND all:CoT AND all:scale',
        ],
        "ss_queries": [
            "token budget chain of thought confounding",
            "test time compute factorial decomposition",
        ]
    },
    "commitment_zone": {
        "description": "Mid-layer probing / internalized reasoning / commitment zone",
        "arxiv_queries": [
            'all:mid-layer AND all:probing AND all:reasoning',
            'all:internalized AND all:reasoning AND all:probing',
            'all:logit AND all:lens AND all:commitment',
            'all:layer AND all:probing AND all:instruction AND all:tuning',
            'all:representation AND all:trajectory AND all:reasoning',
        ],
        "ss_queries": [
            "mid-layer probing internalized reasoning LLM",
            "logit lens instruction tuning commitment",
        ]
    },
    "prompt_model_alignment": {
        "description": "Prompt-dependent CoT effects / prompt sensitivity in TTS",
        "arxiv_queries": [
            'all:prompt AND all:sensitive AND all:"chain of thought"',
            'all:prompt AND all:dependent AND all:reasoning AND all:performance',
            'all:structured AND all:CoT AND all:generic AND all:CoT',
            'all:prompt AND all:strategy AND all:interaction AND all:model',
            'all:"prompt engineering" AND all:reasoning AND all:scale',
        ],
        "ss_queries": [
            "prompt sensitivity chain of thought reasoning",
            "structured vs generic CoT prompt effect",
        ]
    },
    "test_time_compute_scaling": {
        "description": "General TTS scaling papers (broad coverage)",
        "arxiv_queries": [
            'all:"test time compute" AND all:scaling AND all:language',
            'all:"test-time" AND all:compute AND all:optimal',
            'all:inference AND all:time AND all:compute AND all:scaling',
            'all:"test time" AND all:strategy AND all:reversal',
            'all:overthinking AND all:reasoning AND all:model',
        ],
        "ss_queries": [
            "test time compute scaling optimal allocation 2024 2025",
            "overthinking large language models reasoning",
        ]
    },
}

def run_curl(url, timeout=30):
    """Run curl with SSL skip, return text."""
    try:
        result = subprocess.run(
            ["curl", "-ks", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout+5
        )
        return result.stdout
    except Exception as e:
        return f"ERROR: {e}"

def search_arxiv(query, max_results=20):
    """Search arXiv API, return list of paper dicts."""
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query={encoded}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    text = run_curl(url)
    if text.startswith("ERROR"):
        return []

    papers = []
    try:
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ElementTree.fromstring(text)
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')[:500]
            published = entry.find('atom:published', ns).text[:10]
            link = entry.find('atom:id', ns).text
            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
            papers.append({
                "title": title,
                "authors": ", ".join(authors[:3]) + ("..." if len(authors) > 3 else ""),
                "date": published,
                "arxiv": link,
                "abstract": summary,
            })
    except Exception as e:
        pass
    return papers

def search_semantic_scholar(query, limit=20, year="2024-2026"):
    """Search Semantic Scholar API, return list of paper dicts."""
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&year={year}&fields=title,authors,year,abstract,url,externalIds"
    text = run_curl(url)
    if text.startswith("ERROR") or not text.strip():
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
    all_results = {}
    total_papers = 0
    seen_titles = set()  # deduplicate

    for category, config in SEARCH_QUERIES.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Category: {category} — {config['description']}", flush=True)
        print(f"{'='*60}", flush=True)

        category_papers = []

        # Search arXiv
        for qi, q in enumerate(config["arxiv_queries"]):
            print(f"  arXiv query {qi+1}/{len(config['arxiv_queries'])}: {q[:60]}...", flush=True)
            papers = search_arxiv(q, max_results=10)
            for p in papers:
                tkey = p["title"].lower().strip()
                if tkey not in seen_titles:
                    seen_titles.add(tkey)
                    p["source"] = "arxiv"
                    p["search_category"] = category
                    category_papers.append(p)
            time.sleep(1)  # rate limit

        # Search Semantic Scholar
        for qi, q in enumerate(config["ss_queries"]):
            print(f"  SS query {qi+1}/{len(config['ss_queries'])}: {q[:60]}...", flush=True)
            papers = search_semantic_scholar(q, limit=15)
            for p in papers:
                tkey = p["title"].lower().strip()
                if tkey not in seen_titles:
                    seen_titles.add(tkey)
                    p["source"] = "semantic_scholar"
                    p["search_category"] = category
                    category_papers.append(p)
            time.sleep(1)

        all_results[category] = category_papers
        total_papers += len(category_papers)
        print(f"  Found {len(category_papers)} unique papers", flush=True)

    # Save results
    with open(OUTPUT, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}", flush=True)
    print(f"TOTAL: {total_papers} unique papers across {len(SEARCH_QUERIES)} categories", flush=True)
    print(f"Saved to: {OUTPUT}", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
