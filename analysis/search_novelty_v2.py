#!/usr/bin/env python3
"""
Novelty Search V2: Broader queries + HF-Mirror daily papers.
Uses curl + Python only (NO MCP webReader).
"""
import subprocess, json, re, time, sys
from pathlib import Path
from xml.etree import ElementTree
from collections import defaultdict

OUTPUT = Path(__file__).parent / "novelty_search_results_v2.json"

def run_curl(url, timeout=30):
    try:
        result = subprocess.run(
            ["curl", "-ks", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout+5
        )
        return result.stdout
    except Exception as e:
        return ""

def search_arxiv(query, max_results=30):
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query={encoded}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    text = run_curl(url)
    if not text:
        return []
    papers = []
    try:
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ElementTree.fromstring(text)
        for entry in root.findall('atom:entry', ns):
            title_el = entry.find('atom:title', ns)
            if title_el is None: continue
            title = title_el.text.strip().replace('\n', ' ')
            summary_el = entry.find('atom:summary', ns)
            summary = summary_el.text.strip().replace('\n', ' ')[:500] if summary_el is not None else ""
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
        print(f"    arXiv parse error: {e}", flush=True)
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

def fetch_hf_daily(days=7):
    """Fetch recent papers from HF-Mirror daily papers API."""
    all_papers = []
    # Try trending and recent
    for endpoint in ["daily_papers", "trending_papers"]:
        url = f"https://hf-mirror.com/api/{endpoint}"
        text = run_curl(url, timeout=20)
        if not text: continue
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for p in data[:50]:
                    all_papers.append({
                        "title": p.get("title", ""),
                        "authors": "",
                        "date": p.get("publishedAt", "")[:10] if p.get("publishedAt") else "",
                        "arxiv": f"https://arxiv.org/abs/{p['paper']['id']}" if isinstance(p.get('paper'), dict) and p['paper'].get('id') else "",
                        "abstract": "",
                        "source": f"hf_{endpoint}",
                    })
        except Exception as e:
            print(f"  HF {endpoint} error: {e}", flush=True)
    return all_papers

def main():
    seen = set()
    all_results = defaultdict(list)

    # === Broader arXiv queries ===
    queries = {
        "cot_reversal": [
            '"chain of thought" hurts',
            '"chain of thought" negative effect',
            'CoT performance degradation model size',
            'reasoning hurts larger language models',
            '"let\'s think step by step" negative',
            'thoughtful reasoning hurts performance',
            'CoT unhelpful instruction following',
        ],
        "truncation_paradox": [
            'truncation token budget LLM evaluation',
            'generation length benchmark evaluation',
            'token limit reasoning model accuracy',
            'incomplete output evaluation math',
        ],
        "bon_inversion": [
            '"best-of-n" sampling scaling',
            'majority voting language model scale',
            'rejection sampling inference compute',
            'repeated sampling coverage',
            'best-of-n vs majority voting',
            'sampling strategy model size',
        ],
        "commitment_zone": [
            'logit lens reasoning model',
            'midlayer probing language model',
            'representation analysis instruction tuning',
            'hidden state reasoning trace',
            'layer-wise analysis reasoning',
        ],
        "prompt_model_alignment": [
            'prompt design chain of thought effect',
            'prompt sensitivity reasoning LLM',
            'structured prompt vs generic prompt',
            'prompt model interaction scale',
        ],
        "test_time_compute": [
            'test-time compute scaling',
            'inference time compute optimal',
            'test time search strategy',
            'overthinking reasoning model',
            'test time compute model size',
            'compute optimal inference strategy',
            'scaling test-time compute',
        ],
    }

    for cat, qs in queries.items():
        print(f"\n=== {cat} ===", flush=True)
        for q in qs:
            print(f"  arXiv: {q[:50]}...", end="", flush=True)
            papers = search_arxiv(q, max_results=15)
            count = 0
            for p in papers:
                tkey = re.sub(r'\s+', '', p["title"].lower())
                if tkey not in seen and len(tkey) > 10:
                    seen.add(tkey)
                    p["source"] = "arxiv"
                    p["category"] = cat
                    all_results[cat].append(p)
                    count += 1
            print(f" +{count} (total {len(all_results[cat])})", flush=True)
            time.sleep(0.5)

    # === Semantic Scholar queries ===
    ss_queries = {
        "cot_reversal": [
            "chain of thought prompting negative effect large language models",
            "CoT reasoning degradation instruction tuning",
            "when does chain of thought hurt performance",
        ],
        "truncation_paradox": [
            "truncation bias evaluation language models generation length",
            "token budget reasoning evaluation incomplete",
        ],
        "bon_inversion": [
            "best of N sampling scaling law language models",
            "majority voting model size effect",
        ],
        "commitment_zone": [
            "mid layer probing reasoning representation language model",
            "logit lens instruction tuning commitment",
        ],
        "prompt_model_alignment": [
            "prompt engineering chain of thought model dependent",
            "structured reasoning prompt effect scale",
        ],
        "test_time_compute": [
            "test time compute scaling optimal allocation 2024 2025",
            "overthinking reasoning large language models",
            "inference time compute scaling law",
            "compute optimal test time strategy model size",
        ],
    }

    for cat, qs in ss_queries.items():
        for q in qs:
            print(f"  SS: {q[:50]}...", end="", flush=True)
            papers = search_ss(q, limit=15)
            count = 0
            for p in papers:
                tkey = re.sub(r'\s+', '', p["title"].lower())
                if tkey not in seen and len(tkey) > 10:
                    seen.add(tkey)
                    p["source"] = "semantic_scholar"
                    p["category"] = cat
                    all_results[cat].append(p)
                    count += 1
            print(f" +{count}", flush=True)
            time.sleep(1)

    # === HF-Mirror recent papers ===
    print(f"\n=== HF-Mirror ===", flush=True)
    hf_papers = fetch_hf_daily()
    # Filter for TTS-related keywords
    tts_keywords = [
        "test-time", "test time", "inference time", "chain of thought", "cot ",
        "reasoning", "scaling", "overthink", "best-of-n", "majority voting",
        "token budget", "truncation", "commitment zone", "probing",
        "prompt", "model size", "compute optimal",
    ]
    for p in hf_papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        if any(kw in text for kw in tts_keywords):
            tkey = re.sub(r'\s+', '', p["title"].lower())
            if tkey not in seen and len(tkey) > 10:
                seen.add(tkey)
                p["category"] = "hf_tts_related"
                all_results["hf_tts_related"].append(p)
    print(f"  HF TTS-related: {len(all_results.get('hf_tts_related', []))}", flush=True)

    # Summary
    total = sum(len(v) for v in all_results.values())
    print(f"\n{'='*60}", flush=True)
    for cat, papers in all_results.items():
        print(f"  {cat}: {len(papers)} papers", flush=True)
    print(f"  TOTAL: {total} unique papers", flush=True)
    print(f"{'='*60}", flush=True)

    with open(OUTPUT, 'w') as f:
        json.dump(dict(all_results), f, indent=2, ensure_ascii=False)
    print(f"Saved to: {OUTPUT}", flush=True)

if __name__ == "__main__":
    main()
