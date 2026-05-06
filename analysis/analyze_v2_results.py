#!/usr/bin/env python3
"""
TTS-Bench v2 综合分析脚本
- 读取所有checkpoint结果
- Bootstrap 95% CI
- Overthinking分析
- Compute-optimal frontier
- LaTeX表格输出
- 论文级可视化
"""
import json, sys, os
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE = Path(__file__).parent
CKPT = BASE / "results_v2"

# ============================================================
# 1. 加载所有结果
# ============================================================
def load_all_results():
    """加载所有checkpoint，按(model, dataset, method)聚合"""
    groups = defaultdict(list)
    for f in sorted(CKPT.glob("*.json")):
        d = json.load(open(f))
        meta = d.get("metadata", {})
        if not meta.get("done"):
            print(f"  WARNING: incomplete config: {f.name} (n={meta.get('n',0)})")
        key = (meta.get("model",""), meta.get("ds",""), meta.get("method",""))
        groups[key].extend(d.get("results", []))
    return groups

# ============================================================
# 2. 统计分析
# ============================================================
def bootstrap_ci(arr, n_boot=10000, ci=95):
    """Bootstrap confidence interval"""
    boots = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (100 - ci) / 2
    return np.percentile(boots, [alpha, 100-alpha])

def cohens_d(g1, g2):
    """Cohen's d effect size"""
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_sd = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / pooled_sd if pooled_sd > 0 else 0

def analyze_group(key, results):
    """分析单个(model, dataset, method)组合"""
    valid = [r for r in results if "err" not in r]
    if not valid:
        return None

    accs = np.array([float(r["ok"]) for r in valid])
    toks = np.array([r["tok"] for r in valid])
    lats = np.array([r["lat"] for r in valid])

    m = np.mean(accs)
    ci_lo, ci_hi = bootstrap_ci(accs)
    tok_m = np.mean(toks)
    lat_m = np.mean(lats)

    # Per-question accuracy across seeds (for stability)
    q_accs = defaultdict(list)
    for r in valid:
        q_accs[r["q"]].append(float(r["ok"]))
    # Cross-seed agreement rate
    agreement = np.mean([1 if sum(v)/len(v) > 0.5 or sum(v)/len(v) < 0.5 else 0
                         for v in q_accs.values() if len(v) > 1])

    return {
        "model": key[0], "dataset": key[1], "method": key[2],
        "acc": round(m, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
        "tokens": round(tok_m, 1), "latency": round(lat_m, 3),
        "efficiency": round(m/tok_m, 6) if tok_m > 0 else 0,
        "n": len(valid), "seeds": len(set(r.get("q",0) for r in valid)),
        "acc_std": round(np.std(accs), 4),
        "cross_seed_agreement": round(agreement, 4) if len(q_accs) > 0 else 0,
    }

# ============================================================
# 3. Overthinking分析
# ============================================================
def analyze_overthinking(final):
    """分析overthinking现象"""
    results = []
    models = sorted(set(r["model"] for r in final.values()))
    datasets = sorted(set(r["dataset"] for r in final.values()))

    for model in models:
        for ds in datasets:
            bk = f"{model}_{ds}_baseline"
            ck_s = f"{model}_{ds}_cot_short"
            ck_l = f"{model}_{ds}_cot_long"

            if bk not in final or ck_s not in final or ck_l not in final:
                continue

            b, cs, cl = final[bk], final[ck_s], final[ck_l]

            # Overthinking: more tokens but no gain (or loss)
            # Diminishing returns: cot_long vs cot_short
            dim_returns_tok = cl["tokens"] / cs["tokens"] if cs["tokens"] > 0 else 1
            dim_returns_acc = cl["acc"] - cs["acc"]

            # Overthinking: baseline vs cot_long
            ot_tok_ratio = cl["tokens"] / b["tokens"] if b["tokens"] > 0 else 1
            ot_acc_diff = (cl["acc"] - b["acc"]) * 100
            ot_ci_overlap = not (cl["ci_lo"] > b["ci_hi"] or b["ci_lo"] > cl["ci_hi"])

            is_overthinking = ot_acc_diff < 2 and ot_tok_ratio > 1.5 and ot_ci_overlap
            is_diminishing = dim_returns_acc < 0.01 and dim_returns_tok > 1.2

            # Effect size
            d = cohens_d(
                [1 if r["ok"] else 0 for r in []],  # placeholder
                [1 if r["ok"] else 0 for r in []]
            )

            results.append({
                "model": model, "dataset": ds,
                "baseline_acc": b["acc"], "cot_short_acc": cs["acc"], "cot_long_acc": cl["acc"],
                "baseline_tok": b["tokens"], "cot_short_tok": cs["tokens"], "cot_long_tok": cl["tokens"],
                "cot_vs_base_acc": round(ot_acc_diff, 2),
                "cot_vs_base_tok_ratio": round(ot_tok_ratio, 2),
                "dim_returns_tok_ratio": round(dim_returns_tok, 2),
                "dim_returns_acc": round(dim_returns_acc * 100, 2),
                "overthinking": is_overthinking,
                "diminishing_returns": is_diminishing,
                "ci_overlap": ot_ci_overlap,
            })

    return results

# ============================================================
# 4. Compute-Optimal Frontier
# ============================================================
def compute_optimal_frontier(final):
    """Compute-optimal frontier: accuracy vs tokens"""
    methods = ["baseline", "cot_short", "cot_long", "bon4_vote", "bon8_vote", "entropy_stop", "adaptive"]
    models = sorted(set(r["model"] for r in final.values()))
    datasets = sorted(set(r["dataset"] for r in final.values()))

    frontier = {}
    for model in models:
        for ds in datasets:
            points = []
            for method in methods:
                k = f"{model}_{ds}_{method}"
                if k in final:
                    points.append({
                        "method": method,
                        "acc": final[k]["acc"],
                        "tokens": final[k]["tokens"],
                        "latency": final[k]["latency"],
                        "efficiency": final[k]["efficiency"],
                    })
            if points:
                # Sort by tokens (compute budget)
                points.sort(key=lambda x: x["tokens"])
                # Find Pareto frontier
                pareto = []
                max_acc = 0
                for p in points:
                    if p["acc"] >= max_acc:
                        pareto.append(p)
                        max_acc = p["acc"]
                frontier[f"{model}_{ds}"] = {
                    "all_points": points,
                    "pareto": pareto,
                }
    return frontier

# ============================================================
# 5. LaTeX表格生成
# ============================================================
def generate_latex_tables(final, ot_analysis, frontier):
    """生成论文级LaTeX表格"""
    models = sorted(set(r["model"] for r in final.values()))
    datasets = sorted(set(r["dataset"] for r in final.values()))
    methods = ["baseline", "cot_short", "cot_long", "bon4_vote", "bon8_vote", "entropy_stop", "adaptive"]
    method_labels = {
        "baseline": "Baseline",
        "cot_short": "CoT (256)",
        "cot_long": "CoT (512)",
        "bon4_vote": "BoN-4 Vote",
        "bon8_vote": "BoN-8 Vote",
        "entropy_stop": "Entropy Stop",
        "adaptive": "Adaptive",
    }

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{TTS-Bench: Accuracy (\%) across models, datasets, and methods. 95\% CI in brackets. $\uparrow$ indicates compute-optimal method.}")
    latex.append(r"\label{tab:tts_bench_main}")
    latex.append(r"\resizebox{\textwidth}{!}{")
    latex.append(r"\begin{tabular}{llcccccc}")
    latex.append(r"\toprule")

    # Header
    header = "Model & Dataset & " + " & ".join(method_labels[m] for m in methods) + r" \\"
    latex.append(header)
    latex.append(r"\midrule")

    for model in models:
        for di, ds in enumerate(datasets):
            row_parts = []
            if di == 0:
                row_parts.append(r"\multirow{2}{*}{" + model + "}")
            else:
                row_parts.append("")
            row_parts.append(ds)

            for method in methods:
                k = f"{model}_{ds}_{method}"
                if k in final:
                    r = final[k]
                    acc_str = f"{r['acc']*100:.1f}"
                    ci_str = f"[{r['ci_lo']*100:.1f},{r['ci_hi']*100:.1f}]"
                    # Bold best method
                    row_parts.append(f"{acc_str} ({ci_str})")
                else:
                    row_parts.append("--")

            latex.append(" & ".join(row_parts) + r" \\")
        if model != models[-1]:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}}")
    latex.append(r"\end{table}")

    return "\n".join(latex)

# ============================================================
# 6. Overthinking LaTeX Table
# ============================================================
def generate_overthinking_table(ot_analysis):
    """生成overthinking分析LaTeX表格"""
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Overthinking analysis: CoT (512 tokens) vs Baseline. Token ratio shows compute overhead. Overthinking occurs when accuracy does not improve despite significantly more compute.}")
    latex.append(r"\label{tab:overthinking}")
    latex.append(r"\begin{tabular}{llcccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & Dataset & Base Acc & CoT Acc & Token Ratio & Overthinking \\")
    latex.append(r"\midrule")

    for r in ot_analysis:
        ot_label = r"\cmark" if r["overthinking"] else r"\xmark"
        latex.append(
            f"{r['model']} & {r['dataset']} & "
            f"{r['baseline_acc']*100:.1f}\\% & {r['cot_long_acc']*100:.1f}\\% & "
            f"{r['cot_vs_base_tok_ratio']:.1f}$\\times$ & {ot_label} \\\\"
        )

    ot_rate = sum(1 for r in ot_analysis if r["overthinking"]) / max(len(ot_analysis), 1) * 100
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{5}{l}{Overthinking Rate} & " + f"{ot_rate:.0f}\\% \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)

# ============================================================
# 7. 效率分析表格
# ============================================================
def generate_efficiency_table(final):
    """生成效率分析LaTeX表格"""
    models = sorted(set(r["model"] for r in final.values()))
    datasets = sorted(set(r["dataset"] for r in final.values()))
    methods = ["baseline", "cot_short", "bon4_vote", "adaptive", "entropy_stop"]

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Compute efficiency analysis: accuracy per token. Higher is better.}")
    latex.append(r"\label{tab:efficiency}")
    latex.append(r"\begin{tabular}{llccccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & Dataset & " + " & ".join(m.replace("_", " ").title() for m in methods) + r" \\")
    latex.append(r"\midrule")

    for model in models:
        for ds in datasets:
            row = [model if ds == datasets[0] else "", ds]
            for method in methods:
                k = f"{model}_{ds}_{method}"
                if k in final:
                    r = final[k]
                    row.append(f"{r['efficiency']*100:.3f}")
                else:
                    row.append("--")
            latex.append(" & ".join(row) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"TTS-Bench v2 Analysis | {datetime.now()}", flush=True)

    # Load
    print("\n[1] Loading results...", flush=True)
    groups = load_all_results()
    print(f"  Found {len(groups)} configs", flush=True)

    # Analyze each group
    print("\n[2] Computing statistics...", flush=True)
    final = {}
    for key, results in groups.items():
        r = analyze_group(key, results)
        if r:
            k = f"{key[0]}_{key[1]}_{key[2]}"
            final[k] = r

    # Save final JSON
    with open(BASE / "results_v2_final.json", 'w') as f:
        json.dump(final, f, indent=2)
    print(f"  Saved results_v2_final.json ({len(final)} configs)", flush=True)

    # Print main table
    print(f"\n{'='*100}")
    print("MAIN RESULTS TABLE")
    print(f"{'='*100}")
    print(f"{'Model':<14} {'DS':<7} {'Method':<16} {'Acc%':>6} {'95% CI':>22} {'Tok':>7} {'Lat(s)':>7} {'Eff':>10} {'Seeds':>5}")
    print("-"*105)
    for k in sorted(final):
        r = final[k]
        parts = k.split("_")
        model = parts[0]+"_"+parts[1]
        ds = parts[2]
        meth = "_".join(parts[3:])
        print(f"{model:<14} {ds:<7} {meth:<16} {r['acc']*100:>5.1f} [{r['ci_lo']*100:.1f},{r['ci_hi']*100:.1f}] {r['tokens']:>7.0f} {r['latency']:>7.3f} {r['efficiency']:>10.2e} {r['seeds']:>5}")

    # Overthinking analysis
    print(f"\n{'='*100}")
    print("OVERTHINKING ANALYSIS")
    print(f"{'='*100}")
    ot = analyze_overthinking(final)
    for r in ot:
        ot_label = "OVER" if r["overthinking"] else "OK"
        dim_label = "DIM" if r["diminishing_returns"] else "GAIN"
        print(f"\n{r['model']}/{r['dataset']}:")
        print(f"  Base={r['baseline_acc']*100:.1f}% CoT_S={r['cot_short_acc']*100:.1f}% CoT_L={r['cot_long_acc']*100:.1f}%")
        print(f"  CoT_L vs Base: Δacc={r['cot_vs_base_acc']:+.1f}%, tok_ratio={r['cot_vs_base_tok_ratio']:.1f}x, {ot_label}")
        print(f"  CoT_L vs CoT_S: Δacc={r['dim_returns_acc']:+.1f}%, tok_ratio={r['dim_returns_tok_ratio']:.1f}x, {dim_label}")

    if ot:
        ot_rate = sum(1 for r in ot if r["overthinking"]) / len(ot) * 100
        dim_rate = sum(1 for r in ot if r["diminishing_returns"]) / len(ot) * 100
        print(f"\n  Overthinking Rate: {ot_rate:.0f}% ({sum(1 for r in ot if r['overthinking'])}/{len(ot)})")
        print(f"  Diminishing Returns Rate: {dim_rate:.0f}% ({sum(1 for r in ot if r['diminishing_returns'])}/{len(ot)})")

    # Compute-optimal frontier
    print(f"\n{'='*100}")
    print("COMPUTE-OPTIMAL FRONTIER")
    print(f"{'='*100}")
    frontier = compute_optimal_frontier(final)
    for k, v in frontier.items():
        print(f"\n{k}:")
        print(f"  Pareto frontier: {' → '.join(f'{p['method']}({p['acc']*100:.1f}%@{p['tokens']:.0f}tok)' for p in v['pareto'])}")
        # Best efficiency
        best_eff = max(v['all_points'], key=lambda x: x['efficiency'])
        best_acc = max(v['all_points'], key=lambda x: x['acc'])
        print(f"  Best efficiency: {best_eff['method']} ({best_eff['efficiency']*100:.4f})")
        print(f"  Best accuracy: {best_acc['method']} ({best_acc['acc']*100:.1f}%)")

    # LaTeX tables
    print(f"\n{'='*100}")
    print("LATEX TABLES")
    print(f"{'='*100}")

    main_table = generate_latex_tables(final, ot, frontier)
    print("\n% Main Results Table")
    print(main_table)

    if ot:
        ot_table = generate_overthinking_table(ot)
        print("\n% Overthinking Table")
        print(ot_table)

    eff_table = generate_efficiency_table(final)
    print("\n% Efficiency Table")
    print(eff_table)

    # Summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}")
    n_total = sum(r["n"] for r in final.values())
    n_configs = len(final)
    n_done = sum(1 for r in final.values() if r["seeds"] >= 3)
    print(f"  Total samples: {n_total}")
    print(f"  Completed configs: {n_configs}/126")
    print(f"  Configs with 3 seeds: {n_done}")

    # Best method per model
    print(f"\n  BEST METHOD PER MODEL:")
    for model in sorted(set(r["model"] for r in final.values())):
        model_results = {k: v for k, v in final.items() if v["model"] == model}
        best_acc_k = max(model_results, key=lambda k: model_results[k]["acc"])
        best_eff_k = max(model_results, key=lambda k: model_results[k]["efficiency"])
        print(f"    {model}: Best Acc={final[best_acc_k]['method']}({final[best_acc_k]['acc']*100:.1f}%), "
              f"Best Eff={final[best_eff_k]['method']}({final[best_eff_k]['efficiency']*100:.4f})")

    print(f"\nDone: {datetime.now()}")

if __name__ == "__main__":
    main()
