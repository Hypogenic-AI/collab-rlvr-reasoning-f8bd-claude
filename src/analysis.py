"""
Statistical analysis and visualization for Collaborative RLVR Debate experiments.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_DIR = Path("../results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})


def load_results():
    with open(RESULTS_DIR / "all_results.json") as f:
        return json.load(f)


def bootstrap_ci(data, n_boot=2000, alpha=0.05):
    """Bootstrap 95% confidence interval for the mean."""
    data = np.array(data, dtype=float)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(data), lo, hi


def mcnemar_test(correct_a, correct_b):
    """McNemar's test for paired binary data."""
    a = np.array(correct_a, dtype=bool)
    b = np.array(correct_b, dtype=bool)
    # Discordant pairs
    b_wins = np.sum(~a & b)  # a wrong, b right
    a_wins = np.sum(a & ~b)  # a right, b wrong
    n_disc = b_wins + a_wins
    if n_disc == 0:
        return {"chi2": 0, "p_value": 1.0, "b_wins": int(b_wins), "a_wins": int(a_wins)}
    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b_wins - a_wins) - 1) ** 2 / (b_wins + a_wins) if (b_wins + a_wins) > 0 else 0
    p = 1 - stats.chi2.cdf(chi2, df=1) if chi2 > 0 else 1.0
    return {"chi2": float(chi2), "p_value": float(p), "b_wins": int(b_wins), "a_wins": int(a_wins)}


def analyze_gsm8k(rows):
    """Analyze GSM8K experiment results."""
    n = len(rows)
    print(f"\n{'='*60}")
    print(f"GSM8K ANALYSIS (n={n})")
    print(f"{'='*60}")

    # Accuracy with CIs
    metrics = {
        "GPT-4.1 single": [r["sg"] for r in rows],
        "Claude single": [r["sc"] for r in rows],
        "Self-consist(3)": [r["sc3"] for r in rows],
        "Cross-debate (any)": [r["cx_any"] for r in rows],
        "Cross-debate (GPT)": [r["cx_r1a"] for r in rows],
        "Cross-debate (Claude)": [r["cx_r1b"] for r in rows],
        "Same-debate (any)": [r["sm_any"] for r in rows],
    }

    results_table = {}
    print(f"\n{'Method':<25} {'Accuracy':>10} {'95% CI':>20}")
    print("-" * 60)
    for name, vals in metrics.items():
        mean, lo, hi = bootstrap_ci(vals)
        print(f"{name:<25} {mean:>10.1%} [{lo:.1%}, {hi:.1%}]")
        results_table[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_correct": sum(vals), "n": n}

    # Statistical tests
    print(f"\n--- Statistical Tests ---")

    # McNemar: single GPT vs cross-debate-any
    test1 = mcnemar_test([r["sg"] for r in rows], [r["cx_any"] for r in rows])
    print(f"Single GPT vs Cross-debate: χ²={test1['chi2']:.2f}, p={test1['p_value']:.4f} "
          f"(debate wins {test1['b_wins']}, single wins {test1['a_wins']})")

    # McNemar: single GPT vs same-debate-any
    test2 = mcnemar_test([r["sg"] for r in rows], [r["sm_any"] for r in rows])
    print(f"Single GPT vs Same-debate:  χ²={test2['chi2']:.2f}, p={test2['p_value']:.4f} "
          f"(debate wins {test2['b_wins']}, single wins {test2['a_wins']})")

    # McNemar: cross-debate vs self-consistency
    test3 = mcnemar_test([r["sc3"] for r in rows], [r["cx_any"] for r in rows])
    print(f"SC(3) vs Cross-debate:      χ²={test3['chi2']:.2f}, p={test3['p_value']:.4f}")

    # Outcome breakdown
    print(f"\n--- Cross-Debate Outcome Breakdown ---")
    ocs = Counter(r["cx_oc"] for r in rows)
    for k, v in sorted(ocs.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:>3} ({v/n:.1%})")

    # Error correction analysis
    print(f"\n--- Error Correction Analysis ---")
    pos_corr = ocs.get("positive_correction", 0)
    neg_pers = ocs.get("negative_persuasion", 0)
    both_wrong_to_corr = ocs.get("both_wrong_to_correct", 0)
    mixed = ocs.get("mixed", 0)
    net = pos_corr + both_wrong_to_corr - neg_pers
    print(f"  Positive corrections:  {pos_corr} ({pos_corr/n:.1%})")
    print(f"  Both wrong→correct:    {both_wrong_to_corr} ({both_wrong_to_corr/n:.1%})")
    print(f"  Negative persuasion:   {neg_pers} ({neg_pers/n:.1%})")
    print(f"  Mixed:                 {mixed} ({mixed/n:.1%})")
    print(f"  Net correction:        {net} ({net/n:.1%})")

    return {
        "accuracy_table": results_table,
        "tests": {"single_vs_cross": test1, "single_vs_same": test2, "sc3_vs_cross": test3},
        "outcomes": dict(ocs),
        "error_correction": {"pos_corr": pos_corr, "neg_pers": neg_pers,
                             "both_wrong_to_corr": both_wrong_to_corr, "mixed": mixed, "net": net},
    }


def analyze_math500(rows):
    """Analyze MATH-500 results."""
    n = len(rows)
    print(f"\n{'='*60}")
    print(f"MATH-500 ANALYSIS (n={n})")
    print(f"{'='*60}")

    metrics = {
        "GPT-4.1 single": [r["sg"] for r in rows],
        "Claude single": [r["sc"] for r in rows],
        "Cross-debate (any)": [r["cx_any"] for r in rows],
        "Cross-debate (GPT)": [r["cx_r1a"] for r in rows],
        "Cross-debate (Claude)": [r["cx_r1b"] for r in rows],
    }

    results_table = {}
    print(f"\n{'Method':<25} {'Accuracy':>10} {'95% CI':>20}")
    print("-" * 60)
    for name, vals in metrics.items():
        mean, lo, hi = bootstrap_ci(vals)
        print(f"{name:<25} {mean:>10.1%} [{lo:.1%}, {hi:.1%}]")
        results_table[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    test1 = mcnemar_test([r["sg"] for r in rows], [r["cx_any"] for r in rows])
    print(f"\nSingle GPT vs Cross-debate: χ²={test1['chi2']:.2f}, p={test1['p_value']:.4f}")

    ocs = Counter(r["cx_oc"] for r in rows)
    print(f"\nOutcome breakdown: {dict(ocs)}")

    # By difficulty level
    levels = set(r.get("level", "") for r in rows)
    if levels - {""}:
        print(f"\n--- By Difficulty Level ---")
        for lv in sorted(levels):
            if not lv: continue
            subset = [r for r in rows if r.get("level") == lv]
            if len(subset) < 3: continue
            sg = sum(r["sg"] for r in subset) / len(subset)
            cx = sum(r["cx_any"] for r in subset) / len(subset)
            print(f"  {lv}: Single GPT {sg:.0%} → Debate {cx:.0%} (n={len(subset)})")

    return {"accuracy_table": results_table, "tests": {"single_vs_cross": test1}, "outcomes": dict(ocs)}


def analyze_robustness(rows):
    """Analyze robustness experiment."""
    n = len(rows)
    print(f"\n{'='*60}")
    print(f"ROBUSTNESS ANALYSIS (n={n})")
    print(f"{'='*60}")

    s_orig = [r["s_orig"] for r in rows]
    s_rep = [r["s_rep"] for r in rows]
    d_orig = [r["d_orig"] for r in rows]
    d_rep = [r["d_rep"] for r in rows]

    so_m, so_lo, so_hi = bootstrap_ci(s_orig)
    sr_m, sr_lo, sr_hi = bootstrap_ci(s_rep)
    do_m, do_lo, do_hi = bootstrap_ci(d_orig)
    dr_m, dr_lo, dr_hi = bootstrap_ci(d_rep)

    s_drop = so_m - sr_m
    d_drop = do_m - dr_m

    print(f"\n{'Condition':<25} {'Original':>10} {'Rephrased':>10} {'Drop':>10}")
    print("-" * 60)
    print(f"{'Single GPT-4.1':<25} {so_m:>10.1%} {sr_m:>10.1%} {s_drop:>10.1%}")
    print(f"{'Cross-model debate':<25} {do_m:>10.1%} {dr_m:>10.1%} {d_drop:>10.1%}")

    # Test if debate has smaller drop
    # Compare drops via bootstrap
    drops_single = np.array(s_orig, float) - np.array(s_rep, float)
    drops_debate = np.array(d_orig, float) - np.array(d_rep, float)
    diff_drops = drops_single - drops_debate  # positive = debate more robust
    mean_diff, lo_diff, hi_diff = bootstrap_ci(diff_drops)
    # Paired t-test on drops
    t_stat, p_val = stats.ttest_rel(drops_single, drops_debate)
    print(f"\nDrop difference (single - debate): {mean_diff:.1%} [{lo_diff:.1%}, {hi_diff:.1%}]")
    print(f"Paired t-test: t={t_stat:.2f}, p={p_val:.4f}")

    return {
        "single_orig": so_m, "single_rep": sr_m, "single_drop": s_drop,
        "debate_orig": do_m, "debate_rep": dr_m, "debate_drop": d_drop,
        "drop_diff": mean_diff, "p_value": p_val,
    }


def plot_accuracy_comparison(gsm_analysis, math_analysis):
    """Bar chart comparing methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, analysis, title in [(axes[0], gsm_analysis, "GSM8K"), (axes[1], math_analysis, "MATH-500")]:
        table = analysis["accuracy_table"]
        methods = list(table.keys())
        means = [table[m]["mean"] for m in methods]
        ci_lo = [table[m]["ci_lo"] for m in methods]
        ci_hi = [table[m]["ci_hi"] for m in methods]
        errors = [[m - l for m, l in zip(means, ci_lo)],
                  [h - m for m, h in zip(means, ci_hi)]]

        colors = ['#4C72B0', '#4C72B0', '#55A868', '#DD8452', '#C44E52', '#8172B2', '#937860']
        bars = ax.bar(range(len(methods)), means, yerr=errors, capsize=5,
                      color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.3, label='GPT-4.1 baseline')
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'accuracy_comparison.png'}")


def plot_outcome_breakdown(gsm_analysis):
    """Pie/bar chart of debate outcome categories."""
    ocs = gsm_analysis["outcomes"]
    labels = list(ocs.keys())
    values = list(ocs.values())

    color_map = {
        "both_correct_stay": "#55A868",
        "positive_correction": "#4C72B0",
        "both_wrong_to_correct": "#1F77B4",
        "negative_persuasion": "#D62728",
        "both_wrong_stay": "#7F7F7F",
        "mixed": "#FF7F0E",
        "no_change": "#BCBD22",
    }
    colors = [color_map.get(l, "#999999") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.replace("_", " ").title() for l in labels])
    ax.set_xlabel("Number of Problems")
    ax.set_title("Cross-Model Debate Outcome Breakdown (GSM8K)")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "outcome_breakdown.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'outcome_breakdown.png'}")


def plot_robustness(rob_analysis):
    """Bar chart comparing accuracy drops on original vs rephrased problems."""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ["Single GPT-4.1", "Cross-model Debate"]
    orig = [rob_analysis["single_orig"], rob_analysis["debate_orig"]]
    rep = [rob_analysis["single_rep"], rob_analysis["debate_rep"]]

    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax.bar(x - w/2, orig, w, label='Original', color='#4C72B0', alpha=0.8)
    bars2 = ax.bar(x + w/2, rep, w, label='Rephrased', color='#DD8452', alpha=0.8)

    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness to Rephrasings (GSM8K)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add annotations for drops
    for i, (o, r) in enumerate(zip(orig, rep)):
        drop = o - r
        ax.annotate(f'Drop: {drop:.1%}', xy=(i, min(o, r) - 0.03),
                    ha='center', fontsize=10, color='red' if drop > 0.05 else 'green')

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "robustness_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'robustness_comparison.png'}")


def plot_error_correction_flow(gsm_rows):
    """Visualize how answers change through debate."""
    # Count transitions
    categories = {
        "Both correct → Both correct": 0,
        "One correct → Both correct": 0,
        "One correct → Both wrong": 0,
        "One correct → No change": 0,
        "Both wrong → One/both correct": 0,
        "Both wrong → Both wrong": 0,
    }

    for r in gsm_rows:
        r0a, r0b = r["cx_r0a"], r["cx_r0b"]
        r1a, r1b = r["cx_r1a"], r["cx_r1b"]

        if r0a and r0b:
            categories["Both correct → Both correct"] += 1
        elif (r0a or r0b) and not (r0a and r0b):
            if r1a and r1b:
                categories["One correct → Both correct"] += 1
            elif not r1a and not r1b:
                categories["One correct → Both wrong"] += 1
            else:
                categories["One correct → No change"] += 1
        else:  # both wrong
            if r1a or r1b:
                categories["Both wrong → One/both correct"] += 1
            else:
                categories["Both wrong → Both wrong"] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_map = {
        "Both correct → Both correct": "#55A868",
        "One correct → Both correct": "#4C72B0",
        "One correct → Both wrong": "#D62728",
        "One correct → No change": "#FF7F0E",
        "Both wrong → One/both correct": "#1F77B4",
        "Both wrong → Both wrong": "#7F7F7F",
    }
    labels = list(categories.keys())
    values = list(categories.values())
    colors = [colors_map.get(l, "#999999") for l in labels]

    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of Problems")
    ax.set_title("Error Correction Flow in Cross-Model Debate (GSM8K)")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "error_correction_flow.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'error_correction_flow.png'}")


def run_analysis():
    """Run full analysis pipeline."""
    R = load_results()

    gsm_analysis = analyze_gsm8k(R["gsm8k"])
    math_analysis = analyze_math500(R["math500"])
    rob_analysis = analyze_robustness(R["robustness"])

    # Plots
    plot_accuracy_comparison(gsm_analysis, math_analysis)
    plot_outcome_breakdown(gsm_analysis)
    plot_error_correction_flow(R["gsm8k"])
    if R["robustness"]:
        plot_robustness(rob_analysis)

    # Save analysis results
    analysis = {
        "gsm8k": gsm_analysis,
        "math500": math_analysis,
        "robustness": rob_analysis,
    }
    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))

    print(f"\nAnalysis saved to {RESULTS_DIR / 'analysis.json'}")
    return analysis


if __name__ == "__main__":
    run_analysis()
