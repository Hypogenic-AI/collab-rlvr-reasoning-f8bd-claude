"""
Main experiment runner for Collaborative RLVR Debate Study.

Experiments:
1. Cross-model debate (GPT-4.1 + Claude Sonnet): Different inductive biases
2. Same-model debate with temperature (GPT-4.1 temp=0.7): Diversity via sampling
3. Robustness to rephrasings
"""

import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from llm_client import call_llm
from answer_extraction import (
    extract_answer_from_response,
    extract_gold_answer_gsm8k,
    extract_gold_answer_math,
    answers_match,
    normalize_answer,
)
from prompts import *

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)

# Model configs
GPT41 = {"model": "gpt-4.1", "provider": "openai"}
GPT41_MINI = {"model": "gpt-4.1-mini", "provider": "openai"}
CLAUDE_SONNET = {"model": "anthropic/claude-sonnet-4", "provider": "openrouter"}


def load_gsm8k(n):
    ds = load_from_disk("../datasets/gsm8k")["test"]
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return [{
        "id": f"gsm8k_{i}", "question": ds[i]["question"],
        "gold_answer": extract_gold_answer_gsm8k(ds[i]["answer"]),
        "dataset": "gsm8k",
    } for i in indices]


def load_math500(n):
    ds = load_from_disk("../datasets/math500")["test"]
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return [{
        "id": f"math500_{i}", "question": ds[i]["problem"],
        "gold_answer": str(ds[i]["answer"]), "dataset": "math500",
        "level": ds[i].get("level", ""), "subject": ds[i].get("subject", ""),
    } for i in indices]


def solve(problem, model_cfg, temperature=0.0):
    """Single-agent solve."""
    is_gsm = problem["dataset"] == "gsm8k"
    tmpl = SINGLE_AGENT_COT if is_gsm else SINGLE_AGENT_COT_MATH
    prompt = tmpl.format(problem=problem["question"])
    resp = call_llm([{"role": "user", "content": prompt}],
                    model=model_cfg["model"], provider=model_cfg["provider"],
                    temperature=temperature)
    dtype = "gsm8k" if is_gsm else "math"
    ans = extract_answer_from_response(resp, dtype)
    correct = answers_match(ans, problem["gold_answer"], dtype)
    return {"response": resp, "answer": ans, "correct": correct}


def debate(problem, cfg_a, cfg_b, temp_a=0.0, temp_b=0.0):
    """Two-model debate: independent solve → share → critique → final."""
    is_gsm = problem["dataset"] == "gsm8k"
    ind_tmpl = DEBATE_INDEPENDENT if is_gsm else DEBATE_INDEPENDENT_MATH
    crit_tmpl = DEBATE_CRITIQUE if is_gsm else DEBATE_CRITIQUE_MATH
    dtype = "gsm8k" if is_gsm else "math"

    # Round 0: Independent
    prompt = ind_tmpl.format(problem=problem["question"])
    sol_a = call_llm([{"role": "user", "content": prompt}],
                     model=cfg_a["model"], provider=cfg_a["provider"], temperature=temp_a)
    sol_b = call_llm([{"role": "user", "content": prompt}],
                     model=cfg_b["model"], provider=cfg_b["provider"], temperature=temp_b)

    ans_a0 = extract_answer_from_response(sol_a, dtype)
    ans_b0 = extract_answer_from_response(sol_b, dtype)
    cor_a0 = answers_match(ans_a0, problem["gold_answer"], dtype)
    cor_b0 = answers_match(ans_b0, problem["gold_answer"], dtype)

    # Round 1: Critique
    cp_a = crit_tmpl.format(problem=problem["question"], own_solution=sol_a, other_solution=sol_b)
    cp_b = crit_tmpl.format(problem=problem["question"], own_solution=sol_b, other_solution=sol_a)

    rev_a = call_llm([{"role": "user", "content": cp_a}],
                     model=cfg_a["model"], provider=cfg_a["provider"], temperature=temp_a)
    rev_b = call_llm([{"role": "user", "content": cp_b}],
                     model=cfg_b["model"], provider=cfg_b["provider"], temperature=temp_b)

    ans_a1 = extract_answer_from_response(rev_a, dtype)
    ans_b1 = extract_answer_from_response(rev_b, dtype)
    cor_a1 = answers_match(ans_a1, problem["gold_answer"], dtype)
    cor_b1 = answers_match(ans_b1, problem["gold_answer"], dtype)

    return {
        "r0": {"a": {"resp": sol_a, "ans": ans_a0, "cor": cor_a0},
               "b": {"resp": sol_b, "ans": ans_b0, "cor": cor_b0}},
        "r1": {"a": {"resp": rev_a, "ans": ans_a1, "cor": cor_a1},
               "b": {"resp": rev_b, "ans": ans_b1, "cor": cor_b1}},
    }


def classify_outcome(r0a, r0b, r1a, r1b):
    """Classify debate outcome."""
    if r0a and r0b and r1a and r1b:
        return "both_correct_stay"
    if not r0a and not r0b and not r1a and not r1b:
        return "both_wrong_stay"

    corrections = (not r0a and r1a) + (not r0b and r1b)
    persuasions = (r0a and not r1a) + (r0b and not r1b)

    if corrections > 0 and persuasions == 0:
        if not r0a and not r0b:
            return "both_wrong_to_correct"
        return "positive_correction"
    if persuasions > 0 and corrections == 0:
        return "negative_persuasion"
    if corrections > 0 and persuasions > 0:
        return "mixed"
    return "no_change"


def self_consistency(problem, cfg, n=5):
    """Self-consistency: majority vote over n samples at temp=0.7."""
    answers = []
    for _ in range(n):
        r = solve(problem, cfg, temperature=0.7)
        answers.append(normalize_answer(r["answer"]))
    votes = Counter(a for a in answers if a is not None)
    if not votes:
        return {"answer": None, "correct": False, "votes": {}}
    best = votes.most_common(1)[0][0]
    dtype = "gsm8k" if problem["dataset"] == "gsm8k" else "math"
    return {"answer": best, "correct": answers_match(best, problem["gold_answer"], dtype), "votes": dict(votes)}


def rephrase(problem, cfg):
    """Rephrase a problem."""
    prompt = REPHRASE_PROMPT.format(problem=problem["question"])
    return call_llm([{"role": "user", "content": prompt}],
                    model=cfg["model"], provider=cfg["provider"], temperature=0.7).strip()


def save(data, name):
    def ser(o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)
    with open(RESULTS_DIR / name, "w") as f:
        json.dump(data, f, indent=2, default=ser)


def run_all():
    ts = datetime.now().isoformat()
    print(f"{'='*70}")
    print(f"Collaborative RLVR Debate Experiment — {ts}")
    print(f"{'='*70}")

    # Load data
    gsm_problems = load_gsm8k(100)
    math_problems = load_math500(50)
    print(f"Loaded {len(gsm_problems)} GSM8K, {len(math_problems)} MATH-500 problems")

    all_results = {"config": {"seed": SEED, "timestamp": ts,
                              "model_a": "gpt-4.1", "model_b": "anthropic/claude-sonnet-4"},
                   "gsm8k": [], "math500": [], "robustness": []}

    # ── Experiment 1: GSM8K ──
    print(f"\n{'='*70}")
    print("EXP 1A: GSM8K — Single-agent, Self-consistency, Cross-model Debate")
    print(f"{'='*70}")

    for i, p in enumerate(tqdm(gsm_problems, desc="GSM8K")):
        row = {"id": p["id"], "gold": p["gold_answer"]}

        # Baselines
        sa_gpt = solve(p, GPT41)
        sa_claude = solve(p, CLAUDE_SONNET)
        sc_gpt = self_consistency(p, GPT41, n=3)

        # Cross-model debate (GPT-4.1 + Claude)
        d_cross = debate(p, GPT41, CLAUDE_SONNET)
        oc_cross = classify_outcome(d_cross["r0"]["a"]["cor"], d_cross["r0"]["b"]["cor"],
                                    d_cross["r1"]["a"]["cor"], d_cross["r1"]["b"]["cor"])

        # Same-model debate with diversity (GPT-4.1 temp=0.7)
        d_same = debate(p, GPT41, GPT41, temp_a=0.0, temp_b=0.7)
        oc_same = classify_outcome(d_same["r0"]["a"]["cor"], d_same["r0"]["b"]["cor"],
                                   d_same["r1"]["a"]["cor"], d_same["r1"]["b"]["cor"])

        row.update({
            "single_gpt": sa_gpt["correct"],
            "single_claude": sa_claude["correct"],
            "sc_gpt3": sc_gpt["correct"],
            # Cross-model debate
            "cross_r0a": d_cross["r0"]["a"]["cor"], "cross_r0b": d_cross["r0"]["b"]["cor"],
            "cross_r1a": d_cross["r1"]["a"]["cor"], "cross_r1b": d_cross["r1"]["b"]["cor"],
            "cross_any_r1": d_cross["r1"]["a"]["cor"] or d_cross["r1"]["b"]["cor"],
            "cross_outcome": oc_cross,
            # Same-model debate
            "same_r0a": d_same["r0"]["a"]["cor"], "same_r0b": d_same["r0"]["b"]["cor"],
            "same_r1a": d_same["r1"]["a"]["cor"], "same_r1b": d_same["r1"]["b"]["cor"],
            "same_any_r1": d_same["r1"]["a"]["cor"] or d_same["r1"]["b"]["cor"],
            "same_outcome": oc_same,
        })
        # Store responses for qualitative analysis (first 10 only to save space)
        if i < 10:
            row["_responses"] = {
                "single_gpt_resp": sa_gpt["response"][:500],
                "single_claude_resp": sa_claude["response"][:500],
                "cross_debate_r0a": d_cross["r0"]["a"]["resp"][:500],
                "cross_debate_r0b": d_cross["r0"]["b"]["resp"][:500],
                "cross_debate_r1a": d_cross["r1"]["a"]["resp"][:500],
                "cross_debate_r1b": d_cross["r1"]["b"]["resp"][:500],
            }

        all_results["gsm8k"].append(row)

        if (i + 1) % 10 == 0:
            save(all_results, "all_results_incremental.json")
            # Print running stats
            rows = all_results["gsm8k"]
            n = len(rows)
            print(f"\n  [{n}] Single GPT: {sum(r['single_gpt'] for r in rows)/n:.1%} | "
                  f"Single Claude: {sum(r['single_claude'] for r in rows)/n:.1%} | "
                  f"SC(3): {sum(r['sc_gpt3'] for r in rows)/n:.1%} | "
                  f"Cross-debate: {sum(r['cross_any_r1'] for r in rows)/n:.1%} | "
                  f"Same-debate: {sum(r['same_any_r1'] for r in rows)/n:.1%}")

    # ── Experiment 1B: MATH-500 ──
    print(f"\n{'='*70}")
    print("EXP 1B: MATH-500 — Single-agent, Cross-model Debate")
    print(f"{'='*70}")

    for i, p in enumerate(tqdm(math_problems, desc="MATH-500")):
        row = {"id": p["id"], "gold": p["gold_answer"], "level": p.get("level", ""), "subject": p.get("subject", "")}

        sa_gpt = solve(p, GPT41)
        sa_claude = solve(p, CLAUDE_SONNET)
        d_cross = debate(p, GPT41, CLAUDE_SONNET)
        oc = classify_outcome(d_cross["r0"]["a"]["cor"], d_cross["r0"]["b"]["cor"],
                              d_cross["r1"]["a"]["cor"], d_cross["r1"]["b"]["cor"])

        row.update({
            "single_gpt": sa_gpt["correct"],
            "single_claude": sa_claude["correct"],
            "cross_r0a": d_cross["r0"]["a"]["cor"], "cross_r0b": d_cross["r0"]["b"]["cor"],
            "cross_r1a": d_cross["r1"]["a"]["cor"], "cross_r1b": d_cross["r1"]["b"]["cor"],
            "cross_any_r1": d_cross["r1"]["a"]["cor"] or d_cross["r1"]["b"]["cor"],
            "cross_outcome": oc,
        })

        if i < 5:
            row["_responses"] = {
                "single_gpt_resp": sa_gpt["response"][:500],
                "cross_debate_r0a": d_cross["r0"]["a"]["resp"][:500],
                "cross_debate_r0b": d_cross["r0"]["b"]["resp"][:500],
                "cross_debate_r1a": d_cross["r1"]["a"]["resp"][:500],
                "cross_debate_r1b": d_cross["r1"]["b"]["resp"][:500],
            }

        all_results["math500"].append(row)

        if (i + 1) % 10 == 0:
            save(all_results, "all_results_incremental.json")
            rows = all_results["math500"]
            n = len(rows)
            print(f"\n  [{n}] Single GPT: {sum(r['single_gpt'] for r in rows)/n:.1%} | "
                  f"Single Claude: {sum(r['single_claude'] for r in rows)/n:.1%} | "
                  f"Cross-debate: {sum(r['cross_any_r1'] for r in rows)/n:.1%}")

    # ── Experiment 2: Robustness ──
    print(f"\n{'='*70}")
    print("EXP 2: Robustness to Rephrasings (GSM8K, first 50)")
    print(f"{'='*70}")

    for i, p in enumerate(tqdm(gsm_problems[:50], desc="Robustness")):
        rep_text = rephrase(p, GPT41)
        p_rep = dict(p, question=rep_text, id=p["id"] + "_rep")

        sa_orig = solve(p, GPT41)
        sa_rep = solve(p_rep, GPT41)
        d_orig = debate(p, GPT41, CLAUDE_SONNET)
        d_rep = debate(p_rep, GPT41, CLAUDE_SONNET)

        all_results["robustness"].append({
            "id": p["id"], "gold": p["gold_answer"],
            "original_q": p["question"][:200],
            "rephrased_q": rep_text[:200],
            "single_orig": sa_orig["correct"],
            "single_rep": sa_rep["correct"],
            "debate_orig_any": d_orig["r1"]["a"]["cor"] or d_orig["r1"]["b"]["cor"],
            "debate_rep_any": d_rep["r1"]["a"]["cor"] or d_rep["r1"]["b"]["cor"],
        })

        if (i + 1) % 10 == 0:
            save(all_results, "all_results_incremental.json")
            rows = all_results["robustness"]
            n = len(rows)
            print(f"\n  [{n}] Single orig: {sum(r['single_orig'] for r in rows)/n:.1%} | "
                  f"Single rep: {sum(r['single_rep'] for r in rows)/n:.1%} | "
                  f"Debate orig: {sum(r['debate_orig_any'] for r in rows)/n:.1%} | "
                  f"Debate rep: {sum(r['debate_rep_any'] for r in rows)/n:.1%}")

    # Final save
    save(all_results, "all_results.json")
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to {RESULTS_DIR / 'all_results.json'}")
    print(f"{'='*70}")

    # Print final summary
    print_summary(all_results)


def print_summary(results):
    """Print a summary table."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for ds_name in ["gsm8k", "math500"]:
        rows = results[ds_name]
        if not rows:
            continue
        n = len(rows)
        print(f"\n{ds_name.upper()} (n={n}):")
        print(f"  Single-agent GPT-4.1:     {sum(r['single_gpt'] for r in rows)/n:.1%}")
        if "single_claude" in rows[0]:
            print(f"  Single-agent Claude:      {sum(r['single_claude'] for r in rows)/n:.1%}")
        if "sc_gpt3" in rows[0]:
            print(f"  Self-consistency(3):      {sum(r['sc_gpt3'] for r in rows)/n:.1%}")
        print(f"  Cross-model debate (any): {sum(r['cross_any_r1'] for r in rows)/n:.1%}")
        if "same_any_r1" in rows[0]:
            print(f"  Same-model debate (any):  {sum(r['same_any_r1'] for r in rows)/n:.1%}")

        # Outcome breakdown for cross-model debate
        outcomes = Counter(r["cross_outcome"] for r in rows)
        print(f"  Debate outcomes: {dict(outcomes)}")

    rows = results.get("robustness", [])
    if rows:
        n = len(rows)
        print(f"\nROBUSTNESS (n={n}):")
        print(f"  Single orig: {sum(r['single_orig'] for r in rows)/n:.1%}")
        print(f"  Single rep:  {sum(r['single_rep'] for r in rows)/n:.1%}")
        so = sum(r['single_orig'] for r in rows) / n
        sr = sum(r['single_rep'] for r in rows) / n
        print(f"  Single drop: {so - sr:.1%}")
        print(f"  Debate orig: {sum(r['debate_orig_any'] for r in rows)/n:.1%}")
        print(f"  Debate rep:  {sum(r['debate_rep_any'] for r in rows)/n:.1%}")
        do = sum(r['debate_orig_any'] for r in rows) / n
        dr = sum(r['debate_rep_any'] for r in rows) / n
        print(f"  Debate drop: {do - dr:.1%}")


if __name__ == "__main__":
    run_all()
