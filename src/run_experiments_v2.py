"""
Optimized experiment runner with concurrent API calls.
Uses asyncio + httpx for parallel requests.
"""

import asyncio
import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from datasets import load_from_disk

from llm_client import call_llm
from answer_extraction import (
    extract_answer_from_response,
    extract_gold_answer_gsm8k,
    answers_match,
    normalize_answer,
)
from prompts import *

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)

GPT41 = {"model": "gpt-4.1", "provider": "openai"}
GPT41_MINI = {"model": "gpt-4.1-mini", "provider": "openai"}
CLAUDE = {"model": "anthropic/claude-sonnet-4", "provider": "openrouter"}

executor = ThreadPoolExecutor(max_workers=6)


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


def make_prompt(problem, template_gsm, template_math):
    is_gsm = problem["dataset"] == "gsm8k"
    return (template_gsm if is_gsm else template_math).format(problem=problem["question"])


def extract_and_check(resp, problem):
    dtype = "gsm8k" if problem["dataset"] == "gsm8k" else "math"
    ans = extract_answer_from_response(resp, dtype)
    cor = answers_match(ans, problem["gold_answer"], dtype)
    return ans, cor


def solve_single(problem, cfg, temp=0.0):
    prompt = make_prompt(problem, SINGLE_AGENT_COT, SINGLE_AGENT_COT_MATH)
    resp = call_llm([{"role": "user", "content": prompt}],
                    model=cfg["model"], provider=cfg["provider"], temperature=temp)
    ans, cor = extract_and_check(resp, problem)
    return {"resp": resp, "ans": ans, "cor": cor}


def run_debate(problem, cfg_a, cfg_b, ta=0.0, tb=0.0):
    """Run debate with parallel API calls where possible."""
    is_gsm = problem["dataset"] == "gsm8k"
    ind_t = DEBATE_INDEPENDENT if is_gsm else DEBATE_INDEPENDENT_MATH
    crit_t = DEBATE_CRITIQUE if is_gsm else DEBATE_CRITIQUE_MATH
    prompt_ind = ind_t.format(problem=problem["question"])

    # Round 0: Both agents solve in parallel
    fut_a = executor.submit(call_llm, [{"role": "user", "content": prompt_ind}],
                            cfg_a["model"], ta, 2048, cfg_a["provider"])
    fut_b = executor.submit(call_llm, [{"role": "user", "content": prompt_ind}],
                            cfg_b["model"], tb, 2048, cfg_b["provider"])
    sol_a = fut_a.result()
    sol_b = fut_b.result()

    ans_a0, cor_a0 = extract_and_check(sol_a, problem)
    ans_b0, cor_b0 = extract_and_check(sol_b, problem)

    # Round 1: Both agents critique in parallel
    cp_a = crit_t.format(problem=problem["question"], own_solution=sol_a, other_solution=sol_b)
    cp_b = crit_t.format(problem=problem["question"], own_solution=sol_b, other_solution=sol_a)

    fut_ra = executor.submit(call_llm, [{"role": "user", "content": cp_a}],
                             cfg_a["model"], ta, 2048, cfg_a["provider"])
    fut_rb = executor.submit(call_llm, [{"role": "user", "content": cp_b}],
                             cfg_b["model"], tb, 2048, cfg_b["provider"])
    rev_a = fut_ra.result()
    rev_b = fut_rb.result()

    ans_a1, cor_a1 = extract_and_check(rev_a, problem)
    ans_b1, cor_b1 = extract_and_check(rev_b, problem)

    return {
        "r0a_cor": cor_a0, "r0b_cor": cor_b0,
        "r1a_cor": cor_a1, "r1b_cor": cor_b1,
        "r0a_ans": ans_a0, "r0b_ans": ans_b0,
        "r1a_ans": ans_a1, "r1b_ans": ans_b1,
        "any_r1": cor_a1 or cor_b1,
        "sol_a": sol_a[:600], "sol_b": sol_b[:600],
        "rev_a": rev_a[:600], "rev_b": rev_b[:600],
    }


def classify(r0a, r0b, r1a, r1b):
    if r0a and r0b and r1a and r1b: return "both_correct_stay"
    if not r0a and not r0b and not r1a and not r1b: return "both_wrong_stay"
    corr = (not r0a and r1a) + (not r0b and r1b)
    pers = (r0a and not r1a) + (r0b and not r1b)
    if corr > 0 and pers == 0:
        return "both_wrong_to_correct" if (not r0a and not r0b) else "positive_correction"
    if pers > 0 and corr == 0: return "negative_persuasion"
    if corr > 0 and pers > 0: return "mixed"
    return "no_change"


def sc3(problem, cfg):
    answers = []
    for _ in range(3):
        r = solve_single(problem, cfg, temp=0.7)
        answers.append(normalize_answer(r["ans"]))
    votes = Counter(a for a in answers if a is not None)
    if not votes: return False
    best = votes.most_common(1)[0][0]
    dtype = "gsm8k" if problem["dataset"] == "gsm8k" else "math"
    return answers_match(best, problem["gold_answer"], dtype)


def rephrase_q(problem, cfg):
    prompt = REPHRASE_PROMPT.format(problem=problem["question"])
    return call_llm([{"role": "user", "content": prompt}],
                    model=cfg["model"], provider=cfg["provider"], temperature=0.7).strip()


def save(data, name):
    def ser(o):
        if isinstance(o, (np.bool_, np.integer)): return int(o)
        if isinstance(o, np.floating): return float(o)
        return str(o)
    with open(RESULTS_DIR / name, "w") as f:
        json.dump(data, f, indent=2, default=ser)


def main():
    ts = datetime.now().isoformat()
    print(f"{'='*70}")
    print(f"Collaborative RLVR Debate — {ts}")
    print(f"{'='*70}")

    gsm = load_gsm8k(80)  # 80 GSM8K problems
    math_p = load_math500(40)  # 40 MATH-500 problems
    print(f"Loaded {len(gsm)} GSM8K, {len(math_p)} MATH-500")

    results = {"config": {"seed": SEED, "ts": ts,
                          "model_a": "gpt-4.1", "model_b": "claude-sonnet-4"},
               "gsm8k": [], "math500": [], "robustness": []}

    # ═══════════════ GSM8K ═══════════════
    print(f"\n{'='*60}")
    print("EXP 1: GSM8K")
    print(f"{'='*60}")

    for i, p in enumerate(gsm):
        t0 = time.time()
        row = {"id": p["id"], "gold": p["gold_answer"]}

        # Single-agent baselines (parallel: GPT + Claude)
        fut_gpt = executor.submit(solve_single, p, GPT41)
        fut_claude = executor.submit(solve_single, p, CLAUDE)
        sa_gpt = fut_gpt.result()
        sa_claude = fut_claude.result()

        # Self-consistency
        sc = sc3(p, GPT41)

        # Cross-model debate (GPT-4.1 vs Claude)
        d_cross = run_debate(p, GPT41, CLAUDE)
        oc_cross = classify(d_cross["r0a_cor"], d_cross["r0b_cor"],
                            d_cross["r1a_cor"], d_cross["r1b_cor"])

        # Same-model debate (GPT-4.1 temp=0 vs temp=0.7)
        d_same = run_debate(p, GPT41, GPT41, ta=0.0, tb=0.7)
        oc_same = classify(d_same["r0a_cor"], d_same["r0b_cor"],
                           d_same["r1a_cor"], d_same["r1b_cor"])

        row.update({
            "sg": sa_gpt["cor"], "sc": sa_claude["cor"], "sc3": sc,
            "cx_r0a": d_cross["r0a_cor"], "cx_r0b": d_cross["r0b_cor"],
            "cx_r1a": d_cross["r1a_cor"], "cx_r1b": d_cross["r1b_cor"],
            "cx_any": d_cross["any_r1"], "cx_oc": oc_cross,
            "sm_r0a": d_same["r0a_cor"], "sm_r0b": d_same["r0b_cor"],
            "sm_r1a": d_same["r1a_cor"], "sm_r1b": d_same["r1b_cor"],
            "sm_any": d_same["any_r1"], "sm_oc": oc_same,
        })

        # Store some responses
        if i < 15:
            row["_resp"] = {
                "gpt_r": sa_gpt["resp"][:500], "cl_r": sa_claude["resp"][:500],
                "cx_r0a": d_cross["sol_a"], "cx_r0b": d_cross["sol_b"],
                "cx_r1a": d_cross["rev_a"], "cx_r1b": d_cross["rev_b"],
            }

        results["gsm8k"].append(row)
        elapsed = time.time() - t0

        if (i + 1) % 5 == 0:
            save(results, "all_results.json")
            rows = results["gsm8k"]
            n = len(rows)
            print(f"  [{n:3d}/{len(gsm)}] {elapsed:.1f}s | "
                  f"GPT:{sum(r['sg'] for r in rows)/n:.0%} "
                  f"Claude:{sum(r['sc'] for r in rows)/n:.0%} "
                  f"SC3:{sum(r['sc3'] for r in rows)/n:.0%} "
                  f"CxDebate:{sum(r['cx_any'] for r in rows)/n:.0%} "
                  f"SmDebate:{sum(r['sm_any'] for r in rows)/n:.0%}")

    # ═══════════════ MATH-500 ═══════════════
    print(f"\n{'='*60}")
    print("EXP 2: MATH-500")
    print(f"{'='*60}")

    for i, p in enumerate(math_p):
        t0 = time.time()
        row = {"id": p["id"], "gold": p["gold_answer"],
               "level": p.get("level",""), "subject": p.get("subject","")}

        fut_gpt = executor.submit(solve_single, p, GPT41)
        fut_claude = executor.submit(solve_single, p, CLAUDE)
        sa_gpt = fut_gpt.result()
        sa_claude = fut_claude.result()

        d_cross = run_debate(p, GPT41, CLAUDE)
        oc = classify(d_cross["r0a_cor"], d_cross["r0b_cor"],
                      d_cross["r1a_cor"], d_cross["r1b_cor"])

        row.update({
            "sg": sa_gpt["cor"], "sc": sa_claude["cor"],
            "cx_r0a": d_cross["r0a_cor"], "cx_r0b": d_cross["r0b_cor"],
            "cx_r1a": d_cross["r1a_cor"], "cx_r1b": d_cross["r1b_cor"],
            "cx_any": d_cross["any_r1"], "cx_oc": oc,
        })

        if i < 5:
            row["_resp"] = {
                "cx_r0a": d_cross["sol_a"], "cx_r0b": d_cross["sol_b"],
                "cx_r1a": d_cross["rev_a"], "cx_r1b": d_cross["rev_b"],
            }

        results["math500"].append(row)
        elapsed = time.time() - t0

        if (i + 1) % 5 == 0:
            save(results, "all_results.json")
            rows = results["math500"]
            n = len(rows)
            print(f"  [{n:3d}/{len(math_p)}] {elapsed:.1f}s | "
                  f"GPT:{sum(r['sg'] for r in rows)/n:.0%} "
                  f"Claude:{sum(r['sc'] for r in rows)/n:.0%} "
                  f"CxDebate:{sum(r['cx_any'] for r in rows)/n:.0%}")

    # ═══════════════ ROBUSTNESS ═══════════════
    print(f"\n{'='*60}")
    print("EXP 3: Robustness (GSM8K rephrasings)")
    print(f"{'='*60}")

    for i, p in enumerate(gsm[:40]):
        t0 = time.time()
        rep_text = rephrase_q(p, GPT41)
        p_rep = dict(p, question=rep_text, id=p["id"]+"_rep")

        # Parallel: single orig + single rep + debate orig + debate rep
        fut_so = executor.submit(solve_single, p, GPT41)
        fut_sr = executor.submit(solve_single, p_rep, GPT41)
        sa_orig = fut_so.result()
        sa_rep = fut_sr.result()

        d_orig = run_debate(p, GPT41, CLAUDE)
        d_rep = run_debate(p_rep, GPT41, CLAUDE)

        results["robustness"].append({
            "id": p["id"], "gold": p["gold_answer"],
            "orig_q": p["question"][:200], "rep_q": rep_text[:200],
            "s_orig": sa_orig["cor"], "s_rep": sa_rep["cor"],
            "d_orig": d_orig["any_r1"], "d_rep": d_rep["any_r1"],
        })
        elapsed = time.time() - t0

        if (i + 1) % 5 == 0:
            save(results, "all_results.json")
            rows = results["robustness"]
            n = len(rows)
            print(f"  [{n:3d}/{len(gsm[:40])}] {elapsed:.1f}s | "
                  f"S_orig:{sum(r['s_orig'] for r in rows)/n:.0%} "
                  f"S_rep:{sum(r['s_rep'] for r in rows)/n:.0%} "
                  f"D_orig:{sum(r['d_orig'] for r in rows)/n:.0%} "
                  f"D_rep:{sum(r['d_rep'] for r in rows)/n:.0%}")

    save(results, "all_results.json")
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*60}")
    print_summary(results)
    return results


def print_summary(R):
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for ds in ["gsm8k", "math500"]:
        rows = R[ds]
        if not rows: continue
        n = len(rows)
        print(f"\n{ds.upper()} (n={n}):")
        print(f"  GPT-4.1 single:    {sum(r['sg'] for r in rows)/n:.1%} ({sum(r['sg'] for r in rows)}/{n})")
        if "sc" in rows[0]:
            print(f"  Claude single:     {sum(r['sc'] for r in rows)/n:.1%} ({sum(r['sc'] for r in rows)}/{n})")
        if "sc3" in rows[0]:
            print(f"  Self-consist(3):   {sum(r['sc3'] for r in rows)/n:.1%}")
        print(f"  Cross debate (any):{sum(r['cx_any'] for r in rows)/n:.1%} ({sum(r['cx_any'] for r in rows)}/{n})")
        if "sm_any" in rows[0]:
            print(f"  Same debate (any): {sum(r['sm_any'] for r in rows)/n:.1%} ({sum(r['sm_any'] for r in rows)}/{n})")

        ocs = Counter(r["cx_oc"] for r in rows)
        print(f"  Cross-debate outcomes: {dict(ocs)}")

    rows = R.get("robustness", [])
    if rows:
        n = len(rows)
        so = sum(r['s_orig'] for r in rows)/n
        sr = sum(r['s_rep'] for r in rows)/n
        do = sum(r['d_orig'] for r in rows)/n
        dr = sum(r['d_rep'] for r in rows)/n
        print(f"\nROBUSTNESS (n={n}):")
        print(f"  Single orig: {so:.1%}  rep: {sr:.1%}  drop: {so-sr:.1%}")
        print(f"  Debate orig: {do:.1%}  rep: {dr:.1%}  drop: {do-dr:.1%}")


if __name__ == "__main__":
    main()
