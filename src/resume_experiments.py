"""Resume experiments from saved checkpoint."""

import json
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
CLAUDE = {"model": "anthropic/claude-sonnet-4", "provider": "openrouter"}

executor = ThreadPoolExecutor(max_workers=4)  # reduced concurrency


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
    is_gsm = problem["dataset"] == "gsm8k"
    ind_t = DEBATE_INDEPENDENT if is_gsm else DEBATE_INDEPENDENT_MATH
    crit_t = DEBATE_CRITIQUE if is_gsm else DEBATE_CRITIQUE_MATH
    prompt_ind = ind_t.format(problem=problem["question"])

    # Round 0: Sequential to avoid rate limits
    sol_a = call_llm([{"role": "user", "content": prompt_ind}],
                     model=cfg_a["model"], provider=cfg_a["provider"], temperature=ta)
    sol_b = call_llm([{"role": "user", "content": prompt_ind}],
                     model=cfg_b["model"], provider=cfg_b["provider"], temperature=tb)

    ans_a0, cor_a0 = extract_and_check(sol_a, problem)
    ans_b0, cor_b0 = extract_and_check(sol_b, problem)

    # Round 1: Sequential
    cp_a = crit_t.format(problem=problem["question"], own_solution=sol_a, other_solution=sol_b)
    cp_b = crit_t.format(problem=problem["question"], own_solution=sol_b, other_solution=sol_a)

    rev_a = call_llm([{"role": "user", "content": cp_a}],
                     model=cfg_a["model"], provider=cfg_a["provider"], temperature=ta)
    rev_b = call_llm([{"role": "user", "content": cp_b}],
                     model=cfg_b["model"], provider=cfg_b["provider"], temperature=tb)

    ans_a1, cor_a1 = extract_and_check(rev_a, problem)
    ans_b1, cor_b1 = extract_and_check(rev_b, problem)

    return {
        "r0a_cor": cor_a0, "r0b_cor": cor_b0,
        "r1a_cor": cor_a1, "r1b_cor": cor_b1,
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
    # Load existing results
    with open(RESULTS_DIR / "all_results.json") as f:
        results = json.load(f)

    gsm_done = len(results["gsm8k"])
    math_done = len(results["math500"])
    rob_done = len(results["robustness"])
    print(f"Resuming: GSM8K={gsm_done}, MATH-500={math_done}, Robustness={rob_done}")

    # Recreate the same problem lists (same seed = same problems)
    gsm = load_gsm8k(80)
    math_p = load_math500(40)

    # ═══════════ Continue GSM8K ═══════════
    if gsm_done < len(gsm):
        print(f"\n{'='*60}")
        print(f"Continuing GSM8K from {gsm_done}")
        print(f"{'='*60}")

        for i in range(gsm_done, len(gsm)):
            p = gsm[i]
            t0 = time.time()
            row = {"id": p["id"], "gold": p["gold_answer"]}

            try:
                sa_gpt = solve_single(p, GPT41)
                sa_claude = solve_single(p, CLAUDE)
                sc = sc3(p, GPT41)
                d_cross = run_debate(p, GPT41, CLAUDE)
                oc_cross = classify(d_cross["r0a_cor"], d_cross["r0b_cor"],
                                    d_cross["r1a_cor"], d_cross["r1b_cor"])
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
                if i < 15:
                    row["_resp"] = {
                        "cx_r0a": d_cross["sol_a"], "cx_r0b": d_cross["sol_b"],
                        "cx_r1a": d_cross["rev_a"], "cx_r1b": d_cross["rev_b"],
                    }
            except Exception as e:
                print(f"  ERROR on problem {i}: {e}")
                time.sleep(30)
                continue

            results["gsm8k"].append(row)
            elapsed = time.time() - t0

            if (len(results["gsm8k"])) % 5 == 0:
                save(results, "all_results.json")
                rows = results["gsm8k"]
                n = len(rows)
                print(f"  [{n:3d}/{len(gsm)}] {elapsed:.1f}s | "
                      f"GPT:{sum(r['sg'] for r in rows)/n:.0%} "
                      f"Claude:{sum(r['sc'] for r in rows)/n:.0%} "
                      f"SC3:{sum(r['sc3'] for r in rows)/n:.0%} "
                      f"CxDebate:{sum(r['cx_any'] for r in rows)/n:.0%} "
                      f"SmDebate:{sum(r['sm_any'] for r in rows)/n:.0%}")

    save(results, "all_results.json")

    # ═══════════ MATH-500 ═══════════
    if math_done < len(math_p):
        print(f"\n{'='*60}")
        print(f"MATH-500 (from {math_done})")
        print(f"{'='*60}")

        for i in range(math_done, len(math_p)):
            p = math_p[i]
            t0 = time.time()
            row = {"id": p["id"], "gold": p["gold_answer"],
                   "level": p.get("level",""), "subject": p.get("subject","")}

            try:
                sa_gpt = solve_single(p, GPT41)
                sa_claude = solve_single(p, CLAUDE)
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
            except Exception as e:
                print(f"  ERROR on problem {i}: {e}")
                time.sleep(30)
                continue

            results["math500"].append(row)
            elapsed = time.time() - t0

            if (len(results["math500"])) % 5 == 0:
                save(results, "all_results.json")
                rows = results["math500"]
                n = len(rows)
                print(f"  [{n:3d}/{len(math_p)}] {elapsed:.1f}s | "
                      f"GPT:{sum(r['sg'] for r in rows)/n:.0%} "
                      f"Claude:{sum(r['sc'] for r in rows)/n:.0%} "
                      f"CxDebate:{sum(r['cx_any'] for r in rows)/n:.0%}")

    save(results, "all_results.json")

    # ═══════════ ROBUSTNESS ═══════════
    rob_target = 40
    if rob_done < rob_target:
        print(f"\n{'='*60}")
        print(f"ROBUSTNESS (from {rob_done})")
        print(f"{'='*60}")

        for i in range(rob_done, min(rob_target, len(gsm))):
            p = gsm[i]
            t0 = time.time()

            try:
                rep_text = rephrase_q(p, GPT41)
                p_rep = dict(p, question=rep_text, id=p["id"]+"_rep")

                sa_orig = solve_single(p, GPT41)
                sa_rep = solve_single(p_rep, GPT41)
                d_orig = run_debate(p, GPT41, CLAUDE)
                d_rep = run_debate(p_rep, GPT41, CLAUDE)

                results["robustness"].append({
                    "id": p["id"], "gold": p["gold_answer"],
                    "orig_q": p["question"][:200], "rep_q": rep_text[:200],
                    "s_orig": sa_orig["cor"], "s_rep": sa_rep["cor"],
                    "d_orig": d_orig["any_r1"], "d_rep": d_rep["any_r1"],
                })
            except Exception as e:
                print(f"  ERROR on problem {i}: {e}")
                time.sleep(30)
                continue

            elapsed = time.time() - t0

            if (len(results["robustness"])) % 5 == 0:
                save(results, "all_results.json")
                rows = results["robustness"]
                n = len(rows)
                print(f"  [{n:3d}/{rob_target}] {elapsed:.1f}s | "
                      f"S_orig:{sum(r['s_orig'] for r in rows)/n:.0%} "
                      f"S_rep:{sum(r['s_rep'] for r in rows)/n:.0%} "
                      f"D_orig:{sum(r['d_orig'] for r in rows)/n:.0%} "
                      f"D_rep:{sum(r['d_rep'] for r in rows)/n:.0%}")

    save(results, "all_results.json")
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*60}")

    # Print final summary
    for ds in ["gsm8k", "math500"]:
        rows = results[ds]
        if not rows: continue
        n = len(rows)
        print(f"\n{ds.upper()} (n={n}):")
        print(f"  GPT-4.1 single:    {sum(r['sg'] for r in rows)/n:.1%}")
        if "sc" in rows[0]:
            print(f"  Claude single:     {sum(r['sc'] for r in rows)/n:.1%}")
        if "sc3" in rows[0]:
            print(f"  Self-consist(3):   {sum(r['sc3'] for r in rows)/n:.1%}")
        print(f"  Cross debate (any):{sum(r['cx_any'] for r in rows)/n:.1%}")
        if "sm_any" in rows[0]:
            print(f"  Same debate (any): {sum(r['sm_any'] for r in rows)/n:.1%}")
        ocs = Counter(r["cx_oc"] for r in rows)
        print(f"  Outcomes: {dict(ocs)}")

    rows = results.get("robustness", [])
    if rows:
        n = len(rows)
        so = sum(r['s_orig'] for r in rows)/n
        sr = sum(r['s_rep'] for r in rows)/n
        do = sum(r['d_orig'] for r in rows)/n
        dr = sum(r['d_rep'] for r in rows)/n
        print(f"\nROBUSTNESS (n={n}):")
        print(f"  Single: orig={so:.1%} rep={sr:.1%} drop={so-sr:.1%}")
        print(f"  Debate: orig={do:.1%} rep={dr:.1%} drop={do-dr:.1%}")


if __name__ == "__main__":
    main()
