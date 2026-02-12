"""Resume experiments using only OpenAI models (OpenRouter quota exceeded).
For remaining GSM8K: use GPT-4.1 + GPT-4.1-mini (cross-model)
For MATH-500 and robustness: use GPT-4.1 + GPT-4.1-mini
"""

import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

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

GPT41 = {"model": "gpt-4.1", "provider": "openai"}
GPT41_MINI = {"model": "gpt-4.1-mini", "provider": "openai"}


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


def solve_single(problem, cfg, temp=0.0):
    is_gsm = problem["dataset"] == "gsm8k"
    tmpl = SINGLE_AGENT_COT if is_gsm else SINGLE_AGENT_COT_MATH
    prompt = tmpl.format(problem=problem["question"])
    resp = call_llm([{"role": "user", "content": prompt}],
                    model=cfg["model"], provider=cfg["provider"], temperature=temp)
    dtype = "gsm8k" if is_gsm else "math"
    ans = extract_answer_from_response(resp, dtype)
    cor = answers_match(ans, problem["gold_answer"], dtype)
    return {"resp": resp, "ans": ans, "cor": cor}


def run_debate(problem, cfg_a, cfg_b, ta=0.0, tb=0.0):
    is_gsm = problem["dataset"] == "gsm8k"
    ind_t = DEBATE_INDEPENDENT if is_gsm else DEBATE_INDEPENDENT_MATH
    crit_t = DEBATE_CRITIQUE if is_gsm else DEBATE_CRITIQUE_MATH
    prompt_ind = ind_t.format(problem=problem["question"])

    sol_a = call_llm([{"role": "user", "content": prompt_ind}],
                     model=cfg_a["model"], provider=cfg_a["provider"], temperature=ta)
    sol_b = call_llm([{"role": "user", "content": prompt_ind}],
                     model=cfg_b["model"], provider=cfg_b["provider"], temperature=tb)

    dtype = "gsm8k" if is_gsm else "math"
    ans_a0 = extract_answer_from_response(sol_a, dtype)
    ans_b0 = extract_answer_from_response(sol_b, dtype)
    cor_a0 = answers_match(ans_a0, problem["gold_answer"], dtype)
    cor_b0 = answers_match(ans_b0, problem["gold_answer"], dtype)

    cp_a = crit_t.format(problem=problem["question"], own_solution=sol_a, other_solution=sol_b)
    cp_b = crit_t.format(problem=problem["question"], own_solution=sol_b, other_solution=sol_a)

    rev_a = call_llm([{"role": "user", "content": cp_a}],
                     model=cfg_a["model"], provider=cfg_a["provider"], temperature=ta)
    rev_b = call_llm([{"role": "user", "content": cp_b}],
                     model=cfg_b["model"], provider=cfg_b["provider"], temperature=tb)

    ans_a1 = extract_answer_from_response(rev_a, dtype)
    ans_b1 = extract_answer_from_response(rev_b, dtype)
    cor_a1 = answers_match(ans_a1, problem["gold_answer"], dtype)
    cor_b1 = answers_match(ans_b1, problem["gold_answer"], dtype)

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
    with open(RESULTS_DIR / "all_results.json") as f:
        results = json.load(f)

    # Update config note
    results["config"]["note"] = "First 40 GSM8K used Claude via OpenRouter; remaining use GPT-4.1-mini"
    results["config"]["model_b_remaining"] = "gpt-4.1-mini"

    gsm_done = len(results["gsm8k"])
    math_done = len(results["math500"])
    rob_done = len(results["robustness"])
    print(f"Resuming: GSM8K={gsm_done}, MATH-500={math_done}, Robustness={rob_done}")

    gsm = load_gsm8k(80)
    math_p = load_math500(40)

    # ═══════════ GSM8K (remaining) ═══════════
    if gsm_done < len(gsm):
        print(f"\nGSM8K from {gsm_done} (using GPT-4.1 + GPT-4.1-mini)")
        for i in range(gsm_done, len(gsm)):
            p = gsm[i]
            t0 = time.time()
            row = {"id": p["id"], "gold": p["gold_answer"]}

            try:
                sa_gpt = solve_single(p, GPT41)
                sa_mini = solve_single(p, GPT41_MINI)
                sc = sc3(p, GPT41)

                d_cross = run_debate(p, GPT41, GPT41_MINI)
                oc_cross = classify(d_cross["r0a_cor"], d_cross["r0b_cor"],
                                    d_cross["r1a_cor"], d_cross["r1b_cor"])

                d_same = run_debate(p, GPT41, GPT41, ta=0.0, tb=0.7)
                oc_same = classify(d_same["r0a_cor"], d_same["r0b_cor"],
                                   d_same["r1a_cor"], d_same["r1b_cor"])

                row.update({
                    "sg": sa_gpt["cor"], "sc": sa_mini["cor"], "sc3": sc,
                    "cx_r0a": d_cross["r0a_cor"], "cx_r0b": d_cross["r0b_cor"],
                    "cx_r1a": d_cross["r1a_cor"], "cx_r1b": d_cross["r1b_cor"],
                    "cx_any": d_cross["any_r1"], "cx_oc": oc_cross,
                    "sm_r0a": d_same["r0a_cor"], "sm_r0b": d_same["r0b_cor"],
                    "sm_r1a": d_same["r1a_cor"], "sm_r1b": d_same["r1b_cor"],
                    "sm_any": d_same["any_r1"], "sm_oc": oc_same,
                })
                if i < 55:
                    row["_resp"] = {
                        "cx_r0a": d_cross["sol_a"], "cx_r0b": d_cross["sol_b"],
                        "cx_r1a": d_cross["rev_a"], "cx_r1b": d_cross["rev_b"],
                    }
            except Exception as e:
                print(f"  ERROR problem {i}: {e}")
                time.sleep(15)
                continue

            results["gsm8k"].append(row)
            n = len(results["gsm8k"])
            if n % 5 == 0:
                save(results, "all_results.json")
                rows = results["gsm8k"]
                print(f"  [{n:3d}/{len(gsm)}] {time.time()-t0:.1f}s | "
                      f"GPT:{sum(r['sg'] for r in rows)/n:.0%} "
                      f"Model2:{sum(r['sc'] for r in rows)/n:.0%} "
                      f"SC3:{sum(r['sc3'] for r in rows)/n:.0%} "
                      f"CxDebate:{sum(r['cx_any'] for r in rows)/n:.0%} "
                      f"SmDebate:{sum(r['sm_any'] for r in rows)/n:.0%}")

    save(results, "all_results.json")

    # ═══════════ MATH-500 ═══════════
    if math_done < len(math_p):
        print(f"\nMATH-500 (using GPT-4.1 + GPT-4.1-mini)")
        for i in range(math_done, len(math_p)):
            p = math_p[i]
            t0 = time.time()
            row = {"id": p["id"], "gold": p["gold_answer"],
                   "level": p.get("level",""), "subject": p.get("subject","")}

            try:
                sa_gpt = solve_single(p, GPT41)
                sa_mini = solve_single(p, GPT41_MINI)
                d_cross = run_debate(p, GPT41, GPT41_MINI)
                oc = classify(d_cross["r0a_cor"], d_cross["r0b_cor"],
                              d_cross["r1a_cor"], d_cross["r1b_cor"])

                row.update({
                    "sg": sa_gpt["cor"], "sc": sa_mini["cor"],
                    "cx_r0a": d_cross["r0a_cor"], "cx_r0b": d_cross["r0b_cor"],
                    "cx_r1a": d_cross["r1a_cor"], "cx_r1b": d_cross["r1b_cor"],
                    "cx_any": d_cross["any_r1"], "cx_oc": oc,
                })
            except Exception as e:
                print(f"  ERROR problem {i}: {e}")
                time.sleep(15)
                continue

            results["math500"].append(row)
            n = len(results["math500"])
            if n % 5 == 0:
                save(results, "all_results.json")
                rows = results["math500"]
                print(f"  [{n:3d}/{len(math_p)}] {time.time()-t0:.1f}s | "
                      f"GPT:{sum(r['sg'] for r in rows)/n:.0%} "
                      f"Mini:{sum(r['sc'] for r in rows)/n:.0%} "
                      f"CxDebate:{sum(r['cx_any'] for r in rows)/n:.0%}")

    save(results, "all_results.json")

    # ═══════════ ROBUSTNESS ═══════════
    rob_target = 40
    if rob_done < rob_target:
        print(f"\nROBUSTNESS (using GPT-4.1 + GPT-4.1-mini)")
        for i in range(rob_done, min(rob_target, len(gsm))):
            p = gsm[i]
            t0 = time.time()

            try:
                rep_text = rephrase_q(p, GPT41)
                p_rep = dict(p, question=rep_text, id=p["id"]+"_rep")

                sa_orig = solve_single(p, GPT41)
                sa_rep = solve_single(p_rep, GPT41)
                d_orig = run_debate(p, GPT41, GPT41_MINI)
                d_rep = run_debate(p_rep, GPT41, GPT41_MINI)

                results["robustness"].append({
                    "id": p["id"], "gold": p["gold_answer"],
                    "orig_q": p["question"][:200], "rep_q": rep_text[:200],
                    "s_orig": sa_orig["cor"], "s_rep": sa_rep["cor"],
                    "d_orig": d_orig["any_r1"], "d_rep": d_rep["any_r1"],
                })
            except Exception as e:
                print(f"  ERROR problem {i}: {e}")
                time.sleep(15)
                continue

            n = len(results["robustness"])
            if n % 5 == 0:
                save(results, "all_results.json")
                rows = results["robustness"]
                print(f"  [{n:3d}/{rob_target}] {time.time()-t0:.1f}s | "
                      f"S_orig:{sum(r['s_orig'] for r in rows)/n:.0%} "
                      f"S_rep:{sum(r['s_rep'] for r in rows)/n:.0%} "
                      f"D_orig:{sum(r['d_orig'] for r in rows)/n:.0%} "
                      f"D_rep:{sum(r['d_rep'] for r in rows)/n:.0%}")

    save(results, "all_results.json")
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")

    # Summary
    for ds in ["gsm8k", "math500"]:
        rows = results[ds]
        if not rows: continue
        n = len(rows)
        print(f"\n{ds.upper()} (n={n}):")
        print(f"  GPT-4.1 single:    {sum(r['sg'] for r in rows)/n:.1%}")
        print(f"  Model-B single:    {sum(r['sc'] for r in rows)/n:.1%}")
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
        so, sr = sum(r['s_orig'] for r in rows)/n, sum(r['s_rep'] for r in rows)/n
        do, dr = sum(r['d_orig'] for r in rows)/n, sum(r['d_rep'] for r in rows)/n
        print(f"\nROBUSTNESS (n={n}):")
        print(f"  Single: orig={so:.1%} rep={sr:.1%} drop={so-sr:.1%}")
        print(f"  Debate: orig={do:.1%} rep={dr:.1%} drop={do-dr:.1%}")


if __name__ == "__main__":
    main()
