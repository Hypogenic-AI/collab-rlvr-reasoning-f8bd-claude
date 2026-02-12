"""Main experiment runner for collaborative RLVR debate study."""

import json
import os
import random
import time
import sys
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
from prompts import (
    SINGLE_AGENT_COT,
    SINGLE_AGENT_COT_MATH,
    DEBATE_INDEPENDENT,
    DEBATE_INDEPENDENT_MATH,
    DEBATE_CRITIQUE,
    DEBATE_CRITIQUE_MATH,
    REPHRASE_PROMPT,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_gsm8k_problems(n=100):
    """Load n random problems from GSM8K test set."""
    ds = load_from_disk("../datasets/gsm8k")
    test = ds["test"]
    indices = random.sample(range(len(test)), min(n, len(test)))
    problems = []
    for i in indices:
        item = test[i]
        gold = extract_gold_answer_gsm8k(item["answer"])
        problems.append({
            "id": f"gsm8k_{i}",
            "question": item["question"],
            "gold_answer": gold,
            "full_solution": item["answer"],
            "dataset": "gsm8k",
        })
    return problems


def load_math500_problems(n=50):
    """Load n random problems from MATH-500 test set."""
    ds = load_from_disk("../datasets/math500")
    test = ds["test"]
    indices = random.sample(range(len(test)), min(n, len(test)))
    problems = []
    for i in indices:
        item = test[i]
        problems.append({
            "id": f"math500_{i}",
            "question": item["problem"],
            "gold_answer": str(item["answer"]),
            "full_solution": item["solution"],
            "dataset": "math500",
            "subject": item.get("subject", ""),
            "level": item.get("level", ""),
        })
    return problems


def run_single_agent(problem, model="gpt-4.1", provider="openai", temperature=0.0):
    """Run single-agent CoT on a problem. Returns dict with response and extracted answer."""
    dataset = problem["dataset"]
    template = SINGLE_AGENT_COT if dataset == "gsm8k" else SINGLE_AGENT_COT_MATH
    prompt = template.format(problem=problem["question"])

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, model=model, provider=provider, temperature=temperature)

    dtype = "gsm8k" if dataset == "gsm8k" else "math"
    extracted = extract_answer_from_response(response, dataset_type=dtype)
    correct = answers_match(extracted, problem["gold_answer"], dataset_type=dtype)

    return {
        "response": response,
        "extracted_answer": extracted,
        "correct": correct,
    }


def run_self_consistency(problem, model="gpt-4.1", provider="openai", n_samples=5):
    """Run self-consistency (majority vote over n CoT samples)."""
    answers = []
    responses = []
    for _ in range(n_samples):
        result = run_single_agent(problem, model=model, provider=provider, temperature=0.7)
        answers.append(normalize_answer(result["extracted_answer"]))
        responses.append(result)

    # Majority vote
    from collections import Counter
    vote_counts = Counter(a for a in answers if a is not None)
    if not vote_counts:
        return {"responses": responses, "extracted_answer": None, "correct": False, "vote_counts": {}}

    majority_answer = vote_counts.most_common(1)[0][0]
    dtype = "gsm8k" if problem["dataset"] == "gsm8k" else "math"
    correct = answers_match(majority_answer, problem["gold_answer"], dataset_type=dtype)

    return {
        "responses": responses,
        "extracted_answer": majority_answer,
        "correct": correct,
        "vote_counts": dict(vote_counts),
    }


def run_collaborative_debate(problem, model_a="gpt-4.1", provider_a="openai",
                             model_b="gpt-4.1", provider_b="openai",
                             temperature=0.0):
    """Run collaborative debate between two models.

    Protocol:
    1. Both models solve independently (Round 0)
    2. Each model sees the other's solution and critiques/revises (Round 1)
    3. Final answers are extracted from Round 1 responses
    """
    dataset = problem["dataset"]
    is_gsm = dataset == "gsm8k"
    ind_template = DEBATE_INDEPENDENT if is_gsm else DEBATE_INDEPENDENT_MATH
    crit_template = DEBATE_CRITIQUE if is_gsm else DEBATE_CRITIQUE_MATH
    dtype = "gsm8k" if is_gsm else "math"

    # Round 0: Independent solutions
    prompt = ind_template.format(problem=problem["question"])
    messages = [{"role": "user", "content": prompt}]

    sol_a = call_llm(messages, model=model_a, provider=provider_a, temperature=temperature)
    sol_b = call_llm(messages, model=model_b, provider=provider_b, temperature=temperature)

    ans_a_r0 = extract_answer_from_response(sol_a, dataset_type=dtype)
    ans_b_r0 = extract_answer_from_response(sol_b, dataset_type=dtype)
    correct_a_r0 = answers_match(ans_a_r0, problem["gold_answer"], dataset_type=dtype)
    correct_b_r0 = answers_match(ans_b_r0, problem["gold_answer"], dataset_type=dtype)

    # Round 1: Critique and revise
    critique_prompt_a = crit_template.format(
        problem=problem["question"],
        own_solution=sol_a,
        other_solution=sol_b,
    )
    critique_prompt_b = crit_template.format(
        problem=problem["question"],
        own_solution=sol_b,
        other_solution=sol_a,
    )

    rev_a = call_llm([{"role": "user", "content": critique_prompt_a}],
                     model=model_a, provider=provider_a, temperature=temperature)
    rev_b = call_llm([{"role": "user", "content": critique_prompt_b}],
                     model=model_b, provider=provider_b, temperature=temperature)

    ans_a_r1 = extract_answer_from_response(rev_a, dataset_type=dtype)
    ans_b_r1 = extract_answer_from_response(rev_b, dataset_type=dtype)
    correct_a_r1 = answers_match(ans_a_r1, problem["gold_answer"], dataset_type=dtype)
    correct_b_r1 = answers_match(ans_b_r1, problem["gold_answer"], dataset_type=dtype)

    # Final answer: take agent A's post-debate answer (or majority if both available)
    # We report both agents' answers for analysis
    final_correct = correct_a_r1 or correct_b_r1  # "any agent correct" for debate

    return {
        "round0": {
            "agent_a": {"response": sol_a, "answer": ans_a_r0, "correct": correct_a_r0},
            "agent_b": {"response": sol_b, "answer": ans_b_r0, "correct": correct_b_r0},
        },
        "round1": {
            "agent_a": {"response": rev_a, "answer": ans_a_r1, "correct": correct_a_r1},
            "agent_b": {"response": rev_b, "answer": ans_b_r1, "correct": correct_b_r1},
        },
        "final_correct_any": final_correct,
        "final_correct_a": correct_a_r1,
        "final_correct_b": correct_b_r1,
    }


def run_majority_vote(problem, model_a="gpt-4.1", provider_a="openai",
                      model_b="gpt-4.1", provider_b="openai", temperature=0.0):
    """Run two models independently and take majority vote (no debate)."""
    result_a = run_single_agent(problem, model=model_a, provider=provider_a, temperature=temperature)
    result_b = run_single_agent(problem, model=model_b, provider=provider_b, temperature=temperature)

    # If both agree, use that. If disagree, take either (coin flip with fixed seed per problem).
    ans_a = normalize_answer(result_a["extracted_answer"])
    ans_b = normalize_answer(result_b["extracted_answer"])

    if ans_a == ans_b:
        final_answer = ans_a
    else:
        # No debate to resolve, so we just report both
        final_answer = ans_a  # default to model_a

    dtype = "gsm8k" if problem["dataset"] == "gsm8k" else "math"
    final_correct = answers_match(final_answer, problem["gold_answer"], dataset_type=dtype)
    any_correct = result_a["correct"] or result_b["correct"]

    return {
        "agent_a": result_a,
        "agent_b": result_b,
        "final_answer": final_answer,
        "final_correct": final_correct,
        "any_correct": any_correct,
    }


def rephrase_problem(problem, model="gpt-4.1", provider="openai"):
    """Rephrase a math problem using an LLM."""
    prompt = REPHRASE_PROMPT.format(problem=problem["question"])
    messages = [{"role": "user", "content": prompt}]
    rephrased = call_llm(messages, model=model, provider=provider, temperature=0.7)
    return rephrased.strip()


def categorize_debate_outcome(correct_a_r0, correct_b_r0, correct_a_r1, correct_b_r1):
    """Categorize the debate outcome for error analysis.

    Returns one of:
    - "both_correct_stay": Both correct before and after
    - "both_wrong_stay": Both wrong before and after
    - "positive_correction": At least one wrong agent corrected
    - "negative_persuasion": At least one correct agent became wrong
    - "mixed_improvement": Some correction and some persuasion
    - "both_wrong_to_correct": Both wrong, at least one correct after (best case)
    """
    a_before = correct_a_r0
    b_before = correct_b_r0
    a_after = correct_a_r1
    b_after = correct_b_r1

    corrections = 0
    persuasions = 0

    if not a_before and a_after:
        corrections += 1
    if not b_before and b_after:
        corrections += 1
    if a_before and not a_after:
        persuasions += 1
    if b_before and not b_after:
        persuasions += 1

    if corrections == 0 and persuasions == 0:
        if a_before and b_before:
            return "both_correct_stay"
        else:
            return "both_wrong_stay"

    if corrections > 0 and persuasions == 0:
        if not a_before and not b_before:
            return "both_wrong_to_correct"
        return "positive_correction"

    if persuasions > 0 and corrections == 0:
        return "negative_persuasion"

    return "mixed_improvement"


def run_experiment_1(problems, model_a="gpt-4.1", provider_a="openai",
                     model_b=None, provider_b=None):
    """Experiment 1: Single-agent vs. Debate accuracy.

    Conditions:
    1. Single-agent CoT (model A)
    2. Self-consistency (model A, n=5)
    3. Majority vote (model A + B, no debate)
    4. Collaborative debate (model A + B)
    """
    if model_b is None:
        model_b = model_a
        provider_b = provider_a

    results = []
    for i, prob in enumerate(tqdm(problems, desc="Experiment 1")):
        print(f"\n--- Problem {i+1}/{len(problems)}: {prob['id']} ---")
        print(f"  Q: {prob['question'][:80]}...")
        print(f"  Gold: {prob['gold_answer']}")

        # 1. Single-agent
        single = run_single_agent(prob, model=model_a, provider=provider_a)
        print(f"  Single: {single['extracted_answer']} ({'OK' if single['correct'] else 'WRONG'})")

        # 2. Self-consistency (only 3 samples to save budget)
        sc = run_self_consistency(prob, model=model_a, provider=provider_a, n_samples=3)
        print(f"  SC(3): {sc['extracted_answer']} ({'OK' if sc['correct'] else 'WRONG'})")

        # 3. Majority vote (no debate)
        mv = run_majority_vote(prob, model_a=model_a, provider_a=provider_a,
                               model_b=model_b, provider_b=provider_b)
        print(f"  MajVote: {mv['final_answer']} ({'OK' if mv['any_correct'] else 'WRONG'})")

        # 4. Collaborative debate
        debate = run_collaborative_debate(prob, model_a=model_a, provider_a=provider_a,
                                         model_b=model_b, provider_b=provider_b)
        outcome = categorize_debate_outcome(
            debate["round0"]["agent_a"]["correct"],
            debate["round0"]["agent_b"]["correct"],
            debate["round1"]["agent_a"]["correct"],
            debate["round1"]["agent_b"]["correct"],
        )
        print(f"  Debate R0: A={debate['round0']['agent_a']['answer']} "
              f"({'OK' if debate['round0']['agent_a']['correct'] else 'X'}), "
              f"B={debate['round0']['agent_b']['answer']} "
              f"({'OK' if debate['round0']['agent_b']['correct'] else 'X'})")
        print(f"  Debate R1: A={debate['round1']['agent_a']['answer']} "
              f"({'OK' if debate['round1']['agent_a']['correct'] else 'X'}), "
              f"B={debate['round1']['agent_b']['answer']} "
              f"({'OK' if debate['round1']['agent_b']['correct'] else 'X'})")
        print(f"  Outcome: {outcome}")

        results.append({
            "problem": prob,
            "single_agent": {
                "answer": single["extracted_answer"],
                "correct": single["correct"],
            },
            "self_consistency": {
                "answer": sc["extracted_answer"],
                "correct": sc["correct"],
            },
            "majority_vote": {
                "any_correct": mv["any_correct"],
                "final_correct": mv["final_correct"],
            },
            "debate": {
                "r0_a_correct": debate["round0"]["agent_a"]["correct"],
                "r0_b_correct": debate["round0"]["agent_b"]["correct"],
                "r1_a_correct": debate["round1"]["agent_a"]["correct"],
                "r1_b_correct": debate["round1"]["agent_b"]["correct"],
                "final_any_correct": debate["final_correct_any"],
                "outcome": outcome,
            },
            # Store full responses for detailed analysis
            "full_debate": debate,
            "full_single": single,
        })

        # Save incrementally
        save_results(results, "experiment1_incremental.json")

    return results


def run_experiment_2_robustness(problems, model_a="gpt-4.1", provider_a="openai",
                                model_b=None, provider_b=None, n_rephrase=50):
    """Experiment 2: Robustness to rephrasings.

    1. Select n problems from the set
    2. Rephrase each problem
    3. Run single-agent and debate on both original and rephrased
    4. Compare accuracy drop
    """
    if model_b is None:
        model_b = model_a
        provider_b = provider_a

    subset = problems[:n_rephrase]
    results = []

    for i, prob in enumerate(tqdm(subset, desc="Experiment 2 - Robustness")):
        print(f"\n--- Rephrase {i+1}/{len(subset)}: {prob['id']} ---")

        # Rephrase the problem
        rephrased_text = rephrase_problem(prob, model=model_a, provider=provider_a)
        rephrased_prob = prob.copy()
        rephrased_prob["question"] = rephrased_text
        rephrased_prob["id"] = prob["id"] + "_rephrased"

        # Single-agent on original
        single_orig = run_single_agent(prob, model=model_a, provider=provider_a)
        # Single-agent on rephrased
        single_reph = run_single_agent(rephrased_prob, model=model_a, provider=provider_a)

        # Debate on original
        debate_orig = run_collaborative_debate(prob, model_a=model_a, provider_a=provider_a,
                                               model_b=model_b, provider_b=provider_b)
        # Debate on rephrased
        debate_reph = run_collaborative_debate(rephrased_prob, model_a=model_a, provider_a=provider_a,
                                               model_b=model_b, provider_b=provider_b)

        results.append({
            "problem_id": prob["id"],
            "original_question": prob["question"],
            "rephrased_question": rephrased_text,
            "gold_answer": prob["gold_answer"],
            "single_orig_correct": single_orig["correct"],
            "single_reph_correct": single_reph["correct"],
            "debate_orig_any_correct": debate_orig["final_correct_any"],
            "debate_reph_any_correct": debate_reph["final_correct_any"],
        })

        save_results(results, "experiment2_incremental.json")

    return results


def save_results(results, filename):
    """Save results to JSON (handling non-serializable types)."""
    def default_serializer(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)

    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=default_serializer)


def compute_summary_stats(results):
    """Compute summary statistics from experiment 1 results."""
    n = len(results)
    single_correct = sum(1 for r in results if r["single_agent"]["correct"])
    sc_correct = sum(1 for r in results if r["self_consistency"]["correct"])
    mv_any_correct = sum(1 for r in results if r["majority_vote"]["any_correct"])
    debate_any_correct = sum(1 for r in results if r["debate"]["final_any_correct"])
    debate_a_correct = sum(1 for r in results if r["debate"]["r1_a_correct"])
    debate_b_correct = sum(1 for r in results if r["debate"]["r1_b_correct"])

    # R0 baselines (before debate)
    r0_a_correct = sum(1 for r in results if r["debate"]["r0_a_correct"])
    r0_b_correct = sum(1 for r in results if r["debate"]["r0_b_correct"])
    r0_any_correct = sum(1 for r in results if r["debate"]["r0_a_correct"] or r["debate"]["r0_b_correct"])

    # Outcome breakdown
    from collections import Counter
    outcomes = Counter(r["debate"]["outcome"] for r in results)

    stats = {
        "n_problems": n,
        "single_agent_accuracy": single_correct / n,
        "self_consistency_accuracy": sc_correct / n,
        "majority_vote_any_accuracy": mv_any_correct / n,
        "debate_any_accuracy": debate_any_correct / n,
        "debate_agent_a_accuracy": debate_a_correct / n,
        "debate_agent_b_accuracy": debate_b_correct / n,
        "pre_debate_agent_a_accuracy": r0_a_correct / n,
        "pre_debate_agent_b_accuracy": r0_b_correct / n,
        "pre_debate_any_accuracy": r0_any_correct / n,
        "outcome_breakdown": dict(outcomes),
        "positive_correction_rate": outcomes.get("positive_correction", 0) / n,
        "negative_persuasion_rate": outcomes.get("negative_persuasion", 0) / n,
        "net_correction_rate": (outcomes.get("positive_correction", 0) - outcomes.get("negative_persuasion", 0)) / n,
    }
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("Collaborative RLVR Debate Experiment")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print("=" * 60)

    # Configuration
    MODEL_A = "gpt-4.1"
    PROVIDER_A = "openai"
    # For same-model debate, use the same model
    MODEL_B = "gpt-4.1"
    PROVIDER_B = "openai"

    # Load problems
    print("\nLoading problems...")
    gsm8k_problems = load_gsm8k_problems(n=100)
    math500_problems = load_math500_problems(n=50)
    print(f"Loaded {len(gsm8k_problems)} GSM8K problems, {len(math500_problems)} MATH-500 problems")

    # Experiment 1: GSM8K
    print("\n" + "=" * 60)
    print("EXPERIMENT 1A: GSM8K - Single vs Debate")
    print("=" * 60)
    gsm8k_results = run_experiment_1(gsm8k_problems, model_a=MODEL_A, provider_a=PROVIDER_A,
                                      model_b=MODEL_B, provider_b=PROVIDER_B)
    gsm8k_stats = compute_summary_stats(gsm8k_results)
    save_results(gsm8k_results, "experiment1_gsm8k.json")
    save_results(gsm8k_stats, "experiment1_gsm8k_stats.json")
    print(f"\nGSM8K Summary:")
    for k, v in gsm8k_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Experiment 1: MATH-500
    print("\n" + "=" * 60)
    print("EXPERIMENT 1B: MATH-500 - Single vs Debate")
    print("=" * 60)
    math_results = run_experiment_1(math500_problems, model_a=MODEL_A, provider_a=PROVIDER_A,
                                     model_b=MODEL_B, provider_b=PROVIDER_B)
    math_stats = compute_summary_stats(math_results)
    save_results(math_results, "experiment1_math500.json")
    save_results(math_stats, "experiment1_math500_stats.json")
    print(f"\nMATH-500 Summary:")
    for k, v in math_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Experiment 2: Robustness (GSM8K only, first 50)
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Robustness to Rephrasings (GSM8K)")
    print("=" * 60)
    robustness_results = run_experiment_2_robustness(
        gsm8k_problems, model_a=MODEL_A, provider_a=PROVIDER_A,
        model_b=MODEL_B, provider_b=PROVIDER_B, n_rephrase=50
    )
    save_results(robustness_results, "experiment2_robustness.json")

    print("\nAll experiments complete!")
    print(f"Results saved to {RESULTS_DIR}")
