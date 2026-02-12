# Research Plan: Collaborative RLVR for Robust Reasoning

## Motivation & Novelty Assessment

### Why This Research Matters
RLVR is the dominant paradigm for training LLM reasoning (DeepSeek-R1, DAPO), but produces brittle behavior: models exploit dataset patterns, generate unfaithful reasoning chains, and fail under rephrasings or distribution shifts. If collaborative debate can force reasoning to be externalized and defended, it could produce fundamentally more robust reasoners — addressing a core limitation of the most important LLM training paradigm.

### Gap in Existing Work
The literature review identifies a clear gap: **no prior work combines RLVR training with multi-agent debate**. Existing work falls into two buckets:
1. **RLVR (single-agent)**: DeepSeek-R1, DAPO train single models with verifiable rewards — effective but brittle
2. **Multi-agent debate (inference-time)**: Du et al., ReConcile apply debate at inference time without training — improves accuracy but doesn't improve the underlying model

MAPoRL (closest work) uses PPO + learned verifier for multi-agent training, but differs fundamentally: it uses sequential turn-taking, learned rewards, and targets collaboration behavior rather than reasoning robustness.

### Our Novel Contribution
We test whether **inference-time collaborative debate between LLMs improves mathematical reasoning accuracy and robustness** compared to single-agent reasoning, using real state-of-the-art LLMs. This directly validates the hypothesis that externalizing reasoning through debate forces more robust problem-solving — the key premise that would justify investing in collaborative RLVR training.

### Experiment Justification
- **Experiment 1 (Single vs. Collaborative Accuracy)**: Establishes whether debate improves raw accuracy on GSM8K and MATH — the most basic prediction of the hypothesis
- **Experiment 2 (Robustness to Rephrasings)**: Tests whether debate specifically helps with rephrased problems where single-agent RLVR is known to be brittle
- **Experiment 3 (Error Correction Analysis)**: Analyzes whether debate enables genuine error correction (agents changing wrong answers to correct ones) vs. mere agreement amplification
- **Experiment 4 (Cross-Model Debate)**: Tests whether different models debating (different inductive biases) produce better outcomes than same-model debate

---

## Research Question
Does collaborative debate between LLMs produce more accurate and robust mathematical reasoning compared to single-agent reasoning, and does this improvement come from genuine error correction rather than simple agreement?

## Hypothesis Decomposition
1. **H1 (Accuracy)**: Two models debating before answering achieve higher accuracy than either model alone on GSM8K and MATH
2. **H2 (Error Correction)**: Debate enables genuine error correction — agents change wrong answers to correct ones, not just correct-to-wrong
3. **H3 (Robustness)**: Debate provides larger accuracy gains on rephrased/modified problems than on original problems (suggesting it counters pattern exploitation)
4. **H4 (Diversity Helps)**: Cross-model debate (models with different biases) produces better outcomes than same-model debate

## Proposed Methodology

### Approach
We use real LLM APIs to conduct a controlled study of collaborative debate vs. single-agent reasoning on mathematical problems. Each model independently solves a problem, then they share solutions and critique each other in 1-2 debate rounds, then produce a final answer. We compare accuracy across conditions.

**Why this approach**: Rather than training RLVR from scratch (which would require massive compute), we validate the core premise empirically: does debate improve reasoning? This is the necessary prerequisite for investing in collaborative RLVR training.

### Models
- **Primary**: GPT-4.1 (via OpenAI API) — strong mathematical reasoner
- **Secondary**: Claude Sonnet 4.5 (via OpenRouter) or another model — different architecture/training for cross-model debate
- **Budget model**: GPT-4.1-mini or similar — for same-model debate baseline

### Experimental Steps

1. **Data Preparation**: Sample 100 problems from GSM8K test set and 50 from MATH-500 for core experiments. Create 50 rephrased versions of GSM8K problems for robustness testing.

2. **Baseline (Single-Agent)**: Each model solves each problem independently with chain-of-thought prompting. Record accuracy and reasoning chains. Run 3 times with different seeds.

3. **Collaborative Debate**:
   - Both models solve independently (Round 0)
   - Share solutions with each other (Round 1 debate)
   - Each model critiques the other's solution and revises its answer
   - Record final answers and all intermediate reasoning

4. **Robustness Test**: Run both conditions on rephrased problems. Compare accuracy drop between single-agent and debate conditions.

5. **Error Correction Analysis**: Categorize all debate outcomes:
   - Both initially correct → still correct (agreement)
   - Both initially wrong → still wrong (shared failure)
   - One correct, one wrong → final answer correct (positive correction)
   - One correct, one wrong → final answer wrong (negative persuasion)
   - Both wrong → one or both correct (genuine debate benefit)

### Baselines
1. **Single-agent CoT**: Standard chain-of-thought, one model
2. **Self-consistency (SC)**: Same model, multiple samples, majority vote (n=5)
3. **Majority voting**: Two models vote independently (no debate)

### Evaluation Metrics
- **Accuracy**: Percentage of problems solved correctly (primary)
- **Error correction rate**: Fraction of initially-wrong answers corrected through debate
- **Negative persuasion rate**: Fraction of initially-correct answers changed to wrong
- **Net correction**: Error correction rate minus negative persuasion rate
- **Robustness delta**: Accuracy drop on rephrased problems (lower is better)

### Statistical Analysis Plan
- McNemar's test for paired accuracy comparisons (same problems, different conditions)
- Bootstrap confidence intervals (n=1000) for all metrics
- Effect sizes via odds ratios
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- **Support H1**: Debate accuracy > single-agent accuracy by 3-8% on GSM8K, more on MATH
- **Support H2**: Net correction rate > 0 (more errors corrected than introduced)
- **Support H3**: Accuracy drop on rephrased problems smaller for debate (< 50% of single-agent drop)
- **Support H4**: Cross-model debate outperforms same-model debate

Refutation would be: debate shows no improvement, or negative persuasion dominates error correction.

## Timeline and Milestones
- Environment setup & data prep: 15 min
- Implement experiment framework: 30 min
- Run Experiment 1 (accuracy): 30 min
- Run Experiment 2 (robustness): 20 min
- Run Experiment 3 (error correction): 10 min (analysis of Exp 1 data)
- Run Experiment 4 (cross-model): 20 min
- Analysis & visualization: 30 min
- Documentation: 30 min
- Buffer: 30 min

## Potential Challenges
1. **API rate limits**: Mitigate by batching requests and using exponential backoff
2. **Cost**: ~200 problems × 3 conditions × 2 models × ~1000 tokens = ~$20-50 budget
3. **Rephrasing quality**: Use GPT to rephrase, then manually validate a sample
4. **Answer extraction**: Math answers need careful parsing; use regex for GSM8K #### format and MATH \boxed{} format
5. **Model agreement**: Models might always agree, producing trivial debate. Monitor agreement rates.

## Success Criteria
1. Complete experiments with ≥100 GSM8K and ≥50 MATH problems
2. Statistical tests with p < 0.05 for at least one hypothesis
3. Clear error correction analysis with categorized outcomes
4. All results reproducible with documented code and seeds
