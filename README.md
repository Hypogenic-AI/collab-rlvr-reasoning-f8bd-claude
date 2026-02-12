# Collaborative RLVR for Robust Reasoning

Empirical investigation of whether collaborative debate between LLMs improves mathematical reasoning, testing the premise that underlies collaborative RLVR.

## Key Findings

- **+7.5% accuracy on MATH-500** (77.5% → 85.0%) from cross-model debate vs. best single agent
- **Genuine error correction**: 33% of cases where both models independently failed were corrected through debate
- **Most effective on harder problems**: Level 4: +13%, Level 5: +11%
- **Minimal benefit on easy problems** (GSM8K: +1.3%, models already at 92%+)
- **No robustness benefit** to problem rephrasings (equal accuracy drop for single vs. debate)
- **Negative persuasion is a real risk**: 3 cases on GSM8K where debate caused correct→wrong switches

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai httpx numpy scipy matplotlib seaborn pandas datasets tqdm

# Set API keys
export OPENAI_API_KEY=your_key

# Run experiments
cd src && python run_experiments_v2.py

# Run analysis
python analysis.py
```

## File Structure

```
├── REPORT.md              # Full research report with results
├── README.md              # This file
├── planning.md            # Research plan
├── literature_review.md   # Literature review
├── resources.md           # Resource catalog
├── src/
│   ├── llm_client.py           # LLM API client (OpenAI + OpenRouter)
│   ├── answer_extraction.py    # Answer parsing for GSM8K/MATH
│   ├── prompts.py              # Prompt templates
│   ├── run_experiments_v2.py   # Main experiment runner
│   ├── resume_v2.py            # Resume script (handles API limits)
│   └── analysis.py             # Statistical analysis and plotting
├── results/
│   ├── all_results.json        # All experimental results
│   ├── analysis.json           # Analysis outputs
│   └── plots/                  # Visualizations
│       ├── accuracy_comparison.png
│       ├── summary_comparison.png
│       ├── outcome_breakdown.png
│       ├── math_by_level.png
│       ├── math_answer_flow.png
│       ├── error_correction_flow.png
│       └── robustness_comparison.png
├── datasets/                   # Downloaded datasets (GSM8K, MATH-500, ANLI)
├── papers/                     # Reference papers (PDFs)
└── code/                       # Reference codebases (MAPoRL, verl, DAPO, reasoning-gym)
```

## Models Used

| Model | Provider | Role |
|-------|----------|------|
| GPT-4.1 | OpenAI | Primary agent, baselines |
| GPT-4.1-mini | OpenAI | Cross-model debate partner |
| Claude Sonnet 4 | OpenRouter | Cross-model debate (first 40 GSM8K) |

## See Also

- [REPORT.md](REPORT.md) — Full research report with detailed analysis
- [planning.md](planning.md) — Original research plan
- [literature_review.md](literature_review.md) — Comprehensive literature review
