# Datasets

Datasets for the "Collaborative RLVR for Robust Reasoning" research project.

## Available Datasets

### GSM8K (Primary)
- **Location**: `gsm8k/`
- **Source**: `openai/gsm8k`
- **Splits**: Train (7,473) / Test (1,319)
- **Description**: Grade school math word problems requiring multi-step arithmetic reasoning. Each problem includes a detailed step-by-step solution and numerical final answer. Primary training dataset used by DeepSeek-R1 and MAPoRL.
- **Sample**: See `gsm8k_samples.json`

### MATH (Competition Level)
- **Location**: `math/`
- **Source**: `EleutherAI/hendrycks_math`
- **Splits**: Train (7,500) / Test (5,000)
- **Subjects**: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus
- **Description**: Competition-level mathematics problems. Harder than GSM8K, useful for evaluating generalization.
- **Sample**: See `math_samples.json`

### MATH-500 (Evaluation)
- **Location**: `math500/`
- **Source**: `HuggingFaceTB/MATH-500`
- **Splits**: Test (500)
- **Description**: Curated 500-problem subset of MATH commonly used for standardized evaluation in RLVR papers.

### ANLI (Cross-Domain)
- **Location**: `anli/`
- **Source**: `facebook/anli`
- **Splits**: Train/Dev/Test per round (R1, R2, R3)
- **Description**: Adversarial Natural Language Inference. Used by MAPoRL to test cross-domain transfer of collaboration skills.

## Download Script

Large data files are excluded from git. To re-download:

```python
from datasets import load_dataset

# GSM8K
load_dataset("openai/gsm8k", "main").save_to_disk("gsm8k")

# MATH (all subjects)
subjects = ["algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
for s in subjects:
    load_dataset("EleutherAI/hendrycks_math", s).save_to_disk(f"math/{s}")

# MATH-500
load_dataset("HuggingFaceTB/MATH-500").save_to_disk("math500")

# ANLI
load_dataset("facebook/anli").save_to_disk("anli")
```

## Git Policy

See `.gitignore` -- large data files are excluded. Only README, sample files (`*_samples.json`), and download scripts are tracked.
