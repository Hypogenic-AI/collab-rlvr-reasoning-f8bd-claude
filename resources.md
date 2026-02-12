# Resources Catalog: Collaborative RLVR for Robust Reasoning

## Papers

All papers are stored in `papers/` as PDF files. Chunked versions for detailed reading are in `papers/pages/`.

### Core RLVR Papers
| File | Title | Authors | Year | Notes |
|------|-------|---------|------|-------|
| `2501.12948_deepseek_r1.pdf` | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | Guo et al. | 2025 | Foundational RLVR paper; GRPO + R1-Zero |
| `2402.03300_deepseekmath_grpo.pdf` | DeepSeekMath: Pushing the Limits of Mathematical Reasoning | Shao et al. | 2024 | Introduces GRPO algorithm |
| `2503.14476_dapo.pdf` | DAPO: Open-Source LLM Reinforcement Learning System | Yu et al. | 2025 | Open-source RLVR at scale |
| `2503.20783_understanding_r1_zero.pdf` | Understanding R1-Zero-Like Training | Zhao et al. | 2025 | Systematic analysis of R1-Zero training |
| `2503.13555_grpo_analysis.pdf` | GRPO Analysis | Zhong et al. | 2025 | Theoretical GRPO analysis |

### Critical RLVR Analysis
| File | Title | Authors | Year | Notes |
|------|-------|---------|------|-------|
| `2504.13837_does_rl_incentivize_reasoning.pdf` | Does RL Really Incentivize Reasoning? | Yue et al. | 2025 | Critical study on what RLVR learns |
| `2506.03691_spurious_rewards.pdf` | Spurious Rewards: Rethinking Training Signals | Chen et al. | 2025 | RLVR works with imperfect rewards |
| `2503.09476_negative_reinforcement.pdf` | Surprising Effectiveness of Negative Reinforcement | Liu et al. | 2025 | Negative-only rewards effective |
| `2505.07532_rlvr_incentivizes_reasoning.pdf` | RLVR Implicitly Incentivizes Correct Reasoning | Liu et al. | 2025 | Counter-argument: RLVR does help reasoning |
| `2505.05551_entropy_mechanism_rl.pdf` | Entropy Mechanism of RL for Reasoning LLMs | Zhu et al. | 2025 | Entropy dynamics analysis |

### Multi-Agent Debate
| File | Title | Authors | Year | Notes |
|------|-------|---------|------|-------|
| `2305.14325_multiagent_debate.pdf` | Improving Factuality and Reasoning through Multiagent Debate | Du et al. | 2023 | Seminal debate paper |
| `2305.19118_mad_divergent_thinking.pdf` | Encouraging Divergent Thinking through MAD | Liang et al. | 2023 | Debate for divergent thinking |
| `2309.13007_reconcile.pdf` | ReConcile: Round-Table Conference for Reasoning | Chen et al. | 2023 | Confidence-weighted discussion |
| `2312.01823_exchange_of_thought.pdf` | Exchange-of-Thought: Cross-Model Communication | Yin et al. | 2023 | Structured reasoning exchange |
| `2311.17371_going_mad.pdf` | Should We Be Going MAD? | Smit et al. | 2023 | Benchmarking debate strategies |
| `2503.17510_talk_isnt_cheap.pdf` | Talk Isn't Always Cheap | Wu et al. | 2025 | Failure modes in debate |
| `2402.18272_rethinking_bounds_mad.pdf` | Rethinking Bounds of LLM Reasoning | Xu et al. | 2024 | Multi-agent vs single-model limits |
| `2402.18176_multi_llm_debate_framework.pdf` | Multi-LLM Debate Framework | Zhao et al. | 2024 | Theoretical convergence analysis |

### Collaborative/Multi-Agent Systems
| File | Title | Authors | Year | Notes |
|------|-------|---------|------|-------|
| `2502.18439_maporl.pdf` | MAPoRL: Multi-Agent Post-Co-Training with RL | Li et al. | 2025 | **Closest prior work** |
| `2310.02124_collaboration_social_psych.pdf` | Exploring Collaboration Mechanisms (Social Psych) | Zhang et al. | 2023 | Social psychology frameworks |
| `2501.06322_multi_agent_collab_survey.pdf` | Multi-Agent Collaboration Mechanisms Survey | Yang et al. | 2025 | Comprehensive survey |
| `2506.01773_scalable_oversight_mad.pdf` | Scalable Oversight with Collaborative MAD | Wang et al. | 2025 | Scaling debate oversight |
| `2506.07596_dont_lie_collaborative_selfplay.pdf` | Don't Lie to Your Friends | Akata et al. | 2025 | Honest communication via self-play |
| `2404.09960_coevolving.pdf` | Coevolving with the Other You | Chen et al. | 2024 | Cooperative multi-agent RL |

### Reasoning and Self-Improvement
| File | Title | Authors | Year | Notes |
|------|-------|---------|------|-------|
| `2305.20050_lets_verify_step_by_step.pdf` | Let's Verify Step by Step | Lightman et al. | 2023 | Process reward models |
| `2203.14465_star_self_taught_reasoner.pdf` | STaR: Self-Taught Reasoner | Zelikman et al. | 2022 | Bootstrapping reasoning |
| `2504.20571_rl_one_example.pdf` | RL for Reasoning with One Training Example | Yuan et al. | 2025 | Minimal data RL for reasoning |
| `2505.01939_reasoning_gym.pdf` | Reasoning Gym | Bakhtin et al. | 2025 | 100+ RLVR environments |

### Deep-Read Chunks Available
Papers with detailed page-by-page chunks in `papers/pages/`:
- `maporl/` - MAPoRL (3 chunks, fully read)
- `deepseek_r1/` - DeepSeek-R1 (chunked, partially read)
- `debate/` - Du et al. multiagent debate (fully read via agent)
- `dapo/` - DAPO
- `deepseekmath/` - DeepSeekMath/GRPO
- `r1zero/` - Understanding R1-Zero
- `rl_incentivize/` - Does RL Incentivize Reasoning
- `neg_rl/` - Negative Reinforcement
- `rlvr_correct/` - RLVR Incentivizes Reasoning
- `spurious/` - Spurious Rewards
- `reconcile/` - ReConcile
- `scalable_mad/` - Scalable Oversight MAD
- `selfplay/` - Don't Lie to Your Friends

---

## Datasets

All datasets are stored in `datasets/` and downloaded via HuggingFace.

### Primary Dataset
| Dataset | Location | Split Sizes | Description | Source |
|---------|----------|-------------|-------------|--------|
| **GSM8K** | `datasets/gsm8k/` | Train: 7,473 / Test: 1,319 | Grade school math word problems | `openai/gsm8k` |

### Secondary / Evaluation Datasets
| Dataset | Location | Split Sizes | Description | Source |
|---------|----------|-------------|-------------|--------|
| **MATH** | `datasets/math/` | Train: 7,500 / Test: 5,000 | Competition math (7 subjects) | `EleutherAI/hendrycks_math` |
| **MATH-500** | `datasets/math500/` | Test: 500 | Commonly used eval subset | `HuggingFaceTB/MATH-500` |
| **ANLI** | `datasets/anli/` | Train/Dev/Test per round | Adversarial NLI (3 rounds) | `facebook/anli` |

### Dataset Details

**GSM8K** is the primary training dataset, used by both DeepSeek-R1 and MAPoRL. Problems are grade-school level math word problems requiring multi-step arithmetic reasoning. Each problem has a detailed solution with step-by-step reasoning and a final numerical answer, making it ideal for verifiable rewards.

**MATH** provides harder competition-level problems across algebra, counting & probability, geometry, intermediate algebra, number theory, prealgebra, and precalculus. Useful for testing generalization of reasoning improvements.

**MATH-500** is a curated 500-problem evaluation subset commonly used in RLVR papers for standardized comparison.

**ANLI** (Adversarial Natural Language Inference) is used by MAPoRL to test cross-domain transfer of collaboration skills. Not a reasoning dataset per se, but relevant for testing whether collaborative training generalizes.

### Sample Data
- `datasets/gsm8k_samples.json` - Sample GSM8K problems for quick inspection
- `datasets/math_samples.json` - Sample MATH problems for quick inspection

### Downloading
Large data files are excluded from git (see `datasets/.gitignore`). To reproduce:
```python
from datasets import load_dataset

# GSM8K
gsm8k = load_dataset("openai/gsm8k", "main")
gsm8k.save_to_disk("datasets/gsm8k")

# MATH (all subjects)
import os
subjects = ["algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
for subject in subjects:
    ds = load_dataset("EleutherAI/hendrycks_math", subject)
    ds.save_to_disk(f"datasets/math/{subject}")

# MATH-500
math500 = load_dataset("HuggingFaceTB/MATH-500")
math500.save_to_disk("datasets/math500")

# ANLI
anli = load_dataset("facebook/anli")
anli.save_to_disk("datasets/anli")
```

---

## Code Repositories

All repositories are cloned into `code/`.

### MAPoRL (`code/maporl/`)
- **Source**: https://github.com/Mao-KU/MaPoRL (or similar)
- **Description**: Multi-Agent Post-Co-Training implementation. The closest codebase to our research.
- **Key files**:
  - `train_ppo_v2_multi_agent_multi_model.py` - Main training script for multi-agent PPO
  - `trl/trl/trainer/ppov2_trainer_multi_different_model.py` - Extended PPOv2 trainer for multiple agents
  - `trl/trl/trainer/utils_multi_unified_chat.py` - Chat template utilities for multi-agent
  - `trl/trl/trainer/utils_multi_unified.py` - General multi-agent utilities
  - `config/ppo_config/` - Configuration files for different training setups
  - `reward_server.py` - Separate reward computation server
  - `reward_gen_data.py` - Reward training data generation
- **Framework**: Based on TRL (Transformer Reinforcement Learning)
- **Relevance**: Direct starting point for multi-agent RL training; needs adaptation from PPO to GRPO and from turn-taking to debate format

### verl (`code/verl/`)
- **Source**: https://github.com/volcengine/verl
- **Description**: Volcano Engine Reinforcement Learning framework for LLMs. Supports GRPO, PPO, and other RL algorithms at scale.
- **Relevance**: Infrastructure for scalable RLVR training; used by DAPO. Potential base for implementing collaborative GRPO.

### DAPO (`code/dapo/`)
- **Source**: https://github.com/BytedTsinghua-SIA/DAPO
- **Description**: Open-source RLVR system achieving strong results on AIME 2024. Implements decoupled clip ratios and dynamic sampling.
- **Relevance**: State-of-the-art RLVR implementation; reference for training recipes, hyperparameters, and reward design. Built on verl.

### Reasoning Gym (`code/reasoning-gym/`)
- **Source**: https://github.com/open-thought/reasoning-gym
- **Description**: Collection of 100+ procedurally generated reasoning environments for RLVR. Covers math, logic, code, and more.
- **Relevance**: Evaluation suite for testing reasoning capabilities; provides diverse verifiable tasks beyond GSM8K/MATH.

---

## Paper Search Results

Raw search results from paper-finder are stored in `paper_search_results/` as JSONL files:
- Results cover ~248 papers across RLVR, multi-agent debate, and related topics
- Used to identify the 28 core papers downloaded above

---

## Recommended Reading Order

For someone new to this research area:

1. **DeepSeek-R1** (`2501.12948`) - Understand RLVR fundamentals
2. **DeepSeekMath** (`2402.03300`) - Understand GRPO algorithm
3. **Du et al. Multiagent Debate** (`2305.14325`) - Understand debate mechanism
4. **MAPoRL** (`2502.18439`) - Understand closest prior work
5. **Does RL Incentivize Reasoning** (`2504.13837`) - Critical perspective on RLVR
6. **DAPO** (`2503.14476`) - Practical RLVR implementation details
7. **Talk Isn't Cheap** (`2503.17510`) - Failure modes to avoid
8. **Don't Lie to Your Friends** (`2506.07596`) - Training collaborative communication

---

## Environment Setup

The project uses an isolated Python environment managed by `uv`:

```bash
# Create and activate environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install pypdf requests arxiv httpx datasets transformers trl
```

The `pyproject.toml` at the workspace root defines the project configuration.
