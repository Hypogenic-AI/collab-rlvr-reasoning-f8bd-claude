# Code Repositories

Cloned repositories relevant to the "Collaborative RLVR for Robust Reasoning" research project.

## Repositories

### MAPoRL (`maporl/`)
- **Source**: Multi-Agent Post-Co-Training for Collaborative LLMs with RL
- **Framework**: TRL (Transformer Reinforcement Learning)
- **Key files**:
  - `train_ppo_v2_multi_agent_multi_model.py` - Multi-agent PPO training
  - `trl/trl/trainer/ppov2_trainer_multi_different_model.py` - Extended PPOv2 trainer
  - `trl/trl/trainer/utils_multi_unified_chat.py` - Multi-agent chat utilities
  - `config/ppo_config/` - Training configurations
  - `reward_server.py` - Separate reward computation server
- **Relevance**: Closest codebase to our research. Needs adaptation from PPO to GRPO and from turn-taking to debate format.

### verl (`verl/`)
- **Source**: https://github.com/volcengine/verl
- **Framework**: Volcano Engine RL for LLMs
- **Description**: Scalable RL framework supporting GRPO, PPO, and other algorithms. Handles distributed training, model parallelism, and efficient rollout generation.
- **Relevance**: Primary infrastructure for implementing scalable RLVR training. Used by DAPO.

### DAPO (`dapo/`)
- **Source**: https://github.com/BytedTsinghua-SIA/DAPO
- **Framework**: Built on verl
- **Description**: Open-source RLVR system with decoupled clip ratios and dynamic sampling. Achieves strong results on AIME 2024.
- **Relevance**: State-of-the-art RLVR reference implementation. Provides training recipes, hyperparameters, and reward design patterns.

### Reasoning Gym (`reasoning-gym/`)
- **Source**: https://github.com/open-thought/reasoning-gym
- **Description**: 100+ procedurally generated reasoning environments for RLVR. Covers arithmetic, algebra, logic, code, and more.
- **Relevance**: Broad evaluation suite for testing reasoning capabilities across diverse task types.

## Implementation Strategy

For the collaborative RLVR project, the likely approach is:

1. **Base framework**: Use verl/DAPO for GRPO training infrastructure
2. **Multi-agent extension**: Adapt MAPoRL's multi-agent training patterns from PPO to GRPO
3. **Debate protocol**: Implement debate communication as a new multi-agent interaction format
4. **Evaluation**: Use Reasoning Gym + GSM8K/MATH for comprehensive evaluation
