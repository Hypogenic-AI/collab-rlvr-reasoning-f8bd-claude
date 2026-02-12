# Literature Review: Collaborative RLVR for Robust Reasoning

## Research Hypothesis
Making RLVR collaborative -- by having two models solve mathematical reasoning tasks independently and then discuss before answering -- will force reasoning to be externalized, challenged, and defended, resulting in more robust and faithful reasoning compared to standard single-agent RLVR.

---

## 1. Reinforcement Learning with Verifiable Rewards (RLVR)

### 1.1 Foundations: DeepSeek-R1 and GRPO

**DeepSeek-R1** (Guo et al., 2025; `2501.12948`) demonstrated that pure reinforcement learning without supervised fine-tuning (SFT) can produce emergent reasoning behaviors in LLMs. The key innovation is training directly on tasks with verifiable answers using rule-based reward signals.

**GRPO (Group Relative Policy Optimization)**, introduced in **DeepSeekMath** (Shao et al., 2024; `2402.03300`), is the core RL algorithm. For each prompt, GRPO:
1. Samples a group of G outputs from the current policy
2. Computes rewards for each output
3. Normalizes advantages within the group: A_i = (r_i - mean(r)) / std(r)
4. Optimizes a clipped surrogate objective with KL penalty against a reference policy

This eliminates the need for a learned value function (critic), making it significantly more memory-efficient than PPO.

**DeepSeek-R1-Zero** showed that applying GRPO with only accuracy + format rewards to a base model (no SFT warmup) produces emergent chain-of-thought reasoning, self-verification, and even an "aha moment" where the model spontaneously discovers new reasoning strategies. However, R1-Zero suffers from readability issues, language mixing, and sometimes infinite reasoning loops.

The full **DeepSeek-R1** pipeline uses a multi-stage approach: (1) cold-start SFT data, (2) reasoning-oriented RL, (3) rejection sampling to create new SFT data, (4) RL on broader tasks including helpfulness.

### 1.2 Critical Analysis of RLVR

Several papers raise important questions about what RLVR actually learns:

- **"Does RL Really Incentivize Reasoning?"** (Yue et al., 2025; `2504.13837`) presents evidence that RL may primarily be selecting and reinforcing reasoning patterns already present in the base model, rather than teaching genuinely new reasoning capabilities. This has direct implications for our hypothesis: if RLVR alone doesn't create new reasoning, perhaps the collaborative debate mechanism can push models beyond their individual capabilities.

- **"Spurious Rewards"** (Chen et al., 2025; `2506.03691`) demonstrates the surprising finding that RLVR can improve performance even with partially random or spurious reward signals. This suggests the training signal may be less about reward accuracy and more about the optimization process itself -- providing important context for designing collaborative reward signals.

- **"Surprising Effectiveness of Negative Reinforcement"** (Liu et al., 2025; `2503.09476`) shows that negative-only rewards (penalizing wrong answers without rewarding correct ones) can be surprisingly effective in RLVR, suggesting that the learning signal structure has unexplored dimensions.

- **"RLVR Implicitly Incentivizes Correct Reasoning"** (Liu et al., 2025; `2505.07532`) provides a counter-perspective, arguing that RLVR does genuinely encourage correct reasoning chains, not just correct final answers. This supports the use of RLVR as a training signal for reasoning quality.

- **"Entropy Mechanism of RL for Reasoning LLMs"** (Zhu et al., 2025; `2505.05551`) analyzes the entropy dynamics during RL training for reasoning, identifying entropy collapse as a key failure mode where the policy loses diversity too quickly.

### 1.3 RLVR at Scale

- **DAPO** (Yu et al., 2025; `2503.14476`) provides an open-source RLVR system achieving strong results on AIME 2024. Key innovations include decoupled clip ratios for positive/negative samples, dynamic sampling to handle varying difficulty, and token-level loss computation. DAPO's codebase (built on verl) is directly relevant as infrastructure for our experiments.

- **Understanding R1-Zero-Like Training** (Zhao et al., 2025; `2503.20783`) provides systematic analysis of training dynamics, reward design, and failure modes in R1-Zero-style training.

- **GRPO Analysis** (Zhong et al., 2025; `2503.13555`) offers deeper theoretical understanding of how GRPO optimizes and its relationship to other policy gradient methods.

### 1.4 Key Takeaways for Our Research
- RLVR with GRPO is an effective and efficient framework for training reasoning capabilities
- The debate about whether RL creates new reasoning or amplifies existing patterns motivates combining RLVR with multi-agent collaboration as a mechanism to push beyond individual model limits
- Entropy management and reward signal design are critical challenges
- Existing infrastructure (DAPO/verl, TRL) provides practical starting points

---

## 2. Multi-Agent Debate and Collaboration

### 2.1 Foundational Multi-Agent Debate

**"Improving Factuality and Reasoning through Multiagent Debate"** (Du et al., 2023; `2305.14325`) is the seminal work establishing that multiple LLM agents debating can improve both factuality and reasoning beyond any single agent. Key findings from our deep reading:

- **Mechanism**: Multiple agents (default: 3) independently generate answers, then iteratively read each other's responses and update their own over multiple rounds (default: 2).
- **Not just majority voting**: Debate enables genuine error correction -- even when all agents initially give wrong answers, the debate process can lead to correct convergence. This is the critical finding for our hypothesis.
- **Scaling properties**: Performance monotonically improves with more agents and more rounds, though with diminishing returns after ~4 rounds.
- **Compatible with CoT**: Debate is orthogonal to chain-of-thought prompting; combining them yields the best results (GSM8K: single 77%, debate+CoT 85%).
- **Cross-model debate works**: Different models (ChatGPT + Bard) debating outperform either model alone.
- **Failure mode**: When all agents share the same systematic bias, debate cannot correct it. This motivates training agents to develop diverse reasoning strategies.
- **Agreeable agent problem**: RLHF-trained models tend to be too agreeable for productive debate, converging too quickly. This is directly relevant -- our RLVR training should reward productive disagreement.

### 2.2 Debate Variations and Extensions

- **MAD (Multi-Agent Debate for Divergent Thinking)** (Liang et al., 2023; `2305.19118`) introduces a structured debate framework with a judge, showing that debate is especially beneficial for tasks requiring divergent thinking rather than convergent tasks.

- **ReConcile** (Chen et al., 2023; `2309.13007`) proposes round-table conference-style multi-agent discussion with weighted voting based on agent confidence, showing improvements on reasoning benchmarks.

- **Exchange-of-Thought** (Yin et al., 2023; `2312.01823`) introduces cross-model communication protocols for multi-step reasoning, showing that structured exchange of intermediate reasoning steps outperforms simple answer sharing.

- **"Should We Be Going MAD?"** (Smit et al., 2023; `2311.17371`) provides empirical benchmarking of debate strategies, finding that debate effectiveness depends heavily on task type and model capability.

- **"Talk Isn't Always Cheap"** (Wu et al., 2025; `2503.17510`) identifies failure modes in multi-agent debate, including persuasion cascades, groupthink, and computational costs, providing important cautions for our design.

- **"Rethinking Bounds of LLM Reasoning"** (Xu et al., 2024; `2402.18272`) analyzes theoretical limits of multi-agent debate vs. single-model approaches, finding that debate can exceed individual model performance bounds under certain conditions.

- **Multi-LLM Debate Framework** (Zhao et al., 2024; `2402.18176`) provides theoretical analysis of convergence properties in multi-LLM debates.

### 2.3 Collaborative Multi-Agent Systems

- **"Exploring Collaboration Mechanisms"** (Zhang et al., 2023; `2310.02124`) applies social psychology frameworks to LLM collaboration, comparing debate, negotiation, and consensus-building approaches.

- **Multi-Agent Collaboration Mechanisms Survey** (Yang et al., 2025; `2501.06322`) provides a comprehensive survey of different collaboration paradigms for LLM agents.

- **"Towards Scalable Oversight with Collaborative MAD"** (Wang et al., 2025; `2506.01773`) examines how multi-agent debate can scale to more complex oversight tasks.

- **"Don't Lie to Your Friends"** (Akata et al., 2025; `2506.07596`) introduces collaborative self-play where agents learn to be more honest and informative when communicating, directly relevant to training collaborative reasoning.

- **"Coevolving with the Other You"** (Chen et al., 2024; `2404.09960`) explores cooperative multi-agent RL where agents co-evolve strategies, providing RL-specific insights for our multi-agent training.

### 2.4 Key Takeaways for Our Research
- Multi-agent debate consistently improves reasoning beyond individual model capabilities
- The improvement goes beyond ensemble/voting effects -- genuine error correction occurs through debate
- Failure modes (groupthink, agreeable agents, shared biases) provide clear design targets
- Structured communication protocols and diverse agent strategies improve debate quality
- No prior work combines debate with RLVR training -- this is our key contribution

---

## 3. The Closest Prior Work: MAPoRL

### 3.1 Overview

**MAPoRL (Multi-Agent Post-Co-Training for Collaborative LLMs with RL)** (Li et al., 2025; `2502.18439`) is the closest existing work to our proposed research. We performed an extensive deep-read of this paper.

### 3.2 Method

MAPoRL uses multi-agent PPO (not GRPO) to train multiple agents to collaborate:

1. **Multi-turn interaction**: Agents take turns generating responses in a sequential communication protocol
2. **Influence-aware verification reward**: The reward considers not just answer correctness but also how much each agent influenced the group's final answer
3. **Collaboration incentive**: Two hyperparameters (alpha, beta) control the balance between individual accuracy and collaborative synergy
4. **Game-theoretic framing**: Agents choose between "Collaborate" (a0) and "Act Independently" (a1), with synergy rewards requiring threshold participation

### 3.3 Key Findings

- Single-agent training is **insufficient** for learning collaboration -- multi-agent training is essential
- Collaboration skills **transfer across domains** (train on GSM8K, improve on ANLI)
- SFT alone **fails** to induce genuine collaborative behavior; RL is necessary
- The method extends TRL's PPO trainer to support multiple agents with different model architectures

### 3.4 Differences from Our Proposal

| Aspect | MAPoRL | Our Proposal |
|--------|--------|-------------|
| RL Algorithm | PPO (with value function) | GRPO (value-function-free) |
| Reward Type | Learned verifier + influence | Verifiable rewards (rule-based) |
| Communication | Sequential turn-taking | Simultaneous debate rounds |
| Training Focus | Collaboration behavior | Robust reasoning through debate |
| Core Mechanism | Influence-aware rewards | Externalized reasoning + critique |
| Scalability | Limited by PPO critic | More scalable via GRPO |

### 3.5 Key Takeaways for Our Research
- MAPoRL validates that multi-agent RL training can produce genuine collaboration
- Using GRPO instead of PPO would be more memory-efficient and potentially more stable
- Verifiable rewards (RLVR-style) could replace the learned verifier, simplifying the pipeline
- The debate format (simultaneous reasoning + critique) may produce stronger externalization than sequential turn-taking
- Cross-domain transfer of collaboration skills is an encouraging finding

---

## 4. Process Rewards and Reasoning Quality

- **"Let's Verify Step by Step"** (Lightman et al., 2023; `2305.20050`) introduces process reward models (PRMs) that evaluate each step of reasoning rather than just the final answer. This is relevant because debate naturally provides process-level feedback through step-by-step critique.

- **STaR (Self-Taught Reasoner)** (Zelikman et al., 2022; `2203.14465`) introduces bootstrapping reasoning through self-generated rationales, directly in the RLVR lineage. Debate could serve as a higher-quality form of rationale generation.

- **"Reinforcement Learning for Reasoning with One Training Example"** (Yuan et al., 2025; `2504.20571`) shows that RL for reasoning can work with extremely limited data, suggesting that the quality of the training signal matters more than quantity.

- **Reasoning Gym** (Bakhtin et al., 2025; `2505.01939`) provides 100+ environments for RLVR training, offering a broad evaluation suite for testing collaborative reasoning.

---

## 5. Research Gaps and Our Contribution

### 5.1 Identified Gaps

1. **No RLVR + Debate combination**: Existing work either uses RLVR for single-agent reasoning (DeepSeek-R1, DAPO) or uses debate at inference time without RL training (Du et al.). No work trains agents with RLVR specifically to debate and improve reasoning through collaboration.

2. **Agreeable agent problem unaddressed by training**: Du et al. identified that RLHF makes agents too agreeable for productive debate. No work has used RL training to specifically address this -- our RLVR training could reward productive disagreement and principled stubbornness.

3. **Shared bias failure mode**: When all agents have the same misconception, debate fails. RLVR training that rewards reasoning diversity across agents could mitigate this.

4. **GRPO for multi-agent training**: MAPoRL uses PPO with a learned critic. GRPO's group-relative advantages are naturally suited for multi-agent settings where outputs from collaborating agents can be compared.

5. **Externalized reasoning as training signal**: The debate process forces reasoning to be explicit and articulable. This externalized reasoning could serve as a higher-quality training signal than single-agent chain-of-thought.

### 5.2 Proposed Approach Summary

Our proposed approach would:
1. Have two agents independently solve math problems using chain-of-thought
2. Share their solutions and engage in structured debate (1-2 rounds)
3. Each agent produces a final answer after considering the debate
4. Use verifiable rewards (correct/incorrect final answer) to train both agents via GRPO
5. The key insight: agents that produce clear, defensible reasoning and effectively critique flawed reasoning will consistently achieve better final answers, so RLVR will naturally select for these capabilities

### 5.3 Expected Benefits
- **Externalized reasoning**: Debate forces reasoning to be explicit, making it verifiable and trainable
- **Error correction**: Agents learn to identify and correct reasoning errors through critique
- **Reasoning robustness**: Agents that survive debate challenges develop more robust reasoning strategies
- **Diversity**: Multi-agent training with different initializations or architectures naturally promotes reasoning diversity
- **Scalability**: GRPO eliminates the need for a learned critic, making multi-agent training more practical

---

## 6. Summary of Key Papers

| Paper | Year | Relevance | Key Contribution |
|-------|------|-----------|-----------------|
| DeepSeek-R1 | 2025 | Core RLVR | Pure RL produces emergent reasoning |
| DeepSeekMath (GRPO) | 2024 | Core algorithm | Efficient RL without value function |
| Du et al. (Debate) | 2023 | Core mechanism | Debate improves reasoning beyond voting |
| MAPoRL | 2025 | Closest work | Multi-agent PPO for collaboration |
| DAPO | 2025 | Infrastructure | Open-source RLVR at scale |
| Does RL Incentivize Reasoning | 2025 | Motivation | RL may only select existing patterns |
| Spurious Rewards | 2025 | Reward design | RLVR works with imperfect rewards |
| Let's Verify Step by Step | 2023 | Process reward | Step-level reasoning evaluation |
| STaR | 2022 | Self-improvement | Bootstrapping reasoning with reasoning |
| Don't Lie to Your Friends | 2025 | Training signal | Collaborative self-play for honesty |

---

## References

1. Guo, D. et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948
2. Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300
3. Du, Y. et al. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate. arXiv:2305.14325
4. Liang, T. et al. (2023). Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. arXiv:2305.19118
5. Li, C. et al. (2025). MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning. arXiv:2502.18439
6. Yu, Z. et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System. arXiv:2503.14476
7. Zhao, J. et al. (2025). Understanding R1-Zero-Like Training: A Critical Study. arXiv:2503.20783
8. Yue, Y. et al. (2025). Does RL Really Incentivize Reasoning Capability in LLMs? arXiv:2504.13837
9. Chen, X. et al. (2025). Spurious Rewards: Rethinking Training Signals in RLVR. arXiv:2506.03691
10. Liu, Z. et al. (2025). The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning. arXiv:2503.09476
11. Liu, W. et al. (2025). RLVR Implicitly Incentivizes Correct Reasoning. arXiv:2505.07532
12. Zhu, H. et al. (2025). Entropy Mechanism of RL for Reasoning LLMs. arXiv:2505.05551
13. Chen, J. et al. (2023). ReConcile: Round-Table Conference Improves Reasoning. arXiv:2309.13007
14. Yin, D. et al. (2023). Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication. arXiv:2312.01823
15. Zhang, Y. et al. (2023). Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View. arXiv:2310.02124
16. Smit, A. et al. (2023). Should We Be Going MAD? A Look at Multi-Agent Debate Strategies. arXiv:2311.17371
17. Wu, K. et al. (2025). Talk Isn't Always Cheap: When Multi-Agent Debate Fails. arXiv:2503.17510
18. Xu, Y. et al. (2024). Rethinking the Bounds of LLM Reasoning. arXiv:2402.18272
19. Zhao, L. et al. (2024). Multi-LLM Debate Framework. arXiv:2402.18176
20. Lightman, H. et al. (2023). Let's Verify Step by Step. arXiv:2305.20050
21. Zelikman, E. et al. (2022). STaR: Self-Taught Reasoner. arXiv:2203.14465
22. Zhong, W. et al. (2025). Reinforcement Learning with Verifiable Rewards: GRPO Analysis. arXiv:2503.13555
23. Yang, Z. et al. (2025). Multi-Agent Collaboration Mechanisms: A Survey. arXiv:2501.06322
24. Chen, Z. et al. (2024). Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent RL. arXiv:2404.09960
25. Yuan, W. et al. (2025). Reinforcement Learning for Reasoning with One Training Example. arXiv:2504.20571
26. Bakhtin, A. et al. (2025). Reasoning Gym: Verifiable Reasoning Environments. arXiv:2505.01939
27. Wang, T. et al. (2025). Towards Scalable Oversight with Collaborative Multi-Agent Debate. arXiv:2506.01773
28. Akata, E. et al. (2025). Don't Lie to Your Friends: Learning Honest Communication in Collaborative Self-Play. arXiv:2506.07596
