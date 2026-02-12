#!/bin/bash
# Core papers for Collaborative RLVR research

# 1. DeepSeek-R1 (foundational RLVR paper)
wget -q "https://arxiv.org/pdf/2501.12948.pdf" -O "2501.12948_deepseek_r1.pdf" &

# 2. DeepSeekMath (GRPO algorithm origin)
wget -q "https://arxiv.org/pdf/2402.03300.pdf" -O "2402.03300_deepseekmath_grpo.pdf" &

# 3. Improving Factuality and Reasoning through Multiagent Debate (Du et al.)
wget -q "https://arxiv.org/pdf/2305.14325.pdf" -O "2305.14325_multiagent_debate.pdf" &

# 4. Encouraging Divergent Thinking through Multi-Agent Debate (MAD)
wget -q "https://arxiv.org/pdf/2305.19118.pdf" -O "2305.19118_mad_divergent_thinking.pdf" &

# 5. DAPO - open-source RLVR at scale
wget -q "https://arxiv.org/pdf/2503.14476.pdf" -O "2503.14476_dapo.pdf" &

# 6. Understanding R1-Zero-Like Training
wget -q "https://arxiv.org/pdf/2503.20783.pdf" -O "2503.20783_understanding_r1_zero.pdf" &

# 7. Does RL Really Incentivize Reasoning (critical study)
wget -q "https://arxiv.org/pdf/2504.13837.pdf" -O "2504.13837_does_rl_incentivize_reasoning.pdf" &

# 8. MAPoRL - Multi-Agent Post-Co-Training with RL (CLOSEST prior work)
wget -q "https://arxiv.org/pdf/2503.18559.pdf" -O "2503.18559_maporl.pdf" &

# 9. ReConcile - Round-Table Conference for reasoning
wget -q "https://arxiv.org/pdf/2309.13007.pdf" -O "2309.13007_reconcile.pdf" &

# 10. Spurious Rewards - rethinking RLVR signals
wget -q "https://arxiv.org/pdf/2506.03691.pdf" -O "2506.03691_spurious_rewards.pdf" &

# 11. Surprising Effectiveness of Negative Reinforcement in RLVR  
wget -q "https://arxiv.org/pdf/2503.09476.pdf" -O "2503.09476_negative_reinforcement.pdf" &

# 12. Exchange-of-Thought cross-model communication
wget -q "https://arxiv.org/pdf/2312.01823.pdf" -O "2312.01823_exchange_of_thought.pdf" &

# 13. Exploring Collaboration Mechanisms (Social Psychology)
wget -q "https://arxiv.org/pdf/2310.02124.pdf" -O "2310.02124_collaboration_social_psych.pdf" &

# 14. RLVR Implicitly Incentivizes Correct Reasoning
wget -q "https://arxiv.org/pdf/2505.07532.pdf" -O "2505.07532_rlvr_incentivizes_reasoning.pdf" &

# 15. Entropy Mechanism of RL for Reasoning LLMs
wget -q "https://arxiv.org/pdf/2505.05551.pdf" -O "2505.05551_entropy_mechanism_rl.pdf" &

# 16. Multi-Agent Collaboration Mechanisms Survey
wget -q "https://arxiv.org/pdf/2501.06322.pdf" -O "2501.06322_multi_agent_collab_survey.pdf" &

# 17. Let's Verify Step by Step (process reward)
wget -q "https://arxiv.org/pdf/2305.20050.pdf" -O "2305.20050_lets_verify_step_by_step.pdf" &

# 18. STaR - Self-Taught Reasoner (bootstrapping reasoning)
wget -q "https://arxiv.org/pdf/2203.14465.pdf" -O "2203.14465_star_self_taught_reasoner.pdf" &

# 19. Reinforcement Learning with Verifiable Rewards: GRPO analysis
wget -q "https://arxiv.org/pdf/2503.13555.pdf" -O "2503.13555_grpo_analysis.pdf" &

# 20. Rethinking Bounds of LLM Reasoning (multi-agent vs single)
wget -q "https://arxiv.org/pdf/2402.18272.pdf" -O "2402.18272_rethinking_bounds_mad.pdf" &

# 21. Don't lie to your friends (collaborative self-play)
wget -q "https://arxiv.org/pdf/2506.07596.pdf" -O "2506.07596_dont_lie_collaborative_selfplay.pdf" &

# 22. Should we be going MAD (benchmarking debate strategies)
wget -q "https://arxiv.org/pdf/2311.17371.pdf" -O "2311.17371_going_mad.pdf" &

# 23. Multi-LLM Debate Framework (theoretical)
wget -q "https://arxiv.org/pdf/2402.18176.pdf" -O "2402.18176_multi_llm_debate_framework.pdf" &

# 24. Talk Isn't Always Cheap (failure modes in debate)
wget -q "https://arxiv.org/pdf/2503.17510.pdf" -O "2503.17510_talk_isnt_cheap.pdf" &

# 25. Coevolving with the Other You (cooperative multi-agent RL)
wget -q "https://arxiv.org/pdf/2404.09960.pdf" -O "2404.09960_coevolving.pdf" &

# 26. Reinforcement Learning for Reasoning with One Training Example  
wget -q "https://arxiv.org/pdf/2504.20571.pdf" -O "2504.20571_rl_one_example.pdf" &

# 27. Reasoning Gym (RLVR environments)
wget -q "https://arxiv.org/pdf/2505.01939.pdf" -O "2505.01939_reasoning_gym.pdf" &

# 28. Towards Scalable Oversight with Collaborative MAD
wget -q "https://arxiv.org/pdf/2506.01773.pdf" -O "2506.01773_scalable_oversight_mad.pdf" &

wait
echo "All downloads initiated"
