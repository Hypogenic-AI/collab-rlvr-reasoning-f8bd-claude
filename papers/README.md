# Papers

28 papers downloaded for the "Collaborative RLVR for Robust Reasoning" research project.

## Organization

- `*.pdf` - Full paper PDFs downloaded from arXiv
- `pages/` - Chunked versions of key papers for detailed reading
- `download_papers.sh` - Script to re-download all papers

## Paper List

### Core RLVR
1. `2501.12948_deepseek_r1.pdf` - DeepSeek-R1 (foundational RLVR)
2. `2402.03300_deepseekmath_grpo.pdf` - DeepSeekMath (GRPO algorithm)
3. `2503.14476_dapo.pdf` - DAPO (open-source RLVR at scale)
4. `2503.20783_understanding_r1_zero.pdf` - Understanding R1-Zero
5. `2503.13555_grpo_analysis.pdf` - GRPO analysis

### Critical RLVR Analysis
6. `2504.13837_does_rl_incentivize_reasoning.pdf` - Does RL incentivize reasoning?
7. `2506.03691_spurious_rewards.pdf` - Spurious rewards analysis
8. `2503.09476_negative_reinforcement.pdf` - Negative reinforcement effectiveness
9. `2505.07532_rlvr_incentivizes_reasoning.pdf` - RLVR incentivizes correct reasoning
10. `2505.05551_entropy_mechanism_rl.pdf` - Entropy mechanism in RL

### Multi-Agent Debate
11. `2305.14325_multiagent_debate.pdf` - Multiagent debate (seminal)
12. `2305.19118_mad_divergent_thinking.pdf` - MAD for divergent thinking
13. `2309.13007_reconcile.pdf` - ReConcile round-table conference
14. `2312.01823_exchange_of_thought.pdf` - Exchange-of-Thought
15. `2311.17371_going_mad.pdf` - Benchmarking debate strategies
16. `2503.17510_talk_isnt_cheap.pdf` - Failure modes in debate
17. `2402.18272_rethinking_bounds_mad.pdf` - Rethinking bounds of reasoning
18. `2402.18176_multi_llm_debate_framework.pdf` - Debate framework theory

### Collaborative Systems
19. `2502.18439_maporl.pdf` - MAPoRL (closest prior work)
20. `2310.02124_collaboration_social_psych.pdf` - Collaboration via social psych
21. `2501.06322_multi_agent_collab_survey.pdf` - Multi-agent collaboration survey
22. `2506.01773_scalable_oversight_mad.pdf` - Scalable oversight with debate
23. `2506.07596_dont_lie_collaborative_selfplay.pdf` - Collaborative self-play
24. `2404.09960_coevolving.pdf` - Cooperative multi-agent RL

### Reasoning Foundations
25. `2305.20050_lets_verify_step_by_step.pdf` - Process reward models
26. `2203.14465_star_self_taught_reasoner.pdf` - STaR self-taught reasoner
27. `2504.20571_rl_one_example.pdf` - RL with minimal data
28. `2505.01939_reasoning_gym.pdf` - Reasoning gym environments

## Re-downloading

```bash
# Uses wget (if available) or Python requests
bash download_papers.sh
```

Note: `download_papers.sh` uses wget. If wget is unavailable, use Python:
```python
import requests
url = "https://arxiv.org/pdf/PAPER_ID.pdf"
r = requests.get(url)
with open("output.pdf", "wb") as f:
    f.write(r.content)
```
