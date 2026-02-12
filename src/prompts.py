"""Prompt templates for single-agent and collaborative debate experiments."""

# Single-agent CoT prompt
SINGLE_AGENT_COT = """Solve the following math problem step by step. Show your reasoning clearly, then give your final answer.

Problem: {problem}

Think through this step by step, then provide your final answer in the format:
#### [numerical answer]"""

SINGLE_AGENT_COT_MATH = """Solve the following math problem step by step. Show your reasoning clearly, then give your final answer.

Problem: {problem}

Think through this step by step, then provide your final answer in the format:
\\boxed{{answer}}"""

# Debate Round 0: Independent solution
DEBATE_INDEPENDENT = """Solve the following math problem step by step. Show your reasoning clearly.

Problem: {problem}

Think through this step by step, then provide your final answer in the format:
#### [numerical answer]"""

DEBATE_INDEPENDENT_MATH = """Solve the following math problem step by step. Show your reasoning clearly.

Problem: {problem}

Think through this step by step, then provide your final answer in the format:
\\boxed{{answer}}"""

# Debate Round 1: Critique and revise
DEBATE_CRITIQUE = """You previously solved a math problem. Another mathematician also solved it independently. Compare solutions and determine the correct answer.

Problem: {problem}

YOUR SOLUTION:
{own_solution}

OTHER MATHEMATICIAN'S SOLUTION:
{other_solution}

Instructions:
1. Carefully examine both solutions for correctness
2. Identify any errors in reasoning or calculation in either solution
3. If you find errors in the other solution, explain why it's wrong
4. If you find errors in your own solution, acknowledge and correct them
5. If you still believe your original answer is correct, defend it with clear reasoning
6. Give your final answer

Final answer format:
#### [numerical answer]"""

DEBATE_CRITIQUE_MATH = """You previously solved a math problem. Another mathematician also solved it independently. Compare solutions and determine the correct answer.

Problem: {problem}

YOUR SOLUTION:
{own_solution}

OTHER MATHEMATICIAN'S SOLUTION:
{other_solution}

Instructions:
1. Carefully examine both solutions for correctness
2. Identify any errors in reasoning or calculation in either solution
3. If you find errors in the other solution, explain why it's wrong
4. If you find errors in your own solution, acknowledge and correct them
5. If you still believe your original answer is correct, defend it with clear reasoning
6. Give your final answer

Final answer format:
\\boxed{{answer}}"""

# Rephrase prompt
REPHRASE_PROMPT = """Rephrase the following math problem. Keep the exact same mathematical content and answer, but change the wording, names, and context. The problem should be equally difficult.

Original problem: {problem}

Provide ONLY the rephrased problem, nothing else."""
