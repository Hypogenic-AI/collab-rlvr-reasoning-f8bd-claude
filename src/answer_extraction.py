"""Extract and compare answers from GSM8K and MATH formatted responses."""

import re


def extract_gsm8k_answer(solution_text):
    """Extract numerical answer from GSM8K format (#### N)."""
    # First try #### format
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', solution_text)
    if match:
        return match.group(1).replace(',', '').strip()

    # Try "the answer is X" pattern
    match = re.search(r'(?:the answer is|answer:\s*)[\$]?\s*(-?[\d,]+\.?\d*)', solution_text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '').strip()

    # Try to find the last number in the text
    numbers = re.findall(r'(-?[\d,]+\.?\d*)', solution_text)
    if numbers:
        return numbers[-1].replace(',', '').strip()

    return None


def extract_math_answer(solution_text):
    """Extract answer from MATH format (\\boxed{...}), handling nested braces."""
    # Try \boxed{} format with nested brace support
    idx = solution_text.find('\\boxed{')
    if idx != -1:
        start = idx + len('\\boxed{')
        depth = 1
        i = start
        while i < len(solution_text) and depth > 0:
            if solution_text[i] == '{':
                depth += 1
            elif solution_text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return solution_text[start:i-1].strip()

    # Try "the answer is" pattern
    match = re.search(r'(?:the answer is|answer:\s*)(.*?)(?:\.|$)', solution_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def extract_answer_from_response(response_text, dataset_type="gsm8k"):
    """Extract answer from an LLM response."""
    if dataset_type == "gsm8k":
        return extract_gsm8k_answer(response_text)
    elif dataset_type in ("math", "math500"):
        return extract_math_answer(response_text)
    return None


def normalize_answer(answer_str):
    """Normalize an answer string for comparison."""
    if answer_str is None:
        return None
    s = str(answer_str).strip()
    # Remove $ signs, trailing periods
    s = s.replace('$', '').replace('%', '').rstrip('.')
    # Remove commas from numbers
    s = s.replace(',', '')
    # Try to convert to float for numerical comparison
    try:
        val = float(s)
        # If it's an integer, return as int string
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s.lower().strip()


def normalize_latex(s):
    """Normalize LaTeX for comparison (strip whitespace, common variations)."""
    if s is None:
        return None
    s = s.strip()
    # Remove \left and \right
    s = s.replace('\\left', '').replace('\\right', '')
    # Remove \, spacing commands
    s = re.sub(r'\\[,;:!]', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', '', s)
    # Remove trailing periods
    s = s.rstrip('.')
    return s


def answers_match(predicted, gold, dataset_type="gsm8k"):
    """Check if predicted answer matches gold answer."""
    if predicted is None or gold is None:
        return False

    if dataset_type == "gsm8k":
        pred_norm = normalize_answer(predicted)
        gold_norm = normalize_answer(gold)
        return pred_norm is not None and gold_norm is not None and pred_norm == gold_norm
    else:
        # For MATH: try numeric comparison first, then LaTeX string comparison
        pred_num = normalize_answer(predicted)
        gold_num = normalize_answer(gold)
        if pred_num is not None and gold_num is not None:
            try:
                if float(pred_num) == float(gold_num):
                    return True
            except ValueError:
                pass
        # Fall back to LaTeX string match
        pred_latex = normalize_latex(predicted)
        gold_latex = normalize_latex(gold)
        if pred_latex is not None and gold_latex is not None:
            return pred_latex == gold_latex
        return False


def extract_gold_answer_gsm8k(answer_field):
    """Extract the gold numerical answer from GSM8K answer field."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_field)
    if match:
        return match.group(1).replace(',', '').strip()
    return None


def extract_gold_answer_math(answer_field):
    """Extract gold answer from MATH answer field."""
    return str(answer_field).strip()
