#!/usr/bin/env python3
"""Shared evaluation utilities for answer extraction and checking."""

import re


def extract_ans(text):
    """Extract numerical answer from model output.

    Priority: \boxed{} > "the answer is" / "therefore" > last number.
    """
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums:
            return nums[-1]
    for pat in [r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+)([^\n.]+)',
                r'answer[:\s]+([^\n.]+)']:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            nums = re.findall(r'-?\d+\.?\d*', matches[-1].group(1))
            if nums:
                return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]


def check(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    p = predicted.strip().replace(',', '').replace(' ', '')
    g = str(ground_truth).strip().replace(',', '').replace(' ', '')
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return p.lower() == g.lower()


def gt_in_output(text, gt):
    """Check if ground truth number appears anywhere in output text.

    Returns (found: bool, matched_value: str|None).
    Used for semantic accuracy (extraction artifact detection).
    """
    gt_clean = str(gt).strip().replace(',', '').replace(' ', '')
    gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
    if not gt_nums:
        return False, None
    target = gt_nums[-1]
    output_nums = re.findall(r'-?\d+\.?\d*', text)
    for num in output_nums:
        try:
            if abs(float(num) - float(target)) < 1e-6:
                return True, num
        except Exception:
            pass
    return False, None
