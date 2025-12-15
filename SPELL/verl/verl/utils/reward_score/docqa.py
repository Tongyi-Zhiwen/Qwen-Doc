import os
import re
from typing import Dict, Tuple, Optional
import math
import numpy as np
import sys
import re
import string
from collections import Counter, defaultdict
import pickle
from pathlib import Path
import jsonlines
import json
from tqdm import tqdm

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def last_text_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\text{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_text(s: str) -> str:
    """Remove the LaTeX text command from a string.

    Args:
        s: String with format "\\text{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\text{"
    assert s[: len(left)] == left, f"text error: {s}"
    assert s[-1] == "}", f"text error: {s}"
    return s[len(left) : -1]
    
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    ZERO_METRIC = (0, 0, 0)
    
    if len(normalized_prediction) == 0 or len(normalized_ground_truth) == 0:
        return ZERO_METRIC
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall
def half_em_subem(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    if len(ground_truth) == 0 or len(prediction) == 0:
        return 0
    if ground_truth == prediction:
        return 1
    elif ground_truth in prediction or prediction in ground_truth:
        return 0.5
    else:
        return 0

def sub_em_strict(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    if len(ground_truth) == 0 or len(prediction) == 0:
        return 0
    return ground_truth in prediction

def sub_em_loose(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    if len(ground_truth) == 0 or len(prediction) == 0:
        return 0
    return (ground_truth in prediction) or (prediction in ground_truth)
def exact_match_score(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    if len(prediction) == 0 or len(ground_truth) == 0:
        return 0
    return ground_truth == prediction
def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    subem_strict = sub_em_strict(prediction, gold)
    subem_loose = sub_em_loose(prediction, gold)
    half_em = half_em_subem(prediction, gold)

    f1, prec, recall = f1_score(prediction, gold)
    metrics['sub_em_strict'] += subem_strict
    metrics['sub_em_loose'] += subem_loose
    metrics['half_em'] += half_em
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    metrics['total_num'] += 1
    return em, prec, recall

def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'half_em': 0, 'sub_em_strict': 0, 'sub_em_loose': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics

def extract_solution(solution_str: str) -> str:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    
    # Extract final answer using XML-style tags for R1 type thinking models
    if "<think>" in solution_str:
        if "</think>" not in solution_str:
            print("[Error] The thinkinig length is too long.")
            return "Empty"
        else:
            final_answer = solution_str.split("</think>")[-1].strip()
    elif "</think>" in solution_str or "answer is" in solution_str:
        final_answer = solution_str.split("</think>")[-1].strip()
    else:
        final_answer = solution_str[-1000:]

    return final_answer

def parse_model_answer(response: str) -> Optional[str]:
    if response is None or not isinstance(response, str):
        return None

    response = response.replace('*', '').replace("<｜Assistant｜>", '').replace("<｜end▁of▁sentence｜>", '')
    def clean_and_match_answer(candidate, pattern):
        # split the and get the answer
        candidate = candidate.rsplit(pattern, 1)[-1]
        # get the return 
        candidate = candidate.strip().strip('.').strip()

        return candidate

    for pattern in ["the answer is", "correct answer is", "answer is"]:
        if pattern in response:
            answer = clean_and_match_answer(response, pattern)
            # bad format
            if "(insert answer here)" in answer or "insert answer here" in answer:
                return None
            else:
                # remove new line
                return answer.split("\n")[0].strip().strip('.').strip()

    return None
def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 prompt_str: str,
                 format_reward: float = 0.0,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    answer_text = extract_solution(solution_str)
    answer_score = 0
    pred_status, gt_status = None, None
    metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'half_em': 0, 'sub_em_strict': 0, 'sub_em_loose': 0, 'total_num': 0}
    if answer_text:
        pred_status = parse_model_answer(answer_text)
        gt_status = parse_model_answer(ground_truth)
        # print(pred_status, gt_status)
        
        if pred_status and gt_status:
            metrics = calc_metrics([pred_status], [gt_status])
            metric_key = os.getenv('DOC_QA_METRIC', "sub_em_loose") 
            assert metric_key in metrics, f"Metric key {metric_key} not in {metrics.keys()}"
            answer_score = metrics[metric_key]
        else:
            answer_score = 0.0

    # store the f1 score for weight calculation
    return {
        "score": answer_score,
        "weight": metrics['f1'],
        "pred": pred_status,
        "gt": gt_status,
    }