import os
import re
from typing import Dict, Tuple, Optional
import math
from sympy import Rational
import numpy as np

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

def round_up_to_decimal(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def is_number(string):
    pattern = r'^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$'
    match = re.match(pattern, string)
    return bool(match)

def is_scientific_number(string):
    pattern = r'^[-+]?\d+(\.\d+)?e[-]?\d+$'
    match = re.match(pattern, string)
    return bool(match)

def normalize(prediction: str):
    if prediction is None:
        return None
    # Preprocessing the string [Stage 1]
    prediction = prediction.strip()
    prediction = prediction.rstrip('.')
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else '999999999'

    for money in ["£", "€", "¥", "million", "billion", "thousand", "US", "USD", "RMB"]:
        prediction = prediction.replace(money, '')
        
    # Replace special tokens
    if '=' in prediction:
        prediction = prediction.split('=')[-1].strip()
    if '≈' in prediction:
        prediction = prediction.split('≈')[-1].strip()
    if '`' in prediction:
        prediction = prediction.replace('`', '')
    if '%' in prediction:
        prediction = prediction.replace('%', '')
    if '$' in prediction:
        prediction = prediction.replace('$', '')
    if '°' in prediction:
        prediction = prediction.replace('°', '')

    # Detect the boolean keyword in the generation
    if prediction in ['true', 'yes', 'false', 'no']:
        if prediction == 'true' or prediction == 'yes':
            prediction = 'True'
        else:
            prediction = 'False'
    if 'True' in prediction or 'False' in prediction:
        prediction = 'True' if 'True' in prediction else 'False'

    # Detect the approximation keyword
    if 'approximately' in prediction:
        prediction = prediction.replace('approximately', '').strip()
    if ' or ' in prediction:
        prediction = prediction.split(' or ')[0]

    # Drop the units before and after the number
    if re.match(r'[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$', prediction):
        prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$', prediction).group(1)
    if re.match(r'[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(r'[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$', prediction).group(1)
    if re.match(r'[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$', prediction):
        prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$', prediction).group(1)
    if re.match(r'[^-+\d]{1,2}(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(r'[^-+\d]{1,2}((?:[\d,]*\.*\d+))$', prediction).group(1)

    # Preprocessing the number [Stage 1]
    if '10^' in prediction:
        prediction = re.sub(r'10\^(-?\d+)', r'math.pow(10, \1)', prediction)
    if ' x ' in prediction:
        prediction = prediction.replace(' x ', '*')
    if ' × ' in prediction:
        prediction = prediction.replace(' × ', '*')
    if is_number(prediction):
        prediction = prediction.replace(',', '')

    # Preprocessing the option [Stage 3]
    if '(a)' in prediction or '(b)' in prediction or '(c)' in prediction or '(d)' in prediction:
        prediction = '"' + re.search(r'\([a-d]\)', prediction).group(0) + '"'

    # If the prediction is empty, use dummy '0'
    if not prediction:
        prediction = '999999999'

    # Converting the string answer to a number/list/bool/option
    try:
        prediction = eval(prediction)
    except Exception:
        # TO CHECK
        prediction = 999999999

    # Performing common type conversion
    if isinstance(prediction, (set, tuple)):
        prediction = list(prediction)
        if isinstance(prediction[0], complex):
            prediction = [tmp.real for tmp in prediction]
        elif isinstance(prediction[0], Rational):
            prediction = [float(tmp) for tmp in prediction]
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    else:
        if isinstance(prediction, complex):
            prediction = prediction.real
        elif isinstance(prediction, Rational):
            prediction = float(prediction)

    return prediction

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
    # maybe don;t have <think> due to different tokenzier
    elif "</think>" in solution_str or "\\boxed{" in solution_str or "answer is" in solution_str:
        final_answer = solution_str.split("</think>")[-1].strip()
    else:
        # return the last 2000 characters to avoid overlong answers
        final_answer = solution_str[-300:]     

    return final_answer
def parse_model_answer(response: str) -> Optional[str]:
    """Parses the final answer from the model's response text.
    
    Args:
        response: Text extracted from the model's response
        
    Returns:
        The final answer as a numeric value (string), or None if not found
    """

    # Extract and check the boxed answer
    # boxed_pred = last_boxed_only_string(response)
    # extracted_pred = None
    # if boxed_pred:
    #     extracted_pred = remove_boxed(boxed_pred).replace(',', '').rstrip('.')
    # return extracted_pred
    # Remove any asterisks or other unwanted characters
    if response is None or not isinstance(response, str):
        return None
    response = response.replace('*', '')
    
    if "\\boxed{" in response:
        boxed_pred = last_boxed_only_string(response)
        extracted_pred = None
        if boxed_pred:
            extracted_pred = remove_boxed(boxed_pred).replace(',', '').rstrip('.')
        return extracted_pred

    patterns = [
        r'the answer is \((\=?\≈?\`?\%?\$?\°?\£?\€?\¥?-?[0-9\.,]+)\)',
        r'the answer is (\=?\≈?\`?\%?\$?\°?\£?\€?\¥?-?[0-9\.,]+)',
        r'answer is \((\=?\≈?\`?\%?\$?\°?\£?\€?\¥?-?[0-9\.,]+)\)',
        r'answer is (\=?\≈?\`?\%?\$?\°?\£?\€?\¥?-?[0-9\.,]+)',
    ]
    
    for pattern in patterns:
        match = re.findall(pattern, response, re.IGNORECASE)
        if match:
            return match[-1].replace(',', '').rstrip('.')

    return None

def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.0015
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False

def compare_two_numbers(p, gt):
    if isinstance(p, int) or isinstance(p, float):
        pass
    elif isinstance(p, list) or isinstance(p, bool) or isinstance(p, str):
        return False
    elif isinstance(p, tuple) or isinstance(p, complex) or isinstance(p, dict):
        return False
    else:
        raise ValueError(p)

    try:
        valid_scales = [100,1000,1000000,1000000000]
        v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))

        for valid_scale_value in valid_scales:
            if v2 <= 2 * v1 / valid_scale_value and within_eps(pred=v2*valid_scale_value, gt=v1):
                return True

        if round_up_to_decimal(v1, 3) == round_up_to_decimal(v2, 3):
            return True

        return within_eps(pred=p, gt=gt)
    except OverflowError:
        return False

def get_acc(prediction, gt, cot=True):
    if cot:
        prediction = normalize(prediction)
        gt = normalize(gt)
        if prediction is None or gt is None:
            return 0
    else:
        prediction = float(prediction)
    
    answer_type = type(gt).__name__
    # print(f"answer_type::{answer_type}")
    assert answer_type in ["int", "float", "float64", "bool"], f"Unsupported answer_type::{answer_type}"
    if isinstance(prediction, (str, int, float, bool)) or isinstance(prediction, list):
        # Comparing prediction against the reference
        if answer_type in ['bool']:
            acc = int(prediction == gt)
        elif answer_type == 'int':
            acc = int(compare_two_numbers(prediction, gt))
        elif answer_type == 'float' or answer_type == 'float64':
            acc = int(compare_two_numbers(prediction, gt))
        else:
            acc = 0
    else:
        acc = 0
        print("Error: ", prediction, type(prediction))
    return acc
    
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
    if answer_text:
        pred_status = parse_model_answer(answer_text)
        gt_status = parse_model_answer(ground_truth)
        
        if pred_status and gt_status:
            answer_score = get_acc(pred_status, gt_status)
        else:
            answer_score = 0
    return {
        "score": answer_score,
        "weight": 1.0,
        "pred": pred_status,
        "gt": gt_status
    }
    