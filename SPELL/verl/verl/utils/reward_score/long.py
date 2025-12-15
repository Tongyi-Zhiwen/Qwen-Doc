import re
from typing import Dict, Tuple, Optional


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
            # print("[Error] The thinkinig length is too long.")
            return "Empty"
        else:
            final_answer = solution_str.split("</think>")[-1].strip()
    # maybe don;t have <think> due to different tokenzier
    elif "</think>" in solution_str or "answer is" in solution_str:
        final_answer = solution_str.split("</think>")[-1].strip()
    else:
        # return the last 500 characters to avoid overlong answers
        final_answer = solution_str[-300:]        

    return final_answer

def parse_model_answer(response: str) -> Optional[str]:
    """Parses model's multiple-choice answer text (A-D) from response.
    
    Optimizations:
    - Consolidated regex patterns
    - More efficient pattern matching order
    - Better error handling
    - Clearer return type annotation
    
    Args:
        response: Text extracted from model's response
        
    Returns:
        Extracted answer (A-D) or None if not found
    """
    if not response or not isinstance(response, str):
        return None
    
    response = response.replace('*', '').strip()
    
    # Consolidated regex patterns ordered by likelihood/importance
    patterns = [
        r'\\boxed\{([A-D])\}',                   # Boxed format
        r'correct answer is \(([A-D])\)',  # Most specific format
        r'correct answer is ([A-D])',      # Common format
        r'answer is \(([A-D])\)',              # Less specific format
        r'answer is ([A-D])',                  # Least specific format
    ]
    
    # Check other patterns
    for pattern in patterns:
        match = re.findall(pattern, response, re.IGNORECASE)
        if match:
            if match[-1] in ['A', 'B', 'C', 'D']:
                return match[-1]
            else:
                return None
                
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
    # Validate answer content
    answer_score = 0
    pred_status, gt_status = None, None
    if answer_text:
        try:
            pred_status = parse_model_answer(answer_text)
            gt_status = parse_model_answer(ground_truth)
            if pred_status and gt_status:

                if pred_status == gt_status:
                    answer_score = 1
                else:
                    answer_score = 0
            else:
                answer_score = 0
        except Exception as e:
            print(e)
            answer_score = 0
    return {
        "score": answer_score,
        "weight": 1.0,
        "pred": pred_status,
        "gt": gt_status,
    }
