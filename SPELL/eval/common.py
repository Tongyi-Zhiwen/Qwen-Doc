import json
import pandas as pd

LLM_JUDGE_PROMPT = """## TASK 
I need your help in evaluating an answer provided by an LLM against a ground truth answer. Your task is to determine if the ground truth answer is present in the LLM's response. Please analyze the provided data and make a decision.
        
## Instruction 
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer". 
2. Consider the substance of the answers - look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning. 
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answer" are present in the "Predicted Answer".
4. Your decision **must be** one of the "[[YES]]" or "[[NO]]".

## Input Data
- Question: {problem}
- Predicted Answer: {answer_1}
- Ground Truth Answer: {answer_2} 

## Output Format 
Provide your final evaluation in the following format: 
"Explanation:" "How you made the decision
"Decision:" "[[YES]]" or "[[NO]]"

Please proceed with the evaluation."""



docmath_qa_template = """You are an expert in document analysis and numeric reasoning, you are supposed to answer the given question based on the provided context. You need to first think through the problem step by step, documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as "Therefore, the answer is (insert answer here)". The final answer should be a numeric value.

<text>
{content}
</text>

Question: {question}

Please reason step by step, and format your answer as follows: "Therefore, the answer is (insert answer here)."
"""

doc_mc_template = """Please read the following text and answer the question below.

<text>
{content}
</text>

{question}

Format your answer as follows: "The correct answer is (insert answer here)."
"""

doc_general_qa_template = """Please read the following text and answer the question below. 

<text>
{content}
</text>

Question: {question}

Format your answer as follows: "The correct answer is (insert answer here)."
"""


SOLVER_PROMPT_MAPPING = {
    "docmath_qa": docmath_qa_template,
    "doc_mc": doc_mc_template,
    "doc_general_qa": doc_general_qa_template,
    "default": docmath_qa_template,
}

def read_json(file_path):
    """Read json file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_jsonl(file_path):
    """Read jsonl file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def save_json(data, file_path):
    """Save json file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Save to: {file_path}")

def save_jsonl(data, file_path):
    """Save jsonl file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Save to: {file_path}")


def read_parquet(file_path):
    try:
        df = pd.read_parquet(file_path)
        records = df.to_dict('records')
        return records
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def save_parquet(data, file_path):
    try:
        df = pd.DataFrame(data)
        df.to_parquet(file_path)
        print(f"Save to: {file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

