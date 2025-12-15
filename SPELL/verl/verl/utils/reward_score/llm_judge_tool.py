import openai
import time
import os


ORM_USER_TEMPLATE = """## TASK 
You are an expert in verifying if two answers are the same. Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent. Your task is to determine two answers are equivalent, without attempting to solve the original problem. 
        
## Instruction 
1. Carefully compare the Answer 1 and Answer 2. 
2. Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.
3. For numerical answers, you should allow a **±0.15% tolerance**.
4. Your decision **must be** one of the "[[YES]]" or "[[NO]]".

## Input Data
- Problem: {problem}
- Answer 1: {answer_1}
- Answer 2: {answer_2} 

## Output Format 
Provide your final evaluation in the following format: 
"Explanation:" Provide an explanation for why the answers are equivalent or not.
"Decision:" "[[YES]]" or "[[NO]]"

Please proceed with the evaluation."""

def call_oai_rm_llm(
    prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model_id: str = "gpt-4o",
    retry_count: int = 3
) -> tuple[str, list[str]]:
    """Call OpenAI API with retry logic.

    Args:
        prompt: The text prompt to send to the model
        system_prompt: System instruction for the model
        n: Number of completions to generate
        temperature: Sampling temperature
        model_id: OpenAI model ID to use
        retry_count: Number of retries on rate limit errors

    Returns:
        Generated text(s) from the model
    """
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{os.getenv('VERIFIER_HOST')}:{os.getenv('VERIFIER_PORT')}/v1"
    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    backoff = 1
    retry_count = int(retry_count)

    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as exc:
            if "429" in str(exc):
                print("Retry due to rate limit: ", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            print("Exception: ", exc)
            return []

    if n == 1:
        return response.choices[0].message.content
    return [choice.message.content for choice in response.choices]

def call_reward_model(problem: str, model_answer: str, ground_truth: str):
    orm_response = call_oai_rm_llm(
        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
        temperature=0.0,
        model_id=os.getenv("VERIFIER_PATH"),
        retry_count=3,
    )
    if "[[YES]]" in orm_response and "[[NO]]" not in orm_response:
        return 1.0
    else:
        return 0.0
