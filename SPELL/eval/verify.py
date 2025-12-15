import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import *
from datetime import datetime
from transformers import AutoTokenizer
from pathlib import Path
from apis import *
from generate import process_answer
from metric_utils import *
    
def process_single_item(item, args):
    # from .prompts import prompt
    try:
        # Choose the appropriate prompt version

        if isinstance(item['judge_prompt'], list):
            messages = item['judge_prompt']
        else:
            messages = [{"role": "user", "content": item['judge_prompt']}]
        
        ans = query_local_vllm(messages=messages, args=args)

        # store the llm judge
        judge = extract_solution(ans)
        item["judge_output"] = judge
        if "YES" in judge and "NO" not in judge:
            item["judge_gpt"] = 1.0
            item['judge_pred'] = 'YES'
        else:
            item["judge_gpt"] = 0.0
            if 'NO' in judge :
                item['judge_pred'] = 'NO'
            else:
                item['judge_pred'] = 'INVALID'
        return item

    except Exception as e:
        traceback.print_exc()
        # Handle exceptions gracefully
        print(f"Error processing id {item.get('task', 'N/A')}: {e}")
        item["judge_output"] = "Empty"
        item["judge_gpt"] = 0.0
        return item

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    output_map = {}
    input_map = {}
    data_all = []
    judge_model_name = args.model.split("/")[-1]
    for task in args.tasks:
        in_file = f"{args.save_dir}/{task}/{args.save_file}.jsonl"
        out_file = f"{args.save_dir}/{task}/{args.save_file}_{judge_model_name}_judges.jsonl"
        input_map[task] = in_file 
        output_map[task] = out_file
        input_data = read_jsonl(in_file)
        print(f"Reading {len(input_data)} examples from {in_file}")
        data_all.extend(input_data)

    result_data = []
    exist_idx = []
    all_exist_data = []
    if args.overwrite:
        data_all = [process_answer(item, item['response']) for item in data_all]
        for out_file in output_map.values():
            with open(out_file, "w") as f:
                pass
    for out_file in output_map.values():
    # cache
        if os.path.exists(out_file):
            exist_data = read_jsonl(out_file)
            exist_idx.extend([item['_id'] for item in exist_data])
            all_exist_data.extend(exist_data)
    exist_idx = set(exist_idx)
    for item in data_all:
        if item["_id"] not in exist_idx:
            result_data.append(item)

    if len(result_data) == 0 or len(all_exist_data) == len(data_all):
        compute_metrics(all_exist_data, args)
        return
        
    fout_map = {}
    for key,val in output_map.items():
        fout_map[key] = open(val, 'a', encoding='utf-8')
    
    new_result_data = all_exist_data
    data_to_eval = []
    for item in result_data:
        # print(item.keys())
        answer = item["reward_model"]["ground_truth"]
        pred = item["pred"]  
        # for multiple-choice tasks, we don't apply llm judge
        if item['task'] == "longbench-v2" or (not args.judge_all and float(item['judge']) == 1.0):
            # question = item['prompt'].split("</text>\n\n")[-1].split("\n\nPlease reason step by step")[0]
            item["judge_gpt"] = 0.0
            item["judge_output"] = "Empty"
            item["judge_model"] = judge_model_name
            new_result_data.append(item)
            continue

        if "ori_question" in item:
            question = item['ori_question']
        else:
            question = item['prompt']
            if isinstance(question, list):
                # Extract user content from chat format
                for msg in reversed(question):
                    if msg.get('role') == 'user':
                        question = msg.get('content', '')
                        break
            
            if "</text>\n\n" in question:
                question = question.split("</text>\n\n")[-1].split("\n\n")[0]

        messages=[
            {"role": "user", "content": LLM_JUDGE_PROMPT.format(problem=question, answer_1=pred, answer_2=answer)}
        ]
        item['judge_prompt'] = messages
        data_to_eval.append(item)

    print(f"Samples to evaluate: {len(data_to_eval)}")


    # compute metrics use api
    max_workers = args.n_proc  
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_single_item, item, args): item 
            for item in data_to_eval
        }

        with tqdm(total=len(data_to_eval)) as pbar:
            for future in as_completed(future_to_item):
                try:
                    item = future.result()
                    item["judge_model"] = judge_model_name
                    new_result_data.append(item)
                    fout_map[item['task']].write(json.dumps(item, ensure_ascii=False) + '\n')
                    fout_map[item['task']].flush()
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"Error: {str(e)}")
                    pbar.update(1)
                    
    if len(new_result_data) == len(data_all):
        compute_metrics(new_result_data, args)
    else:
        print(f"Left {len(data_all) - len(new_result_data)} need to compute")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results/docmath")
    parser.add_argument("--save_file", "-f", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model", "-j", type=str, default="deepseek-v3")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--n_proc", type=int, default=100)
    parser.add_argument("--tasks", nargs="+", required=True, 
                       help="List of tasks to process (docmath, 2wikimqa, hotpotqa, etc.)")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_input_len", type=int, default=120000)
    parser.add_argument("--max_output_len", type=int, default=10000)
    parser.add_argument("--temperature",  type=float, default=0.7)
    parser.add_argument("--gpu_memory_utilization",  type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--judge_all", action="store_true")
    parser.add_argument("--thinking_budget", type=int, default=4096)
    args = parser.parse_args()
    main()