import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import numpy as np
from verl.utils.reward_score import docmath, long, docqa
from datasets import load_dataset, concatenate_datasets
from common import *
from metric_utils import *

def format_prompt(item):
    prompt_template =  SOLVER_PROMPT_MAPPING

    if item['task'] == 'docmath':
        template = prompt_template["docmath_qa"]
        context = '\n'.join(item['paragraphs'])
        question = item["question"].strip()
    elif item['task'] == 'frames':
        template = prompt_template["doc_general_qa"]
        context = "\n".join([f"{i['title']}\n{i['text']}" for i in item['wiki_items']])
        question = item['Prompt'].strip()
    elif item['task'] in ["2wikimqa", "hotpotqa", "musique"]:
        template = prompt_template["doc_general_qa"]
        context = item["context"]
        question = item['input'].strip()
    elif item['task'] in ["longbench-v2"]:
        template = prompt_template["doc_mc"]
        context = item["context"]
        instruction = """What is the correct answer to this question: {question}
    Choices:
    (A) {choice_A}
    (B) {choice_B}
    (C) {choice_C}
    (D) {choice_D}
    """
        question = instruction.format(question=item['question'], choice_A=item['choice_A'], choice_B=item['choice_B'], choice_C=item['choice_C'], choice_D=item['choice_D'])
        # return [{"role":"user","content": prompt}]
    else:
        raise ValueError(f"Unknown task: {item['task']}, please add your task definition")
    # template = template_0shot
    prompt = template.replace('{content}', context.strip()).replace('{question}', question)
    return [{"role":"user","content": prompt}]

def process_answer(item,response):
    gt = item["reward_model"]["ground_truth"]
    item['response'] = response
    if item['task'] == 'docmath':
        pred = docmath.extract_solution(response)
        item['pred'] = docmath.parse_model_answer(pred) if pred else docmath.parse_model_answer(response)
        item['judge'] = docmath.get_acc(str(item['pred']), str(gt)) if item['pred'] else 0

    elif item['task'] in ["2wikimqa", "hotpotqa", "musique", 'frames']:
        pred = docqa.extract_solution(response)
        item['pred'] = docqa.parse_model_answer(pred) if pred else docqa.parse_model_answer(response)
        metrics = docqa.calc_metrics([str(item["pred"])], [str(gt)])
        # sub-em score
        item['judge'] = metrics.get('sub_em', 0.0)
        item['f1_score'] = metrics.get("f1", 0.0)
        item['em_score'] = metrics.get('em', 0.0)

    elif item['task'] in ["longbench-v2"]:
        pred = long.extract_solution(response)
        item['pred'] = long.parse_model_answer(pred) if pred else long.parse_model_answer(response)
        item['judge'] = 1 if item['pred'] and gt == item['pred'] else 0

    else:
        raise ValueError(f"Unknown task: {item['task']}, please add your task definition")

    return item

# get evaluation dataset
def construct_data(args):
    input_dir, tasks = args.input_dir, args.tasks
    dataset = []
    for task in tasks:
        if task == "docmath":
            new_data = []
            all_data = load_dataset("yale-nlp/DocMath-Eval")
            for sub_domain in ["complong", "compshort", "simplong", "simpshort"]:
                data = all_data[f"{sub_domain}-testmini"]
                for i,item in enumerate(data):
                    item['task'] = task
                    new_item = dict(
                        task=task,
                        sub_task=sub_domain,
                        data_source=f"{task}_{sub_domain}",
                        prompt=format_prompt(item),
                        ability="doc-qa",
                        reward_model={"style": "rule", "ground_truth": item['ground_truth']},
                        id=f"{task}_{sub_domain}_{i}"
                    )     
                    new_data.append(new_item)         
            dataset.extend(new_data)
        elif task in ["2wikimqa", "hotpotqa", "musique"]:
        # longbench: 
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            new_data = []
            for i,item in enumerate(data):
                item['task'] = task
                new_item = dict(
                    task=task,
                    sub_task=task,
                    data_source=f"longbench-{task}",
                    prompt=format_prompt(item),
                    ability="doc-qa",
                    reward_model={"style": "rule", "ground_truth": item["answers"][0]},
                    id=f"longbench_{task}_{i}"
                )          
                new_data.append(new_item)         
            dataset.extend(new_data)
        elif task == "frames":
            data = load_dataset('Tongyi-Zhiwen/frames')
            new_data = []
            for i,item in enumerate(data):
                item['task'] = task
                new_item = dict(
                    task=task,
                    sub_task=task,
                    data_source=task,
                    prompt=format_prompt(item),
                    ability="doc-qa",
                    reward_model={"style": "rule", "ground_truth": item["Answer"]},
                    id=f"{task}_{i}"
                )          
                new_data.append(new_item)         
            dataset.extend(new_data)
        elif task == "longbench-v2":
            data = load_dataset('THUDM/LongBench-v2', split='train')
            new_data = []
            domain_to_new_data = defaultdict(list)
            for i,item in enumerate(data):
                item['task'] = task
                new_item = dict(
                    task=task,
                    sub_task=item['domain'],
                    data_source=f"{task}_{item['domain']}",
                    prompt=format_prompt(item),
                    ability="doc-qa",
                    reward_model={"style": "rule", "ground_truth": str(item['answer'])},
                    id=item['_id'],
                    # extra_info={
                    #     "domain": row['domain'],
                    #     "sub_domain": row["sub_domain"],
                    #     "difficulty": row['difficulty'],                       
                    # }
                )          
                new_data.append(new_item) 
                domain_to_new_data[item['domain']].append(new_item)       
            dataset.extend(new_data)

    return dataset


def get_tokenized_data(args):    
    max_len=args.max_input_len
    # tokenized the prompt and get chunk
    model_prefix = args.save_file.split("_")[0]
    # store the tokenized data, incase repeat tokenize process
    tokenzied_datadet_save_dir = f"{args.input_dir}/{model_prefix}_tokenized_prompt_max_len{max_len}.jsonl"
    dataset = construct_data(args)
    print(f"original data len {len(dataset)}")
    # contruct data only

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    # chunk the input to calid length 
    for item in tqdm(dataset):
        prompt = tokenizer.apply_chat_template(item["prompt"], add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
        item["prompt"] = prompt
    if not args.overwrite:
        save_jsonl(dataset, tokenzied_datadet_save_dir)
    return dataset
        
def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    output_map = {}
    for task in args.tasks:
        os.makedirs(f"{args.save_dir}/{task}", exist_ok=True)
        out_file = f"{args.save_dir}/{task}/{args.save_file}.jsonl"
        output_map[task] = out_file
    # get testing data
    # recompute all metrics
    if args.recompute_metric_only:
        all_data = []
        for out_file in output_map.values():
            if os.path.exists(out_file):
                with open(out_file, encoding='utf-8') as f:
                    all_data.extend([json.loads(line) for line in f])
        all_data = [process_answer(item, item['response']) for item in all_data]
        compute_metrics(all_data, args)
        exit(0)

    dataset = get_tokenized_data(args)
    if args.debug:
        import random
        random.seed(42)
        dataset = random.sample(dataset,100)

    import copy
    # repeat interleave
    dataset = [copy.deepcopy(item) for item in dataset for _ in range(args.n_sampling) ]
    print(f"n_sampling data len {len(dataset)}")

    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx  
        data_all.append(item)

    print(data_all[0]["_id"])
    print(data_all[-1]["_id"])
    # cache
    has_data = {}
    for out_file in output_map.values():
        if os.path.exists(out_file):
            with open(out_file, encoding='utf-8') as f:
                has_data.update({json.loads(line)["_id"]: 0 for line in f})
    fout_map = {}
    for key,val in output_map.items():
        fout_map[key] = open(val, 'a', encoding='utf-8')

    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    if len(data) == 0:
        print("No new data")
        exit(0)

    from vllm import LLM, SamplingParams   
    import torch

    llm = LLM(
        model=args.model,
        max_model_len=args.max_input_len + args.max_output_len,
        # enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=torch.cuda.device_count(), # set to auto
        trust_remote_code=True
    )# Split data into chunks

    chunk_size = max(1, len(data) // args.split)  # Ensure at least 1 item per chunk
    data_chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    all_output = []

    for chunk_idx, chunk in enumerate(data_chunks):
        print(f"Processing chunk {chunk_idx+1}/{len(data_chunks)} (size={len(chunk)})")
        
        prompts = [item["prompt"] for item in chunk]
        
        # Print first prompt of each chunk for debugging
        print(f"First prompt in chunk {chunk_idx+1}:")
        print(prompts[0])
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            n=1,
            max_tokens=args.max_output_len,
            seed=args.seed
        )
        
        outputs = llm.generate(
            prompts,
            sampling_params
        )
        
        outputs = [output.outputs[0].text for output in outputs]
        
        for item, response in zip(chunk, outputs):
            item = process_answer(item, response)
            all_output.append(item)
            fout_map[item['task']].write(json.dumps(item, ensure_ascii=False) + '\n')
            fout_map[item['task']].flush()
        
        print(f"Completed chunk {chunk_idx+1}/{len(data_chunks)}")

    compute_metrics(all_output, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, default=None)
    parser.add_argument("--save_dir", "-s", type=str, default=None)
    parser.add_argument("--save_file", "-f", type=str, default="model_output")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--tasks", nargs="+", default=["docmath", "frames", "2wikimqa", "hotpotqa", "musique", "longbench-v2"],
                       help="List of tasks to process (docmath, 2wikimqa, hotpotqa, etc.)")
    parser.add_argument("--tokenizer", "-t", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--split", type=int, default=8)
    parser.add_argument("--n_sampling", "-p", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_input_len", type=int, default=120000)
    parser.add_argument("--max_output_len", type=int, default=10000)
    parser.add_argument("--temperature",  type=float, default=0.7)
    parser.add_argument("--gpu_memory_utilization",  type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--recompute_metric_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main()