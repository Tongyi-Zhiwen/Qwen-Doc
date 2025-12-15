from collections import Counter, defaultdict
from functools import partial
from typing import Any, Dict, List, Callable, Union

import numpy as np
import torch
import pandas as pd
import os
from datetime import datetime
from common import save_json
from pathlib import Path


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
    else:
        # for none thinking model, keep the last 300 characters
        final_answer = solution_str.split("</think>")[-1].strip()

    return final_answer
def update_or_create_excel(file_path: str, new_row: dict, task_domain: str = "doc-eval") -> None:
    file = Path(file_path)
    sheets_dict = {}

    # Read existing worksheets (if file exists)
    if file.exists():
        with pd.ExcelFile(file_path, engine='openpyxl') as xls:
            sheets_dict = {sheet_name: pd.read_excel(xls, sheet_name) for sheet_name in xls.sheet_names}

    # Get target worksheet or create new
    if task_domain in sheets_dict:
        df = sheets_dict[task_domain]
        result_list = df.to_dict(orient='records')
    else:
        result_list = []
        df = pd.DataFrame(columns=new_row.keys())
        sheets_dict[task_domain] = df

    # Update or add entry
    in_list = False
    for i, d in enumerate(result_list):
        if d.get('model') == new_row.get('model'):
            in_list = True
            for k in new_row.keys():
                # Round numerical values to 2 decimals when updating
                if isinstance(new_row[k], (int, float)):
                    result_list[i][k] = round(new_row[k] * 100, 2)
                else:
                    result_list[i][k] = new_row[k]
            break

    if not in_list:
        # Round numerical values in new_row before adding
        rounded_row = {k: round(v * 100, 2) if isinstance(v, (int, float)) else v 
                      for k, v in new_row.items()}
        result_list.append(rounded_row)

    # Update DataFrame with rounded values
    updated_df = pd.DataFrame(result_list)
    
    # Apply rounding to all numeric columns (except 'model')
    numeric_cols = updated_df.select_dtypes(include=['number']).columns
    updated_df[numeric_cols] = updated_df[numeric_cols].round(2)
    
    sheets_dict[task_domain] = updated_df

    # Write all worksheets (preserving others)
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        for sheet_name, df_sheet in sheets_dict.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"File {file_path} '{task_domain}' Upated Successfully!")

def compute_metrics(data: List[Dict[str, Any]], args: Any):
    """
    Computes and aggregates evaluation metrics, sending question-level aggregated
    data to process_spr_validation_metrics for further processing.

    Args:
        data: A list of dictionaries, where each dictionary represents an item
              with metrics like 'task', 'id', 'judge', 'f1_score', 'em_score', 'judge_gpt'.
        args: An object containing arguments like save_file and save_dir.
    """

    # 1. Group data by (task, id) to aggregate metrics per unique question.
    # This prepares the data so each (task, id) pair represents one 'sample'
    # for process_spr_validation_metrics.

    save_dir = f"{args.save_dir}/all_metrics"
    judge_model = getattr(args, "model", "rule").split("/")[-1]

    os.makedirs(save_dir, exist_ok=True)
    grouped_data_by_question_id = defaultdict(lambda: defaultdict(list))
    for item in data:
        task_name = item.get('task', 'unknown')
        # 'id' is assumed to be the unique identifier for a question
        question_id = item.get('id', 'unknown_id') 

        rule_score = item.get("judge", 0)
        f1_score = item.get("f1_score", 0.0)
        em_score = item.get('em_score', 0.0)
        llm_score = item.get("judge_gpt", 0.0)
        
        # Calculate 'score' for the current item as in the original function
        current_item_score = max(rule_score, llm_score)

        # Store all individual scores for later aggregation per question_id
        grouped_data_by_question_id[(task_name, question_id)]['rule_scores'].append(rule_score)
        grouped_data_by_question_id[(task_name, question_id)]['llm_scores'].append(llm_score)
        grouped_data_by_question_id[(task_name, question_id)]['f1_scores'].append(f1_score)
        grouped_data_by_question_id[(task_name, question_id)]['em_scores'].append(em_score)
        grouped_data_by_question_id[(task_name, question_id)]['total_scores'].append(current_item_score)

    # print(grouped_data_by_question_id)

    # 2. Prepare data for process_spr_validation_metrics.
    # Each entry in these lists will correspond to a unique (task, question_id) pair.
    data_sources_for_spr = []   # List of task names
    sample_inputs_for_spr = []  # List of question IDs (used as 'prompts' in spr function)
    infos_dict_for_spr = defaultdict(list) # Stores aggregated metric values for each (task, question_id)

    # Aggregate scores for each unique (task, question_id) pair.
    # For binary/score metrics, taking the maximum value across multiple responses for the same question
    # implies that if any response was correct/good, the question is considered correct/good.
    for (task_name, question_id), scores_lists in grouped_data_by_question_id.items():
        data_sources_for_spr.append(task_name)
        sample_inputs_for_spr.append(question_id) # Using question_id as the prompt identifier

        # Aggregate metrics for this specific question_id within this task.
        # Use max for scores where a single "success" for a question is sufficient.
        infos_dict_for_spr['rule_score'].append(scores_lists['rule_scores'] if scores_lists['rule_scores'] else [0])
        infos_dict_for_spr['llm_score'].append(scores_lists['llm_scores'] if scores_lists['llm_scores'] else [0.0])
        infos_dict_for_spr['f1_score'].append(scores_lists['f1_scores'] if scores_lists['f1_scores'] else [0.0])
        infos_dict_for_spr['em_score'].append(scores_lists['em_scores'] if scores_lists['em_scores'] else [0.0])
        infos_dict_for_spr['total_score'].append(scores_lists['total_scores'] if scores_lists['total_scores'] else [0.0])
        
        # Note: 'pred' is not directly aggregated here as it's typically a textual prediction
        # used for majority voting on individual responses. After question-level aggregation,
        # its role would need redefinition if it's meant to be passed to process_spr_validation_metrics.

    # 3. Call process_spr_validation_metrics with the prepared data.
    # The 'group_prefix' argument is passed through from 'args' if it exists, otherwise defaults to False.
    processed_metrics = process_spr_validation_metrics(
        data_sources=data_sources_for_spr,
        sample_inputs=sample_inputs_for_spr,
        infos_dict=dict(infos_dict_for_spr), # Convert defaultdict to dict before passing
        group_prefix=True # Assuming group_prefix might be in args
    )

    metric_dict = {}
    n_max = 0
    for data_source, var2metric2val in processed_metrics.items():
        # core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
            for metric_name, metric_val in metric2val.items():
                pfx = f"{data_source}/{var_name}/{metric_name}"
                metric_dict[pfx] = metric_val

    print(f"{n_max=}")

    # 4. Re-structure the output for the original compute_metrics DataFrame and print/save logic.
    metrics_for_df = []
    scores_for_excel_update = {}
    scores_for_excel_update['model'] = args.save_file + "_" + judge_model
    if args.overwrite:
        scores_for_excel_update['model'] = scores_for_excel_update['model'] + "_recompute"


    # To get the count of unique questions per task, re-count from the original data.
    unique_questions_per_task = defaultdict(set)
    for item in data:
        task_name_item = item.get('task', 'unknown')
        question_id_item = item.get('id', 'unknown_id')
        unique_questions_per_task[task_name_item].add(question_id_item)

    # Iterate through processed_metrics to extract task-level averages.
    # The 'mean@1' key is used because process_spr_validation_metrics operates on
    # already-aggregated question-level data, effectively having one response per sample.
    for data_source_name, var_metrics in processed_metrics.items():
        if data_source_name == "overall" or '_' in data_source_name :
            # Skip 'overall' and prefix-grouped results for this loop; handle them later or separately
            # if group_prefix is enabled, the prefix-grouped results are handled explicitly later.
            # Here, we only want the original task names as data_sources_for_spr will contain them.
            if data_source_name not in unique_questions_per_task and data_source_name != "overall":
                continue # Skip if it's a prefix-grouped item and not an original task name

        # Get aggregated scores for the current task
        total_score_metric = var_metrics.get('total_score', {})
        task_avg_score = total_score_metric.get(f'mean@{n_max}', 0.0)
        task_rule_score = var_metrics.get('rule_score', {}).get(f'mean@{n_max}', 0.0)
        task_llm_score = var_metrics.get('llm_score', {}).get(f'mean@{n_max}', 0.0)
        task_f1_score = var_metrics.get('f1_score', {}).get(f'mean@{n_max}', 0.0)
        task_em_score = var_metrics.get('em_score', {}).get(f'mean@{n_max}', 0.0)
        
        # Get the count of unique questions for this task
        task_count = len(unique_questions_per_task[data_source_name])

        cur_metric = {
            'Model': args.save_file,
            'Task': data_source_name,
            'Average Score': task_avg_score,
            "Rule Score": task_rule_score,
            "F1 Score": task_f1_score,
            "EM Score": task_em_score,
            "LLM Score": task_llm_score,
            "Judge Model": judge_model,
            'Count': task_count
        }

        for i in range(2, n_max + 1):
            if f'best@{i}/mean' in total_score_metric.keys():
                cur_metric[f'Pass@{i}'] = total_score_metric[f'best@{i}/mean']
                cur_metric[f'Rule Pass@{i}'] = var_metrics.get('rule_score', {}).get(f'best@{i}/mean', 0.0)
                cur_metric[f'LLM Pass@{i}'] = var_metrics.get('llm_score', {}).get(f'best@{i}/mean', 0.0)

        metrics_for_df.append(cur_metric)
        scores_for_excel_update[data_source_name] = task_avg_score

    # Add overall metrics to the DataFrame
    if "overall" in processed_metrics:
        overall_metrics = processed_metrics["overall"]
        total_score_metric = overall_metrics.get('total_score', {})
        overall_avg_score = total_score_metric.get(f'mean@{n_max}', 0.0)
        overall_rule_score = overall_metrics.get('rule_score', {}).get(f'mean@{n_max}', 0.0)
        overall_llm_score = overall_metrics.get('llm_score', {}).get(f'mean@{n_max}', 0.0)
        overall_f1_score = overall_metrics.get('f1_score', {}).get(f'mean@{n_max}', 0.0)
        overall_em_score = overall_metrics.get('em_score', {}).get(f'mean@{n_max}', 0.0)
        
        total_unique_questions = sum(len(ids) for ids in unique_questions_per_task.values())

        cur_metric = {
            'Model': args.save_file,
            'Task': 'OVERALL',
            'Average Score': overall_avg_score,
            "Rule Score": overall_rule_score,
            "F1 Score": overall_f1_score,
            "EM Score": overall_em_score,
            "LLM Score": overall_llm_score,
            "Judge Model": judge_model,
            'Count': total_unique_questions
        }
        
        for i in range(2, n_max + 1):
            if f'best@{i}/mean' in total_score_metric.keys():
                cur_metric[f'Pass@{i}'] = total_score_metric[f'best@{i}/mean']
                cur_metric[f'Rule Pass@{i}'] = overall_metrics.get('rule_score', {}).get(f'best@{i}/mean', 0.0)
                cur_metric[f'LLM Pass@{i}'] =  overall_metrics.get('llm_score', {}).get(f'best@{i}/mean', 0.0)

        metrics_for_df.append(cur_metric)
        # scores_for_excel_update["Average"] = overall_avg_score
        
    # Create DataFrame
    df = pd.DataFrame(metrics_for_df)

    # Print results
    print("\n" + "="*50)
    print("Evaluation Results Summary")
    print(df)
    
    # Generate filename with timestamp
    xlsx_filename = f"{args.save_file}_judge{judge_model}_evaluation_metrics.xlsx"
    xlsx_path = os.path.join(save_dir, xlsx_filename)


    save_json([metric_dict],f"{save_dir}/{args.save_file}_judge{judge_model}_evaluation_metrics.json")
    
    # Save to Excel
    os.makedirs(save_dir, exist_ok=True) # Ensure save directory exists
    df.to_excel(xlsx_path, index=False)
    # scores_for_excel_update['judge_model'] = 
    update_or_create_excel(os.path.join(save_dir, "all_results.xlsx") ,scores_for_excel_update, "doc-eval")
    print(f"\nMetrics saved to: {xlsx_path}")


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    # Handle empty data to prevent errors
    if not data:
        return [(0.0, 0.0) for _ in range(len(reduce_fns))]

    for _ in range(n_bootstrap):
        # Ensure subset_size does not exceed len(data) when sampling without replacement
        actual_subset_size = min(subset_size, len(data))
        bootstrap_idxs = np.random.choice(len(data), size=actual_subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
            
    return [(np.mean(lst), np.std(lst)) if lst else (0.0, 0.0) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.
    """
    if not data:
        return 0.0 # Return default if data is empty

    vote2vals = defaultdict(list)
    for d in data:
        if vote_key in d and val_key in d: # Ensure keys exist
            vote2vals[d[vote_key]].append(d[val_key])

    if not vote2vals: # If no valid votes were collected
        return 0.0

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    # Return the first value for the majority vote, or 0.0 if not found
    maj_val = vote2vals[maj_vote][0] if vote2vals[maj_vote] else 0.0

    return maj_val


def process_spr_validation_metrics(data_sources: list[str],
                               sample_inputs: list[str],
                               infos_dict: dict[str, list[Any]],
                               seed: int = 42,
                               group_prefix: bool = False) -> dict[str, dict[str, dict[str, float]]]:
    """
    Processes validation metrics into a structured format, performing aggregation
    at the (data_source, prompt) level and then aggregating across prompts.

    Args:
        data_sources: A list of data source identifiers (e.g., task names) for each sample.
        sample_inputs: A list of input prompts (e.g., question IDs) for each sample.
        infos_dict: A dictionary where keys are metric names (e.g., 'f1_score') and values
                    are lists of corresponding metric values for each sample.
                    (After compute_metrics's initial aggregation, each sample here
                    represents a unique question with a single aggregated score).
        seed: Random seed for reproducibility. Defaults to 42.
        group_prefix: If True, group data sources with the same prefix and calculate
                      an overall average for that prefix.

    Returns:
        dict[str, dict[str, dict[str, float]]]: A nested dictionary containing
        aggregated metrics: data_source -> variable_name -> metric_name -> metric_value.
    """
    # Group metrics by data source (task) and prompt (question_id)
    # Since compute_metrics has already aggregated per (task, id), each var_vals_list here
    # will contain exactly one element (the aggregated score for that specific question).
    # print(infos_dict)
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals_list_all_samples in infos_dict.items():
            # Append the single aggregated value for this specific sample (task, id)
            if sample_idx < len(var_vals_list_all_samples):
                if isinstance(var_vals_list_all_samples[sample_idx],list):
                    var2vals[var_name].extend(var_vals_list_all_samples[sample_idx])
                else:
                    var2vals[var_name].append(var_vals_list_all_samples[sample_idx])

    # print(data_src2prompt2var2vals)

    # Calculate metrics for each group (task, question_id).
    # 'n_resps' will typically be 1 here, meaning 'mean@1', 'best@1', etc., will just be the value itself.
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals_list in var2vals.items():
                # Skip if the value is a string or empty to prevent numerical errors
                if not var_vals_list or isinstance(var_vals_list[0], str):
                    continue
                
                # Ensure values are numerical
                var_vals = [v if v is not None and isinstance(v, (int, float)) else 0 for v in var_vals_list]
                
                metric = {}
                n_resps = len(var_vals) 
                
                if n_resps > 0:
                    metric[f"mean@{n_resps}"] = np.mean(var_vals)
                    # metric[f"std@{n_resps}"] = np.std(var_vals)
                else:
                    metric[f"mean@{n_resps}"] = 0.0
                    # metric[f"std@{n_resps}"] = 0.0

                # Define 'n' values for bootstrap. If n_resps is 1, this will typically result in [1].
                ns = []
                n_val_iter = 2
                while n_val_iter < n_resps:
                    ns.append(n_val_iter)
                    n_val_iter *= 2
                if n_resps > 0 and n_resps not in ns: # Ensure n_resps itself is included if not already
                    ns.append(n_resps)

                for current_n in ns:
                    # Best/Worst-of-N:
                    # For n_resps = 1, bootstrap_metric on [value] with subset_size=1 will return (value, 0).
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals,
                                                                                  subset_size=current_n,
                                                                                  reduce_fns=[np.max, np.min],
                                                                                  seed=seed)
                    metric[f"best@{current_n}/mean"], metric[f"best@{current_n}/std"] = bon_mean, bon_std
                    metric[f"worst@{current_n}/mean"], metric[f"worst@{current_n}/std"] = won_mean, won_std
                    
                    # Majority voting:
                    # This block will only be relevant if 'pred' was included in infos_dict and has a value.
                    # As discussed, with question-level aggregation, 'pred' might not be passed this way.
                    if "pred" in var2vals and var2vals["pred"] and len(var2vals["pred"]) == n_resps:
                        vote_data = [{"val": val, "pred": pred_val} for val, pred_val in zip(var_vals, var2vals["pred"])]
                        if vote_data:
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(data=vote_data,
                                                                        subset_size=current_n,
                                                                        reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                                                        seed=seed)
                            metric[f"maj@{current_n}/mean"], metric[f"maj@{current_n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts (question_ids) for each data_source (task).
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    # Accumulate values for overall average across all data sources (benchmarks).
    overall_metrics_collector = defaultdict(lambda: defaultdict(list))
    
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                if prompt_vals: # Ensure there are values to average
                    avg_val = np.mean(prompt_vals)
                    data_src2var2metric2val[data_source][var_name][metric_name] = avg_val
                    overall_metrics_collector[var_name][metric_name].append(avg_val)
                else:
                    data_src2var2metric2val[data_source][var_name][metric_name] = 0.0

    overall_aggregated_metrics = defaultdict(lambda: defaultdict(float))
    for var_name, metric_vals_dict in overall_metrics_collector.items():
        for metric_name, values in metric_vals_dict.items():
            if values:
                overall_aggregated_metrics[var_name][metric_name] = np.mean(values)
            else:
                overall_aggregated_metrics[var_name][metric_name] = 0.0
    
    # Add "overall" results to the final output if available
    if overall_aggregated_metrics:
        data_src2var2metric2val["overall"] = overall_aggregated_metrics

    # New functionality: Group by data_source prefix.
    if group_prefix:
        prefix2var2metric2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for data_source, var2metric in data_src2var2metric2val.items():
            # Skip "overall" and ensure data_source has a prefix to group
            if data_source == "overall" or '_' not in data_source:
                continue

            prefix = data_source.split('_')[0]
            # Only aggregate if there's actually a prefix difference (i.e., not the full name)
            if prefix != data_source:
                for var_name, metric_dict in var2metric.items():
                    for metric_name, metric_val in metric_dict.items():
                        prefix2var2metric2vals[prefix][var_name][metric_name].append(metric_val)

        # Calculate final aggregated metrics for prefixes (mean of aggregated task metrics).
        prefix_aggregated_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for prefix, var2metric_vals in prefix2var2metric2vals.items():
            for var_name, metric_vals_dict in var2metric_vals.items():
                for metric_name, values in metric_vals_dict.items():
                    if values:
                        prefix_aggregated_metrics[prefix][var_name][metric_name] = np.mean(values)
                    else:
                        prefix_aggregated_metrics[prefix][var_name][metric_name] = 0.0

        data_src2var2metric2val.update(prefix_aggregated_metrics)

    return data_src2var2metric2val
