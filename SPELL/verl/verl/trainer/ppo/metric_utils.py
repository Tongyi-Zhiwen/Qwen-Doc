# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

from collections import Counter, defaultdict
from functools import partial
from typing import Any, Dict, List, Callable, Union

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_spell_data_metrics(batch: DataProto, use_critic: bool = True, update_questioner: bool = False) -> Dict[str, Any]:
    
    def _safe_metric_agg(tensor: torch.Tensor, agg_fn, default_val=float('nan')):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return default_val
        try:
            return agg_fn(tensor).detach().item()
        except RuntimeError: # Catches errors like max/min on empty tensor if somehow numel check is bypassed
            return default_val

    def _safe_var_metric(tensor: torch.Tensor, default_val=float('nan')):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 1: # variance is undefined for <= 1 element
            return default_val
        return torch.var(tensor).detach().item()

    # --- Original computations ---
    # sequence_score = batch.batch['token_level_scores'].sum(-1)
    # sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    if 'score' in batch.non_tensor_batch:
        scores = batch.non_tensor_batch['score']
        # print(scores)
        numeric_scores = np.array([float(x) for x in scores], dtype=np.float32)
        sequence_reward = torch.from_numpy(numeric_scores) 
    else:
        sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    if 'llm_reward' in batch.non_tensor_batch and 'rule_based_reward' in batch.non_tensor_batch:
        llm_rewards = batch.non_tensor_batch['llm_reward']
        rule_based_rewards = batch.non_tensor_batch['rule_based_reward']
        # print(scores)
        llm_rewards_np = np.array([float(x) for x in llm_rewards], dtype=np.float32)
        llm_rewards_tensor = torch.from_numpy(llm_rewards_np) 
        rule_rewards_np = np.array([float(x) for x in rule_based_rewards], dtype=np.float32)
        rule_rewards_tensor = torch.from_numpy(rule_rewards_np) 
    else:
        llm_rewards_tensor, rule_rewards_tensor = torch.zeros_like(sequence_reward), sequence_reward

    if 'f1_score' in batch.non_tensor_batch: 
        f1_scores = batch.non_tensor_batch['f1_score']
        f1_scores_np = np.array([float(x) for x in f1_scores], dtype=np.float32)
        f1_scores_tensor = torch.from_numpy(f1_scores_np) 
    else:
        f1_scores_tensor = torch.zeros_like(sequence_reward)

    if 'entropys' in batch.batch.keys(): 
        entropys_tensor = batch.batch['entropys']
    else:
        entropys_tensor = torch.zeros_like(sequence_reward)

    if 'meta_info' in batch.non_tensor_batch and 'from_cache' in batch.non_tensor_batch['meta_info']:
        samples_from_cache = sum(batch.non_tensor_batch['meta_info']['from_cache'])
    else:
        samples_from_cache = 0

    advantages = batch.batch['advantages']
    advantages_seq = advantages.cpu().tolist()
    valid_adv = torch.tensor([advs[0] for advs in advantages_seq])
    # returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    # These are masks over the full sequence length, distinguishing prompt and response parts
    # Original code uses 'prompt_mask' and 'response_mask' for these.
    # Let's stick to original naming for these specific masks.
    prompt_attention_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    # response_attention_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_attention_mask.size(-1)

    # Assuming _compute_response_info is available in the scope.
    # This function is expected to return actual (unpadded) lengths.
    response_info = _compute_response_info(batch) 
    prompt_length = response_info['prompt_length'] # Shape: (batch_size,)
    response_length = response_info['response_length'] # Shape: (batch_size,)

    # valid_adv = torch.masked_select(advantages, response_attention_mask)
    # valid_returns = torch.masked_select(returns, response_attention_mask)
    
    # Base metrics dictionary
    metrics = {}


    # if use_critic:
    #     values = batch.batch['values']
    #     valid_values = torch.masked_select(values, response_attention_mask)
    #     return_diff_var_val = _safe_var_metric(valid_returns - valid_values)
    #     return_var_val = _safe_var_metric(valid_returns)
        
    #     vf_explained_var_val = float('nan')
    #     if return_var_val != float('nan') and return_diff_var_val != float('nan'):
    #         if (return_var_val + 1e-5) != 0: # Avoid division by zero if return_var_val is ~ -1e-5
    #             vf_explained_var_val = (1.0 - return_diff_var_val / (return_var_val + 1e-5))
        
    #     # Ensure it's a detached item if not nan
    #     if isinstance(vf_explained_var_val, torch.Tensor):
    #         vf_explained_var_val = vf_explained_var_val.detach().item()


    metrics.update({
        # score
        # 'critic/score/mean': _safe_metric_agg(sequence_score, torch.mean),
        # 'critic/score/max': _safe_metric_agg(sequence_score, torch.max),
        # 'critic/score/min': _safe_metric_agg(sequence_score, torch.min),
        # reward
        'critic/rewards/mean': _safe_metric_agg(sequence_reward, torch.mean),
        'critic/rewards/max': _safe_metric_agg(sequence_reward, torch.max),
        'critic/rewards/min': _safe_metric_agg(sequence_reward, torch.min),
        # 'critic/llm_rewards/mean': _safe_metric_agg(llm_rewards_tensor, torch.mean),
        # 'critic/rule_rewards/mean': _safe_metric_agg(rule_rewards_tensor, torch.mean),
        # adv
        'critic/advantages/mean': _safe_metric_agg(valid_adv, torch.mean),
        'critic/advantages/max': _safe_metric_agg(valid_adv, torch.max),
        'critic/advantages/min': _safe_metric_agg(valid_adv, torch.min),
        'actor/entropy/mean': _safe_metric_agg(entropys_tensor, torch.mean),
        'actor/entropy/max': _safe_metric_agg(entropys_tensor, torch.max),
        'actor/entropy/min': _safe_metric_agg(entropys_tensor, torch.min),
        'actor/from_cache': samples_from_cache,
        # returns
        # 'critic/returns/mean': _safe_metric_agg(valid_returns, torch.mean),
        # 'critic/returns/max': _safe_metric_agg(valid_returns, torch.max),
        # 'critic/returns/min': _safe_metric_agg(valid_returns, torch.min),
        **({
            # values
            'critic/values/mean': _safe_metric_agg(valid_values, torch.mean),
            # 'critic/values/max': _safe_metric_agg(valid_values, torch.max),
            # 'critic/values/min': _safe_metric_agg(valid_values, torch.min),
            # vf explained var
            'critic/vf_explained_var': vf_explained_var_val,
        } if use_critic else {}),
        # response length
        'response_length/mean': _safe_metric_agg(response_length, torch.mean),
        'response_length/max': _safe_metric_agg(response_length, torch.max),
        'response_length/min': _safe_metric_agg(response_length, torch.min),
        'response_length/clip_ratio': _safe_metric_agg(torch.eq(response_length, max_response_length).float(), torch.mean),
        # prompt length
        'prompt_length/mean': _safe_metric_agg(prompt_length, torch.mean),
        'prompt_length/max': _safe_metric_agg(prompt_length, torch.max),
        'prompt_length/min': _safe_metric_agg(prompt_length, torch.min),
        'prompt_length/clip_ratio': _safe_metric_agg(torch.eq(prompt_length, max_prompt_length).float(), torch.mean),
    })


    batch_size = sequence_reward.shape[0]
    device = sequence_reward.device 

    def _compute_grouped_metrics(
        raw_group_key_names: Union[str, List[str]],
        raw_all_group_values_lists: Union[List[Any], np.ndarray, List[Union[List[Any], np.ndarray]]],
        raw_group_key_prefixes: Union[str, List[str]],
    ):

        key_names: List[str]
        processed_values_lists: List[List[Any]] = [] # Ensure inner lists are Python lists
        prefixes: List[str]

        # Normalize inputs
        if isinstance(raw_group_key_names, str):
            key_names = [raw_group_key_names]
            # raw_all_group_values_lists is a single list/ndarray for this key
            if isinstance(raw_all_group_values_lists, np.ndarray):
                processed_values_lists = [raw_all_group_values_lists.tolist()]
            elif raw_all_group_values_lists is not None:
                processed_values_lists = [list(raw_all_group_values_lists)]
            else: # Handle None case
                 print(f"Warning: Group values list for key '{raw_group_key_names}' is None. Skipping.")
                 return
            prefixes = [raw_group_key_prefixes] if isinstance(raw_group_key_prefixes, str) else raw_group_key_prefixes # original passed prefix could be list even for single key name
            if not isinstance(prefixes, list) or len(prefixes) != 1: # Ensure prefixes matches key_names
                print(f"Warning: Prefix mismatch for single key '{raw_group_key_names}'. Using key name as prefix.")
                prefixes = [raw_group_key_names]


        else: # It's a list of key names
            key_names = raw_group_key_names
            if len(key_names) == 0:
                print("Warning: Empty list of group key names provided. Skipping.")
                return
            
            if not isinstance(raw_all_group_values_lists, list) or len(raw_all_group_values_lists) != len(key_names):
                print(f"Warning: Mismatch between number of key names ({len(key_names)}) and number of values lists. Skipping.")
                return

            for i, val_list in enumerate(raw_all_group_values_lists):
                if val_list is None:
                    print(f"Warning: Group values list for key '{key_names[i]}' is None. Skipping entire group.")
                    return
                if isinstance(val_list, np.ndarray):
                    processed_values_lists.append(val_list.tolist())
                else:
                    processed_values_lists.append(list(val_list))
            
            if not isinstance(raw_group_key_prefixes, list) or len(raw_group_key_prefixes) != len(key_names):
                 print(f"Warning: Mismatch between number of key names ({len(key_names)}) and number of prefixes. Using key names as prefixes.")
                 prefixes = key_names # Fallback to using key names as prefixes
            else:
                prefixes = raw_group_key_prefixes
        
        # Validate lengths of processed_values_lists
        if not processed_values_lists or not processed_values_lists[0]: # Handles case where a list was None and we returned
            # This check might be redundant if None lists cause early return, but good for safety.
            # print(f"Warning: No valid values lists for keys {key_names}. Skipping.") # Already handled by None check
            return

        actual_len = len(processed_values_lists[0])
        if actual_len == 0:
            # print(f"Warning: Group values lists for keys {key_names} are empty. Skipping.") # No need to print if batch_size is 0 (handled by caller)
            return
        
        if actual_len != batch_size:
            print(f"Warning: Group values lists for keys {key_names} have length {actual_len}, expected {batch_size}. Skipping.")
            return
        
        for i, v_list in enumerate(processed_values_lists):
            if len(v_list) != actual_len:
                print(f"Warning: Inconsistent lengths in group values lists for key '{key_names[i]}'. Expected {actual_len}, got {len(v_list)}. Skipping.")
                return

        # Combine group values per sample: e.g., [('alpaca', 'user'), ('dolly', 'questioner'), ...]
        combined_values_per_sample = list(zip(*processed_values_lists))

        # Find unique combinations (tuples). np.nan is handled correctly by set and sort for tuples.
        try:
            # Ensure all elements in tuples are comparable for sorting, or handle TypeError
            # For typical string/numeric/None/NaN data, this should be fine.
            unique_item_tuples = sorted(list(set(combined_values_per_sample)))
        except TypeError as e:
            print(f"Warning: Could not sort unique combined items for keys {key_names}. Error: {e}. Skipping grouped metrics.")
            # Example: if a list contains [1, "a", None], set works, but sort might fail if types are unorderable.
            # For data_source/role (typically strings), this should be rare.
            return

        # print(unique_item_tuples)
        for current_item_tuple in unique_item_tuples:
            # Build metric key suffix, e.g., "task/alpaca/role/questioner"
            metric_suffix_parts = []
            for i in range(len(key_names)):
                p = prefixes[i]
                v_str = str(current_item_tuple[i]) # str(np.nan) -> "nan"
                metric_suffix_parts.append(f"{p}/{v_str}")
            full_metric_item_suffix = "/".join(metric_suffix_parts)

            # Create combined mask
            item_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            for key_idx in range(len(key_names)):
                batch_values_for_this_key = processed_values_lists[key_idx] # This is a Python list
                target_value_for_this_key = current_item_tuple[key_idx]

                is_target_nan = isinstance(target_value_for_this_key, float) and np.isnan(target_value_for_this_key)
                
                if is_target_nan:
                    key_specific_mask_np = np.array([isinstance(v, float) and np.isnan(v) for v in batch_values_for_this_key])
                else:
                    key_specific_mask_np = np.array([v == target_value_for_this_key for v in batch_values_for_this_key])
                
                item_mask &= torch.from_numpy(key_specific_mask_np).to(device)

            if item_mask.sum() == 0:
                continue # Should not happen if current_item_tuple came from unique_item_tuples based on data

            item_advantages_seq = advantages[item_mask] 
            item_advantages_seq = item_advantages_seq.cpu().tolist()
            item_valid_adv = torch.tensor([advs[0] for advs in item_advantages_seq])
            # item_returns_seq = returns[item_mask]       
            # item_response_attention_mask = response_attention_mask[item_mask] 
            # item_valid_adv = torch.masked_select(item_advantages_seq, item_response_attention_mask)
            # print(f"{full_metric_item_suffix} advantage in this batch")
            # print(item_valid_adv.shape)
            # print(item_valid_adv)
            # item_valid_returns = torch.masked_select(item_returns_seq, item_response_attention_mask)

            item_sequence_reward = sequence_reward[item_mask]

            # print(f"{full_metric_item_suffix} rewards in this batch")
            # print(item_sequence_reward.shape)
            # print(item_sequence_reward)

            item_prompt_length = prompt_length[item_mask]
            item_response_length = response_length[item_mask]
            cur_entropys_tensor = entropys_tensor[item_mask]

            group_metrics_dict = {
                f'actor/{full_metric_item_suffix}/entropy/mean': _safe_metric_agg(cur_entropys_tensor, torch.mean),
                f'actor/{full_metric_item_suffix}/entropy/max': _safe_metric_agg(cur_entropys_tensor, torch.max),
                f'actor/{full_metric_item_suffix}/entropy/min': _safe_metric_agg(cur_entropys_tensor, torch.min),
                f'critic/{full_metric_item_suffix}/count': item_prompt_length.shape[0],
                # f'critic/{full_metric_item_suffix}/score/mean': _safe_metric_agg(current_group_score_source, torch.mean),
                f'critic/{full_metric_item_suffix}/rewards/mean': _safe_metric_agg(item_sequence_reward, torch.mean),
                f'critic/{full_metric_item_suffix}/rewards/max': _safe_metric_agg(item_sequence_reward, torch.max),
                f'critic/{full_metric_item_suffix}/rewards/min': _safe_metric_agg(item_sequence_reward, torch.min),
                f'critic/{full_metric_item_suffix}/advantages/mean': _safe_metric_agg(item_valid_adv, torch.mean),
                f'critic/{full_metric_item_suffix}/advantages/max': _safe_metric_agg(item_valid_adv, torch.max),
                f'critic/{full_metric_item_suffix}/advantages/min': _safe_metric_agg(item_valid_adv, torch.min),
                # f'critic/{full_metric_item_suffix}/returns/mean': _safe_metric_agg(item_valid_returns, torch.mean),
                
                f'response_length/{full_metric_item_suffix}/mean': _safe_metric_agg(item_response_length, torch.mean),
                f'response_length/{full_metric_item_suffix}/max': _safe_metric_agg(item_response_length, torch.max),
                f'response_length/{full_metric_item_suffix}/min': _safe_metric_agg(item_response_length, torch.min),
                f'response_length/{full_metric_item_suffix}/clip_ratio': _safe_metric_agg(torch.eq(item_response_length, max_response_length).float(), torch.mean),
                f'prompt_length/{full_metric_item_suffix}/mean': _safe_metric_agg(item_prompt_length, torch.mean),
                f'prompt_length/{full_metric_item_suffix}/max': _safe_metric_agg(item_prompt_length, torch.max),
                f'prompt_length/{full_metric_item_suffix}/min': _safe_metric_agg(item_prompt_length, torch.min),
                f'prompt_length/{full_metric_item_suffix}/clip_ratio': _safe_metric_agg(torch.eq(item_prompt_length, max_prompt_length).float() if max_prompt_length > 0 else torch.tensor(0.0), torch.mean),
            }
            if "responder" in full_metric_item_suffix:
                cur_llm_rewards_tensor = llm_rewards_tensor[item_mask]
                # keep the positive values, mask the value of -100.0 (bad case)
                cur_llm_rewards_tensor = cur_llm_rewards_tensor[cur_llm_rewards_tensor >= 0.0]
                cur_rule_rewards_tensor = rule_rewards_tensor[item_mask]
                cur_f1_scores_tensor = f1_scores_tensor[item_mask]
                
                group_metrics_dict.update({
                    f'critic/{full_metric_item_suffix}/llm_rewards/mean': _safe_metric_agg(cur_llm_rewards_tensor, torch.mean),
                    f'critic/{full_metric_item_suffix}/rule_rewards/mean': _safe_metric_agg(cur_rule_rewards_tensor, torch.mean),
                    f'critic/{full_metric_item_suffix}/f1_score/mean': _safe_metric_agg(cur_f1_scores_tensor, torch.mean),
                })


            # if use_critic:
            #     item_values_group = values[item_mask]
            #     item_valid_values = _get_valid_response_data(item_values_group, item_resp_attn_mask_group)
                
            #     item_return_diff_var_val = _safe_var_metric(item_valid_returns - item_valid_values)
            #     item_return_var_val = _safe_var_metric(item_valid_returns)
                
            #     item_vf_explained_var_val = float('nan')
            #     if not np.isnan(item_return_var_val) and not np.isnan(item_return_diff_var_val):
            #         if abs(item_return_var_val + 1e-5) > 1e-8: 
            #             item_vf_explained_var_val = (1.0 - item_return_diff_var_val / (item_return_var_val + 1e-5))
                
            #     if isinstance(item_vf_explained_var_val, torch.Tensor):
            #         item_vf_explained_var_val = item_vf_explained_var_val.detach().item()

            #     group_metrics_dict.update({
            #         f'critic/{full_metric_item_suffix}/values/mean': _safe_metric_agg(item_valid_values, torch.mean),
            #         f'critic/{full_metric_item_suffix}/vf_explained_var': item_vf_explained_var_val,
            #     })
            metrics.update(group_metrics_dict)


    if 'data_source' in batch.non_tensor_batch and 'role' in batch.non_tensor_batch:
        data_sources_list = batch.non_tensor_batch.get('data_source')
        roles_list = batch.non_tensor_batch.get('role')
        if data_sources_list is not None and roles_list is not None:
            _compute_grouped_metrics(
                raw_group_key_names=['data_source', 'role'],
                raw_all_group_values_lists=[data_sources_list, roles_list],
                raw_group_key_prefixes=['task', 'role'] # Defines the path structure in metric key
            )

    if 'data_source' in batch.non_tensor_batch:
        data_sources_list = batch.non_tensor_batch.get('data_source')
        if data_sources_list is not None:
             _compute_grouped_metrics(
                raw_group_key_names='data_source',
                raw_all_group_values_lists=data_sources_list,
                raw_group_key_prefixes='task'
            )

    if 'role' in batch.non_tensor_batch:
        roles_list = batch.non_tensor_batch.get('role')
        if roles_list is not None:
            _compute_grouped_metrics(
                raw_group_key_names='role',
                raw_all_group_values_lists=roles_list,
                raw_group_key_prefixes='role'
            )

    return metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


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

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


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

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_spell_validation_metrics(data_sources: list[str],
                               sample_inputs: list[str],
                               infos_dict: dict[str, list[Any]],
                               seed: int = 42,
                               group_prefix: bool = False) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.
    
    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample
        group_prefix: group the data source with the same prefix and get the overall average
        
    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        # print(data_source,len(prompt2var2vals))
        for prompt, var2vals in prompt2var2vals.items():
            # print(prompt,len(var2vals))
            for var_name, var_vals in var2vals.items():
                # print(var_name,len(var_vals),var_vals)
                if isinstance(var_vals[0], str):
                    continue
                var_vals = [v if v is not None and isinstance(v, (int, float)) else 0 for v in var_vals]
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                # n = 2
                # while n < n_resps:
                #     ns.append(n)
                #     n *= 2
                # ns.append(n_resps)

                # for n in ns:
                #     # Best/Worst-of-N
                #     [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals,
                #                                                                   subset_size=n,
                #                                                                   reduce_fns=[np.max, np.min],
                #                                                                   seed=seed)
                #     metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                #     metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                #     # Majority voting
                #     if var2vals.get("pred", None) is not None:
                #         vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                #         [(maj_n_mean, maj_n_std)
                #         ] = bootstrap_metric(data=vote_data,
                #                              subset_size=n,
                #                              reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                #                              seed=seed)
                #         metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)
                    
    data_src_counts = {
        ds: len(prompts) for ds, prompts in data_src2prompt2var2vals.items()
    }

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    overall_metrics_collector = defaultdict(lambda: defaultdict(list))
    
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                mean_val = np.mean(prompt_vals)
                data_src2var2metric2val[data_source][var_name][metric_name] = mean_val
                weight = data_src_counts.get(data_source, 0)
                if weight > 0:
                    overall_metrics_collector[var_name][metric_name].append((mean_val, weight))
    
    overall_aggregated_metrics = defaultdict(lambda: defaultdict(float))
    for var_name, metric_vals_dict in overall_metrics_collector.items():
        for metric_name, value_weight_pairs in metric_vals_dict.items():
            if not value_weight_pairs:
                continue
            values, weights = zip(*value_weight_pairs) 
            overall_aggregated_metrics[var_name][metric_name] = np.average(values, weights=weights)
    
    if overall_aggregated_metrics:
        data_src2var2metric2val["overall"] = overall_aggregated_metrics

    if group_prefix:
        prefix2var2metric2value_weight_pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for data_source, var2metric in data_src2var2metric2val.items():
            if data_source == "overall": 
                continue
            prefix = data_source.split('_')[0]
            weight = data_src_counts.get(data_source, 0)
            if weight > 0:
                for var_name, metric_dict in var2metric.items():
                    for metric_name, metric_val in metric_dict.items():
                        prefix2var2metric2value_weight_pairs[prefix][var_name][metric_name].append((metric_val, weight))

        prefix_aggregated_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for prefix, var2metric_vals in prefix2var2metric2value_weight_pairs.items():
            num_sources_for_prefix = sum(1 for ds in data_src_counts if ds.startswith(prefix + '_') or ds == prefix)
            if num_sources_for_prefix <= 1:
                continue
                
            for var_name, metric_vals_dict in var2metric_vals.items():
                for metric_name, value_weight_pairs in metric_vals_dict.items():
                    if not value_weight_pairs:
                        continue
                    values, weights = zip(*value_weight_pairs)
                    prefix_aggregated_metrics[prefix][var_name][metric_name] = np.average(values, weights=weights)

        data_src2var2metric2val.update(prefix_aggregated_metrics)
    

    return data_src2var2metric2val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val