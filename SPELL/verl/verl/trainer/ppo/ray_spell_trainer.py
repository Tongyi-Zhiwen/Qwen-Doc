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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, List, Tuple, Any
from copy import deepcopy
from collections import defaultdict, Counter, deque
from functools import partial
import math

import ray
import numpy as np
import json
from tqdm import tqdm
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem, list_of_dict_to_dict_of_list
from verl.single_controller.base import Worker  
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.metric_utils import compute_spell_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, bootstrap_metric, calc_maj_val, process_spell_validation_metrics, _compute_response_info
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

import torch
import tensordict
from tensordict import TensorDict
from verl.utils.torch_functional import masked_mean

# spell configs
from verl.utils.dataset.spell_dataset import get_record, extract_solution, RESPONDER_PROMPT_MAPPING, VERIFIER_PROMPT, DomainWeightedSPELLDataset, DomainSampler
from verl.utils.reward_score import docmath, long, docqa

import random
random.seed(42)

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]
    
    
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, update_questioner=False, questioner_reward_type="reverse", questioner_group="group"):
    """Computes advantages and returns based on a specified advantage estimator.

    This function serves as a dispatcher to various advantage calculation algorithms,
    such as GAE, GRPO, REINFORCE++, etc. It populates the input `DataProto` with
    'advantages' and 'returns' tensors.

    Args:
        data (DataProto): The data structure containing token-level rewards and other
                          necessary information (e.g., values for GAE).
        adv_estimator: The advantage estimation algorithm to use (e.g., AdvantageEstimator.GAE).
        gamma (float, optional): The discount factor. Defaults to 1.0.
        lam (float, optional): The lambda for GAE advantage computation. Defaults to 1.0.
        num_repeat (int, optional): The number of repeated samples. Defaults to 1.
        update_questioner (bool, optional): Flag to indicate if the questioner role is being updated,
                                            affecting GRPO calculations. Defaults to False.
        questioner_reward_type (str, optional): The reward type for the questioner in GRPO.
                                                Defaults to "reverse".
        questioner_group (str, optional): The grouping strategy for the questioner in GRPO.
                                          Defaults to "group".

    Returns:
        DataProto: The input `DataProto` updated with 'advantages' and 'returns' tensors.
    
    Raises:
        NotImplementedError: If the specified `adv_estimator` is not supported.
    """
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        response_mask = data.batch['response_mask']
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        # roles are needed if update_questioner is True
        roles = data.non_tensor_batch.get('role', None) # Get roles, defaults to None if not present
        index = data.non_tensor_batch['uid']
        case_types = data.non_tensor_batch.get('type', None)
        response_mask = data.batch['response_mask']
        tasks = data.non_tensor_batch['data_source']
        if roles is not None:
            advantages, returns, scores = core_algos.compute_spell_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                    eos_mask=response_mask,
                                                                    index=index,
                                                                    roles=roles,
                                                                    tasks=tasks,
                                                                    case_types=case_types,
                                                                    questioner_reward_type=questioner_reward_type,
                                                                    questioner_group=questioner_group,
                                                                )
            # record the mean reward
            data.non_tensor_batch['score'] = scores 
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.SPELL:
        token_level_rewards = data.batch['token_level_rewards']
        # roles are needed if update_questioner is True
        roles = data.non_tensor_batch.get('role', None) # Get roles, defaults to None if not present
        index = data.non_tensor_batch['uid']
        case_types = data.non_tensor_batch.get('type', None)
        tasks = data.non_tensor_batch['data_source']
        response_mask = data.batch['response_mask']
        if roles is not None:
            advantages, returns, scores = core_algos.compute_spell_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                    eos_mask=response_mask,
                                                                    index=index,
                                                                    roles=roles,
                                                                    tasks=tasks,
                                                                    case_types=case_types,
                                                                    questioner_reward_type=questioner_reward_type,
                                                                    questioner_group=questioner_group,
                                                                )
            # record the mean reward
            data.non_tensor_batch['score'] = scores 
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def prepare_gen_batch(batch: DataProto):
    """Prepares a data batch for sequence generation by removing unnecessary keys.

    This function creates a copy of the input batch and removes tensors and metadata
    that are not required for the generation process, making the data object smaller
    and more efficient to transfer to generation workers.

    Args:
        batch (DataProto): The original data batch.

    Returns:
        DataProto: A new, smaller `DataProto` object suitable for generation.
    """
    # pop those keys for generation
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
    if "multi_modal_data" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("multi_modal_data")
    if "raw_prompt" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("raw_prompt")
    if "tools_kwargs" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("tools_kwargs")
    if "interaction_kwargs" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("interaction_kwargs")
    gen_batch = batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )
    return gen_batch
    
@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] += timer.last  # Allow to accumulate time


class BatchAccumulator:
    """Manages the accumulation of data until a full training batch is ready."""
    def __init__(self, config):
        """Initializes the BatchAccumulator.

        Sets up the configuration for accumulating batches, including batch sizes,
        world size, and buffers for different roles (responder, questioner, verifier).

        Args:
            config: The main configuration object containing trainer, data, and algorithm settings.
        """
        self.config = config
        # number of GPUs total
        self.n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        self.n_rollout_engines = self.n_gpus // self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.world_size = self.n_gpus // self.config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
        self.reset()

        self.prompt_bsz = self.config.data.train_batch_size
        self.traj_bsz = self.prompt_bsz * self.config.actor_rollout_ref.rollout.n
        self.max_gen_batches = self.config.algorithm.filter_groups.get('max_num_gen_batches', 0)
        self.verifier_rollout_n = self.config.algorithm.self_verification.get('n',1)
        # verifier batch ratio to responder, default to reverse of judge n
        self.verifier_ratio = self.config.algorithm.self_verification.get('update_ratio', 1 / self.verifier_rollout_n)

        # store the chunked good questioner question for next bacth generation
        if self.config.data.questioner_prompt_reuse:
            self.questioner_buffer = None

    def reset(self):
        """Resets the internal state for the next accumulation cycle.
        
        Clears all accumulated data batches (responder, questioner, verifier) and
        resets counters to prepare for building the next training batch.
        """
        # Initialize empty batches for each role
        self.responder_batch = None
        self.questioner_batch = None
        self.verifier_batch = None
        self.questioner_bad_case_uids = defaultdict(list)
        self.num_gen_batches = 0

    def is_ready(self) -> bool:
        """Checks if enough data has been accumulated to form a training batch.

        The readiness is determined by whether the number of unique prompts in the
        responder batch has reached the target training batch size (`prompt_bsz`).
        It also includes a safety check to prevent an infinite loop if data quality
        is too low to fill a batch.

        Returns:
            bool: True if a full training batch is ready, False otherwise.
            
        Raises:
            ValueError: If the maximum number of generation batches is reached before
                        a full training batch is accumulated.
        """
        if self.responder_batch is None:
            return False
            
        if not self.config.algorithm.filter_groups.enable:
            return True

        num_prompt_in_batch = len(self.get_unique_uids(self.responder_batch))
        
        if num_prompt_in_batch >= self.prompt_bsz:
            return True
        
        if self.max_gen_batches > 0 and self.num_gen_batches >= self.max_gen_batches:
            raise ValueError(
                f'Generated {self.num_gen_batches} batches but only collected {num_prompt_in_batch}/{self.prompt_bsz} prompts. '
                'Check data quality or increase `max_num_gen_batches`.'
            )
        return False

    def accumulate(self, new_responder_data: DataProto, new_questioner_data: DataProto, bad_case_uids: dict):
        """Accumulates new experience data into the accumulator.

        This method appends new responder and questioner data to their respective
        internal buffers. It also tracks the unique identifiers (UIDs) of bad cases
        generated by the questioner.

        Args:
            new_responder_data (DataProto): New experience data from the responder role.
            new_questioner_data (DataProto): New experience data from the questioner role.
            bad_case_uids (dict): A dictionary mapping bad case types to lists of UIDs.
        """
        self.num_gen_batches += 1
        if self.responder_batch is None:
            self.responder_batch = new_responder_data
        elif len(new_responder_data) > 0:
            self.responder_batch = DataProto.concat([self.responder_batch, new_responder_data])

        if self.config.algorithm.questioner.get("update", False):
            if self.questioner_batch is None:
                self.questioner_batch = new_questioner_data
            elif len(new_questioner_data) > 0:
                self.questioner_batch = DataProto.concat([self.questioner_batch, new_questioner_data])
            
            for bad_type, uids in bad_case_uids.items():
                self.questioner_bad_case_uids[bad_type].extend(uids)

    def get_final_batch(self) -> DataProto:
        """Prepares and returns the final, aligned training batch.

        This method orchestrates the final construction of the training batch by:
        1. Truncating the responder batch to the exact required size.
        2. Building the questioner batch by combining "good" cases (those that led to
           responder data) and a sampled set of "bad" cases.
        3. Building the verifier batch by sampling from accumulated verifier experiences.
        4. Aligning and padding all batches to ensure they have the same data structure.
        5. Concatenating all role batches into a single `DataProto` for training.

        Returns:
            DataProto: The final, combined `DataProto` ready for the PPO update.
        """
        # Truncate responder batch to the exact required size
        # print("getting final batch") 
        chunked_data = None
        if self.responder_batch is not None and len(self.responder_batch) > self.traj_bsz:
            if self.config.data.questioner_prompt_reuse:
                chunked_data = self.responder_batch[self.traj_bsz:]
            self.responder_batch = self.responder_batch[:self.traj_bsz]

        print(f"Number of responder trajectories: {len(self.responder_batch)}")
        self.responder_batch.non_tensor_batch['role'] = np.array(["responder"] * len(self.responder_batch), dtype=object)
        self.responder_batch.non_tensor_batch['type'] = np.array(["good"] * len(self.responder_batch), dtype=object)
            
        final_questioner_batch = None
        if self.config.algorithm.questioner.get("update", False) and self.questioner_batch is not None:
                
            final_responder_uids = self.get_unique_uids(self.responder_batch)
            good_questioner_indices = [i for i, uid in enumerate(self.questioner_batch.non_tensor_batch['uid']) if str(uid) in final_responder_uids]
            # store the chunked question for next batch generation, not used in our paper
            if self.config.data.questioner_prompt_reuse and chunked_data is not None:
                chunked_responder_uids = self.get_unique_uids(chunked_data)
                chunked_good_questioner_indices = [i for i, uid in enumerate(self.questioner_batch.non_tensor_batch['uid']) if str(uid) in chunked_responder_uids]
                self.questioner_buffer = self.questioner_batch[chunked_good_questioner_indices]
                print(f'save {len(self.questioner_buffer)} for next step.')
            
            bad_indices_and_types = self._sample_bad_cases(good_questioner_indices)
            
            final_questioner_batch = self._build_final_questioner_batch(good_questioner_indices, bad_indices_and_types)
            if final_questioner_batch is None and self.verifier_batch is None:
                return self.responder_batch

            final_questioner_batch = self._align_and_pad_batch(final_questioner_batch)

        if self.verifier_batch is None:
            return DataProto.concat([self.responder_batch, final_questioner_batch])

        if final_questioner_batch is not None:
            print(f"Number of questioner trajectories: {len(final_questioner_batch)}")
        else:
            print(f"Number of questioner trajectories: 0")

        final_verifier_batch = self._build_final_judge_batch()
        if final_verifier_batch is None:
            print(f"Number of verifier trajectories: 0")
            return DataProto.concat([self.responder_batch, final_questioner_batch])  

        print(f"Number of verifier trajectories: {len(final_verifier_batch)}")

        return DataProto.concat([self.responder_batch, final_questioner_batch, final_verifier_batch])

    def _build_final_judge_batch(self):
        """Samples and prepares the verifier (judge) batch for training.
        
        It selects a subset of the accumulated verifier data based on a configured
        ratio relative to the responder batch size. It ensures the final batch size
        is divisible by the world size and aligns its structure.

        Returns:
            DataProto | None: The prepared verifier batch, or None if not enough data
                              is available or verifier updates are disabled.
        """
        print(f"Vefifier batch num before select: {len(self.verifier_batch) // self.verifier_rollout_n}")
        samples_to_update = int(len(self.responder_batch) * self.verifier_ratio / self.verifier_rollout_n)
        all_uids = list(set(self.verifier_batch.non_tensor_batch['uid'].tolist()))
        # select through verifier uids
        # NOTE: Vefifier uids are different from responder & questioner uids
        if samples_to_update > len(all_uids):
            print(f"Not enough verifier data to update, disable verifier update!")
            return None
        samples_to_update = min(samples_to_update, len(all_uids))
        sampled_uids = set(random.sample(all_uids, samples_to_update))

        sampled_idxs = []
        for i, uid in enumerate(self.verifier_batch.non_tensor_batch['uid']):
            if uid in sampled_uids:
                sampled_idxs.append(i)
            
        final_verifier_batch = self.verifier_batch[sampled_idxs]

        truncated_length = (len(final_verifier_batch) // self.world_size) * self.world_size
        if truncated_length < len(final_verifier_batch):
            print("Warning: the verifier batch size is not divisble by world size, this will perform chunk process!")
            # Truncate the current DataProto to the nearest smaller divisible size
            final_verifier_batch = final_verifier_batch[:truncated_length]

        final_verifier_batch = self._align_and_pad_batch(final_verifier_batch)
        return final_verifier_batch

    def _sample_bad_cases(self, good_case_indices: List[int]) -> List[Tuple[str, int]]:
        """Samples "bad" questioner cases to include in the training batch.

        This method samples from the collected bad cases (e.g., bad format, bad grounding)
        to include in the questioner's training data. The sampling is stratified by task
        to match the task distribution of the "good" cases, ensuring a balanced update.

        Args:
            good_case_indices (List[int]): The indices of the "good" questioner samples
                                           within the accumulated questioner batch.

        Returns:
            List[Tuple[str, int]]: A list of tuples, where each tuple contains the bad
                                   case type (e.g., 'bad_format') and its index.
        """
        if not self.config.data.questioner_bad_case_ratio > 0:
            return []
        
        bad_case_type_uid_turple_list = []
        for bad_type in self.questioner_bad_case_uids.keys():
            print(f"{bad_type} case count: {len(self.questioner_bad_case_uids[bad_type])}")
            self.questioner_bad_case_uids[bad_type] = set(self.questioner_bad_case_uids[bad_type])

        # all bad cases
        for i, p_uid in enumerate(self.questioner_batch.non_tensor_batch['uid']):
            for bad_type, bad_uid_lists in self.questioner_bad_case_uids.items():
                if p_uid in bad_uid_lists:
                    bad_case_type_uid_turple_list.append((bad_type,i))

        print(f"Num of bad cases:{len(bad_case_type_uid_turple_list)}") 
        if not bad_case_type_uid_turple_list:
            return []

        sampled_bad_indices_to_add = [] # Store newly sampled bad case indices, list of turple(base_type,uid)

        # Assuming 'data_source' field in non_tensor_batch holds task information
        questioner_tasks_all = self.questioner_batch.non_tensor_batch.get('data_source')

        if questioner_tasks_all is not None:
            # 1. Count tasks for "good" cases (already in indices_questioner_final_pass)
            # Ensure indices are valid before accessing questioner_tasks_all
            good_case_tasks = []
            for i in good_case_indices:
                good_case_tasks.append(questioner_tasks_all[i])
            print(f"Num of good cases:{len(good_case_tasks)}")

            task_counts_good_cases = Counter(good_case_tasks).most_common()
            print(f"Task counts for good cases:{task_counts_good_cases}")

            # 2. Group "bad" cases by task
            task_to_bad_case_indices = defaultdict(list)
            for bad_type,bad_idx in bad_case_type_uid_turple_list:
                task = questioner_tasks_all[bad_idx]
                task_to_bad_case_indices[task].append((bad_type,bad_idx))

            # 3. Sample bad cases per task based on good case task distribution
            # buffer_cnt: not enough sample in one task
            buffer_cnt = 0
            for task, num_good_for_task in task_counts_good_cases:
                if num_good_for_task == 0: # Should not happen due to Counter logic but safe check
                    continue

                num_bad_to_sample_for_task = int(self.config.data.questioner_bad_case_ratio * num_good_for_task) + buffer_cnt
                buffer_cnt = 0
                
                available_bad_for_this_task = task_to_bad_case_indices.get(task, [])

                if num_bad_to_sample_for_task > len(available_bad_for_this_task):
                    # move to next task
                    buffer_cnt = num_bad_to_sample_for_task - len(available_bad_for_this_task)
                    num_bad_to_sample_for_task = len(available_bad_for_this_task)


                if num_bad_to_sample_for_task > 0:
                    sampled_indices = random.sample(available_bad_for_this_task, num_bad_to_sample_for_task)
                    sampled_bad_indices_to_add.extend(sampled_indices)

            if buffer_cnt > 0:
                print("Warning, not enough base case in this batch ! This will cause invalid minibatch nums")

        return sampled_bad_indices_to_add


    def _build_final_questioner_batch(self, good_indices, bad_indices_and_types):
        """Constructs the final questioner training batch from good and bad cases.

        It takes the indices of selected good and bad cases, extracts them from the
        main questioner buffer, assigns their respective types ('good', 'bad_format', etc.),
        and concatenates them into a single `DataProto`.

        Args:
            good_indices (List[int]): List of indices for good questioner samples.
            bad_indices_and_types (List[Tuple[str, int]]): List of (type, index) for bad samples.

        Returns:
            DataProto | None: The combined questioner `DataProto`, or None if no questioner
                              data is to be included in the final batch.
        """
        final_questioner_parts = []
        if good_indices:
            good_batch = self.questioner_batch[good_indices]
            good_batch.non_tensor_batch["type"] = np.array(["good"] * len(good_indices), dtype=object)
            final_questioner_parts.append(good_batch)
        
        if bad_indices_and_types:
            truncated_length = (len(bad_indices_and_types) // self.world_size) * self.world_size
            if truncated_length < len(bad_indices_and_types):
                print(f"Warning: Not enough bad case, this will truncate batch to nearest smaller divisible size: {len(bad_indices_and_types)} -> {truncated_length}")
                # Truncate the current DataProto to the nearest smaller divisible size
                bad_indices_and_types = bad_indices_and_types[:truncated_length]
            bad_indices = [item[1] for item in bad_indices_and_types]
            bad_types = [item[0] for item in bad_indices_and_types]
            bad_batch = self.questioner_batch[bad_indices]
            bad_batch.non_tensor_batch["type"] = np.array(bad_types, dtype=object)
            final_questioner_parts.append(bad_batch)

        if not final_questioner_parts:
            return None

        final_batch = DataProto.concat(final_questioner_parts)
        final_batch.non_tensor_batch['role'] = np.array(["questioner"] * len(final_batch), dtype=object)
        return final_batch

    def _align_and_pad_batch(self, batch_to_align: DataProto):
        """Aligns the keys of a given batch with the main responder batch.

        This function ensures that a given batch (e.g., questioner or verifier) has the
        exact same set of tensor and non-tensor keys as the responder batch. It adds
        missing keys with zero-filled tensors or default non-tensor values and removes
        any extra keys not present in the responder batch. This is crucial for
        concatenating batches and for the GRPO update step.

        Args:
            batch_to_align (DataProto): The batch whose keys need to be aligned.

        Returns:
            DataProto: The modified batch with aligned keys.
        """
        target_tensor_keys = set(self.responder_batch.batch.keys())
        target_non_tensor_keys = set(self.responder_batch.non_tensor_batch.keys())
        origin_tensor_keys = set(batch_to_align.batch.keys())
        origin_non_tensor_keys = set(batch_to_align.non_tensor_batch.keys())
        
        num_questioner_samples = len(batch_to_align)

        lose_tensor_keys = target_tensor_keys - origin_tensor_keys
        lose_non_tensor_keys = target_non_tensor_keys - origin_non_tensor_keys

        more_tensor_keys = origin_tensor_keys - target_tensor_keys
        more_non_tensor_keys = origin_non_tensor_keys - target_non_tensor_keys
        
        print(f"Responder keys:\n Tensor:{target_tensor_keys} \nNon Tensor: {target_non_tensor_keys}")
        print(f"Batch to align keys before:\nTensor: {set(batch_to_align.batch.keys())}\nNon Tensor: {set(batch_to_align.non_tensor_batch.keys())}")
        print(f"Lose tensor keys {lose_tensor_keys}")
        print(f"Lose non tensor keys {lose_non_tensor_keys}")
        print(f"More tensor keys {more_tensor_keys}")
        print(f"More non tensor keys {more_non_tensor_keys}")

        for key in more_tensor_keys:
            batch_to_align.batch.pop(key)

        for key in more_non_tensor_keys:
            batch_to_align.non_tensor_batch.pop(key)
        
        for key in lose_tensor_keys:
            template = self.responder_batch.batch[key]
            filler_shape = (num_questioner_samples,) + template.shape[1:]
            batch_to_align.batch[key] = torch.zeros(filler_shape, dtype=template.dtype, device=template.device)
            
        for key in lose_non_tensor_keys:
            template_item = self.responder_batch.non_tensor_batch[key][0]
            filler_list = [deepcopy(template_item) for _ in range(num_questioner_samples)]
            batch_to_align.non_tensor_batch[key] = np.array(filler_list, dtype=object) if isinstance(self.responder_batch.non_tensor_batch[key], np.ndarray) else filler_list

        # print(f"Batch to align keys after:\nTensor: {set(batch_to_align.batch.keys())}\nNon Tensor:{set(batch_to_align.non_tensor_batch.keys())}")
        return batch_to_align

    @staticmethod
    def get_unique_uids(data: DataProto) -> set:
        """Extracts a set of unique string UIDs from a DataProto object.

        Args:
            data (DataProto): The data object containing a 'uid' key in its non_tensor_batch.

        Returns:
            set: A set of unique UIDs. Returns an empty set if the data is None or
                 lacks the 'uid' key.
        """
        if data is None or 'uid' not in data.non_tensor_batch: return set()
        return set(str(uid) for uid in data.non_tensor_batch['uid'])


class RaySPELLTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        # print(role_worker_mapping)
        # print(resource_pool_manager)

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'
        # print(role_worker_mapping)
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.SPELL,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        
        questioner_config = self.config.algorithm.questioner
        # print(f"questioner_config = {questioner_config}")
        self.update_questioner = questioner_config.get("update", False)
        self.questioner_reward_type = questioner_config.get("reward_type", "gaussian")
        self.questioner_group = questioner_config.get("group", "batch")

        self.filter_questioner_prompts = self.config.data.filter_questioner_prompts
        self.questioner_bad_case_ratio = self.config.data.questioner_bad_case_ratio
        self.questioner_prompt_reuse = self.config.data.questioner_prompt_reuse

        veri_config = self.config.algorithm.self_verification

        # config for verifier 
        self.self_verification = veri_config.get("enable", False)
        self.llm_judge_tasks = veri_config.get("tasks", ['doc_general_qa', 'docmath_qa', 'doc_mc'])
        self.update_verifier = veri_config.get("update", False)
        self.verifier_reward_type = veri_config.get("reward_type", "most")
        self.verifier_label_type = veri_config.get("label_type", "maj_cons")
        self.verifier_rollout_n = veri_config.get("n", 1)
        self.verifier_lower_bound = veri_config.get('update_lower_bound', 0.6)
        self.verifier_upper_bound = veri_config.get('update_upper_bound', 0.9)

        # the template mapping
        self.responder_template = RESPONDER_PROMPT_MAPPING

        self._validate_config()
            
        self._create_dataloader()

        # NEW: Initialize task parsers for the dispatch pattern
        self.task_parsers = {
            'docmath_qa': self._parse_docmath_qa,
            'doc_mc': self._parse_doc_mc,
            'doc_general_qa': self._parse_doc_general_qa,
        }

    def _parse_docmath_qa(self, preds: dict) -> dict | int:
        """Parses a model's JSON output for the 'docmath_qa' task.

        This function validates the format of the prediction, extracts the question and answer,
        normalizes the answer for comparison, and prepares a dictionary for the responder prompt.

        Args:
            preds (dict): The dictionary parsed from the model's JSON response.

        Returns:
            dict | int: A dictionary containing the "question" and "ground_truth" if parsing is
                        successful. Returns -1 for a format error or 0 for an invalid answer.
        """
        if "question" not in preds or "answer" not in preds or not isinstance(preds['question'], str) or not isinstance(preds['answer'], str | int | float): 
            print(f"Wrong json format for docmath_qa: {preds}")
            return -1

        gt_answer = preds["answer"]
        if gt_answer is None: 
            print(f"No valid answer find in docmath_qa :{preds['answer']}")
            return -1
        parsed_gt_answer = docmath.parse_model_answer(str(preds["answer"]))
        if parsed_gt_answer:
            gt_answer = parsed_gt_answer
        try:
            gt_responder_answer_str = docmath.normalize(str(gt_answer))
            if not gt_responder_answer_str or int(gt_responder_answer_str) == 999999999 or float(gt_responder_answer_str) == 0.0: 
                print(f"The answer for docmath_qa is invalid :{preds['answer']}")
                return 0
        except (ValueError, TypeError):
            return 0
        return {"question": preds["question"], "ground_truth": f"{gt_responder_answer_str}"}

    def _parse_doc_mc(self, preds: dict) -> dict | int:
        """Parses a model's JSON output for the 'doc_mc' (multiple choice) task.

        This function validates the format, extracts the question, options, and answer,
        and constructs a formatted prompt that includes the question and all choices for the
        responder model.

        Args:
            preds (dict): The dictionary parsed from the model's JSON response.

        Returns:
            dict | int: A dictionary containing the formatted "question" and "ground_truth" answer
                        key if parsing is successful. Returns -1 for a format error.
        """
        if not all(k in preds for k in ["question", "answer", "options"]) or not isinstance(preds['question'], str) or not isinstance(preds['answer'], str): 
            print(f"Format error in for doc_mc :{preds}")
            return -1

        options = preds['options']
        if not isinstance(options, dict) or not all(k in options for k in ["A", "B", "C", "D"]): 
            print(f"Invalid options type find for doc_mc: {options}")
            return -1
        gt_answer = preds["answer"]
        if gt_answer is None :
            print(f"Empty gt for doc_mc: {preds}")
            return -1
        parsed_answer = long.parse_model_answer(str(gt_answer))
        if parsed_answer:
            gt_answer = parsed_answer

        if gt_answer not in ['A', 'B', 'C', 'D']:
            print(f"Invalid gt choice for doc_mc: {preds['answer']}")
            return -1   
        # mc prompt from longbench-v2
        mc_instruction = """What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}
"""
    
        responder_question_str = mc_instruction.format(question=preds['question'], choice_A=options['A'], choice_B=options['B'], choice_C=options['C'], choice_D=options['D'])
        return {"question": responder_question_str, "ground_truth": f"{gt_answer}"}

    def _parse_doc_general_qa(self, preds: dict) -> dict | int:
        """Parses a model's JSON output for the 'doc_general_qa' task.

        This function validates the format, extracts the question and answer, and filters
        out ground truth answers that are excessively long to ensure focused learning.

        Args:
            preds (dict): The dictionary parsed from the model's JSON response.

        Returns:
            dict | int: A dictionary containing the "question" and "ground_truth" if parsing is
                        successful. Returns -1 for a format error or 0 for an invalid (e.g., too long) answer.
        """
        if "question" not in preds or "answer" not in preds or not isinstance(preds['question'], str) or not isinstance(preds['answer'], str):
            print(f"Wrong json format for doc_general_qa: {preds}")
            return -1
        gt_answer = preds["answer"]
        if not gt_answer:
            print(f"No valid answer find in general_qa :{preds['answer']}")
            return -1
        gt_answer = str(gt_answer)
        parsed_gt_answer = docqa.parse_model_answer(gt_answer)
        if parsed_gt_answer:
            gt_answer = parsed_gt_answer   
        # filter out overlong answer
        token_length = len(self.tokenizer.encode(gt_answer))
        # filter overlong answer, we don't penalize this kind of data
        if token_length > 40:
            print(f"Gt in general_qa is more than 40 tokens :{gt_answer}")
            return 0
        
        return {"question": preds["question"], "ground_truth": f"{gt_answer}"}
    
    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        self.n_gpus = n_gpus
        self.n_rollout_engines = self.n_gpus // config.actor_rollout_ref.rollout.tensor_model_parallel_size

        if self.update_verifier:
            assert self.verifier_rollout_n > 1, "update_verifier only support n > 1"

        filter_cfg = config.algorithm.filter_groups
        if filter_cfg.enable:
            assert filter_cfg.max_num_gen_batches > 0, f"{filter_cfg.max_num_gen_batches=}"
            assert filter_cfg.metric is not None, f"{filter_cfg.metric=}"
        else:
            assert config.data.train_batch_size == config.data.gen_batch_size, \
                f"train_batch_size must be equal to gen_batch_size when filter_groups.enable is False, but got {config.data.train_batch_size =} and {config.data.gen_batch_size =}"

        overlong_buffer_cfg = config.custom_reward_function.overlong_buffer
        if overlong_buffer_cfg.enable:
            assert config.data.max_response_length >= overlong_buffer_cfg.len > 0, \
                f"{config.data.max_response_length=} / {overlong_buffer_cfg.len=}"
        else:
            assert overlong_buffer_cfg.len <= 0, f"{overlong_buffer_cfg.len=} > 0 but {overlong_buffer_cfg.enable=}"
            assert overlong_buffer_cfg.penalty_factor <= 0, f"{overlong_buffer_cfg.penalty_factor=} > 0 but {overlong_buffer_cfg.enable=}"
            assert overlong_buffer_cfg.log == False, f"{overlong_buffer_cfg.log=} == True but {overlong_buffer_cfg.enable=}"

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")


    def _dump_generations(self, inputs, outputs, extra_infos_dict, dump_path):
        """Saves model generation samples to JSONL files.

        This function writes inputs, outputs, and additional metadata to files within a
        step-specific directory. If 'data_source' is present in the metadata, it groups
        the samples by their source and saves each group to a separate file.

        Args:
            inputs (List[str]): A list of the input prompts.
            outputs (List[str]): A list of the generated model outputs.
            extra_infos_dict (dict): A dictionary where keys are metadata field names
                                     (e.g., 'score', 'data_source') and values are lists of
                                     corresponding data.
            dump_path (str): The base directory where the generation files will be saved.
        """
        os.makedirs(dump_path, exist_ok=True)
        base_dir = os.path.join(dump_path, f"step_{self.global_steps}")
        os.makedirs(base_dir, exist_ok=True)

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "step": [self.global_steps] * n,
        }
        for k, v in extra_infos_dict.items():
            if v is not None and len(v) == n:
                base_data[k] = v

        if 'data_source' in base_data:
            grouped_entries = defaultdict(list)

            # Group entries by data_source
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                ds = entry['data_source']
                grouped_entries[ds].append(entry)

            # Write each group to its own file
            for data_source, entries in grouped_entries.items():
                file_path = os.path.join(base_dir, f"{data_source}.jsonl")
                with open(file_path, "w", encoding='utf-8') as f:
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"Dumped {len(entries)} entries to {file_path}")
        else:
            filename = os.path.join(base_dir, "all.jsonl")
            with open(filename, "w") as f:
                for i in range(n):
                    entry = {k: v[i] for k, v in base_data.items()}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Dumped generations to {filename}")

    def _create_dataloader(self):
        """Initializes and configures the training and validation dataloaders.

        This method sets up the `DomainWeightedSPELLDataset` for handling different data sources,
        initializes a `DomainSampler` to control the sampling probability of each domain,
        and creates `StatefulDataLoader` instances for both training and validation.
        """
        domain_sampling_config = self.config.algorithm.domain_sampling
        # print(f"[Train DataLoader] Domain sampling enabled. Domain parquet files: {domain_parquet_files}")
        # if self.config.data.use_cache.enable:
        self.train_dataset = DomainWeightedSPELLDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', True),
            truncation=self.config.data.get('truncation', 'error'),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
            n_docs=self.config.data.n_docs,
            tasks=self.config.data.tasks,
            domain_key='data_source',
            use_cache=self.config.data.use_cache.get('enable', False),
            cache_queue_size=int(self.config.data.use_cache.cache_size)
        )
        domains = self.train_dataset.domains

        # Initialize domain weights
        if domain_sampling_config.init_weight_method == "average":
            # method 1. compute the domain weights by the average of domain data nums
            domain_weights = {
                domain: 1.0 / len(domains) for domain in domains
            }
        elif domain_sampling_config.init_weight_method == "predefined":
            # self-defined weights, related to data.tasks
            domain_weights_init = domain_sampling_config.init_weights 
            total_weight = sum(domain_weights_init)
            domain_weights = {
                domain: weight / total_weight
                for weight,domain in zip(domain_weights_init,domains)
            }
        else:
            raise NotImplementedError(f"Domain sampling init_weight_method {domain_sampling_config.init_weight_method} not implemented.")

        print(f"init domain weights:{domain_weights}")
        # Create the sampler
        sampler = DomainSampler(
            dataset=self.train_dataset,
            batch_size=self.config.data.gen_batch_size,
            domain_weights=domain_weights
        )
        self.sampler = sampler

        # Create the dataloader
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collate_fn,
        )
        print("[Train DataLoader] Employing domain sampling for training dataloader.")

        assert len(self.train_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=0,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Logs a sample of validation generations to a configured logger (e.g., W&B).

        It takes a random but deterministic subset of validation samples and logs them as a
        table, allowing for qualitative analysis of model performance over time.

        Args:
            inputs (List[str]): A list of validation input prompts.
            outputs (List[str]): A list of generated outputs from the model.
            scores (List[float]): A list of corresponding scores for each sample.
        """

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        reward_model_lst = []

        # print("Size of validation dataset: ",len(self.val_dataloader))

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # print("Cur batch number samples: ",leb(test_batch))

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                # print("We only do validation on rule-based rm")
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']

            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_infos_dict["response_length"].extend(_compute_response_info(test_batch)['response_length'].squeeze().cpu().tolist())
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    # store only the number output
                    # if isinstance(lst[0], (int,float)):
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            reward_model_lst.append(test_batch.non_tensor_batch.get('reward_model', ['unknown'] * reward_tensor.shape[0]))
            # store the avg length
       
        #  dump generations
        data_sources = np.concatenate(data_source_lst, axis=0)
        reward_models = np.concatenate(reward_model_lst, axis=0)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            extra_info_dict = {
                "data_source": data_sources,
                "reward_model": reward_models
            }
            extra_info_dict.update(reward_extra_infos_dict)
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                extra_infos_dict=extra_info_dict,
                dump_path=val_data_dir
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
        
        data_src2var2metric2val = process_spell_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict, group_prefix = True)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if var_name == core_var and metric_name.startswith("mean@"):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)


        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,

        # save dataloader
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        # AssertionError: 1512 % 16 !=  should be 1536
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _prepare_responder_gen_batch(self, data: DataProto, pre_filter: bool):
        """Prepares a batch of prompts for the responder model.

        This function processes the output from the questioner model. It involves:
        1. Parsing the generated JSON to extract the question and ground truth.
        2. Filtering out malformed or invalid questions.
        3. Applying the appropriate chat template to format the prompt for the responder.
        4. Tokenizing the formatted prompts.
        5. Collating the results into a new `DataProto` batch.

        Args:
            data (DataProto): The `DataProto` containing the questioner's generations.
            pre_filter (bool): A flag indicating if this is for a pre-filtering step,
                               which might use a different context.

        Returns:
            Tuple[DataProto | None, dict]: A tuple containing:
                - The prepared `DataProto` for the responder, or None if no valid items were produced.
                - A dictionary of UIDs for bad cases that were filtered out.
        """
        # Perform grounding filter / format filter
        current_meta_info = data.meta_info if data and hasattr(data, 'meta_info') else {}
        valid_responder_items = []
        bad_case_uids = defaultdict(list)
        responder_tensors_list_unbatched: dict[str, list[torch.Tensor]] = {
            'input_ids': [], 'attention_mask': [], 'position_ids': []
        }
        responder_non_tensors_list: dict[str, list] = {
            'raw_prompt_ids': [], 'raw_prompt': [],'reward_model': [], 'extra_info': [],
            'ability': [], 'data_source': [], 'index': [], 'uid': [], 'is_new': []
        }

        print(f"Len of questioner gen batch before filtering: {len(data)}")
        overlong_prompts = 0
        num_valid_responder_items = 0
        
        for i in range(len(data)):
            item = data[i]
            uid = item.non_tensor_batch.get('uid')
            task = item.non_tensor_batch.get('data_source')
            extra_info = item.non_tensor_batch['extra_info']
            
            if 'responses' not in item.batch: continue
            response_str = self.tokenizer.decode(item.batch['responses'].squeeze().cpu().tolist(), skip_special_tokens=True)
            
            preds = get_record(response_str)
            if isinstance(preds, dict) and "Empty" in preds:
                print(f"Skipping item {task} {i} due to overlong output {response_str}")
                continue

            parser = self.task_parsers.get(task)

            if not isinstance(preds, dict) or parser is None:
                print(f"Skipping item {task} {i} due to bad format {preds}")
                bad_case_uids["bad_format"].append(uid)
                continue

            parsed_info = parser(preds)
            if isinstance(parsed_info, int):
                 # we only penalize bad format
                if parsed_info == -1:
                    print(f"Store item {task} {i} for bad format {preds}")
                    bad_case_uids["bad_format"].append(uid)
                continue

            context = "Empty" if pre_filter else '\n'.join(extra_info["paragraphs"])
            # TypeError: replace() argument 2 must be str, not dict
            user_input = self.responder_template[task].replace("{question}",parsed_info["question"]).replace("{content}",context)

            # cache the old question and answer for cache
            extra_info["ori_question"] = parsed_info["question"]
            extra_info["ori_answer"] = parsed_info["ground_truth"]

            prompt_chat = [{"role": "user", "content": user_input}]
            prompt_templated = self.tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
            prompt_ids = self.tokenizer.encode(prompt_templated, add_special_tokens=False)

            if len(prompt_ids) > self.config.data.max_prompt_length:
                overlong_prompts += 1; print(f"Skipping item {i} due to overlong responder prompt."); continue
            
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_templated, tokenizer=self.tokenizer, max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation='error'
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            
            responder_tensors_list_unbatched['input_ids'].append(input_ids[0])
            responder_tensors_list_unbatched['attention_mask'].append(attention_mask[0])
            responder_tensors_list_unbatched['position_ids'].append(position_ids[0])
            
            # Append non-tensor data to lists
            responder_non_tensors_list['raw_prompt_ids'].append(prompt_ids)
            responder_non_tensors_list['raw_prompt'].append(prompt_chat)
            responder_non_tensors_list['reward_model'].append({"style": "rule", "ground_truth": f"The correct answer is {parsed_info['ground_truth']}"})
            responder_non_tensors_list['extra_info'].append(extra_info) # Contains remaining items
            responder_non_tensors_list['ability'].append(item.non_tensor_batch.get('ability'))
            responder_non_tensors_list['data_source'].append(task)
            responder_non_tensors_list['index'].append(item.non_tensor_batch.get('index'))
            responder_non_tensors_list['uid'].append(item.non_tensor_batch.get('uid'))
            responder_non_tensors_list['is_new'].append(item.non_tensor_batch.get('is_new', True))

            num_valid_responder_items += 1

        print(f"Number of overlong prompts for responder: {overlong_prompts}")
        print(f"Number of valid items prepared for responder: {num_valid_responder_items}")

        if num_valid_responder_items == 0:
            print("Warning: No valid items produced for responder batch.")
            return None, None

        # Collate list of dicts into dict of lists/tensors
        collated_responder_tensors: dict[str, torch.Tensor] = {}
        for key, list_of_tensors in responder_tensors_list_unbatched.items():
            collated_responder_tensors[key] = torch.stack(list_of_tensors,dim=0).contiguous()
        
        final_batch = DataProto.from_dict(
            tensors=collated_responder_tensors, non_tensors=responder_non_tensors_list,
            meta_info=current_meta_info)

        if len(final_batch) > 0:
            print("="*80)
            example_ids = final_batch.non_tensor_batch['raw_prompt_ids'][0]
            print(f"Example prompt for responder (collated): {self.tokenizer.decode(example_ids)}")
            print(f"GT for first responder example: {final_batch.non_tensor_batch['reward_model'][0]['ground_truth']}")
            print("="*80)

        return final_batch, bad_case_uids

    def _questioner_rollout(self, questioner_batch: DataProto) -> DataProto:
        """Handles the generation step for the questioner model.

        Takes an initial batch of prompts, sends them to the actor-rollout worker group
        for generation, and returns the batch augmented with the generated questions.

        Args:
            questioner_batch (DataProto): The initial data batch with prompts for the questioner.

        Returns:
            DataProto: The `DataProto` updated with the generated sequences.
        """
        questioner_gen_batch = prepare_gen_batch(questioner_batch)   
        questioner_gen_batch.meta_info['n'] = 1     
        # rollout for questioner and set n to 1
        # questioner_gen_output = self.actor_rollout_wg.generate_sequences(questioner_gen_batch)
        questioner_gen_batch_padded, pad_size = pad_dataproto_to_divisor(questioner_gen_batch, self.actor_rollout_wg.world_size)
       
        if not self.async_rollout_mode:
            questioner_gen_output_padded = self.actor_rollout_wg.generate_sequences(questioner_gen_batch_padded)
        else:
            self.async_rollout_manager.wake_up()
            questioner_gen_output_padded = self.async_rollout_manager.generate_sequences(questioner_gen_batch_padded)
            self.async_rollout_manager.sleep()

        questioner_gen_output = unpad_dataproto(questioner_gen_output_padded, pad_size=pad_size * questioner_gen_batch.meta_info['n'])
        questioner_batch = questioner_batch.union(questioner_gen_output)
        questioner_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(questioner_batch))], dtype=object)
        return questioner_batch

    def _responder_pipeline(self, questioner_rollout: DataProto, metrics: dict, timing_raw: dict):
        """Orchestrates the full responder data generation and reward calculation pipeline.

        This complex method manages the multi-stage process of generating responder training data:
        1. (Optional) Filters the questioner's generated questions for grounding.
        2. Prepares the final responder prompts from the (filtered) questions.
        3. Performs rollout (generation) with the responder model.
        4. Calculates rewards, potentially combining a rule-based reward with a signal
           from an LLM-based self-verification step.
        5. (Optional) Filters the resulting trajectories based on reward variance to select
           for high-signal training examples.

        Args:
            questioner_rollout (DataProto): The `DataProto` containing the generated questions.
            metrics (dict): A dictionary to accumulate metrics from the pipeline.
            timing_raw (dict): A dictionary to accumulate timing information.

        Returns:
            Tuple[DataProto | None, DataProto]: A tuple containing:
                - The final `DataProto` of responder experiences ready for training, or None if empty.
                - The `DataProto` of the questioner rollouts that were used to generate the experiences.
        """

        cache_config = self.config.data.use_cache

        # reuse the cached prompt from last batch, not used in our paper
        if self.questioner_prompt_reuse and self.accumulator.questioner_buffer is not None:
            questioner_rollout.non_tensor_batch['is_new'] = np.array([True] * len(questioner_rollout), dtype=object)
            self.accumulator.questioner_buffer.non_tensor_batch['is_new'] = np.array([False] * len(self.accumulator.questioner_buffer), dtype=object)
            print(f"Enable questioner prompt reuse, get {len(self.accumulator.questioner_buffer)} from last step")
            questioner_rollout = DataProto.concat([self.accumulator.questioner_buffer, questioner_rollout])
            # clean the buffer
            self.accumulator.questioner_buffer = None

        # --- Stage 1: Optional Grounding Filter of Questioner's questions ---
        if self.filter_questioner_prompts:
            with _timer('responder_filter', timing_raw):
                pre_filter_batch, bad_case_uids = self._prepare_responder_gen_batch(questioner_rollout, pre_filter=True)
                # Update bad cases found during this stage
                for bad_type, uids in bad_case_uids.items():
                    self.accumulator.questioner_bad_case_uids[bad_type].extend(uids)
                filter_gen_batch = prepare_gen_batch(pre_filter_batch)
                
                filter_gen_batch.meta_info['n'] = 1   
                responder_filter_gen_batch_output = None

                filter_gen_batch_padded, pad_size = pad_dataproto_to_divisor(filter_gen_batch, self.actor_rollout_wg.world_size)
                if not self.async_rollout_mode:
                    responder_filter_gen_batch_output_padded = self.actor_rollout_wg.generate_sequences(filter_gen_batch_padded)
                else:
                    self.async_rollout_manager.wake_up()
                    responder_filter_gen_batch_output_padded = self.async_rollout_manager.generate_sequences(filter_gen_batch_padded)
                    self.async_rollout_manager.sleep()

                responder_filter_gen_batch_output = unpad_dataproto(responder_filter_gen_batch_output_padded, pad_size=pad_size * filter_gen_batch.meta_info['n'])
                pre_filter_batch = pre_filter_batch.union(responder_filter_gen_batch_output)
                if self.use_rm:
                    # we first compute reward model score
                    reward_tensor = self.rm_wg.compute_rm_score(pre_filter_batch)
                    pre_filter_batch = pre_filter_batch.union(reward_tensor)

                # we combine with rule-based rm
                reward_extra_infos_dict: dict[str, list]

                
                try:
                    reward_result = self.reward_fn(pre_filter_batch, return_dict=True)
                    reward_extra_infos_dict = reward_result['reward_extra_info']
                except Exception as e:
                    print(f'Error in reward_fn: {e}')
                    reward_tensor = self.reward_fn(pre_filter_batch)
                    reward_extra_infos_dict = {}

                metric_name = self.config.algorithm.filter_groups.metric
                if reward_extra_infos_dict and metric_name in reward_extra_infos_dict:
                    pre_filter_batch.non_tensor_batch[metric_name] = np.array(reward_extra_infos_dict[metric_name])       
                else:
                    print(f"Warning, {metric_name} not found in reward_extra_infos_dict: {list(reward_extra_infos_dict.keys())=}")           

                kept_prompt_uids = []
                filtered_prompt_uids = []
                for uid, metric_val in zip(pre_filter_batch.non_tensor_batch['uid'],
                                        pre_filter_batch.non_tensor_batch[metric_name]):
                    if metric_val < 0.1:
                        kept_prompt_uids.append(uid)
                    else:
                        filtered_prompt_uids.append(uid)
                kept_prompt_uids = set(kept_prompt_uids)
                # questioner bad case for not gounding
                if len(filtered_prompt_uids) > 0:
                    self.accumulator.questioner_bad_case_uids["bad_ground"].extend(filtered_prompt_uids)
                kept_traj_idxs = [idx for idx, traj_from_prompt_uid in enumerate(questioner_rollout.non_tensor_batch['uid']) if traj_from_prompt_uid in kept_prompt_uids]
                # print(f"Before filtering, {len(questioner_rollout)} trajectories")
                questioner_rollout = questioner_rollout[kept_traj_idxs]
                print(f"After grouding filtering, keep {len(questioner_rollout)} trajectories for responder")                    


        # --- Stage 2: Prepare Responder Batch from (filtered) Questioner outputs ---

        responder_batch, bad_case_uids = self._prepare_responder_gen_batch(questioner_rollout, pre_filter=False)

        new_propopser_uids, new_questioner_extra_infos = [], []
        if cache_config.get("enable", False):
            case_types = responder_batch.non_tensor_batch.pop('is_new')
            if 'is_new' in questioner_rollout.non_tensor_batch:
                questioner_rollout.non_tensor_batch.pop('is_new')
            new_idxs = [i for i, is_new in enumerate(case_types) if is_new]
            all_uids = responder_batch.non_tensor_batch['uid']
            new_propopser_uids.extend([all_uids[i] for i in new_idxs])
            all_extra_infos = responder_batch.non_tensor_batch['extra_info']
            new_questioner_extra_infos.extend([all_extra_infos[i] for i in new_idxs])

        # Update bad cases collected in this stage
        for bad_type, uids in bad_case_uids.items():
            self.accumulator.questioner_bad_case_uids[bad_type].extend(uids)

        if len(responder_batch) == 0:

            return None, questioner_rollout
        
        # --- Stage 3: Responder Rollout ---
        responder_gen_batch = prepare_gen_batch(responder_batch)
        responder_gen_output = None
        responder_gen_batch_padded, pad_size = pad_dataproto_to_divisor(responder_gen_batch, self.actor_rollout_wg.world_size)
        with _timer('responder_gen', timing_raw):
            if not self.async_rollout_mode:
                responder_gen_output_padded = self.actor_rollout_wg.generate_sequences(responder_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                responder_gen_output_padded = self.async_rollout_manager.generate_sequences(responder_gen_batch_padded)
                self.async_rollout_manager.sleep()
        responder_gen_output = unpad_dataproto(responder_gen_output_padded, pad_size=pad_size * int(self.config.actor_rollout_ref.rollout.n))

        responder_batch = responder_batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True)
        responder_batch = responder_batch.union(responder_gen_output)


        # --- Stage 4: Reward Calculation (with Self-Verification) ---
        with _timer('reward', timing_raw):
            if self.use_rm:
                rm_score_tensor = self.rm_wg.compute_rm_score(responder_batch)
                responder_batch = responder_batch.union(rm_score_tensor)

            reward_extra_infos_dict: dict[str, list]
            try:
                reward_result = self.reward_fn(responder_batch, return_dict=True)
                reward_tensor_fn = reward_result['reward_tensor']
                reward_extra_infos_dict = reward_result['reward_extra_info']
            except Exception as e:
                print(f'Error in reward_fn: {e}')
                reward_tensor_fn = self.reward_fn(responder_batch)
                reward_extra_infos_dict = {}


            # Whether to include verifier role
            if self.self_verification:
                responder_batch.meta_info['response_length'] = reward_tensor_fn.shape[-1]
                with _timer('self_verification', timing_raw):
                    # This call will return a tensor of shape [batch_size] with 1.0 for "YES" and 0.0 for "NO"
                    llm_rewards, llm_judges = self._self_verification(responder_batch, reward_extra_infos_dict)

                # SUB-STEP 4.3: Combine rewards: final_reward = max(llm_reward, reward_fn)
                # Get sequence-level score from reward_fn
                scores_fn = reward_tensor_fn.sum(dim=-1)
                print(f"Rule-based scores (sample): {scores_fn[:5]}")
                print(f"LLM scores (sample): {llm_rewards[:5]}")
                
                REWARD_FUNCTIONS = {
                    "max": lambda x, y: torch.max(x, y),
                    "mean": lambda x, y: (x + y) / 2,
                    "min": lambda x, y: torch.min(x, y),
                    "sum": lambda x, y: x + y,
                }

                llm_rewards = llm_rewards.to(scores_fn.device, dtype=scores_fn.dtype)

                # get the valid mask
                valid_mask = (llm_rewards >= 0)

                # Align scores_fn with llm_rewards using the same mask
                scores_fn_valid = scores_fn[valid_mask]
                llm_rewards_valid = llm_rewards[valid_mask]

                # Apply selected reward function, we use maximum combination as default setting, see Eq.(6)
                combined_scores = REWARD_FUNCTIONS[self.config.algorithm.reward_combined_function](
                    scores_fn_valid, llm_rewards_valid
                )

                # Build final_scores using the same mask
                final_scores = torch.zeros_like(scores_fn)
                final_scores[valid_mask] = combined_scores
                final_scores[~valid_mask] = scores_fn[~valid_mask]  # Keep original score where LLM judge is negative

                print(f"Combined final scores (sample): {final_scores[:5]}")

                # Create new token-level scores where the final score is placed at the last token
                # Maybe exists bugs for index, however, if we don't apply ppo algorithm, calculate the grpo advantage will ignore the true index
                response_len = responder_batch.batch['responses'].size(1)
                response_attention_mask = responder_batch.batch['attention_mask'][:, -response_len:]
                sequence_lengths = response_attention_mask.sum(dim=1)
                last_token_indices = (sequence_lengths - 1).clamp(min=0) # handle empty sequences

                new_token_level_scores = torch.zeros_like(reward_tensor_fn)
                new_token_level_scores.scatter_(1, last_token_indices.unsqueeze(1), final_scores.unsqueeze(1))
                
                # SUB-STEP 4.4: Update the batch with final combined reward and logging info
                responder_batch.batch['token_level_scores'] = new_token_level_scores
                # Add llm_reward to extra info for logging purposes
                responder_batch.non_tensor_batch['llm_reward'] = np.array(llm_rewards.cpu().tolist(), dtype=object)
                responder_batch.non_tensor_batch['llm_judge'] = llm_judges
                responder_batch.non_tensor_batch['rule_based_reward'] = np.array(scores_fn.cpu().tolist(), dtype=object)
                responder_batch.non_tensor_batch['score'] = np.array(final_scores.cpu().tolist(), dtype=object)
                # save the parsed prediction
                responder_batch.non_tensor_batch['pred'] = np.array(reward_extra_infos_dict.get('pred', ["empty"] * final_scores.shape[0]), dtype=object)

            # rule-based reward only
            else:
                responder_batch.batch['token_level_scores'] = reward_tensor_fn

                print(f'{list(reward_extra_infos_dict.keys())=}')
                if reward_extra_infos_dict and 'score' in reward_extra_infos_dict:
                    responder_batch.non_tensor_batch.update({"score": np.array(reward_extra_infos_dict['score'])})

            # compute rewards 
            if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                responder_batch, kl_metrics = apply_kl_penalty(responder_batch,
                                                            kl_ctrl=self.kl_ctrl,
                                                            kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                responder_batch.batch['token_level_rewards'] = responder_batch.batch['token_level_scores']   

        # Step5: Dynamic Sampling, filter out those group with reward std = 0
        if self.config.algorithm.filter_groups.enable:
            
            metric_name = self.config.algorithm.filter_groups.metric
            if metric_name == "seq_final_reward":
                # Turn to numpy for easier filtering
                responder_batch.non_tensor_batch["seq_final_reward"] = responder_batch.batch['token_level_scores'].sum(
                    dim=-1).numpy()
            elif metric_name == "seq_reward":
                responder_batch.non_tensor_batch["seq_reward"] = responder_batch.batch['token_level_scores'].sum(
                    dim=-1).numpy()

            # Collect the sequence reward for each trajectory
            prompt_uid2metric_vals = defaultdict(list)
            for uid, metric_val in zip(responder_batch.non_tensor_batch['uid'],
                                        responder_batch.non_tensor_batch[metric_name]):
                prompt_uid2metric_vals[uid].append(metric_val)

            
            prompt_uid2metric_std = {}
            prompt_uid2metric_mean = {} 
            for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                prompt_uid2metric_mean[prompt_uid] = np.mean(metric_vals)

            if self.config.algorithm.filter_groups.filter_by_mean:
                low, high = self.config.algorithm.filter_groups.mean_lower_bound, self.config.algorithm.filter_groups.mean_upper_bound
                kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 and low < prompt_uid2metric_mean[uid] < high]  
            else:
                kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0]
            
            if cache_config.get("enable", False) and len(new_propopser_uids) > 0:
                assert len(new_propopser_uids) == len(new_questioner_extra_infos), "The uids and extra_infos should have the same dimension!"
                print(new_questioner_extra_infos[0].keys())
                cached_prompt_uids = set([uid for uid, mean in prompt_uid2metric_mean.items() if cache_config.get("cached_lower", 0.5) <= mean <= cache_config.get("cached_higher", 1.0)])
                cnt = 0
                for idx, uid in enumerate(new_propopser_uids):
                    if uid in cached_prompt_uids:
                        extra_info = new_questioner_extra_infos[idx]
                        cnt += 1
                        cache_key = f"{extra_info['task']}_{extra_info['question_id']}"
                        cache_value = (extra_info["evidence"], extra_info["ori_question"], extra_info["ori_answer"])
                        for cache_item in self.train_dataset.cache[cache_key]:
                            if cache_value[0] == cache_item[1] or cache_value[1] == cache_item[1] or cache_value[2] == cache_item[2]:
                                print(f"Skipping {cache_value} due repeative question: {cache_item}")
                                continue
                        print(f"Caching id {cache_key} \n {cache_value}")
                        self.train_dataset.cache[cache_key].append(cache_value)

                print(f"Save {cnt} questions to cache...")

            zero_variance_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std == 0 and not (0.0 < prompt_uid2metric_mean[uid] < 1.0)]

            if len(zero_variance_prompt_uids) > 0:
                self.accumulator.questioner_bad_case_uids["zero_variance"].extend(zero_variance_prompt_uids)

            kept_traj_idxs = []
            for idx, traj_from_prompt_uid in enumerate(responder_batch.non_tensor_batch['uid']):
                if traj_from_prompt_uid in kept_prompt_uids:
                    kept_traj_idxs.append(idx)

            responder_batch = responder_batch[kept_traj_idxs]
            print(f"After reward std filtering, kept {len(kept_traj_idxs) // self.config.actor_rollout_ref.rollout.n} prompts")
            
        return responder_batch, questioner_rollout

    def _self_verification(self, batch: DataProto, reward_extra_infos_dict: dict) -> torch.Tensor:
        """Performs a self-verification step using the model as an LLM judge.

        This method formulates a new prompt asking the model to compare the responder's
        answer with the ground truth and to output a "YES" or "NO" judgment. It runs
        this generation and parses the output to produce a binary reward signal. This
        signal can be combined with the primary rule-based reward. It also accumulates
        the judge's generations for potential training of the verifier role.

        Args:
            batch (DataProto): The responder batch, containing generated responses and ground truth.
            reward_extra_infos_dict (dict): A dictionary containing parsed predictions ('pred')
                                            and ground truths ('gt') from the rule-based reward function.

        Returns:
            torch.Tensor: A tensor of shape [batch_size] containing the LLM-based rewards
                          (1.0 for "YES", 0.0 for "NO", -100.0 for invalid cases).
        """
        batch_size = len(batch)

        response_length = batch.meta_info.pop('response_length')
        extra_infos = batch.non_tensor_batch['extra_info']
        # deepcopy the verifier batch
        # 1. Prepare batch for verification generation
        prompts_text = [item["ori_question"] for item in extra_infos]
        tasks = batch.non_tensor_batch['data_source']
        indexs = batch.non_tensor_batch['index']
        preds = reward_extra_infos_dict.get("pred",[])
        gts = reward_extra_infos_dict.get("gt",[])
        rule_scores = reward_extra_infos_dict.get("score",[])
        # new uids for verifier
        verifier_uids = [str(uuid.uuid4()) for _ in range(batch_size)]
        if len(prompts_text) != batch_size or len(preds) != batch_size or len(gts) != batch_size:
            print("Warning: the size for prompt, gt, and pred is not equal! This will skip the llm judge step.")
            return torch.tensor([0.0] * batch_size, dtype=torch.float32)

        verifier_tensors_list_unbatched: dict[str, list[torch.Tensor]] = {
            'input_ids': [], 'attention_mask': [], 'position_ids': []
        }
        verifier_non_tensors_list: dict[str, list] = {
            'raw_prompt_ids': [], 'raw_prompt': [], 'uid': [],
            'extra_info': [], 'index': [], 'data_source': []
        }
        # The sample index really in the judge batch, a turple
        valid_idxs = []
        uid_2idxs = dict()
        for i in range(batch_size):
            uid_2idxs[verifier_uids[i]] = i
            if preds[i] is None or gts[i] is None or prompts_text[i] is None or tasks[i] not in self.llm_judge_tasks:
                # user_content = "Just output Hello."
                continue
            else:
                valid_idxs.append(i)

            user_content = VERIFIER_PROMPT.format(
                problem=prompts_text[i],
                answer_1=preds[i],
                answer_2=gts[i]
            )
            chat = [{"role": "user", "content": user_content}]
            templated_prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            # verification_prompts.append(templated_prompt)

            prompt_templated_ids = self.tokenizer.encode(templated_prompt, add_special_tokens=False)

            verifier_input_ids_b, verifier_attention_mask_b = verl_F.tokenize_and_postprocess_data(
                prompt=templated_prompt, tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True, truncation='error')
            verifier_position_ids_b = compute_position_id_with_mask(verifier_attention_mask_b)

            verifier_tensors_list_unbatched['input_ids'].append(verifier_input_ids_b[0])
            verifier_tensors_list_unbatched['attention_mask'].append(verifier_attention_mask_b[0])
            verifier_tensors_list_unbatched['position_ids'].append(verifier_position_ids_b[0])

            verifier_non_tensors_list['raw_prompt_ids'].append(prompt_templated_ids)
            verifier_non_tensors_list['raw_prompt'].append(chat)
            verifier_non_tensors_list['uid'].append(verifier_uids[i])
            verifier_non_tensors_list['data_source'].append(tasks[i])
            verifier_non_tensors_list['extra_info'].append(extra_infos[i])
            verifier_non_tensors_list['index'].append(indexs[i])

        collated_judge_tensors: dict[str, torch.Tensor] = {}
        for key, list_of_tensors in verifier_tensors_list_unbatched.items():
            collated_judge_tensors[key] = torch.stack(list_of_tensors,dim=0).contiguous()
        
        # Prepare verifier batch
        final_verifier_data = DataProto.from_dict(
            tensors=collated_judge_tensors, non_tensors=verifier_non_tensors_list,
            meta_info=None)

        verifier_gen_batch = prepare_gen_batch(final_verifier_data)
        # set the generation number, default to 1 for verifier
        verifier_gen_batch.meta_info['n'] = self.verifier_rollout_n
        ver_output_batch = None

        verifier_gen_batch_padded, pad_size = pad_dataproto_to_divisor(verifier_gen_batch, self.actor_rollout_wg.world_size)
        if not self.async_rollout_mode:
            ver_output_batch_padded = self.actor_rollout_wg.generate_sequences(verifier_gen_batch_padded)
        else:
            self.async_rollout_manager.wake_up()
            ver_output_batch_padded = self.async_rollout_manager.generate_sequences(verifier_gen_batch_padded)
            self.async_rollout_manager.sleep()
        # 2. Generate verification responses
        # ver_output_batch_padded = self.actor_rollout_wg.generate_sequences(verifier_gen_batch_padded)
        # unpad
        ver_output_batch = unpad_dataproto(ver_output_batch_padded, pad_size=pad_size * self.verifier_rollout_n)

        if self.verifier_rollout_n > 1:
            final_verifier_data = final_verifier_data.repeat(self.verifier_rollout_n, interleave=True)
        final_verifier_data = final_verifier_data.union(ver_output_batch)

        uids = final_verifier_data.non_tensor_batch['uid']
        ver_responses_text = final_verifier_data.batch['responses']

        # 3. Parse responses and create llm_reward tensor
        # in this version, we don't apply llm judge for multi-choice question
        # mask the invalid place and set the rm to -100.0
        llm_rewards = [-100.0] * batch_size
        llm_judges = [[] for _ in range(batch_size)]

        # samples_to_print = random.sample(valid_idxs, min(5,len(valid_idxs)))
        valid_idxs = set(valid_idxs)
        idx_2judge = defaultdict(list)
        
        all_judges = []
        extracted_judges = []
        for response, uid in zip(ver_responses_text, uids):
            idx = uid_2idxs[uid]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            # get rid of the <think> token
            judge = extract_solution(response)
            extracted_judges.append(judge)
            # only yes will be assigned a reward
            # save the result
            llm_judges[idx].append(response)
            if judge is not None and ("[[YES]]" in judge and "[[NO]]" not in judge) or ("YES" in judge and "NO" not in judge):
                idx_2judge[idx].append(1.0)
                all_judges.append(1.0)
            # [[NO]] or bad case
            elif judge is not None and ("[[NO]]" in judge and "[[YES]]" not in judge) or ("NO" in judge and "YES" not in judge):
                idx_2judge[idx].append(0.0)
                all_judges.append(0.0)
            # overlong output or bad case
            else:
                # format penalty = -1
                idx_2judge[idx].append(-1.0)
                all_judges.append(-1.0)

        for idx in range(batch_size):
            if idx not in valid_idxs:
                for _ in range(self.verifier_rollout_n):
                    llm_judges[idx].append("Empty")
                    idx_2judge[idx].append(-100.0)

        # gather the results
        for idx, llm_rms in idx_2judge.items():
            # majority voting
            llm_rewards[idx] = Counter(llm_rms).most_common()[0][0]
            
        if self.update_verifier:
            judge_rms = []
            good_cases = []
            for idx, (uid, judge) in enumerate(zip(uids, all_judges)):
                ori_idx = uid_2idxs[uid]
                maj_rm = llm_rewards[ori_idx]
                # All bad cases, skip
                if maj_rm == -1.0:
                    judge_rms.append(-1.0)
                    continue
                llm_rms = idx_2judge[ori_idx]
                valid_llm_rms = [rm for rm in llm_rms if rm != -1.0]
                mean_rm = np.mean(valid_llm_rms)

                if maj_rm == 0.0:
                    mean_rm = 1 - mean_rm

                # set veirifier reward
                if judge == -1.0:
                    judge_rms.append(-1.0)
                elif maj_rm == judge:
                    judge_rms.append(1.0)
                else:
                    judge_rms.append(0.0)

                # dynamic sampling and check llm judge and rule-rm consistency
                # maj_cons: only use the group where majority vote == rule rm to update verifier
                # maj: only use the group where majority vote is not -1 to update verifier
                if self.verifier_lower_bound <= mean_rm <= self.verifier_upper_bound :
                    if self.verifier_label_type == "maj_cons":
                        if maj_rm == rule_scores[ori_idx]:
                            good_cases.append(idx)  
                    # ablate consistency update
                    elif self.verifier_label_type == "maj":
                        good_cases.append(idx)
                    else:
                        raise NotImplementedError

            print(f"Number of consistency judge batch count :{len(good_cases)}")
            if len(good_cases) > 0:
                final_verifier_data.non_tensor_batch["score"] = np.array(judge_rms, dtype=object)
                final_verifier_data.non_tensor_batch["rule_based_reward"] = np.array(judge_rms, dtype=object)
                final_verifier_data.non_tensor_batch["llm_reward"] = np.array(judge_rms, dtype=object)
                final_verifier_data.non_tensor_batch["pred"] = np.array(extracted_judges, dtype=object)
                final_verifier_data.non_tensor_batch["reward_model"] = np.array([{"ground_truth": "consistency"}] * len(judge_rms), dtype=object)
                final_verifier_data.non_tensor_batch["type"] = np.array(["good"] * len(judge_rms), dtype=object)
                final_verifier_data.non_tensor_batch["role"] = np.array(["verifier"] * len(judge_rms), dtype=object)
                final_verifier_data.batch["token_level_scores"] = torch.tensor([[item] + [0] * (response_length - 1) for item in judge_rms], dtype=torch.float32)          
                final_verifier_data_to_train = final_verifier_data[good_cases]
                assert len(final_verifier_data_to_train) % self.verifier_rollout_n == 0, "verifier batch size should be divisible by rollout_n"

                # Accumulate veirifer experience
                if self.accumulator.verifier_batch is None:
                    tensor_keys = final_verifier_data_to_train.batch.keys()
                    self.accumulator.verifier_batch = final_verifier_data_to_train
                elif len(final_verifier_data_to_train) > 0:
                    # check dim consistency before concat
                    tensor_keys = self.accumulator.verifier_batch.batch.keys()
                    for key in tensor_keys:
                        if self.accumulator.verifier_batch.batch[key].shape[1:] != final_verifier_data_to_train.batch[key].shape[1:]:
                            raise ValueError(f"Tensor dim unmatch, {key}: Old dim: {self.accumulator.verifier_batch.batch[key].shape[1:]} New dim: {final_verifier_data_to_train.batch[key].shape[1:]}")
                    self.accumulator.verifier_batch = DataProto.concat([self.accumulator.verifier_batch, final_verifier_data_to_train])
        return torch.tensor(llm_rewards, dtype=torch.float32), np.array([item[0] for item in llm_judges], dtype=object)

    def _perform_ppo_update(self, batch: DataProto, timing_raw: dict):
        """Performs one full PPO (Proximal Policy Optimization) update step.

        This method orchestrates the core learning algorithm. It takes a prepared batch
        of experience data and performs the following steps:
        1. Computes old log probabilities and values for the trajectories.
        2. Computes advantages.
        3. Performs multiple epochs of gradient updates on the actor and critic models.
        4. Gathers and returns all relevant metrics from the update.
        5. Handles validation and checkpointing based on the current global step.

        Args:
            batch (DataProto): The final, collated `DataProto` containing all experiences
                               for the update.
            timing_raw (dict): A dictionary for recording the timing of different sub-steps.

        Returns:
            Tuple[DataProto, dict]: A tuple containing the processed batch and a dictionary of
                                    metrics from the update step.
        """
        metrics = {}

        batch.batch["response_mask"] = compute_response_mask(batch)
        
        # Balance tokens across DP ranks
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

         # compute global_valid tokens
        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

        # Compute logprobs and values
        with _timer('old_log_prob', timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            # save the entropy for metrics
            masked_entropys = torch.sum(entropys * response_masks, dim=-1) / torch.sum(response_masks, dim=-1)  # token-mean
            batch.batch['entropys'] = masked_entropys 
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with _timer('ref', timing_raw):
                batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(batch))
        if self.use_critic:
            with _timer('values', timing_raw):
                batch = batch.union(self.critic_wg.compute_values(batch))

        # Compute rewards (KL penalty) and advantage
        with _timer('adv', timing_raw):
            if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
            
            batch = compute_advantage(
                batch, self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma, lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                update_questioner=self.update_questioner,
                questioner_reward_type=self.questioner_reward_type,
                questioner_group=self.questioner_group
            )


        # Update models
        if self.use_critic:
            with _timer('update_critic', timing_raw):
                critic_output = self.critic_wg.update_critic(batch)
                metrics.update(reduce_metrics(critic_output.meta_info['metrics']))
        
        if self.config.trainer.critic_warmup <= self.global_steps:
            with _timer('update_actor', timing_raw):
                if 'temperature' not in batch.meta_info:
                    print(f'temperature not in bacth.meta_info, set {temperature=}')
                    batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature

                actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_metrics = actor_output.meta_info['metrics']
                metrics.update(reduce_metrics(actor_metrics))
        
        # Log rollout generations if enabled
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with _timer("dump_rollout_generations", timing_raw):
                # print(batch.batch.keys())
                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                extra_info_dict = {
                    "self_verification": batch.non_tensor_batch.get("llm_judge",[]),
                    "data_source": batch.non_tensor_batch.get("data_source",[]),
                    "uid": batch.non_tensor_batch.get("uid",[]),
                    "role": batch.non_tensor_batch.get("role",[]),
                    "advantage": [item[0] for item in batch.batch["advantages"].cpu().tolist()],
                    "entropy": batch.batch["entropys"].cpu().tolist(),
                    "score": batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                    "rule_score": batch.non_tensor_batch.get("rule_based_reward",[]),
                    "llm_score": batch.non_tensor_batch.get("llm_reward",[]),
                    "pred": batch.non_tensor_batch.get("pred",[]),
                    "reward_model": batch.non_tensor_batch.get("reward_model",[])
                }           
                extra_infos = batch.non_tensor_batch.get('extra_info',[]) # list of dict
                if len(extra_infos) > 0:
                    all_keys = extra_infos[0].keys()
                    if 'question_id' in all_keys:
                        extra_info_dict.update({"question_id": [item['question_id'] for item in extra_infos]})
                    if 'evidence' in all_keys:
                        extra_info_dict.update({"evidence": [item['evidence'] for item in extra_infos]})
                # print("Sentence level advantages in this batch:", sentence_advantages)
                # the uid will be the same for questioner and responder
                self._dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    extra_infos_dict=extra_info_dict,
                    dump_path=rollout_data_dir
                )


        # Runs validation and saves checkpoints based on frequency settings.
        is_last_step = self.global_steps >= self.total_training_steps
        if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
            (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
            with _timer('testing', timing_raw):
                val_metrics: dict = self._validate()
                if is_last_step:
                    pprint(f'Final validation metrics: {val_metrics}')
            metrics.update(val_metrics)

        if self.config.trainer.save_freq > 0 and ( is_last_step or \
                self.global_steps % self.config.trainer.save_freq == 0):
            with _timer('save_checkpoint', timing_raw):
                self._save_checkpoint()


        return batch, metrics

    def fit(self):
        """The main training loop for the SPELL trainer.
        """
        from verl.utils.tracking import Tracking
        
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                               experiment_name=self.config.trainer.experiment_name,
                               default_backend=self.config.trainer.logger,
                               config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = self._load_checkpoint() or 0
        self.accumulator = BatchAccumulator(self.config)
        
        if self.val_reward_fn and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False): return

        # log the initial domain weights
        if self.config.algorithm.domain_sampling.enable:
            self.latest_domain_weights = self.sampler.domain_weights
            train_batch_domain_weights = {f"train/domain_weights/{k}": v for k, v in self.latest_domain_weights.items()}
            self.logger.log(data=train_batch_domain_weights, step=self.global_steps)

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        
        for epoch in range(self.config.trainer.total_epochs):
            # for batch_dict in self.train_dataloader:
            for batch_dict in self.train_dataloader:
                questioner_batch = DataProto.from_single_dict(batch_dict)    
                timing_raw = defaultdict(float)
                metrics = {}
                
                with _timer('step', timing_raw):
                    # --- Phase 1: Questioner Rollout ---
                    with _timer('questioner_gen', timing_raw):
                        questioner_rollout = self._questioner_rollout(questioner_batch)
                    if len(questioner_rollout) == 0: continue

                    # --- Phase 2: Responder Pipeline ---
                    # This include gen without text to filter and second round of rollou
                    with _timer('responder_rollout', timing_raw):
                        responder_experience, _ = self._responder_pipeline(questioner_rollout, metrics, timing_raw)
                    
                    # --- Phase 3: Role-Specific Dynamic Sampling ---
                    print("responder rollout end")
                    if self.update_questioner:
                        print("Update questioner, accumulating responder and questioner batch")
                        self.accumulator.accumulate(responder_experience, questioner_rollout, {})
                    else:
                        print("Don't update questioner, accumulating responder batch")
                        self.accumulator.accumulate(responder_experience, None, {})
                    print("accumulating batches")
                    
                    if not self.accumulator.is_ready():
                        continue

                    final_batch = self.accumulator.get_final_batch()

                    # --- Phase 4: Unified Policy Update ---
                    final_batch, ppo_metrics = self._perform_ppo_update(final_batch, timing_raw)
                    metrics.update(ppo_metrics)

                metrics.update(compute_spell_data_metrics(final_batch, self.use_critic, self.update_questioner))
                metrics.update(compute_timing_metrics(final_batch, timing_raw))
                metrics["train/num_gen_batches"] = self.accumulator.num_gen_batches
                self.logger.log(data=metrics, step=self.global_steps)

                # Reset for the next PPO step
                self.accumulator.reset()

                if self.global_steps >= self.total_training_steps:
                    progress_bar.close()
                    return
                
                progress_bar.update(1)
                self.global_steps += 1
