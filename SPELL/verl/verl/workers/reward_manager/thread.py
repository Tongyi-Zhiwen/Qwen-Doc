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
import asyncio
import json
import logging
import time
import heapq
from collections import defaultdict
# this file is from rllm
from concurrent.futures import ProcessPoolExecutor

import ray
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@ray.remote
class RewardActor:
    def __init__(self, preload_test_file) -> None:
        import os
        logger.setLevel('INFO') # just need this for RAY
        # FIXME: use ray info to get the concurrency, or let it decided by RewardManager
        self.concurrency = os.cpu_count() // 2
        self.executor = ProcessPoolExecutor(max_workers=self.concurrency)
        logger.info(f"concurrency: {self.concurrency}")
        self.tests = {}
        self.preload_test_file = preload_test_file
        if self.preload_test_file:
            assert self.preload_test_file.endswith(".jsonl"), "only support jsonl fornow"
            with open(self.preload_test_file) as f:
                for line in f:
                    data = json.loads(line)
                    self.tests.update(data)
            logger.info(f"{len(self.tests)} tests preloaded")
        else:
            logger.info(f"no tests preloaded")
        self.initialized = True

    async def reward(self, i, method, preload_test_key=None, **kwargs):
        if self.tests and preload_test_key:
            # ground_truth will be a uid when preload_test_file is enabled
            kwargs[preload_test_key] = self.tests[kwargs[preload_test_key]]
        fut = self.executor.submit(method, **kwargs)
        return i, await asyncio.wrap_future(fut)
    
    def batch_reward(self, method, args: list[tuple[int, dict]], **const):
        logger.info(f"actor enter") # monitor TTFT
        futures = []
        for i, kwargs in args:
            all_kwargs = {**const, **kwargs}
            preload_test_key = all_kwargs.pop("preload_test_key", None)
            if self.tests and preload_test_key:
                # ground_truth will be a uid when preload_test_file is enabled
                all_kwargs[preload_test_key] = self.tests[kwargs[preload_test_key]]
            fut = self.executor.submit(method, **all_kwargs)
            futures.append((i,fut))
        for i in range(len(futures)):
            idx, fut = futures[i]
            futures[i] = (idx, fut.result())
        return futures
        
    
    async def prepared(self):
        return self.initialized
    
    def __del__(self):
        self.executor.shutdown(wait=False)

class RewardActorPool:
    def __init__(self, create_if_not_exists=True, preload_test_file=None):
        """
        Create actors on ray for reward computation.
        """
        actors = []
        nodes = ray.nodes()
        from ray.util.scheduling_strategies import \
                NodeAffinitySchedulingStrategy
        for i, node in enumerate(nodes):
            if not node.get("Alive", False):
                continue
            resources = node.get("Resources", {})
            if "GPU" in resources: 
                logger.info(f"node {i} {resources=}")
            else:
                continue
            try:
                actor = ray.get_actor(f"RewardActor_{i}")
                actors.append(actor)
            except ValueError as e:
                if not create_if_not_exists:
                    raise e
                actors.append(RewardActor.options(
                        name=f"RewardActor_{i}", 
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=node["NodeID"],
                            soft=False,
                        ),).remote(
                            preload_test_file=preload_test_file
                    ))
        # actor lazily created by default, so make sure that they are ready
        ray.get([actor.prepared.remote() for actor in actors])
        logger.info("App created")
        self.actors = actors
        assert len(self.actors) > 0, "No actors created"
        
        # Least requests load balancing
        self.weighted_actors = [[0, i, actor] for i, actor in enumerate(actors)]
        heapq.heapify(self.weighted_actors)

    async def submit(self, method, args, **const):
        """
        method: a picklable function
        args: a tuple of (i, kwargs), i is request_id, kwargs is the arguments
        const: a dict of constants that will be passed to the method along with the args
        e.g. pass preload_test_key="ground_truth" so that every call to `actor.reward`
             will get
        """
        actor = self.weighted_actors[0][-1]
        self.weighted_actors[0][0] += 1
        heapq.heapreplace(self.weighted_actors, self.weighted_actors[0])
        return await actor.reward.remote(args[0], method, **const, **args[1])

    def map(self, method, args, **const):
        """
        method: a picklable function
        args: a list of (i, kwargs), i is request_id, kwargs is the arguments
        const: a dict of constants that will be passed to the method along with the args
        e.g. pass preload_test_key="ground_truth" so that every call to `actor.reward` 
             will get the same preload_test_key.
        """
        # average split, 10 // 4 -> 3,3,2,2 instead of 3,3,3,1
        import numpy as np
        indicies = np.array_split(np.arange(len(args)), len(self.actors))
        ray_futs = []
        results = []
        for actor_id, indices in enumerate(indicies):
            logger.debug(f"actor {actor_id}, sample len={len(indices)}")
            ray_futs.append(
                self.actors[actor_id].batch_reward.remote(method, args[indices[0]:indices[-1]+1], **const)
            )
        for fut in ray_futs:
            results.extend(ray.get(fut))
        return results

@register("thread")
class ThreadRewardManager:
    """The reward manager.
    """
    actors = None
    def __init__(
            self, 
            tokenizer, 
            num_examine, 
            compute_score=None,
            reward_fn_key="data_source",
            max_resp_len=None,
            overlong_buffer_cfg=None,
            preload_test_file=None,
            **reward_kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_kwargs = reward_kwargs
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

        from ray import cloudpickle

        cloudpickle.dumps_debug(self.compute_score) # raise error if it is not picklable
        cloudpickle.dumps_debug(self.reward_kwargs)

        self.actor_pool = RewardActorPool(
            create_if_not_exists=True,
            preload_test_file=preload_test_file
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        logger.info(f"called {len(data)}")
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        valid_response_lengths = []
        sequences_strs = []
        prompt_strs = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # NOTE: we assume the prompt is not needed, please make sure that is true!
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_lengths.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            sequences_strs.append(sequences_str)
            prompt_strs.append(prompt_str)
        # Process items in parallel using ThreadPoolExecutor
        args = [(i, 
                 dict(data_source=data[i].non_tensor_batch['data_source'], 
                      ground_truth=data[i].non_tensor_batch['reward_model']['ground_truth'],
                      solution_str=sequences_strs[i], 
                      prompt_str=prompt_strs[i]
                      ))
                 for i in range(len(data))]
        const = dict(**self.reward_kwargs, preload_test_key='ground_truth')
        logger.info("decoded")
        
        start = time.time()
        results = self.actor_pool.map(self.compute_score, args, **const)
        # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # results = list(executor.map(process_row, args))
        reward_cost = time.time() - start
        logger.info(f"{reward_cost=}s")
        reward_extra_info = defaultdict(list)
        
        # Fill reward tensor with results
        for (i, result), valid_response_length in zip(results, valid_response_lengths):
            score: float
            if isinstance(result, dict):
                score = result["score"] # for GRPO, do not want duplicate score/reward
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score
            
            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward
            # reward_tensor[i, valid_response_length - 1] = score
    
        print(sequences_strs[0])
        print(reward_tensor[0].sum())
        info_list = reward_extra_info.pop("code_oj_info", [])
        if info_list:
            info = info_list[0]        
            print(info[0])
            if len(info) > 1:
                print(info[-1])
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor