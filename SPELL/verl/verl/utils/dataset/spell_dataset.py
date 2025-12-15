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

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict, deque, Counter

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from copy import deepcopy

from verl.utils.dataset.rl_dataset import *
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.prompts import *
import verl.utils.torch_functional as verl_F
import regex as re
import json
import ast
import random
random.seed(42)

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
        final_answer = solution_str.split("</think>")[-1].strip()

    return final_answer

def get_record(pred_str: str) -> dict:
    def clean_string(s):
        return ''.join(c for c in s if ord(c) >= 32 or c in '\n\r\t')

    pred_str = extract_solution(pred_str)
    if pred_str is None :
        return {}
    if pred_str == "Empty":
        return {"Empty": "Empty"}

    # Ensure pred_str is a string before cleaning
    if not isinstance(pred_str, str):
        return {}
    pred_str = clean_string(pred_str)

    # extract Markdown JSON block
    json_blocks = re.findall(r'```json(.*?)```', pred_str, re.DOTALL)
    # print(json_blocks[-1])
    if json_blocks:
        # get the last match
        for block in reversed(json_blocks):
            try:
                result = json.loads(block.strip())
                return result if isinstance(result, dict) else {}
            except json.JSONDecodeError:
                continue

    try:
        result = json.loads(pred_str)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    return {}


class DomainWeightedSPELLDataset(RLHFDataset):
    """
    A dataset that loads RLHF data from multiple domains with configurable sampling weights.
    """

    def __init__(
        self,
        # parquet_files: Dict[str, Union[str, List[str]]],
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key="prompt",
        image_key="images",
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        return_raw_chat=False,
        truncation="error",
        filter_overlong_prompts=False,
        n_docs=None, 
        tasks=None, 
        domain_key: str = "data_source",  
        use_cache=False, 
        cache_queue_size: int = 3, 
    ):
        """
        Initialize a domain-weighted RLHF dataset.

        Args:
            parquet_files: Dictionary of to parquet file paths
            tokenizer: Tokenizer for processing text
            processor: Optional processor for multi-modal data
            prompt_key: Key for prompts in the parquet files
            image_key: Key for images in the parquet files
            max_prompt_length: Maximum length for prompts
            filter_prompts: Whether to filter prompts
            cache_dir: Directory for caching
            chat_template_func: Function for chat templates
            return_raw_chat: Whether to return raw chat
            truncation: How to handle truncation
            filter_overlong_prompts: Whether to filter overlong prompts
            n_docs: a list fo number m to sample from all n docs
            tasks: task list for questioner
            domain_key: key for task mapping
            use_cache: Whether to use caching
            cache_queue_size: Size of the cache queue
        """
        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=prompt_key,
            image_key=image_key,
            max_prompt_length=max_prompt_length,
            filter_prompts=filter_prompts,
            cache_dir=cache_dir,
            chat_template_func=chat_template_func,
            return_raw_chat=return_raw_chat,
            truncation=truncation,
            filter_overlong_prompts=filter_overlong_prompts
        )
        self.n_docs = n_docs
        self.tasks = tasks
        print("DomainWeightedSPELLDataset initialized. Prompts will be generated dynamically in __getitem__.")
        self.domain_key = domain_key
        if isinstance(tasks, list):
            self.domains = sorted(tasks)
        else:
            all_tasks = []
            for source_totask in tasks.values():
                all_tasks.extend(source_totask)
            self.domains = sorted(list(set(all_tasks)))
        self.domain_dataframes = {}
        self.index_mapping = []
        self.total_size = 0

        # cache index：（question_id, task)
        if use_cache:
            self.cache = defaultdict(lambda: deque(maxlen=cache_queue_size))
            self.cache_queue_size = cache_queue_size
            print(f"Dataset initialized with cache support. Queue size k={self.cache_queue_size}")
        else:
            # set to empty, cache the idx
            self.cache = {}

        self._initialize_domains()
    def _initialize_domains(self):
        if self.domain_key not in self.dataframe.columns:
            raise ValueError(f"Domain key '{self.domain_key}' not found in the dataframe columns: {self.dataframe.columns.tolist()}")

        # chunk the dataset to different domains
        source_usage_counts = defaultdict(int)
        for data_source, tasks in self.tasks.items():
            source_usage_counts[data_source] += len(tasks)

        split_source_data = {} 
        for data_source, usage_count in source_usage_counts.items():
            source_df = self.dataframe[self.dataframe[self.domain_key].str.startswith(data_source)].copy()
            assigned_tasks = self.tasks[data_source]

            if usage_count > 1:
                print(f"Data source '{data_source}' is used by {usage_count} tasks. Splitting data...")
                split_dfs = np.array_split(source_df.sample(frac=1).reset_index(drop=True), usage_count) 
                for i, task in enumerate(assigned_tasks):
                    split_source_data[(data_source, task)] = split_dfs[i]
            else:
                if assigned_tasks:
                    task = assigned_tasks[0]
                    split_source_data[(data_source, task)] = source_df

        self.domain_dataframes = {}
        
        task_to_keys = defaultdict(list)
        for data_source, tasks in self.tasks.items():
            for task in tasks:
                task_to_keys[task].append((data_source, task))

        for task, keys in task_to_keys.items():
            data_to_concat = []
            for key in keys:
                if key in split_source_data:
                    data_to_concat.append(split_source_data[key])
            
            if data_to_concat:
                self.domain_dataframes[task] = pd.concat(data_to_concat, ignore_index=True)
            else:
                self.domain_dataframes[task] = pd.DataFrame(columns=self.dataframe.columns)
        
        self.index_mapping = []
        for domain, df in self.domain_dataframes.items():
            for i in range(len(df)):
                self.index_mapping.append((domain, i))

        self.total_size = len(self.index_mapping)
        
        print("\n" + "="*50)
        print("Data Domain Initialization Finished")
        print("="*50)
        print(f"Total unique tasks initialized: {len(self.domain_dataframes)}")
        for domain, df in self.domain_dataframes.items():
            if not df.empty:
                print(f"  - Task '{domain}': {len(df)} samples")
            else:
                print(f"  - Task '{domain}': 0 samples (no data assigned or source was empty)")

    def _initialize_domains_v1(self):
        if self.domain_key not in self.dataframe.columns:
            raise ValueError(f"Domain key '{self.domain_key}' not found in the dataframe columns: {self.dataframe.columns.tolist()}")

        self.domain_dataframes = {}
        task_to_data_sources = defaultdict(list) 
        for data_source, tasks in self.tasks.items():
            for task in tasks:
                task_to_data_sources[task].append(data_source)

        for task, sources in task_to_data_sources.items():
            combined_data = pd.concat([
                self.dataframe[self.dataframe[self.domain_key].str.startswith(src)]
                for src in sources
            ], ignore_index=True)
            self.domain_dataframes[task] = deepcopy(combined_data)

        self.index_mapping = []
        for domain, df in self.domain_dataframes.items():
            for i in range(len(df)):
                self.index_mapping.append((domain, i))

        self.total_size = len(self.index_mapping)
        print(f"Initialized {len(self.domains)} domains: {self.domains}")
        for domain, df in self.domain_dataframes.items():
            print(f"  - Domain '{domain}': {len(df)} samples")

    def get_domain_size(self, domain: str) -> int:
        """Return the size of a specific domain."""
        if domain not in self.domains:
            raise ValueError(f"Domain '{domain}' not found in dataset.")
        return len(self.domain_dataframes[domain])

    def __len__(self):
        """Return the total number of samples across all domains."""
        return self.total_size

    def __getitem__(self, idx: int):
        domain, domain_idx = self.index_mapping[idx]
        row_dict = self.domain_dataframes[domain].iloc[domain_idx].to_dict()
        
        all_paragraphs = row_dict['paragraphs']
        question_id = row_dict['question_id']
        data_source_key = row_dict['data_source'].split('_')[0] # e.g., 'docmath' from 'docmath_'

        task = domain
        row_dict['data_source'] = task
        
        prompt_content = ""
        evidence_idxs = []

        history = self.cache.get(f"{task}_{question_id}", [])

        # construct prommpt with history
        if history:
            print(f"Cache hit for task: {task} question_id: {question_id}. Found {len(history)} entries.")
            history_template = QUESTIONER_PROMPT_WITH_HISTORY_MAPPING.get(task, QUESTIONER_PROMPT_WITH_HISTORY_MAPPING["default"])
            num_all_paragraphs = len(all_paragraphs)
            
            historical_p_idxs = sorted({idx for item in history for idx in item[0]})
            historical_set = set(historical_p_idxs)
            left_idx = [i for i in range(num_all_paragraphs) if i not in historical_set]
            left_paragraphs = len(left_idx)
            n_list = self.n_docs.get(data_source_key, [3]) 
            n = random.choice(n_list)
            n = min(n, left_paragraphs) 
            sampled_idxs = sorted(random.sample(left_idx, n))
            historical_p_idxs.extend(sampled_idxs)
            historical_p_idxs.sort()

            evidence_idxs = historical_p_idxs
            context_str = '\n'.join([all_paragraphs[i] for i in historical_p_idxs])

            examples_str = []
            for i, item in enumerate(history):
                # item is a tuple: (paragraph_idx, question, answer, score)
                q = item[1]
                a = item[2]
                examples_str.append(f"### Example {i+1}:\nQuestion: {q}\nAnswer: {a}\n")
            examples_str = "\n".join(examples_str)

            prompt_content = history_template.replace("{context}",context_str).replace("{examples}",examples_str)
            row_dict['extra_info']['from_cache'] = True
        # construct prompt without history
        else:
            template = QUESTIONER_PROMPT_MAPPING.get(task, QUESTIONER_PROMPT_MAPPING["default"])
            num_all_paragraphs = len(all_paragraphs)
            
            n_list = self.n_docs.get(data_source_key, [3])
            n = random.choice(n_list)
            n = min(n, num_all_paragraphs) 
            
            sampled_idxs = sorted(random.sample(range(num_all_paragraphs), n))
            evidence_idxs = sampled_idxs
            context_str = '\n'.join([all_paragraphs[i] for i in sampled_idxs])
            
            prompt_content = template.replace("{context}", context_str)
            row_dict['extra_info']['from_cache'] = False

        # specify a target choice to balance the choice distribution
        if task == "docmath_mc":
            prompt_content = prompt_content.replace("<correct_answer>", random.choice(["A", "B", "C", "D"]))

        row_dict['extra_info']['evidence'] = evidence_idxs
        row_dict['extra_info']['task'] = task
        row_dict['extra_info']['question_id'] = question_id
        row_dict['extra_info']['paragraphs'] =  all_paragraphs

        chat = [{"role": "user", "content": prompt_content}]
        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)


        raw_prompt = prompt_with_chat_template

        # Tokenize
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)
            
        final_item = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            'extra_info': row_dict['extra_info'],
            'index': row_dict.get("extra_info", {}).get("index", 0), 
            'data_source': row_dict['data_source']
        }

        final_item["raw_prompt"] = chat 
        
        final_item['question_id'] = question_id

        return final_item

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "domain_dataframes" in state:
                del state["domain_dataframes"]
            return state
        return self.__dict__.copy()


class DomainSampler:
    """A batch sampler that ensures each batch has the correct domain proportions."""

    def __init__(
        self,
        dataset: DomainWeightedRLHFDataset,
        batch_size: int,
        domain_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize a domain sampler.

        Args:
            dataset: The dataset to sample from
            batch_size: Size of each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.domain_weights = domain_weights
        self.domains = list(domain_weights.keys())

        # Create domain indices mapping
        self.domain_indices = {domain: [] for domain in self.domains}
        for i, (domain, _) in enumerate(dataset.index_mapping):
            self.domain_indices[domain].append(i)

        # For each domain, create a shuffled list of indices
        self.domain_iterators = {domain: [] for domain in self.domains}
        for domain in self.domains:
            self._refill_domain_indices(domain)

        self.count_weight()

    def domain_weights(self) -> Dict[str, float]:
        """Return the current domain weights."""
        return self.domain_weights
    def count_weight(self):
        self.domain_counts = {}
        remaining = self.batch_size
        for domain, weight in self.domain_weights.items():
            count = int(self.batch_size * weight)
            if count > len(self.domain_indices[domain]):
                count = len(self.domain_indices[domain])
                print(f"Warning: domain {domain} doesn't have enough data points to take.")
            self.domain_counts[domain] = count
            remaining -= count
        sorted_domains = sorted(
            self.domains, key=lambda d: self.domain_weights[d], reverse=True
        )
        while remaining > 0:
            for domain in sorted_domains:
                if remaining > 0:
                    if len(self.domain_indices[domain]) > self.domain_counts[domain]:
                        self.domain_counts[domain] += 1
                        remaining -= 1
                else:
                    break

    def update_weights(self, weights: Optional[Dict[str, float]]=None) -> None:
        """Update the domain weights."""
        if weights is None:
            return
        self.domain_weights = weights
        self.domains = list(weights.keys())
        self.count_weight()

    def _refill_domain_indices(self, domain: str) -> None:
        """Refill indices for a specific domain."""
        indices = self.domain_indices[domain].copy()
        random.shuffle(indices)
        self.domain_iterators[domain] = indices

    def __len__(self):
        """Return the total number of batches."""
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        """Yield batches of indices that respect domain weights."""
        while True:
            batch_indices = []

            # For each domain, select the required number of indices
            for domain, count in self.domain_counts.items():
                # Skip if the domain has no data
                if not self.domain_indices[domain]:
                    continue

                # Ensure we have enough indices
                if len(self.domain_iterators[domain]) < count:
                    self._refill_domain_indices(domain)

                # Get at most count indices (in case domain has fewer samples than needed)
                to_take = min(count, len(self.domain_iterators[domain]))
                domain_batch_indices = self.domain_iterators[domain][:to_take]
                self.domain_iterators[domain] = self.domain_iterators[domain][to_take:]
                batch_indices.extend(domain_batch_indices)

            # Shuffle the batch indices
            random.shuffle(batch_indices)
            yield batch_indices

