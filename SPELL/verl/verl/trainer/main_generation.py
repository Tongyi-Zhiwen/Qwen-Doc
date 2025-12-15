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
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from tqdm import tqdm
@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    if config.data.debug:
        dataset = dataset[:10]
    chat_lst = dataset[config.data.prompt_key].tolist()

    data_sources = dataset['data_source'].tolist()
    reward_functions = dataset['reward_model'].tolist()

    # recompute score, need cpu only 
    if config.data.get("recompute_score", False):
        assert "responses" in dataset.columns, "Responses not in the dataset."
        response_lists = dataset["responses"].tolist()
        from verl.utils.reward_score import default_compute_score

        all_rms = []
        for responses, reward_fn, data_source in tqdm(zip(response_lists, reward_functions, data_sources)):
            cur_rms = []
            gt = reward_fn["ground_truth"]
            # compute all scores
            for response in responses:
                score = default_compute_score(data_source, response, gt)
                cur_rms.append(score["score"])
            all_rms.append(cur_rms)

        all_rms = np.array(all_rms, dtype=object).reshape(-1, config.data.n_samples).tolist()
        mean_rm = np.mean(all_rms, axis=1).tolist()
        dataset["scores"] = all_rms
        dataset["mean_score"] = mean_rm
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(config.data.output_path)
        exit(0)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes, max_colocate_count=6)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(dataset)
    print(f"generating {total_samples} samples")
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = []

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        print(f"DEBUG: Batch {batch_idx+1}, number of prompts in this batch: {len(batch_chat_lst)}")
        all_input_ids, all_attention_mask, all_position_ids = [], [], []
        for i, chat in enumerate(batch_chat_lst):
            inputs = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
            input_ids = inputs["input_ids"]
            all_input_ids.append(input_ids.squeeze(0))
            # print(f"input_ids shape: {input_ids.shape}")
            attention_mask = inputs["attention_mask"]
            all_attention_mask.append(attention_mask.squeeze(0))
            position_ids = compute_position_id_with_mask(attention_mask)
            all_position_ids.append(position_ids.squeeze(0))

        all_input_ids = torch.stack(all_input_ids)
        all_attention_mask = torch.stack(all_attention_mask)
        all_position_ids = torch.stack(all_position_ids)
            
        batch_dict = {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "position_ids": all_position_ids}

        data = DataProto.from_dict(batch_dict)
        print(f"DEBUG: Number of prompts in this batch: {len(data)}")
        data.meta_info['n'] = config.data.n_samples

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        output = wg.generate_sequences(data)

        print(f"DEBUG: Batch {batch_idx+1}, len(output) from unpad_dataproto is: {len(output)}")
        print(f"DEBUG: Expected number of responses for this batch: {len(batch_chat_lst) * config.data.n_samples}")

        for i in range(len(output)):
            data_item = output[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            output_lst.append(response_str)

    # move to offline computation
    if config.data.get("compute_score",False):
        from verl.utils.reward_score import default_compute_score
        all_rms = []
        data = data.repeat(config.data.n_samples, interleave=True)
        for response, reward_fn, data_source in tqdm(zip(output_lst, reward_functions, data_sources)):
            score = default_compute_score(data_source, response, reward_fn["ground_truth"])
            all_rms.append(score["score"])
        all_rms = np.array(all_rms, dtype=object).reshape(-1, config.data.n_samples).tolist()
        mean_rm = np.mean(all_rms, axis=1).tolist()
        dataset["scores"] = all_rms
        dataset["mean_score"] = mean_rm

    print(f"DEBUG: Final length of flat output_lst before reshape: {len(output_lst)}")
    print(f"DEBUG: Expected final length: {total_samples * config.data.n_samples}")
    # convert output_lst from (n_samples * n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object).reshape(-1, config.data.n_samples).tolist()

    # add to the data frame
    dataset["responses"] = output_lst

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
