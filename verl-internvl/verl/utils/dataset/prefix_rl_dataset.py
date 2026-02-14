# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.trainer.ppo.ray_trainer import DataProto


logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}

import uuid

def uuid_str_from_str(s: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    return str(uuid.uuid5(namespace, s))

def repeat_list_of_dicts(data: list[dict], repeat_times: int = 2, interleave: bool = True) -> list[dict]:
    """
    Repeat a list of dicts either interleaved or blockwise, 
    with unique 'traj_id' suffix for each repetition.

    Args:
        data (list[dict]): List of dictionaries to repeat.
        repeat_times (int): Number of times to repeat each element or the whole list.
        interleave (bool): 
            - If True: interleave each element consecutively (a0, a1, b0, b1, ...)
            - If False: repeat the whole list as blocks (a0, b0, a1, b1, ...)

    Returns:
        list[dict]: Repeated list of dicts with modified traj_id.
    """
    old_traj_id_key = 'index'
    new_traj_id_key = 'uid'
    if not data:
        return []

    repeated = []
    if interleave:
        # Interleave: repeat each item consecutively
        for item in data:
            for i in range(repeat_times):
                new_item = copy.deepcopy(item)
                # support both old and new traj_id key
                if new_traj_id_key in new_item['extra_info']:
                    traj_id = new_item['extra_info'][new_traj_id_key]
                elif old_traj_id_key in new_item['extra_info']:
                    traj_id = new_item['extra_info'].pop(old_traj_id_key)
                else:
                    raise ValueError(f"traj_id key {old_traj_id_key} or {new_traj_id_key} not found in extra_info")

                # new_item['extra_info'][new_traj_id_key] = f"{traj_id}_{i}"
                new_item[new_traj_id_key] = f"{traj_id}"
                
                repeated.append(new_item)
    else:
        raise NotImplementedError("Blockwise repetition is not implemented")
        # Blockwise: repeat the entire list
        for i in range(repeat_times):
            for item in data:
                new_item = deepcopy(item)
                if 'traj_id' in new_item['extra_info']:
                    new_item['extra_info']['traj_id'] = f"{new_item['extra_info']['traj_id']}_{i}"
                repeated.append(new_item)

    return repeated


class PrefixRLHFDataset(RLHFDataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training = kwargs.get('training', True)
    def _maybe_add_hint(self, row_dict: dict, idx: int):
        hint_level = -1
        if idx in [0,1,2]:
            hint = row_dict['extra_info']['captions'][idx]
            assistant_prompt = {"content": f"<think>\n{hint}\n", "role": "assistant"}
            hint_level = idx
        else:
            assistant_prompt = {"content": f"", "role": "assistant"}
        row_dict['prompt'].append(assistant_prompt)
        
        return row_dict, hint_level

    def collate_fn(self, batch: list[dict]):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # breakpoint()
        # print(f'in collate_fn')
        # flatten the batch
        batch = [x for sublist in batch for x in sublist]

        tensors = defaultdict(list)
        non_tensors = defaultdict(list)

        for data in batch:
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    tensors[key].append(val)
                else:
                    non_tensors[key].append(val)

        for key, val in tensors.items():
            tensors[key] = torch.stack(val, dim=0)

        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)
        collated_batch = {**tensors, **non_tensors}

        # convert to DataProto
        batch: DataProto = DataProto.from_single_dict(collated_batch)

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        if "input_ids_without_response_prefix" in batch.batch:
            batch_keys_to_pop.append("input_ids_without_response_prefix")
        if "attention_mask_without_response_prefix" in batch.batch:
            batch_keys_to_pop.append("attention_mask_without_response_prefix")
        if "position_ids_without_response_prefix" in batch.batch:
            batch_keys_to_pop.append("position_ids_without_response_prefix")
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "response_prefix" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("response_prefix")
        # non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) 

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        # print(f'batch size: {len(gen_batch.batch["input_ids"])}')
        return batch, gen_batch



    
    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        # we shall build hints and repeat here
        # among 8 rollouts:
        # 5 not hint, 3 with different levels of hints
        if self.training:
            repeat_n = self.config.train_rollout.n
        else:
            repeat_n = self.config.val_rollout.n
        list_row_dict = repeat_list_of_dicts(
            [row_dict], 
            repeat_times=repeat_n,
            interleave=True
        )
        # breakpoint()
        collated_batch = []
        input_length = np.inf

        for idx, row_dict in enumerate(list_row_dict):

            row_dict_with_hint, hint_level = self._maybe_add_hint(row_dict, idx)
            processed_row_dict = self.process_row_dict(row_dict_with_hint, hint_level=hint_level)
            input_length = min(input_length, len(processed_row_dict["raw_prompt_ids"]))
            collated_batch.append(processed_row_dict)

            
        if not self.config.get("update_response_prefix", False):
            return collated_batch
        
        # this should be the same for all samples in the group
        raw_prompt_ids_without_response_prefix = torch.tensor(row_dict["raw_prompt_ids"][:input_length]).unsqueeze(0)
        # row_dict['input_ids_without_response_prefix'], row_dict['attention_mask_without_response_prefix'] = \
        input_ids_without_response_prefix, attention_mask_without_response_prefix = \
        verl_F.postprocess_data(
            input_ids=raw_prompt_ids_without_response_prefix,
            attention_mask=torch.ones_like(raw_prompt_ids_without_response_prefix),
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # update response prefix
        for row_dict in collated_batch:
                row_dict["response_prefix"] = row_dict["raw_prompt_ids"][input_length:]
                
                row_dict['input_ids_without_response_prefix'] = input_ids_without_response_prefix[0]
                row_dict['attention_mask_without_response_prefix'] = attention_mask_without_response_prefix[0]
                row_dict['position_ids_without_response_prefix'] = compute_position_id_with_mask(row_dict['attention_mask_without_response_prefix'])
        return (collated_batch)



    def process_row_dict(self, row_dict: dict, hint_level=-1):
        messages = self._build_messages(row_dict) # sort out image formats
        model_inputs = {}
        row_dict['extra_info']['hint_level'] = hint_level

        if self.processor is not None:
            row_dict, model_inputs, input_ids, attention_mask, raw_prompt = self.preprocessor(messages, row_dict, 
                add_generation_prompt=False, remove_eos_token=True)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")


        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
