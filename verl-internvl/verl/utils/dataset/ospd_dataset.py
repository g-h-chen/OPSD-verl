#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import re
from typing import List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class OSPDRLHFVDataset(RLHFDataset):
    """
    Dataset for OPSD-style training.

    - Student prompt: image + question
    - Teacher prompt: image + question + hint (caption)
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        **kwargs,
    ):
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor, **kwargs)
        self.hint_key = config.get("hint_key", "captions")
        self.hint_index = config.get("hint_index", 0)
        self.hint_role = config.get("hint_role", "system")
        self.hint_prefix = config.get("hint_prefix", "Hint: ")
        if "max_samples" in config and config.max_samples > 0:
            self.dataframe = self.dataframe.select(range(config.max_samples))
            print(f"Using the first {config.max_samples} samples")

    def _normalize_messages(self, messages: list, example: dict) -> list:
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
                        # Strip ONE leading '\n' if the previous item is an
                        # image/video token, because the chat template already
                        # emits '<image>\n' / '<video>\n'.  Without this the
                        # newline is doubled compared to the plain-string path
                        # used during rejection sampling.
                        if content_list and content_list[-1].get("type") in ("image", "video"):
                            if segment.startswith("\n"):
                                segment = segment[1:]
                        if segment:  # skip empty text after stripping
                            content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
        return messages

    def _get_hint(self, example: dict) -> str:
        hint_val = None
        if self.hint_key in example:
            hint_val = example[self.hint_key]
        else:
            extra_info = example.get("extra_info", {})
            if isinstance(extra_info, dict) and self.hint_key in extra_info:
                hint_val = extra_info[self.hint_key]
        if isinstance(hint_val, list):
            if 0 <= self.hint_index < len(hint_val):
                hint_val = hint_val[self.hint_index]
            else:
                hint_val = ""
        if hint_val is None:
            return ""
        return str(hint_val)

    def _content_to_text(self, content) -> str:
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "image":
                    parts.append("<image>")
                elif item_type == "video":
                    parts.append("<video>")
                elif item_type == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        if content is None:
            return ""
        return str(content)

    def _build_messages(self, example: dict, add_hint: bool = False) -> list:
        base_messages = copy.deepcopy(example[self.prompt_key])
        if not add_hint:
            return self._normalize_messages(base_messages, example)

        system_messages = [m for m in base_messages if m.get("role") == "system"]
        user_message = None
        for msg in base_messages:
            if msg.get("role") == "user":
                user_message = msg
                break
        if user_message is None:
            user_message = {"content": example.get("question", ""), "role": "user"}

        question_text = self._content_to_text(user_message.get("content"))
        hint = self._get_hint(example)
        # teacher_user_content = f"This is the caption of the image: {caption}\n{question_text}"
        image_count = question_text.count("<image>")
        # teacher_user_content = f"{image_count*'<image>'}\nThis is the caption of the image: {caption}\n{question_text.replace('<image>', '')}"
        teacher_user_content = f"{image_count*'<image>'}{self.hint_prefix}{hint}\n{question_text.replace('<image>', '')}"

        messages = []
        messages.extend(system_messages)
        messages.append({"role": "user", "content": teacher_user_content})
        return self._normalize_messages(messages, example)

    def _encode_messages(self, messages: list, row_dict: dict):
        if self.processor is not None:
            row_dict, model_inputs, input_ids, attention_mask, raw_prompt = self.preprocessor(messages, row_dict)
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
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        return row_dict, input_ids[0], attention_mask[0], position_ids[0], raw_prompt

    def __getitem__(self, item):
        base_row_dict: dict = copy.deepcopy(self.dataframe[item])

        student_row = copy.deepcopy(base_row_dict)
        teacher_row = copy.deepcopy(base_row_dict)

        student_messages = self._build_messages(student_row, add_hint=False)
        teacher_messages = self._build_messages(teacher_row, add_hint=True)

        student_row, student_input_ids, student_attention_mask, student_position_ids, student_raw_prompt = (
            self._encode_messages(student_messages, student_row)
        )
        teacher_row, teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_raw_prompt = (
            self._encode_messages(teacher_messages, teacher_row)
        )

        ######## 
        # breakpoint()
        # print(f'student_messages: {student_messages}\n\n\n')
        # print(f'teacher_messages: {teacher_messages}\n\n\n')
        # print('--------------------------------')

        row_dict = student_row
        row_dict.pop(self.prompt_key, None)

        row_dict["input_ids"] = student_input_ids
        row_dict["attention_mask"] = student_attention_mask
        row_dict["position_ids"] = student_position_ids
        row_dict["teacher_input_ids"] = teacher_input_ids
        row_dict["teacher_attention_mask"] = teacher_attention_mask
        row_dict["teacher_position_ids"] = teacher_position_ids

        raw_prompt_ids = self.tokenizer.encode(student_raw_prompt, add_special_tokens=False)
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
        row_dict["teacher_raw_prompt_ids"] = self.tokenizer.encode(teacher_raw_prompt, add_special_tokens=False)

        if self.return_raw_chat:
            row_dict["raw_prompt"] = student_messages

        if self.return_full_prompt:
            row_dict["full_prompts"] = student_raw_prompt

        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict.get("data_source"))
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

