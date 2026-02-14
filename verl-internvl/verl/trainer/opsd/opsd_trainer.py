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

import os
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_timing_metrics
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask


class RayOPSDTrainer(RayPPOTrainer):
    def _link_last_step_huggingface(self):
        local_root_folder = os.path.abspath(self.config.trainer.default_local_dir)
        local_global_step_folder = os.path.join(local_root_folder, f"global_step_{self.global_steps}")
        huggingface_dir = os.path.join(local_global_step_folder, "actor", "huggingface")

        if not os.path.isdir(huggingface_dir):
            return

        for entry in os.listdir(huggingface_dir):
            source_path = os.path.abspath(os.path.join(huggingface_dir, entry))
            target_path = os.path.join(local_root_folder, entry)

            if os.path.lexists(target_path):
                if os.path.islink(target_path):
                    os.unlink(target_path)
                else:
                    # Avoid clobbering real files or directories.
                    continue

            os.symlink(source_path, target_path)

    def _build_teacher_full_inputs(self, batch: DataProto) -> None:
        teacher_prompt_ids = batch.batch["teacher_input_ids"]
        teacher_prompt_mask = batch.batch["teacher_attention_mask"]
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]

        teacher_input_ids = torch.cat([teacher_prompt_ids, responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        batch.batch["teacher_input_ids"] = teacher_input_ids
        batch.batch["teacher_attention_mask"] = teacher_attention_mask
        batch.batch["teacher_position_ids"] = teacher_position_ids

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.debug import marked_timer
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                if isinstance(batch_dict, (tuple, list)) and len(batch_dict) == 2:
                    batch, gen_batch = batch_dict
                    if not isinstance(batch, DataProto):
                        batch = DataProto.from_single_dict(batch)
                    if not isinstance(gen_batch, DataProto):
                        gen_batch = DataProto.from_single_dict(gen_batch)
                else:
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "teacher_raw_prompt_ids"]
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
                
                if self.config.actor_rollout_ref.rollout.get('do_sample', True) == False:
                    gen_batch.meta_info["do_sample"] = False

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    reward_extra_infos_dict = {}
                    if self.reward_fn is not None:
                        with marked_timer("reward", timing_raw, color="yellow"):
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        sequence_reward = reward_tensor.sum(-1)
                        metrics.update(
                            {
                                "reward/train_mean": sequence_reward.mean().detach().item(),
                                "reward/train_max": sequence_reward.max().detach().item(),
                                "reward/train_min": sequence_reward.min().detach().item(),
                            }
                        )

                    self._build_teacher_full_inputs(batch)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                    with marked_timer("update_actor_opsd", timing_raw, color="red"):
                        actor_output = self.actor_rollout_wg.update_actor_opsd(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    # # dbg
                    # print(f'"actor/opsd_kl_loss": {actor_output_metrics["actor/opsd_kl_loss"]}')
                    # print(f'"actor/opsd_kl_mean": {actor_output_metrics["actor/opsd_kl_mean"]}')
                    # print(f'"actor/opsd_kl_max": {actor_output_metrics["actor/opsd_kl_max"]}')
                    # print(f'"actor/opsd_kl_min": {actor_output_metrics["actor/opsd_kl_min"]}')
                    # print(f'"actor/opsd_entropy_mean": {actor_output_metrics["actor/opsd_entropy_mean"]}')
                    # print(f'"actor/opsd_entropy_max": {actor_output_metrics["actor/opsd_entropy_max"]}')
                    # print(f'"actor/opsd_entropy_min": {actor_output_metrics["actor/opsd_entropy_min"]}')
                    # print(f'"actor/opsd_entropy_loss": {actor_output_metrics["actor/opsd_entropy_loss"]}')
                    metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=False)
                            inputs = [query.replace("<IMG_CONTEXT>", "") for query in inputs]
                            inputs = [query.replace("<|endoftext|>", "") for query in inputs]
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=False)
                            outputs = [response.replace("<|endoftext|>", "") for response in outputs]
                            scores = None
                            if self.reward_fn is not None:
                                scores = reward_tensor.sum(-1).cpu().tolist()
                            else:
                                scores = [0.0] * len(outputs)
                            extra_infos = batch.non_tensor_batch.get("extra_info", [])
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                extra_infos=extra_infos,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                            if is_last_step:
                                self._link_last_step_huggingface()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_timing_metrics(batch, timing_raw))
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if self.global_steps > self.total_training_steps:
                    return

