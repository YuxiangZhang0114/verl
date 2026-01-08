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
GKD Trainer with Ray-based single controller.

This trainer extends RayPPOTrainer to support online knowledge distillation
from a teacher model, combining RL rewards with distillation loss.
"""

import sys
import time
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType


class RayGKDTrainer(RayPPOTrainer):
    """GKD Trainer that extends PPO training with knowledge distillation.
    
    This trainer adds teacher model knowledge distillation to the standard PPO
    training loop. It supports:
    - Synchronous teacher knowledge retrieval after student rollout
    - Combining RL loss with distillation loss
    - Pure distillation mode (RL loss coefficient = 0)
    - Hybrid engine mode (actor and rollout share GPU pool)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """Initialize GKD trainer with teacher client.
        
        Args:
            config: Configuration containing GKD-specific parameters
            tokenizer: Tokenizer for encoding/decoding
            role_worker_mapping: Mapping from roles to worker classes
            resource_pool_manager: Manager for Ray resource pools
            ray_worker_group_cls: Class for Ray worker groups
            processor: Optional data processor
            reward_fn: Function for computing rewards
            val_reward_fn: Function for computing validation rewards
            train_dataset: Training dataset
            val_dataset: Validation dataset
            collate_fn: Function to collate data samples
            train_sampler: Sampler for training dataset
            device_name: Device name for training
        """
        # Initialize parent PPO trainer
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        
        # Initialize teacher client if enabled
        self.teacher_client = None
        self.use_teacher = config.gkd.get("enable_teacher", True)
        
        if self.use_teacher:
            # Import teacher client from recipe/gkd
            # Direct import works since recipe/gkd is in the project
            from recipe.gkd.teacher.client import TeacherClient
            
            self.teacher_client = TeacherClient(
                server_ip=config.gkd.teacher_ip,
                server_port=config.gkd.teacher_port,
                num_microbatches=config.gkd.get("teacher_num_microbatches", 1),
                max_tokens=config.gkd.get("teacher_max_tokens", 1),
                n_server_workers=config.gkd.get("teacher_n_server_workers", 1),
                temperature=config.gkd.get("teacher_temperature", 1.0),
                only_response=config.gkd.get("teacher_only_response", False),
            )
            
            print(f"[GKD] Teacher client initialized: {config.gkd.teacher_ip}:{config.gkd.teacher_port}")
        else:
            print("[GKD] Teacher distillation disabled")
    
    def _get_teacher_knowledge(self, batch: DataProto) -> DataProto:
        """Synchronously retrieve teacher knowledge for generated sequences.
        
        Args:
            batch: DataProto containing generated sequences
            
        Returns:
            DataProto containing teacher_topk_logps and teacher_topk_indices
        """
        if not self.use_teacher or self.teacher_client is None:
            # Return empty proto if teacher is disabled
            return DataProto(batch={}, non_tensor_batch={})
        
        # Extract input_ids and attention_mask
        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"].to(torch.bool)
        
        # Convert to list of token sequences (removing padding)
        input_ids_list = []
        for ids, mask in zip(input_ids, attention_mask):
            input_ids_list.append(ids[mask].tolist())
        
        # Submit requests to teacher
        batch_size = len(input_ids_list)
        n_workers = self.config.gkd.get("teacher_n_server_workers", 1)
        micro_batch_size = batch_size // n_workers if batch_size >= n_workers else batch_size
        
        futures = []
        for i in range(0, batch_size, micro_batch_size):
            fut = self.teacher_client.submit(input_ids_list[i : i + micro_batch_size])
            futures.append(fut)
        
        # Collect results
        all_teacher_topk_logps = []
        all_teacher_topk_indices = []
        
        for future in futures:
            try:
                responses, teacher_topk_logps, teacher_topk_indices = future.result()
            except Exception as e:
                print(f"[GKD] Teacher request failed: {e}")
                # Return empty result on error
                return DataProto(batch={}, non_tensor_batch={})
            
            all_teacher_topk_logps.extend(teacher_topk_logps)
            all_teacher_topk_indices.extend(teacher_topk_indices)
        
        # Convert to padded tensors
        real_seq_lens = torch.tensor([x.size(0) for x in all_teacher_topk_logps], dtype=torch.int32)
        topk = all_teacher_topk_logps[0].size(-1)
        
        logp_dtype = all_teacher_topk_logps[0].dtype
        idx_dtype = all_teacher_topk_indices[0].dtype
        
        # Create padded tensors matching input_ids shape
        teacher_knowledge_shape = list(input_ids.shape) + [topk]
        teacher_topk_logps_padded = torch.zeros(*teacher_knowledge_shape, dtype=logp_dtype)
        teacher_topk_indices_padded = torch.zeros(*teacher_knowledge_shape, dtype=idx_dtype)
        
        # Fill in valid positions
        for i in range(batch_size):
            teacher_topk_logps_padded[i][attention_mask[i]] = all_teacher_topk_logps[i]
            teacher_topk_indices_padded[i][attention_mask[i]] = all_teacher_topk_indices[i]
        
        # Create output DataProto
        output_batch = DataProto.from_single_dict(
            data={"real_seq_lens": real_seq_lens},
        )
        
        output_batch.non_tensor_batch.update(
            {
                "teacher_topk_logps": teacher_topk_logps_padded.numpy(),
                "teacher_topk_indices": teacher_topk_indices_padded.numpy(),
            }
        )
        
        return output_batch
    
    def _update_actor(self, batch: DataProto) -> DataProto:
        """Update actor with distillation loss.
        
        This method retrieves teacher knowledge before updating the actor,
        then calls the parent class's update_actor method which will handle
        the distillation loss computation in the actor worker.
        
        Args:
            batch: DataProto containing training data
            
        Returns:
            DataProto containing update metrics
        """
        # Get teacher knowledge if enabled
        if self.use_teacher:
            from verl.utils.debug import marked_timer
            
            timing = {}
            with marked_timer("get_teacher_knowledge", timing):
                teacher_output = self._get_teacher_knowledge(batch)
                
                # Only union if teacher returned valid data
                if teacher_output.non_tensor_batch:
                    batch = batch.union(teacher_output)
                    print(f"[GKD] Teacher knowledge added to batch (took {timing['get_teacher_knowledge']:.2f}s)")
                else:
                    print(f"[GKD] WARNING: Teacher knowledge retrieval failed, proceeding without distillation")
        
        # Check if we should use RL loss (for pure distillation mode)
        use_rl_loss = self.config.gkd.get("use_rl_loss", False)
        
        if not use_rl_loss:
            # Pure distillation mode: add dummy advantages and old_log_probs
            # This ensures the actor worker receives the required fields but ignores RL loss
            print("[GKD] Pure distillation mode: skipping RL loss computation")
            
            # Add dummy old_log_probs and advantages (all zeros)
            if "old_log_probs" not in batch.batch:
                batch.batch["old_log_probs"] = torch.zeros_like(
                    batch.batch["attention_mask"], dtype=torch.float32
                )
            if "advantages" not in batch.batch:
                batch.batch["advantages"] = torch.zeros_like(
                    batch.batch["attention_mask"], dtype=torch.float32
                )
            
            # Mark as pure distillation in meta_info
            batch.meta_info["pure_distillation"] = True
            batch.meta_info["use_rl_loss"] = False
        else:
            # RL + distillation mode: ensure advantages are computed by parent trainer
            batch.meta_info["pure_distillation"] = False
            batch.meta_info["use_rl_loss"] = True
        
        # Pass GKD config to actor worker via meta_info
        batch.meta_info["gkd_distill_loss_coef"] = self.config.gkd.get("distill_loss_coef", 1.0)
        batch.meta_info["gkd_distill_loss_type"] = self.config.gkd.get("distill_loss_type", "forward_kl")
        batch.meta_info["gkd_distill_temperature"] = self.config.gkd.get("distill_temperature", 1.0)
        
        # Call parent class's update_actor (which will handle distillation in worker)
        return super()._update_actor(batch)
