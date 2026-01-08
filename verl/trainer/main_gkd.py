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
Main entry point for GKD (Generalized Knowledge Distillation) training.

This module provides the main function to run GKD training, which combines
online knowledge distillation from a teacher model with optional RL rewards.
"""

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunner as PPOTaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_reference_policy, need_critic
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


@hydra.main(config_path="config", config_name="gkd_trainer", version_base=None)
def main(config):
    """Main entry point for GKD training with Hydra configuration management.
    
    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set config.trainer.device for NPU
    auto_set_device(config)
    
    run_gkd(config)


def run_gkd(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed GKD training.
    
    Args:
        config: Training configuration object containing all necessary parameters
                for distributed GKD training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: Optional custom TaskRunner class.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
        
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    
    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(GKDTaskRunner)
    
    # Create a remote instance of the GKDTaskRunner class
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))
    
    # Optional: save timeline trace file
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class GKDTaskRunner(PPOTaskRunner):
    """Ray remote class for executing distributed GKD training tasks.
    
    This class extends PPOTaskRunner to use RayGKDTrainer instead of RayPPOTrainer.
    It reuses all the worker setup and dataset preparation logic from PPO.
    """
    
    def run(self, config):
        """Execute the main GKD training workflow.
        
        This method follows the PPO workflow but replaces the trainer with RayGKDTrainer.
        
        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the GKD training process.
        """
        import os
        import socket
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        
        print(f"GKDTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        
        # Setup workers (reuse PPO logic)
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        
        # Validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )
        
        # Download checkpoint from HDFS to local
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, 
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        
        # Instantiate tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer
        
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # Load reward manager
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        
        # Initialize resource pool manager
        resource_pool_manager = self.init_resource_pool_mgr(config)
        
        from verl.utils.dataset.rl_dataset import collate_fn
        
        # Create datasets
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        
        # Initialize GKD trainer (instead of PPO trainer)
        from verl.trainer.gkd import RayGKDTrainer
        
        trainer = RayGKDTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        
        # Initialize workers
        trainer.init_workers()
        
        # Start training (uses PPO's fit() method, with GKD's _update_actor override)
        trainer.fit()


if __name__ == "__main__":
    main()
