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
GKD (Generalized Knowledge Distillation) Trainer Module.

This module provides a simplified PPO-style GKD training framework that supports:
- Online knowledge distillation from teacher model
- Rule-based reward combined with distillation loss
- Flexible loss combination (alpha * RL_loss + beta * distill_loss)
"""

from verl.trainer.gkd.ray_trainer import RayGKDTrainer
from verl.trainer.gkd.distill_loss import compute_fsdp_kl_divergence

__all__ = ["RayGKDTrainer", "compute_fsdp_kl_divergence"]
