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
ALFWorld Reward Function for VERL.

This module provides the reward computation for ALFWorld embodied AI tasks.
The primary reward signal comes from the environment through tool execution,
but this function serves as a fallback/aggregator for final reward computation.
"""

from typing import Any, Optional


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs
) -> float:
    """
    Compute the reward score for ALFWorld task.
    
    ALFWorld uses environment terminal reward:
    - Task successfully completed: 1.0
    - Task failed/timeout: 0.0
    
    The actual reward is obtained from the environment through tool execution
    and accumulated via tool_rewards. This function checks various sources
    for the success signal.
    
    Args:
        solution_str: The model's response string (not directly used for ALFWorld)
        ground_truth: Ground truth information containing task metadata
        extra_info: Extra information dictionary containing:
            - success: Boolean indicating task success
            - rollout_reward_scores: Dict of accumulated rewards from tools
            - total_reward: Total accumulated reward from environment
            
    Returns:
        float: 1.0 if task succeeded, 0.0 otherwise
    """
    if extra_info is None:
        extra_info = {}
    
    # Method 1: Check explicit success flag
    if extra_info.get("success", False):
        return 1.0
    
    # Method 2: Check rollout_reward_scores from tool execution
    rollout_scores = extra_info.get("rollout_reward_scores", {})
    if isinstance(rollout_scores, dict) and rollout_scores:
        # Sum all tool rewards - positive total indicates success
        total_tool_reward = sum(
            v for v in rollout_scores.values() 
            if isinstance(v, (int, float))
        )
        if total_tool_reward > 0:
            return 1.0
    
    # Method 3: Check total_reward from environment manager
    total_reward = extra_info.get("total_reward", 0.0)
    if isinstance(total_reward, (int, float)) and total_reward > 0:
        return 1.0
    
    # Method 4: Check ground_truth for pre-computed success
    if isinstance(ground_truth, dict):
        if ground_truth.get("success", False):
            return 1.0
        # Check solvable flag - if task is not solvable, don't penalize
        if not ground_truth.get("solvable", True):
            return 0.0
    
    return 0.0


def compute_score_with_steps(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    step_penalty: float = 0.01,
    max_steps: int = 50,
    **kwargs
) -> float:
    """
    Compute reward with step penalty for efficiency.
    
    This variant penalizes longer solutions to encourage efficient behavior.
    
    Args:
        solution_str: The model's response string
        ground_truth: Ground truth information
        extra_info: Extra information dictionary
        step_penalty: Penalty per step taken
        max_steps: Maximum steps (used for normalization)
        
    Returns:
        float: Reward in range [0, 1], with penalty for more steps
    """
    base_reward = compute_score(solution_str, ground_truth, extra_info, **kwargs)
    
    if base_reward <= 0:
        return 0.0
    
    # Get number of steps taken
    steps = 0
    if extra_info:
        steps = extra_info.get("steps", 0)
        if not steps:
            # Try to count from action history
            action_history = extra_info.get("action_history", [])
            steps = len(action_history) if action_history else 0
    
    # Apply step penalty (capped to not go negative)
    penalty = min(step_penalty * steps, 0.5)  # Max 50% penalty
    
    return max(0.0, base_reward - penalty)
