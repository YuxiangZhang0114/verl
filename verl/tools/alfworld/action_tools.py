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
ALFWorld Action Tools for VERL.

This module provides tool implementations for all ALFWorld actions.
Each tool corresponds to one action in the ALFWorld environment.
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from .env_manager import ALFWorldEnvManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ALFWorldActionTool(BaseTool):
    """
    Base class for ALFWorld action tools.
    
    This class provides common functionality for all ALFWorld action tools:
    - Environment initialization on first call
    - State management via agent_data.extra_fields
    - Action formatting and execution
    - Reward calculation
    
    Subclasses should implement:
    - _get_action_string(): Format the action string for the environment
    """
    
    # Class-level environment manager (shared across all tool instances)
    _env_manager: Optional[ALFWorldEnvManager] = None
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # Initialize environment manager if not already done
        if ALFWorldActionTool._env_manager is None:
            alfworld_data_path = config.get("alfworld_data_path", None)
            ALFWorldActionTool._env_manager = ALFWorldEnvManager.get_instance(alfworld_data_path)
    
    @property
    def env_manager(self) -> ALFWorldEnvManager:
        """Get the environment manager."""
        return ALFWorldActionTool._env_manager
    
    async def create(
        self, instance_id: Optional[str] = None, create_kwargs: Optional[dict] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        create_kwargs = create_kwargs or kwargs.get("create_kwargs", {})
        self._instance_dict[instance_id] = {
            "task_id": create_kwargs.get("task_id"),
            "task_type": create_kwargs.get("task_type"),
        }
        
        return instance_id, ToolResponse()
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        """
        Format the action string for the environment.
        
        Args:
            parameters: The tool call parameters.
            
        Returns:
            The formatted action string.
        """
        raise NotImplementedError("Subclasses must implement _get_action_string")
    
    async def _ensure_env_initialized(self, agent_data: Any) -> tuple[str, str]:
        """
        Ensure the environment is initialized for this request.
        
        Args:
            agent_data: The AgentData object from the agent loop.
            
        Returns:
            tuple[str, str]: (initial_observation, goal)
        """
        if "alfworld_env" not in agent_data.extra_fields:
            # Get task info from tools_kwargs
            task_id = None
            task_type = None
            walkthrough = []
            game_file_path = ""
            
            for tool_name in agent_data.tools_kwargs:
                create_kwargs = agent_data.tools_kwargs[tool_name].get("create_kwargs", {})
                if "task_id" in create_kwargs:
                    task_id = create_kwargs["task_id"]
                    task_type = create_kwargs.get("task_type", "unknown")
                    walkthrough = create_kwargs.get("walkthrough", [])
                    game_file_path = create_kwargs.get("game_file_path", "")
                    break
            
            if task_id is None:
                raise ValueError("task_id not found in tools_kwargs")
            
            # Dynamically register task info if not already registered
            # This ensures the simulated environment has the walkthrough information
            if task_id not in self.env_manager.task_registry:
                self.env_manager.task_registry[task_id] = {
                    "task_type": task_type,
                    "walkthrough": walkthrough,
                    "game_file_path": game_file_path,
                }
                logger.debug(f"Registered task {task_id} with {len(walkthrough)} walkthrough steps")
            
            # Create environment with specific game file
            initial_obs, goal = await self.env_manager.create_env(
                task_id, agent_data.request_id, game_file_path=game_file_path
            )
            
            # Store in extra_fields
            agent_data.extra_fields["alfworld_env"] = {
                "task_id": task_id,
                "initialized": True,
                "initial_obs": initial_obs,
                "goal": goal,
            }
            
            return initial_obs, goal
        else:
            env_state = agent_data.extra_fields["alfworld_env"]
            return env_state["initial_obs"], env_state["goal"]
    
    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], agent_data: Any = None, **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute the action in the ALFWorld environment.
        
        Args:
            instance_id: The tool instance ID.
            parameters: The action parameters.
            agent_data: The AgentData object from the agent loop.
            
        Returns:
            tuple[ToolResponse, float, dict]: (response, reward, metrics)
        """
        if agent_data is None:
            return ToolResponse(text="Error: agent_data is required for ALFWorld tools."), 0.0, {}
        
        # Ensure environment is initialized
        try:
            initial_obs, goal = await self._ensure_env_initialized(agent_data)
        except Exception as e:
            logger.warning(f"Failed to initialize environment: {e}")
            return ToolResponse(text=f"Error initializing environment: {e}"), 0.0, {}
        
        # Check if this is the first action - return initial observation
        env_state = agent_data.extra_fields["alfworld_env"]
        if not env_state.get("first_action_taken", False):
            env_state["first_action_taken"] = True
            # For the first action, we need to show the initial observation
            # but still execute the action
        
        # Format and execute the action
        try:
            action_str = self._get_action_string(parameters)
        except Exception as e:
            return ToolResponse(text=f"Error formatting action: {e}"), 0.0, {}
        
        # Execute action in environment
        try:
            obs, reward, done = await self.env_manager.step(agent_data.request_id, action_str)
        except Exception as e:
            logger.warning(f"Error executing action: {e}")
            return ToolResponse(text=f"Error executing action: {e}"), 0.0, {}
        
        # Update extra_fields with done status
        env_state["done"] = done
        
        # Only give reward at the end (result reward only, no process reward)
        # Check if task was won via info dict
        env_state_info = self.env_manager.get_env_state(agent_data.request_id)
        won = env_state_info.get("won", False)
        
        # Also save won to agent_data.extra_fields for tool_agent_loop to access
        env_state["won"] = won
        
        # Result reward: 1.0 if task completed successfully, 0.0 otherwise
        if done and won:
            final_reward = 1.0
            response_text = f"{obs}\n\n✓ Task completed successfully!"
        elif done:
            final_reward = 0.0
            response_text = f"{obs}\n\n✗ Task failed."
        else:
            final_reward = 0.0
            response_text = obs
        
        # DEBUG LOG
        if done:
            print(f"[ALFWORLD DEBUG] request_id={agent_data.request_id}, action={action_str}, done={done}, won={won}, final_reward={final_reward}")
        
        metrics = {
            "action": action_str,
            "done": done,
            "won": won,
            "steps": env_state_info.get("steps", 0),
        }
        
        return ToolResponse(text=response_text), final_reward, metrics
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the final reward for the episode."""
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            return 0.0
        
        return self.env_manager.get_total_reward(agent_data.request_id)
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class GotoTool(ALFWorldActionTool):
    """Tool for navigating to a receptacle in ALFWorld."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        target = parameters.get("target", "")
        return f"go to {target}"


class TakeTool(ALFWorldActionTool):
    """Tool for picking up an object from a receptacle."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        obj = parameters.get("object", "")
        source = parameters.get("source", "")
        return f"take {obj} from {source}"


class PutTool(ALFWorldActionTool):
    """Tool for placing an object in/on a receptacle."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        obj = parameters.get("object", "")
        target = parameters.get("target", "")
        preposition = parameters.get("preposition", "in/on")
        return f"put {obj} {preposition} {target}"


class OpenTool(ALFWorldActionTool):
    """Tool for opening a receptacle."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        target = parameters.get("target", "")
        return f"open {target}"


class CloseTool(ALFWorldActionTool):
    """Tool for closing a receptacle."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        target = parameters.get("target", "")
        return f"close {target}"


class ToggleTool(ALFWorldActionTool):
    """Tool for toggling an object's state (e.g., turning on/off a lamp)."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        target = parameters.get("target", "")
        return f"toggle {target}"


class CleanTool(ALFWorldActionTool):
    """Tool for cleaning an object using a sink/basin."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        obj = parameters.get("object", "")
        receptacle = parameters.get("receptacle", "")
        return f"clean {obj} with {receptacle}"


class HeatTool(ALFWorldActionTool):
    """Tool for heating an object using a microwave."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        obj = parameters.get("object", "")
        receptacle = parameters.get("receptacle", "microwave 1")
        return f"heat {obj} with {receptacle}"


class CoolTool(ALFWorldActionTool):
    """Tool for cooling an object using a fridge."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        obj = parameters.get("object", "")
        receptacle = parameters.get("receptacle", "fridge 1")
        return f"cool {obj} with {receptacle}"


class UseTool(ALFWorldActionTool):
    """Tool for using a receptacle (e.g., turning on a desklamp)."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        target = parameters.get("target", "")
        return f"use {target}"


class LookTool(ALFWorldActionTool):
    """Tool for looking around the environment."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        return "look"


class InventoryTool(ALFWorldActionTool):
    """Tool for checking what the agent is holding."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        return "inventory"


class ExamineTool(ALFWorldActionTool):
    """Tool for examining an object or receptacle closely."""
    
    def _get_action_string(self, parameters: dict[str, Any]) -> str:
        target = parameters.get("target", "")
        return f"examine {target}"


# Tool class registry for dynamic instantiation
ALFWORLD_TOOL_CLASSES = {
    "goto": GotoTool,
    "take": TakeTool,
    "put": PutTool,
    "open": OpenTool,
    "close": CloseTool,
    "toggle": ToggleTool,
    "clean": CleanTool,
    "heat": HeatTool,
    "cool": CoolTool,
    "use": UseTool,
    "look": LookTool,
    "inventory": InventoryTool,
    "examine": ExamineTool,
}
