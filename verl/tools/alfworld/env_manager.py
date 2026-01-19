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
ALFWorld Environment Manager for VERL.

This module provides a singleton environment manager that handles ALFWorld TextWorld
environment instances for multiple concurrent requests.
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ALFWorldEnvManager:
    """
    Singleton environment manager for ALFWorld.
    
    This class manages ALFWorld TextWorld environment instances for multiple
    concurrent training requests. It provides:
    - Dynamic environment creation based on task_id
    - Environment state tracking per request
    - Thread-safe operations for concurrent access
    - Automatic cleanup of completed environments
    
    Usage:
        manager = ALFWorldEnvManager.get_instance()
        manager.load_task_registry(dataset)
        
        # In tool execution:
        obs, goal = manager.create_env(task_id, request_id)
        obs, reward, done = manager.step(request_id, action)
        manager.release_env(request_id)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, alfworld_data_path: Optional[str] = None):
        """
        Initialize the environment manager.
        
        Args:
            alfworld_data_path: Path to ALFWorld data directory.
                               If None, uses ALFWORLD_DATA environment variable.
        """
        if self._initialized:
            return
            
        self.alfworld_data_path = alfworld_data_path or os.environ.get("ALFWORLD_DATA", None)
        self.task_registry: dict[str, dict] = {}  # task_id -> game_content
        self.active_envs: dict[str, Any] = {}  # request_id -> env instance
        self.env_states: dict[str, dict] = {}  # request_id -> state dict
        self._env_lock = asyncio.Lock()
        self._alfworld_available = None
        self._initialized = True
        
    @classmethod
    def get_instance(cls, alfworld_data_path: Optional[str] = None) -> "ALFWorldEnvManager":
        """Get the singleton instance of the environment manager."""
        return cls(alfworld_data_path)
    
    def _check_alfworld_available(self) -> bool:
        """Check if alfworld package is available and properly configured."""
        if self._alfworld_available is None:
            try:
                from alfworld.agents.environment import get_environment
                # Check if ALFWORLD_DATA exists
                alfworld_data = os.environ.get('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
                data_path = os.path.join(alfworld_data, 'json_2.1.1', 'train')
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"ALFWorld data not found at {data_path}")
                self._alfworld_available = True
                logger.info("ALFWorld environment is available")
            except ImportError as e:
                self._alfworld_available = False
                logger.error(
                    f"ALFWorld package not installed: {e}. "
                    "Install with: pip install alfworld && alfworld-download"
                )
            except Exception as e:
                self._alfworld_available = False
                logger.error(
                    f"ALFWorld configuration error: {e}. "
                    "Make sure to run: alfworld-download"
                )
        return self._alfworld_available
    
    def load_task_registry_from_dataset(self, dataset: list[dict]) -> None:
        """
        Load task registry from a dataset.
        
        Args:
            dataset: List of dataset items with 'id' and 'game_content' fields.
        """
        for item in dataset:
            task_id = item.get("id") or item.get("task_id")
            game_content = item.get("game_content")
            if task_id and game_content:
                if isinstance(game_content, str):
                    game_content = json.loads(game_content)
                self.task_registry[task_id] = game_content
        logger.info(f"Loaded {len(self.task_registry)} tasks into registry")
    
    def load_task_registry_from_parquet(self, parquet_path: str) -> None:
        """
        Load task registry from a parquet file.
        
        Args:
            parquet_path: Path to the parquet file.
        """
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        
        for _, row in df.iterrows():
            extra_info = row.get("extra_info", {})
            task_id = extra_info.get("task_id")
            if task_id:
                # Store minimal info needed for task lookup
                self.task_registry[task_id] = {
                    "task_type": extra_info.get("task_type"),
                    "walkthrough": extra_info.get("walkthrough", []),
                    "pddl_goal": extra_info.get("pddl_goal", ""),
                }
        logger.info(f"Loaded {len(self.task_registry)} tasks from {parquet_path}")
    
    async def create_env(self, task_id: str, request_id: str, game_file_path: str = "") -> tuple[str, str]:
        """
        Create an environment for a specific task and request.
        
        Args:
            task_id: The task identifier from the dataset.
            request_id: Unique identifier for this request/episode.
            game_file_path: Optional path to specific game file for task matching.
            
        Returns:
            tuple[str, str]: (initial_observation, goal_description)
            
        Raises:
            RuntimeError: If ALFWorld is not available or environment creation fails.
        """
        async with self._env_lock:
            if request_id in self.active_envs:
                # Environment already exists, return cached initial state
                state = self.env_states[request_id]
                return state["initial_obs"], state["goal"]
            
            if not self._check_alfworld_available():
                raise RuntimeError(
                    "ALFWorld package is not installed. "
                    "Please install with: pip install alfworld && alfworld-download"
                )
            
            return await self._create_real_env(task_id, request_id, game_file_path)
    
    async def _create_real_env(self, task_id: str, request_id: str, game_file_path: str = "") -> tuple[str, str]:
        """Create a real ALFWorld TextWorld environment with optional specific game file."""
        try:
            from alfworld.agents.environment import get_environment
            import yaml
        except ImportError as e:
            raise RuntimeError(f"Failed to import ALFWorld modules: {e}")
        
        # Load ALFWorld config (once)
        if not hasattr(self, '_alfworld_config'):
            try:
                # Use our bundled config file
                config_path = os.path.join(os.path.dirname(__file__), 'base_config.yaml')
                
                if not os.path.exists(config_path):
                    raise FileNotFoundError(
                        f"ALFWorld config file not found at: {config_path}"
                    )
                
                with open(config_path, 'r') as f:
                    self._alfworld_config = yaml.safe_load(f)
                
                # Expand environment variables in paths
                alfworld_data = os.environ.get('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
                
                def expand_path(path_str):
                    if isinstance(path_str, str):
                        return path_str.replace('$ALFWORLD_DATA', alfworld_data)
                    return path_str
                
                # Expand paths in config
                if 'dataset' in self._alfworld_config:
                    for key in self._alfworld_config['dataset']:
                        self._alfworld_config['dataset'][key] = expand_path(self._alfworld_config['dataset'][key])
                if 'logic' in self._alfworld_config:
                    for key in self._alfworld_config['logic']:
                        self._alfworld_config['logic'][key] = expand_path(self._alfworld_config['logic'][key])
                
                logger.info(f"Loaded ALFWorld config from: {config_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load ALFWorld config: {e}. "
                    "Make sure you have run 'alfworld-download' to download required data."
                )
            # Force TextWorld environment type
            self._alfworld_config['env']['type'] = 'AlfredTWEnv'
        
        # Create shared environment instance (once per worker)
        if not hasattr(self, '_shared_env') or self._shared_env is None:
            try:
                logger.info("Creating shared ALFWorld environment instance...")
                EnvClass = get_environment(self._alfworld_config['env']['type'])
                self._shared_env = EnvClass(self._alfworld_config, train_eval='train')
                self._shared_env = self._shared_env.init_env(batch_size=1)
                logger.info("Shared ALFWorld environment created successfully")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create ALFWorld environment: {e}. "
                    "Make sure ALFWorld is properly installed and configured."
                )
        
        # Get task info from registry
        task_info = self.task_registry.get(task_id, {})
        task_type = task_info.get("task_type", "unknown")
        
        # If no game_file_path provided, try to get from task_registry
        if not game_file_path:
            game_file_path = task_info.get("game_file_path", "")
        
        try:
            # Try to load specific game if game_file_path is provided
            if game_file_path and hasattr(self._shared_env, 'game_files'):
                original_game_files = self._shared_env.game_files.copy()
                
                # Find matching game files
                matching_games = [g for g in original_game_files if game_file_path in g]
                
                if matching_games:
                    # Temporarily replace game_files with only the matching game
                    self._shared_env.game_files = matching_games
                    logger.debug(f"Loading specific game: {matching_games[0]}")
                    reset_result = self._shared_env.reset()
                    # Restore original game_files
                    self._shared_env.game_files = original_game_files
                else:
                    logger.warning(
                        f"Game file '{game_file_path}' not found in available games. "
                        f"Using random game instead."
                    )
                    reset_result = self._shared_env.reset()
            else:
                # No specific game requested, use random
                reset_result = self._shared_env.reset()
            
            # Handle reset result - could be (obs_list, info) or just obs_list
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs_list, info = reset_result
            else:
                obs_list = reset_result
                info = {}
            
            # Extract observation from batch
            obs = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list
            
            # Handle info - could be a list or dict
            if isinstance(info, (list, tuple)):
                info = info[0] if len(info) > 0 else {}
            if not isinstance(info, dict):
                info = {}
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to reset ALFWorld environment: {e}. "
                "Make sure ALFWorld is properly installed and configured."
            )
        
        # Extract goal from observation
        goal = self._extract_goal(obs, {"task_type": task_type})
        
        # Get admissible commands from info
        admissible_commands = info.get('admissible_commands', [])
        if isinstance(admissible_commands, (list, tuple)) and len(admissible_commands) > 0:
            if isinstance(admissible_commands[0], (list, tuple)):
                admissible_commands = list(admissible_commands[0])
            else:
                admissible_commands = list(admissible_commands)
        
        # Store reference to shared environment and state
        self.active_envs[request_id] = self._shared_env
        self.env_states[request_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "initial_obs": obs,
            "current_obs": obs,
            "goal": goal,
            "done": False,
            "total_reward": 0.0,
            "steps": 0,
            "action_history": [],
            "admissible_commands": admissible_commands,
        }
        
        return obs, goal
    
    def _create_simulated_env(self, task_id: str, request_id: str) -> tuple[str, str]:
        """Create a simulated environment for testing without ALFWorld installed."""
        task_info = self.task_registry.get(task_id, {})
        task_type = task_info.get("task_type", "unknown")
        walkthrough = task_info.get("walkthrough", [])
        
        # Generate simulated initial observation based on task type
        initial_obs = self._generate_simulated_observation(task_type)
        goal = self._generate_goal_from_task_type(task_type)
        
        # Store simulated environment state
        self.env_states[request_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "initial_obs": initial_obs,
            "current_obs": initial_obs,
            "goal": goal,
            "done": False,
            "total_reward": 0.0,
            "steps": 0,
            "walkthrough": walkthrough,
            "current_step_idx": 0,
            "action_history": [],
            "simulated": True,
        }
        self.active_envs[request_id] = "simulated"
        
        return initial_obs, goal
    
    def _generate_simulated_observation(self, task_type: str) -> str:
        """Generate a simulated initial observation."""
        observations = {
            "pick_and_place_simple": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a countertop 1, a drawer 1, a drawer 2, a fridge 1, a cabinet 1, "
                "a cabinet 2, a sinkbasin 1, and a stoveburner 1."
            ),
            "pick_two_obj_and_place": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a armchair 1, a coffeetable 1, a drawer 1, a garbagecan 1, "
                "a shelf 1, a sidetable 1, a sofa 1, and a tvstand 1."
            ),
            "pick_clean_then_place_in_recep": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a countertop 1, a cabinet 1, a drawer 1, a fridge 1, "
                "a sinkbasin 1, a microwave 1, and a stoveburner 1."
            ),
            "pick_heat_then_place_in_recep": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a countertop 1, a drawer 1, a fridge 1, a microwave 1, "
                "a sinkbasin 1, a cabinet 1, and a diningtable 1."
            ),
            "pick_cool_then_place_in_recep": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a countertop 1, a drawer 1, a fridge 1, a cabinet 1, "
                "a sinkbasin 1, a microwave 1, and a stoveburner 1."
            ),
            "look_at_obj_in_light": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a bed 1, a desk 1, a drawer 1, a drawer 2, a garbagecan 1, "
                "a desklamp 1, a shelf 1, and a sidetable 1."
            ),
        }
        return observations.get(task_type, observations["pick_and_place_simple"])
    
    def _generate_goal_from_task_type(self, task_type: str) -> str:
        """Generate goal description from task type."""
        goals = {
            "pick_and_place_simple": "put some object in/on some receptacle.",
            "pick_two_obj_and_place": "put two objects in/on receptacles.",
            "pick_clean_then_place_in_recep": "clean some object and put it in/on some receptacle.",
            "pick_heat_then_place_in_recep": "heat some object and put it in/on some receptacle.",
            "pick_cool_then_place_in_recep": "cool some object and put it in/on some receptacle.",
            "look_at_obj_in_light": "examine some object with a lamp.",
        }
        return f"Your task is to: {goals.get(task_type, 'complete the household task.')}"
    
    def _extract_goal(self, observation: str, game_content: dict) -> str:
        """Extract goal from observation or game content."""
        # Try to extract from observation first (ALFWorld includes goal in initial obs)
        if "Your task is to" in observation:
            lines = observation.split("\n")
            for line in lines:
                if "Your task is to" in line:
                    return line.strip()
        
        # Fall back to task type
        task_type = game_content.get("task_type", "")
        return self._generate_goal_from_task_type(task_type)
    
    async def step(self, request_id: str, action: str) -> tuple[str, float, bool]:
        """
        Execute an action in the environment.
        
        Args:
            request_id: The request identifier.
            action: The action string to execute.
            
        Returns:
            tuple[str, float, bool]: (observation, reward, done)
        """
        if request_id not in self.env_states:
            return "Error: Environment not initialized.", 0.0, True
        
        state = self.env_states[request_id]
        
        if state.get("done", False):
            return "Episode already finished.", 0.0, True
        
        state["steps"] += 1
        state["action_history"].append(action)
        
        if state.get("simulated", False):
            return self._simulated_step(request_id, action)
        else:
            # Use lock to prevent concurrent access to shared environment
            async with self._env_lock:
                return await self._real_step(request_id, action)
    
    async def _real_step(self, request_id: str, action: str) -> tuple[str, float, bool]:
        """Execute action in real ALFWorld environment."""
        env = self.active_envs[request_id]
        state = self.env_states[request_id]
        
        # ALFWorld expects a list of actions (batched)
        # Returns: obs_list, scores_list, dones_list, infos_list
        step_result = env.step([action])
        
        # Handle different possible return formats
        if len(step_result) == 4:
            obs_list, scores_list, dones_list, infos_list = step_result
        else:
            logger.warning(f"Unexpected step result format: {len(step_result)} elements")
            obs_list, scores_list, dones_list = step_result[:3]
            infos_list = [{}]
        
        # Extract single results from batch
        obs = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list
        
        # Handle scores - could be list, tuple, or single value
        score_raw = scores_list[0] if isinstance(scores_list, (list, tuple)) else scores_list
        # Handle case where score is a tuple (accumulated_score, step_reward)
        if isinstance(score_raw, (list, tuple)):
            # ALFWorld returns accumulated score, we want step reward
            # For simplicity, check if task is done - that's when reward matters
            reward = 0.0
        else:
            reward = float(score_raw) if score_raw is not None else 0.0
        
        # Handle dones
        done_raw = dones_list[0] if isinstance(dones_list, (list, tuple)) else dones_list
        done = bool(done_raw) if done_raw is not None else False
        
        # Handle infos - it's a list of dicts
        info = infos_list[0] if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0 else {}
        if not isinstance(info, dict):
            info = {}
        
        # Check if task was won (successful completion)
        if done:
            won = info.get('won', [False])
            if isinstance(won, (list, tuple)):
                won = won[0] if len(won) > 0 else False
            if won:
                reward = 1.0  # Task completed successfully
        
        state["current_obs"] = obs
        state["total_reward"] += reward
        state["done"] = done
        state["admissible_commands"] = info.get('admissible_commands', [])
        
        return obs, reward, done
    
    def _simulated_step(self, request_id: str, action: str) -> tuple[str, float, bool]:
        """Execute action in simulated environment."""
        state = self.env_states[request_id]
        walkthrough = state.get("walkthrough", [])
        current_idx = state.get("current_step_idx", 0)
        
        action_lower = action.lower().strip()
        
        # Check if action matches expected walkthrough step
        if current_idx < len(walkthrough):
            expected = walkthrough[current_idx].lower().strip()
            if self._actions_match(action_lower, expected):
                state["current_step_idx"] = current_idx + 1
                
                # Check if task is complete
                if state["current_step_idx"] >= len(walkthrough):
                    state["done"] = True
                    state["total_reward"] = 1.0
                    return "Task completed successfully!", 1.0, True
                
                # Generate appropriate observation
                obs = self._generate_step_observation(action_lower, state)
                return obs, 0.0, False
        
        # Action didn't match expected - generate generic response
        obs = self._generate_generic_response(action_lower, state)
        
        # Check for max steps
        if state["steps"] >= 50:
            state["done"] = True
            return "Maximum steps reached. Task failed.", 0.0, True
        
        return obs, 0.0, False
    
    def _actions_match(self, action: str, expected: str) -> bool:
        """Check if action matches expected action (with some flexibility)."""
        # Normalize actions
        action = action.replace("_", " ").strip().lower()
        expected = expected.replace("_", " ").strip().lower()
        
        # Exact match
        if action == expected:
            return True
        
        # Check if key parts match (verb + main target)
        action_parts = action.split()
        expected_parts = expected.split()
        
        if not action_parts or not expected_parts:
            return False
        
        # Verb must match
        if action_parts[0] != expected_parts[0]:
            return False
        
        # For actions with targets, check if target overlaps
        if len(action_parts) >= 2 and len(expected_parts) >= 2:
            # Extract key target words (ignore prepositions like "to", "from", "in", "on")
            prepositions = {"to", "from", "in", "on", "with", "in/on"}
            action_targets = [w for w in action_parts[1:] if w not in prepositions]
            expected_targets = [w for w in expected_parts[1:] if w not in prepositions]
            
            # Check if any target word matches
            if action_targets and expected_targets:
                # At least 50% of expected target words should match
                matches = sum(1 for t in expected_targets if t in action_targets)
                if matches >= len(expected_targets) * 0.5:
                    return True
        
        return False
    
    def _generate_step_observation(self, action: str, state: dict) -> str:
        """Generate observation for a successful step."""
        action_parts = action.split()
        verb = action_parts[0] if action_parts else ""
        
        observations = {
            "go": "You arrive at the location.",
            "goto": "You arrive at the location.",
            "take": "You pick up the object.",
            "put": "You put the object down.",
            "open": "You open the receptacle.",
            "close": "You close the receptacle.",
            "toggle": "You toggle the object.",
            "clean": "You clean the object.",
            "heat": "You heat the object.",
            "cool": "You cool the object.",
            "use": "You use the object.",
            "look": state.get("current_obs", "You look around."),
            "examine": "You examine the object closely.",
            "inventory": "You check your inventory.",
        }
        
        return observations.get(verb, f"You perform the action: {action}")
    
    def _generate_generic_response(self, action: str, state: dict) -> str:
        """Generate a generic response for non-matching actions."""
        action_parts = action.split()
        verb = action_parts[0] if action_parts else ""
        
        responses = {
            "go": "You can't go there.",
            "goto": "You can't go there.",
            "take": "Nothing happens.",
            "put": "Nothing happens.",
            "open": "You can't open that.",
            "close": "You can't close that.",
            "toggle": "You can't toggle that.",
            "clean": "You can't clean that here.",
            "heat": "You can't heat that here.",
            "cool": "You can't cool that here.",
            "use": "Nothing happens.",
            "look": state.get("current_obs", "You look around but see nothing new."),
            "examine": "You see nothing special.",
            "inventory": "You are not carrying anything." if not state.get("holding") else f"You are carrying: {state.get('holding')}",
        }
        
        return responses.get(verb, "Nothing happens.")
    
    async def release_env(self, request_id: str) -> None:
        """
        Release an environment after an episode is complete.
        
        Args:
            request_id: The request identifier.
        """
        async with self._env_lock:
            if request_id in self.active_envs:
                env = self.active_envs[request_id]
                if env != "simulated" and hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception as e:
                        logger.warning(f"Error closing environment: {e}")
                del self.active_envs[request_id]
            
            if request_id in self.env_states:
                state = self.env_states[request_id]
                # Clean up temporary files
                if "game_file" in state:
                    try:
                        os.unlink(state["game_file"])
                    except Exception:
                        pass
                del self.env_states[request_id]
    
    def get_env_state(self, request_id: str) -> Optional[dict]:
        """Get the current state for a request."""
        return self.env_states.get(request_id)
    
    def is_done(self, request_id: str) -> bool:
        """Check if an episode is done."""
        state = self.env_states.get(request_id)
        return state.get("done", True) if state else True
    
    def get_total_reward(self, request_id: str) -> float:
        """Get the total reward for an episode."""
        state = self.env_states.get(request_id)
        return state.get("total_reward", 0.0) if state else 0.0
