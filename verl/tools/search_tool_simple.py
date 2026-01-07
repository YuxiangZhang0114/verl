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
Simple Search Tool - 基础版搜索工具
去掉了 Ray actor 和复杂的 rate limiting 机制，直接同步调用搜索 API
"""

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchToolSimple(BaseTool):
    """简化版搜索工具 - 直接同步调用，无 Ray actor 依赖

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchToolSimple with configuration and schema.

        Args:
            config: Configuration dictionary containing:
                - retrieval_service_url: URL of the retrieval service (required)
                - topk: Number of top results to return (default: 3)
                - timeout: Request timeout in seconds (default: 30)
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Retrieval service configuration
        self.retrieval_service_url = config.get("retrieval_service_url")
        if not self.retrieval_service_url:
            raise ValueError("Configuration must include 'retrieval_service_url'")

        self.topk = config.get("topk", 3)
        self.timeout = config.get("timeout", 30)

        logger.info(f"Initialized SearchToolSimple with url={self.retrieval_service_url}, topk={self.topk}, timeout={self.timeout}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query (or query_list) and optional topk

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        # Support both single 'query' parameter and 'query_list' parameter
        query_from_params = parameters.get("query")
        query_list_from_params = parameters.get("query_list")

        # Convert single query to query_list if needed
        if query_from_params:
            if isinstance(query_from_params, str):
                query_list = [query_from_params]
            else:
                error_msg = "Error: 'query' must be a string."
                logger.error(f"[SearchToolSimple] {error_msg} Received parameters: {parameters}")
                return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}
        elif query_list_from_params:
            if isinstance(query_list_from_params, list):
                query_list = query_list_from_params
            else:
                error_msg = "Error: 'query_list' must be a list."
                logger.error(f"[SearchToolSimple] {error_msg} Received parameters: {parameters}")
                return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}
        else:
            error_msg = "Error: Either 'query' or 'query_list' must be provided in parameters."
            logger.error(f"[SearchToolSimple] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Get topk from parameters or use default from config
        topk = parameters.get("topk", self.topk)
        if not isinstance(topk, int) or topk <= 0:
            topk = self.topk

        # Execute search directly (no Ray actor)
        try:
            result_text, metadata = perform_single_search_batch(
                retrieval_service_url=self.retrieval_service_url,
                query_list=query_list,
                topk=topk,
                concurrent_semaphore=None,
                timeout=self.timeout,
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchToolSimple] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

