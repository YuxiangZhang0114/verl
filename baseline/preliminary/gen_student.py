#!/usr/bin/env python3
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
Generate multi-turn conversation data with search tool calls for data distillation.

Uses vLLM inference with Alibaba-NLP/Tongyi-DeepResearch-30B-A3B model.
Parses tool calls locally using Hermes format and executes search via retrieval server.
"""

import argparse
import asyncio
import json
import logging
import os
from typing import Any, Optional

import pandas as pd
import regex
import requests
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 关闭 httpx 和 openai 的日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger.disabled = True  # 临时启用日志以调试

# Default system prompt is no longer used - we load prompts directly from the dataset

# Tool schema for search function
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the knowledge base for relevant information based on the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A fully-formed semantic query string. The tool will return search results for this query.",
                },
                "topk": {
                    "type": "integer",
                    "description": "Number of top results to return. Default is 5.",
                    "default": 5,
                }
            },
            "required": ["query"],
        },
    },
}

class HermesToolParser:
    """Parser for Hermes-format tool calls: <tool_call>...</tool_call>"""

    def __init__(self):
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)

    def extract_tool_calls(self, text: str) -> tuple[str, list[dict]]:
        """
        Extract tool calls from text.

        Args:
            text: Response text potentially containing tool calls

        Returns:
            Tuple of (content without tool calls, list of tool call dicts)
        """
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []

        for match in matches:
            try:
                function_call = json.loads(match)
                name = function_call.get("name", "")
                arguments = function_call.get("arguments", {})

                # Convert arguments to JSON string if it's a dict
                if isinstance(arguments, dict):
                    arguments_str = json.dumps(arguments, ensure_ascii=False)
                else:
                    arguments_str = arguments

                function_calls.append({
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_str},
                })
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {e}, match: {match}")

        # Remove tool call tags from content
        content = self.tool_call_regex.sub("", text).strip()

        return content, function_calls


class SearchExecutor:
    """Execute search queries via retrieval server"""

    def __init__(self, retrieval_url: str, timeout: int = 30):
        # Ensure the URL ends with /retrieve
        if not retrieval_url.endswith('/retrieve'):
            self.retrieval_url = f"{retrieval_url}/retrieve"
        else:
            self.retrieval_url = retrieval_url
        self.timeout = timeout

    def execute_search(self, query: str, topk: int = 5) -> str:
        """
        Execute a single search query and return formatted results.

        Args:
            query: A single search query string
            topk: Number of top results

        Returns:
            Formatted search results as string
        """
        try:
            logger.debug(f"Calling retrieval server at {self.retrieval_url} with query: {query}")
            response = requests.post(
                self.retrieval_url,
                json={"queries": [query], "topk": topk, "return_scores": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json().get("result", [])
            
            # Return raw JSON response to model (first result since we only sent one query)
            result_data = data[0] if data else []
            result_text = json.dumps(result_data, ensure_ascii=False)
            logger.debug(f"Retrieval server returned data, length: {len(result_text)}")
            return result_text

        except requests.exceptions.RequestException as e:
            logger.error(f"Search request failed: {e}")
            return json.dumps({"error": f"Search request failed: {str(e)}"})
        except Exception as e:
            logger.error(f"Search execution failed: {e}", exc_info=True)
            return json.dumps({"error": f"Search failed: {str(e)}"})


class DataGenerator:
    """Generate multi-turn conversation data with tool calls using local tokenizer"""

    def __init__(
        self,
        vllm_url: str,
        retrieval_url: str,
        model_path: str,
        max_assistant_turns: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")
        self.search_executor = SearchExecutor(retrieval_url)
        self.parser = HermesToolParser()  # Local tool parser
        self.max_assistant_turns = max_assistant_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load tokenizer for local prompt construction
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _build_system_with_tools(self, system_content: str) -> str:
        """Build system message with tools using tokenizer.apply_chat_template"""
        # Create a minimal messages list just to extract system prompt with tools
        dummy_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "dummy"},
        ]
        # Apply chat template with tools to get the formatted system prompt
        full_prompt = self.tokenizer.apply_chat_template(
            dummy_messages,
            tools=[SEARCH_TOOL_SCHEMA],
            add_generation_prompt=False,
            tokenize=False,
        )
        # Extract the system content between <|im_start|>system and <|im_end|>
        import re
        match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', full_prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return original system content + tool description
        return system_content

    async def generate_single_conversation(self, prompt: list, extra_info: dict) -> dict:
        """
        Generate a complete multi-turn conversation for a given prompt.

        Args:
            prompt: The prompt messages list from the dataset (contains system and user messages)
            extra_info: Extra information from the dataset

        Returns:
            Dictionary with messages and metadata
        """
        # Extract system and user content from the prompt
        system_content = ""
        user_content = ""
        for msg in prompt:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        
        # Build system message with tools embedded
        system_with_tools = self._build_system_with_tools(system_content)
        
        # Initialize messages for API calls (with tools in system)
        api_messages = [
            {"role": "system", "content": system_with_tools},
            {"role": "user", "content": user_content},
        ]
        
        # Messages to save (with original system content, without tool embedding)
        messages = [
            {"role": "system", "content": system_content, "raw_content": system_with_tools},
            {"role": "user", "content": user_content},
        ]
        
        assistant_turns = 0
        error_message = None
        
        logger.info(f"\n{'#'*100}")
        logger.info(f"开始生成对话 - 问题: {user_content[:100]}...")
        logger.info(f"最大轮数: {self.max_assistant_turns}")
        logger.info(f"{'#'*100}\n")

        try:
            while assistant_turns < self.max_assistant_turns:
                logger.info(f"\n{'='*80}\n开始第 {assistant_turns + 1} 轮生成...")
                logger.info(f"当前 api_messages 长度: {len(api_messages)}")
                logger.info(f"最后一条消息: role={api_messages[-1]['role']}, content_len={len(api_messages[-1].get('content', ''))}")
                
                # Call vLLM using chat.completions (no tools parameter, tools are in system prompt)
                response = await self.client.chat.completions.create(
                    model="student_model",
                    messages=api_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                raw_content = response.choices[0].message.content or ""
                logger.info(f"生成的原始内容长度: {len(raw_content)}")
                logger.info(f"原始内容前 200 字符: {raw_content[:200]}")
                
                # Parse tool calls locally using Hermes parser
                content, tool_calls = self.parser.extract_tool_calls(raw_content)
                
                # Debug logging
                if tool_calls:
                    logger.info(f"✓ 发现 {len(tool_calls)} 个工具调用")
                    for idx, tc in enumerate(tool_calls):
                        logger.info(f"  工具调用 {idx + 1}: {tc['function']['name']}")
                else:
                    logger.info(f"✗ 未发现工具调用，纯文本内容长度: {len(content)}")
                    logger.info(f"  内容前 200 字符: {content[:200] if content else '(空)'}")

                # Build assistant message for saving - use parsed content (with tool tags removed)
                save_msg = {"role": "assistant", "content": content ,"raw_content": raw_content}
                if tool_calls:
                    save_msg["tool_calls"] = tool_calls
                messages.append(save_msg)
                
                # Build assistant message for API - use raw content (model sees full output)
                api_msg = {"role": "assistant", "content": raw_content}
                api_messages.append(api_msg)

                assistant_turns += 1

                # Check if we should continue
                if not tool_calls:
                    # No tool calls, conversation is complete
                    logger.info(f"✓ 对话完成，共 {assistant_turns} 轮")
                    break

                # Execute tool calls and add tool messages
                logger.info(f"开始执行 {len(tool_calls)} 个工具调用...")
                for idx, tool_call in enumerate(tool_calls):
                    logger.info(f"  执行工具 {idx + 1}/{len(tool_calls)}")
                    function_name = tool_call["function"]["name"]
                    arguments_str = tool_call["function"]["arguments"]
                    tool_call_id = tool_call.get("id")

                    try:
                        # arguments_str is already a JSON string from parser
                        arguments = json.loads(arguments_str)
                    except (json.JSONDecodeError, TypeError):
                        # If it's already a dict, use it directly
                        if isinstance(arguments_str, dict):
                            arguments = arguments_str
                        else:
                            logger.warning(f"Failed to parse arguments: {arguments_str}")
                            arguments = {}

                    if function_name == "search":
                        # Get query parameter as defined in schema (single string)
                        query = arguments.get("query", "")
                        topk = arguments.get("topk", 5)
                        logger.info(f"    执行搜索: query='{query[:50]}...', topk={topk}")
                        if query:
                            try:
                                search_results = self.search_executor.execute_search(query, topk=topk)
                                logger.info(f"    ✓ 搜索完成，结果长度: {len(search_results)} 字符")
                                # Add tool response
                                tool_msg = {"role": "tool", "content": search_results}
                                if tool_call_id:
                                    tool_msg["tool_call_id"] = tool_call_id
                                messages.append(tool_msg)
                                api_messages.append(tool_msg)
                                logger.info(f"    ✓ 工具结果已添加到消息列表")
                            except Exception as e:
                                logger.error(f"Search execution error: {e}")
                                tool_msg = {"role": "tool", "content": json.dumps({"error": f"Search failed: {str(e)}"})}
                                if tool_call_id:
                                    tool_msg["tool_call_id"] = tool_call_id
                                messages.append(tool_msg)
                                api_messages.append(tool_msg)
                        else:
                            logger.warning("No query provided in search tool call")
                            tool_msg = {"role": "tool", "content": json.dumps({"error": "No query provided"})}
                            if tool_call_id:
                                tool_msg["tool_call_id"] = tool_call_id
                            messages.append(tool_msg)
                            api_messages.append(tool_msg)
                    else:
                        tool_msg = {"role": "tool", "content": json.dumps({"error": f"Unknown tool: {function_name}"})}
                        if tool_call_id:
                            tool_msg["tool_call_id"] = tool_call_id
                        messages.append(tool_msg)
                        api_messages.append(tool_msg)
                
                logger.info(f"✓ 第 {assistant_turns} 轮完成，准备进入下一轮...")

        except Exception as e:
            logger.error(f"Error generating conversation: {e}")
            error_message = str(e)

        return {
            "messages": messages,
            "tools": [SEARCH_TOOL_SCHEMA],
            "prompt": prompt,
            "extra_info": extra_info,
            "assistant_turns": assistant_turns,
            "error": error_message,
        }

    async def generate_batch(self, batch_data: list[dict], desc: str = "Processing batch") -> list[dict]:
        """Generate conversations for a batch of prompts"""
        tasks = [
            self.generate_single_conversation(item["prompt"], item.get("extra_info", {})) for item in batch_data
        ]
        # Use tqdm_asyncio.gather to show progress for tasks within the batch
        return await tqdm_asyncio.gather(*tasks, desc=desc)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn conversation data with search tool calls for data distillation"
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://10.244.247.213:8101/v1",
        help="vLLM server endpoint (OpenAI-compatible API)",
    )
    parser.add_argument(
        "--retrieval_url",
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="Retrieval server endpoint for search execution",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model path for loading tokenizer (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--max_assistant_turns", type=int, default=15, help="Maximum assistant turns per conversation"
    )
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/hotpotqa_train/train.parquet",
        help="Input parquet file with HotpotQA data (should contain 'prompt' and 'extra_info' columns)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="baseline/preliminary/data/train_student_4epoch.parquet",
        help="Output parquet file path",
    )
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size for async processing")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per generation")

    args = parser.parse_args()

    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_parquet(args.input_file)

    if args.max_samples > 0 and args.max_samples < len(df):
        df = df.head(args.max_samples)
        logger.info(f"Processing {args.max_samples} samples")
    else:
        logger.info(f"Processing all {len(df)} samples")

    # Prepare data items - directly use 'prompt' field from the dataset
    data_items = []
    for idx, row in df.iterrows():
        # Get extra_info, handle both dict and other types
        extra_info = row.get('extra_info', {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        
        item = {
            "prompt": row['prompt'],  # Use prompt directly from dataset
            "extra_info": {
                "index": idx,
                "answer": extra_info.get("answer", ""),
                "question": extra_info.get("question", ""),
                "data_source": row.get("data_source", "hotpotqa"),
            },
        }
        data_items.append(item)

    # Initialize generator
    generator = DataGenerator(
        vllm_url=args.vllm_url,
        retrieval_url=args.retrieval_url,
        model_path=args.model_path,
        max_assistant_turns=args.max_assistant_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # Process in batches and save incrementally
    num_batches = (len(data_items) + args.batch_size - 1) // args.batch_size
    total_successful = 0
    total_processed = 0

    logger.info(f"Generating conversations in {num_batches} batches...")
    logger.info(f"Results will be saved incrementally to {args.output_file}")

    for i in tqdm(range(0, len(data_items), args.batch_size), desc="Batches", total=num_batches):
        batch = data_items[i : i + args.batch_size]
        batch_num = (i // args.batch_size) + 1
        batch_results = await generator.generate_batch(batch, desc=f"Batch {batch_num}/{num_batches}")
        
        # Filter out failed generations in this batch
        successful_batch = [r for r in batch_results if r.get("error") is None]
        total_successful += len(successful_batch)
        total_processed += len(batch_results)
        
        # Save batch results incrementally
        if successful_batch:
            batch_df = pd.DataFrame(successful_batch)
            
            # Append to existing file or create new one
            if i == 0:
                # First batch: create new file
                batch_df.to_parquet(args.output_file, index=False)
                logger.info(f"Created output file: {args.output_file}")
            else:
                # Subsequent batches: append to existing file
                existing_df = pd.read_parquet(args.output_file)
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                combined_df.to_parquet(args.output_file, index=False)
            
            logger.info(f"Batch {(i // args.batch_size) + 1}/{num_batches}: Saved {len(successful_batch)} conversations (Total: {total_successful}/{total_processed})")
        else:
            logger.warning(f"Batch {(i // args.batch_size) + 1}/{num_batches}: No successful conversations to save")

    logger.info(f"Completed! Successfully generated {total_successful}/{total_processed} conversations")
    logger.info(f"Final output saved to {args.output_file}")

    # Print sample from saved file
    if total_successful > 0:
        logger.info("\n" + "=" * 80)
        logger.info("Sample conversation (from saved file):")
        logger.info("=" * 80)
        saved_df = pd.read_parquet(args.output_file)
        if len(saved_df) > 0:
            sample = saved_df.iloc[0].to_dict()
            for msg in sample["messages"]:
                role = msg["role"]
                content = msg.get("content", "")[:200]
                logger.info(f"\n{role.upper()}: {content}...")
                if "tool_calls" in msg:
                    logger.info(f"  [Tool calls: {len(msg['tool_calls'])}]")


if __name__ == "__main__":
    asyncio.run(main())

