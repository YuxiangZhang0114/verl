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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default system prompt for search tasks
DEFAULT_SYSTEM_CONTENT = (
    "You are a deep research assistant. Your core function is to conduct thorough, multi-source "
    "investigations into any topic. You must handle both broad, open-domain inquiries and queries "
    "within specialized academic fields. For every request, synthesize information from credible, "
    "diverse sources to deliver a comprehensive, accurate, and objective response. When you have "
    "gathered sufficient information and are ready to provide the definitive response, you must enclose "
    "the entire final answer within <answer></answer> tags."
)

# Tool schema for search function
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web for relevant information based on the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of fully-formed semantic queries. The tool will return search results for each query.",
                }
            },
            "required": ["query_list"],
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
        self.retrieval_url = retrieval_url
        self.timeout = timeout

    def execute_search(self, query_list: list[str], topk: int = 5) -> str:
        """
        Execute search queries and return formatted results.

        Args:
            query_list: List of search queries
            topk: Number of top results per query

        Returns:
            Formatted search results as string
        """
        try:
            response = requests.post(
                self.retrieval_url,
                json={"query_list": query_list, "topk": topk},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Format results
            results = []
            for query, docs in zip(query_list, data.get("results", [])):
                results.append(f"Query: {query}")
                for i, doc in enumerate(docs, 1):
                    title = doc.get("title", "")
                    content = doc.get("content", "")
                    results.append(f"[{i}] {title}: {content}")
                results.append("")

            return "\n".join(results)

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return json.dumps({"error": f"Search failed: {str(e)}"})


class DataGenerator:
    """Generate multi-turn conversation data with tool calls"""

    def __init__(
        self,
        vllm_url: str,
        retrieval_url: str,
        max_assistant_turns: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_content: str = DEFAULT_SYSTEM_CONTENT,
    ):
        self.client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")
        self.parser = HermesToolParser()
        self.search_executor = SearchExecutor(retrieval_url)
        self.max_assistant_turns = max_assistant_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_content = system_content

    async def generate_single_conversation(self, question: str, extra_info: dict) -> dict:
        """
        Generate a complete multi-turn conversation for a single question.

        Args:
            question: The question to answer
            extra_info: Extra information from the dataset

        Returns:
            Dictionary with messages and metadata
        """
        # Initialize messages
        messages = [
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": f"Question: {question}"},
        ]

        assistant_turns = 0
        error_message = None

        try:
            while assistant_turns < self.max_assistant_turns:
                # Generate response
                response = await self.client.chat.completions.create(
                    model="/models/Tonyi-deepresearch",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={"skip_special_tokens": False},  # Keep special tokens for parsing
                )

                assistant_message = response.choices[0].message
                raw_content = assistant_message.content or ""

                # Parse tool calls
                content, tool_calls = self.parser.extract_tool_calls(raw_content)

                # Build assistant message
                msg = {"role": "assistant", "content": content}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)

                assistant_turns += 1

                # Check if we should continue
                if not tool_calls:
                    # No tool calls, conversation is complete
                    break

                # Execute tool calls and add tool messages
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments_str = tool_call["function"]["arguments"]

                    try:
                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse arguments: {arguments_str}")
                        arguments = {}

                    if function_name == "search":
                        query_list = arguments.get("query_list", [])
                        if query_list:
                            search_results = self.search_executor.execute_search(query_list)
                            messages.append({"role": "tool", "content": search_results})
                        else:
                            messages.append(
                                {"role": "tool", "content": json.dumps({"error": "No queries provided"})}
                            )
                    else:
                        messages.append(
                            {"role": "tool", "content": json.dumps({"error": f"Unknown tool: {function_name}"})}
                        )

        except Exception as e:
            logger.error(f"Error generating conversation: {e}")
            error_message = str(e)

        return {
            "messages": messages,
            "tools": [SEARCH_TOOL_SCHEMA],
            "question": question,
            "extra_info": extra_info,
            "assistant_turns": assistant_turns,
            "error": error_message,
        }

    async def generate_batch(self, batch_data: list[dict]) -> list[dict]:
        """Generate conversations for a batch of questions"""
        tasks = [
            self.generate_single_conversation(item["question"], item.get("extra_info", {})) for item in batch_data
        ]
        return await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn conversation data with search tool calls for data distillation"
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server endpoint (OpenAI-compatible API)",
    )
    parser.add_argument(
        "--retrieval_url",
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="Retrieval server endpoint for search execution",
    )
    parser.add_argument(
        "--max_assistant_turns", type=int, default=3, help="Maximum assistant turns per conversation"
    )
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/hotpotqa_hard_train/train.parquet",
        help="Input parquet file with HotpotQA data",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="baseline/DataKD/train_distilled.parquet",
        help="Output parquet file path",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for async processing")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per generation")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_CONTENT,
        help="System prompt for the assistant",
    )

    args = parser.parse_args()

    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_parquet(args.input_file)

    if args.max_samples > 0 and args.max_samples < len(df):
        df = df.head(args.max_samples)
        logger.info(f"Processing {args.max_samples} samples")
    else:
        logger.info(f"Processing all {len(df)} samples")

    # Prepare data items
    data_items = []
    for idx, row in df.iterrows():
        item = {
            "question": row.get("question", ""),
            "extra_info": {
                "index": idx,
                "answer": row.get("answer", ""),
                "question": row.get("question", ""),
                "level": row.get("level", ""),
                "data_source": row.get("data_source", "hotpotqa_hard"),
            },
        }
        data_items.append(item)

    # Initialize generator
    generator = DataGenerator(
        vllm_url=args.vllm_url,
        retrieval_url=args.retrieval_url,
        max_assistant_turns=args.max_assistant_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_content=args.system_prompt,
    )

    # Process in batches with progress bar
    results = []
    num_batches = (len(data_items) + args.batch_size - 1) // args.batch_size

    logger.info(f"Generating conversations in {num_batches} batches...")

    for i in range(0, len(data_items), args.batch_size):
        batch = data_items[i : i + args.batch_size]
        batch_results = await generator.generate_batch(batch)
        results.extend(batch_results)

        # Log progress
        logger.info(f"Completed batch {(i // args.batch_size) + 1}/{num_batches}")

    # Filter out failed generations
    successful_results = [r for r in results if r.get("error") is None]
    logger.info(f"Successfully generated {len(successful_results)}/{len(results)} conversations")

    # Save results
    output_df = pd.DataFrame(successful_results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    output_df.to_parquet(args.output_file, index=False)
    logger.info(f"Saved {len(output_df)} conversations to {args.output_file}")

    # Print sample
    if len(successful_results) > 0:
        logger.info("\n" + "=" * 80)
        logger.info("Sample conversation:")
        logger.info("=" * 80)
        sample = successful_results[0]
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg.get("content", "")[:200]
            logger.info(f"\n{role.upper()}: {content}...")
            if "tool_calls" in msg:
                logger.info(f"  [Tool calls: {len(msg['tool_calls'])}]")


if __name__ == "__main__":
    asyncio.run(main())

