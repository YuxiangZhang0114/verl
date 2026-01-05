#!/bin/bash
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

# Bash launcher for data distillation with search tool calls
# Usage: bash baseline/DataKD/run_generate_search_data.sh

set -e

# ============================================================================
# Configuration - Modify these parameters as needed
# ============================================================================

# vLLM server endpoint (OpenAI-compatible API)
VLLM_URL="${VLLM_URL:-http://10.244.50.63:8000/v1}"

# Retrieval server endpoint for search execution
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"

# Maximum assistant turns per conversation
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-10}"

# Number of samples to process (-1 for all)
MAX_SAMPLES="${MAX_SAMPLES:--3000}"

# Input parquet file with HotpotQA data
INPUT_FILE="${INPUT_FILE:-data/hotpotqa_hard_train/train.parquet}"

# Output parquet file path
OUTPUT_FILE="${OUTPUT_FILE:-baseline/DataKD/train_distilled.parquet}"

# Batch size for async processing
BATCH_SIZE="${BATCH_SIZE:-16}"

# Sampling temperature
TEMPERATURE="${TEMPERATURE:-0.7}"

# Max tokens per generation
MAX_TOKENS="${MAX_TOKENS:-2048}"

# System prompt (optional override)
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.}"

# ============================================================================
# Run the data generation script
# ============================================================================

echo "=========================================="
echo "Data Distillation - Search Tool Generation"
echo "=========================================="
echo "vLLM URL: $VLLM_URL"
echo "Retrieval URL: $RETRIEVAL_URL"
echo "Max Assistant Turns: $MAX_ASSISTANT_TURNS"
echo "Max Samples: $MAX_SAMPLES"
echo "Input File: $INPUT_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Batch Size: $BATCH_SIZE"
echo "Temperature: $TEMPERATURE"
echo "Max Tokens: $MAX_TOKENS"
echo "=========================================="

python3 baseline/DataKD/generate_search_data.py \
    --vllm_url "$VLLM_URL" \
    --retrieval_url "$RETRIEVAL_URL" \
    --max_assistant_turns "$MAX_ASSISTANT_TURNS" \
    --max_samples "$MAX_SAMPLES" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --batch_size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --system_prompt "$SYSTEM_PROMPT"

echo "=========================================="
echo "Data generation completed!"
echo "Output saved to: $OUTPUT_FILE"
echo "=========================================="

