# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import argparse
import logging
import os

import pandas as pd
from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = (
    "Role:\nYou are an information-seeking assistant. Your core function is to retrieve information from the knowledge base to answer the question.\n\nBehavior:\n- For every request, synthesize information from credible and diverse sources to find an answer.\n- Always think before taking action. Use <thinking></thinking> tags to show your thinking process before calling tools or providing the final answer.\n- Once you have gathered sufficient information, you must provide the final answer enclosed within <answer></answer> tags at the end of your message (e.g., a person's name, a date, a location, a number, etc.).\n\n"
)
DEFAULT_USER_CONTENT_PREFIX = "Question:\n"


def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.
    Only uses question, answer, and level fields from the dataset.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    # Only use the three available fields: question, answer, level
    question = row.get("question", "")
    answer = row.get("answer", "")
    level = row.get("level", "")

    # Build prompt structure
    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    # Extract ground truth from answer column
    ground_truth = answer

    # Use a fixed data source tag based on the dataset
    data_source_tagged = "searchR1_hotpotqa_hard"

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "split": current_split_name,
        "index": row_index,
        "answer": answer,
        "question": question,
        "need_tools_kwargs": True,
        "tools_kwargs": tools_kwargs,
        "interaction_kwargs": {
            "query": user_content,
            "ground_truth": ground_truth,
        },
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "agent_name": "tool_agent",
            "prompt": prompt,
            "ability": "search",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": extra_info,
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    # Only process train split
    split = "validation"
    logger.info(f"Processing {split} split...")

    try:
        # Load dataset from HuggingFace using datasets library
        # This automatically handles sharded parquet files
        logger.info(f"Loading {split} split from {args.hf_repo_id} with config '{args.config}'")
        dataset = load_dataset(args.hf_repo_id, name=args.config, split=split)
        logger.info(f"Loaded {len(dataset)} rows from {split} split")

        # Convert to pandas DataFrame for easier processing
        df_raw = dataset.to_pandas()
        
        # Filter only level == "hard" rows
        # if "level" in df_raw.columns:
        #     df_raw = df_raw[df_raw["level"] == "hard"]
        #     logger.info(f"Filtered to {len(df_raw)} rows with level=='hard'")
        # else:
        #     logger.warning("'level' column not found in dataset, skipping filter")

        def apply_process_row(row, split_name=split):
            return process_single_row(row, current_split_name=split_name, row_index=row.name)

        df_raw = df_raw.sample(n=200, random_state=42)
        df_processed = df_raw.apply(apply_process_row, axis=1)

        # Save processed DataFrame
        output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
        df_processed.to_parquet(output_file_path, index=False)
        logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
        processed_files.append(output_file_path)

    except Exception as e:
        logger.error(f"Error processing {split} split: {e}")
        raise

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HotpotQA-Hard from HuggingFace, process, and save to Parquet.")
    parser.add_argument(
        "--hf_repo_id", default="hotpotqa/hotpot_qa", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--config",
        default="distractor",
        choices=["distractor", "fullwiki"],
        help="Dataset configuration name. Available: 'distractor' or 'fullwiki'. Default: 'distractor'.",
    )
    parser.add_argument(
        "--local_dir",
        default="data/hotpotqa_hard_train",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()

    # System and user content configuration
    system_content = DEFAULT_SYSTEM_CONTENT
    user_content_prefix = DEFAULT_USER_CONTENT_PREFIX

    main()
