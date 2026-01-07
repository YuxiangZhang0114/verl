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
DEFAULT_SYSTEM_CONTENT = """Role:
You are an information-seeking assistant. Your core function is to retrieve information from the knowledge base to answer the question.

Behavior:
- For every request, reason step by step and use tools to synthesize information from credible and diverse sources to find an answer.
- Always think before taking action. Use <thinking></thinking> tags to show your thinking process before calling tools.
- Once you have gathered sufficient information, provide the final answer within <answer>Answer</answer> tags at the end of your message (e.g., a person's name, a date, a location, a number, etc.).

"""
DEFAULT_USER_CONTENT_PREFIX = "Question:\n"


def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for ASearcher format.
    Uses question, aug_answer fields from the dataset.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    # Extract fields from the dataset
    question = row.get("question", "")
    answer = row.get("answer", [])
    aug_answer = row.get("aug_answer", [])  # aug_answer is a list field (in ASearcherBase35k)
    source = row.get("source", "")
    
    # Different configs have different id fields
    qid = row.get("qid", None)  # ASearcherBase35k uses qid
    if qid is None:
        # ASearcherLRM35k uses id field instead
        qid = row.get("id", None)
    
    # Convert numpy arrays to Python lists for proper serialization
    if hasattr(answer, 'tolist'):
        answer = answer.tolist()
    if hasattr(aug_answer, 'tolist'):
        aug_answer = aug_answer.tolist()

    # Build prompt structure
    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    # Use aug_answer as ground truth (it's already a list with more answer variants)
    # Format ground_truth as dict with "target" key for ASearcher reward function
    # Note: both answer and aug_answer are lists in the ASearcher-Base dataset
    if isinstance(aug_answer, list) and len(aug_answer) > 0:
        ground_truth = {"target": aug_answer}
    elif isinstance(answer, list) and len(answer) > 0:
        # Fallback to answer field if aug_answer is empty
        ground_truth = {"target": answer}
    else:
        # Edge case: if both are empty or not lists
        ground_truth = {"target": [str(answer) if answer else ""]}

    # Use ASearcher as data source tag to trigger compute_score_subem
    data_source_tagged = "ASearcher"

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
        "aug_answer": aug_answer,
        "question": question,
        "qid": qid,
        "source": source,
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

    # Process dataset
    split = "train"  # Use 'train' as the output split name
    logger.info(f"Processing dataset from {args.data_file}...")

    try:
        # Load dataset from HuggingFace using datasets library
        # Use data_files to load only the specific file since the two files have different schemas
        logger.info(f"Loading {args.data_file} from {args.hf_repo_id}")
        dataset = load_dataset(args.hf_repo_id, data_files=args.data_file, split="train")
        logger.info(f"Loaded {len(dataset)} rows from dataset")

        # Convert to pandas DataFrame for easier processing
        df_raw = dataset.to_pandas()


        # 过滤无效数据 - 更严格的过滤策略（宁可错杀，不能放过）
        invalid_keywords = [
            # 问题本身有问题
            "invalid", "question", "flawed", "erroneous", "misleading", 
            "inaccurate", "incorrect", "wrong", "not valid", "not accurate",
            "contains error", "based on error", "based on false",
            
            # 答案不存在
            "there is no", "no such", "does not exist", "not known",
            "not exist", "doesn't exist", "do not exist", "don't exist",
            "not found", "no record", "no information", "no data",
            
            # 未知/不可用
            "unknown", "not available", "unavailable", 
            "not specified", "unspecified",
            
            # 无法确定/回答
            "cannot be determined", "unable to", "cannot answer",
            "impossible to", "unclear", "ambiguous",
            "insufficient", "not enough", "lack of",
            "not determined", "cannot verify", "unverified",
            "not confirmed", "no evidence", "no proof",
            "not clear", "uncertain", "disputed", "debatable",
            
            # 假设/错误信息
            "hypothetical", "based on incorrect", "based on wrong",
            "assuming", "if we assume", "could be", "might be",
            
            # 其他无效标记
            "the same", "equal", "no difference", "identical",
            "unanswerable", "no answer", "no valid"
        ]
        
        def is_invalid_entry(row):
            """
            严格检查一个数据条目是否包含无效关键词
            检查 aug_answer, answer, question 字段（不区分大小写）
            """
            # 检查 aug_answer（主要字段）
            aug_answer = row.get('aug_answer', [])
            if isinstance(aug_answer, list):
                for ans in aug_answer:
                    ans_lower = str(ans).lower()
                    for keyword in invalid_keywords:
                        if keyword.lower() in ans_lower:
                            return True
            else:
                # 如果不是列表，转为字符串检查
                ans_lower = str(aug_answer).lower()
                for keyword in invalid_keywords:
                    if keyword.lower() in ans_lower:
                        return True
            
            # 也检查 answer 字段
            answer = row.get('answer', [])
            if isinstance(answer, list):
                for ans in answer:
                    ans_lower = str(ans).lower()
                    for keyword in invalid_keywords:
                        if keyword.lower() in ans_lower:
                            return True
            else:
                ans_lower = str(answer).lower()
                for keyword in invalid_keywords:
                    if keyword.lower() in ans_lower:
                        return True
            
            # 可选：也检查 question 字段（如果问题本身包含这些词可能也有问题）
            # question = str(row.get('question', '')).lower()
            # for keyword in ["invalid", "incorrect", "wrong", "error"]:
            #     if keyword in question:
            #         return True
            
            return False
        
        print("before filtering: ", len(df_raw))
        df_raw = df_raw[~df_raw.apply(is_invalid_entry, axis=1)]
        print("after filtering: ", len(df_raw))
        # Apply sampling if specified
        if args.sample_size and args.sample_size > 0:
            if len(df_raw) > args.sample_size:
                df_raw = df_raw.sample(n=args.sample_size, random_state=args.random_state)
                logger.info(f"Sampled {len(df_raw)} rows from the dataset")

        def apply_process_row(row, split_name=split):
            return process_single_row(row, current_split_name=split_name, row_index=row.name)

        df_processed = df_raw.apply(apply_process_row, axis=1)

        # Save processed DataFrame
        output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
        df_processed.to_parquet(output_file_path, index=False)
        logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
        processed_files.append(output_file_path)

    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
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
    parser = argparse.ArgumentParser(description="Download ASearcher-train-data from HuggingFace, process, and save to Parquet.")
    parser.add_argument(
        "--hf_repo_id", default="inclusionAI/ASearcher-train-data", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--data_file",
        default="ASearcher-Base-35k.jsonl",
        help="Data file to load. Available: 'ASearcher-Base-35k.jsonl' (with aug_answer, 35k examples) or 'ASearcher-LRM-35k.jsonl' (with id/idx). Default: 'ASearcher-Base-35k.jsonl'.",
    )
    parser.add_argument(
        "--local_dir",
        default="data/asearcher_train",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional: number of samples to randomly select from the dataset. If None, uses all data.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for sampling. Default: 42.",
    )

    args = parser.parse_args()

    # System and user content configuration
    system_content = DEFAULT_SYSTEM_CONTENT
    user_content_prefix = DEFAULT_USER_CONTENT_PREFIX

    main()

