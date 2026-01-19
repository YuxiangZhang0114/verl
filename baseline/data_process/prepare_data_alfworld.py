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
"""
从 HuggingFace 下载 ALFWorld 数据并转换为 VERL 训练格式。

数据来源: awawa-agi/alfworld-raw
- 训练集: 3,553 个任务
- 验证集 (in-distribution): 140 个任务
- 测试集 (out-of-distribution): 134 个任务

任务类型:
- pick_two_obj_and_place: 813
- pick_and_place_simple: 790
- pick_clean_then_place_in_recep: 650
- pick_cool_then_place_in_recep: 533
- pick_heat_then_place_in_recep: 459
- look_at_obj_in_light: 308
"""

import argparse
import json
import logging
import os
import re

import pandas as pd
from datasets import load_dataset

# Optional HDFS support - only import if available
try:
    from verl.utils.hdfs_io import copy as hdfs_copy, makedirs as hdfs_makedirs
    HAS_HDFS = True
except ImportError:
    HAS_HDFS = False
    hdfs_copy = None
    hdfs_makedirs = None

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ALFWorld 工具列表
ALFWORLD_TOOLS = [
    "goto",
    "take",
    "put",
    "open",
    "close",
    "toggle",
    "clean",
    "heat",
    "cool",
    "use",
    "look",
    "inventory",
    "examine",
]

# 任务类型到自然语言描述的映射
TASK_TYPE_DESCRIPTIONS = {
    "pick_and_place_simple": "pick up an object and place it in/on a receptacle",
    "pick_two_obj_and_place": "pick up two objects and place them in/on receptacles",
    "pick_clean_then_place_in_recep": "pick up an object, clean it, then place it in/on a receptacle",
    "pick_heat_then_place_in_recep": "pick up an object, heat it, then place it in/on a receptacle",
    "pick_cool_then_place_in_recep": "pick up an object, cool it, then place it in/on a receptacle",
    "look_at_obj_in_light": "examine an object under a light source",
}

# System prompt for embodied AI agent
DEFAULT_SYSTEM_CONTENT = """You are an embodied AI agent operating in a household environment. Your task is to complete household tasks by interacting with objects and receptacles in the environment.

Behavior:
- Think step by step about what actions are needed to complete the task.
- Use the available tools to interact with the environment one action at a time.
- After each action, observe the result and plan your next move.
- Continue until you have completed the task successfully.

Important constraints:
- You can only hold ONE object at a time. Put down any held object before picking up another.
- Some receptacles (like fridge, cabinet, drawer) need to be opened before you can see or take objects inside.
- To clean an object, use a sink or basin. To heat an object, use a microwave. To cool an object, use a fridge.
- When looking for objects, check likely locations based on the object type."""


def extract_goal_from_pddl(pddl_problem: str) -> str:
    """
    从 PDDL problem 中提取 goal 描述。
    
    PDDL goal 格式通常为:
    (:goal (and
        (exists (?a - agent ?o - object ?r - receptacle)
            (and (on ?o ?r) (heated ?o))
        )
    ))
    
    Args:
        pddl_problem: PDDL problem 字符串
        
    Returns:
        提取的 goal 字符串（原始 LISP 格式）
    """
    # 使用正则表达式匹配 :goal 部分
    goal_match = re.search(r'\(:goal\s+(.*?)\)\s*(?:\(:|\Z)', pddl_problem, re.DOTALL)
    if goal_match:
        return goal_match.group(1).strip()
    return ""


def pddl_goal_to_natural_language(pddl_goal: str, task_type: str) -> str:
    """
    将 PDDL goal 转换为自然语言描述。
    
    Args:
        pddl_goal: PDDL goal 字符串
        task_type: 任务类型
        
    Returns:
        自然语言描述的 goal
    """
    # 如果有任务类型描述，使用它
    if task_type in TASK_TYPE_DESCRIPTIONS:
        return f"Your task is to {TASK_TYPE_DESCRIPTIONS[task_type]}."
    
    # 否则返回一个通用描述
    return "Complete the household task as described."


def parse_walkthrough_to_actions(walkthrough: list) -> list:
    """
    解析 walkthrough 为标准动作列表。
    
    Args:
        walkthrough: 专家轨迹列表，如 ["go to fridge 1", "open fridge 1", ...]
        
    Returns:
        解析后的动作列表
    """
    actions = []
    for step in walkthrough:
        step = step.strip().lower()
        actions.append(step)
    return actions


def process_single_row(row, game_content: dict, split_name: str, row_index: int):
    """
    处理单行数据为 VERL 训练格式。
    
    Args:
        row: 原始数据行
        game_content: 解析后的 game_content 字典
        split_name: 数据集划分名称 (train/eval_id/eval_ood)
        row_index: 行索引
        
    Returns:
        pd.Series: 处理后的数据行
    """
    task_id = row["id"]
    task_type = row["task_type"]
    
    # 提取 goal
    pddl_goal = extract_goal_from_pddl(game_content.get("pddl_problem", ""))
    natural_goal = pddl_goal_to_natural_language(pddl_goal, task_type)
    
    # 解析专家轨迹
    walkthrough = game_content.get("walkthrough", [])
    expert_actions = parse_walkthrough_to_actions(walkthrough)
    
    # 构建 prompt
    # Note: 初始观察将在环境创建时由 Tool 提供
    user_content = f"""Task: {natural_goal}

The environment will provide you with an initial observation. Use the available tools to complete the task step by step.

Think about what you need to do and start executing actions."""

    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": user_content},
    ]
    
    # 构建 tools_kwargs - 所有工具共享同一个 task_id 和任务信息
    game_file_path = row.get("game_file_path", "")
    tools_kwargs = {
        tool_name: {
            "create_kwargs": {
                "task_id": task_id,
                "task_type": task_type,
                "walkthrough": walkthrough,  # 用于模拟环境判断成功
                "game_file_path": game_file_path,  # 用于加载特定游戏
            }
        }
        for tool_name in ALFWORLD_TOOLS
    }
    
    # Ground truth 用于奖励计算
    ground_truth = {
        "task_type": task_type,
        "walkthrough": walkthrough,
        "expert_actions": expert_actions,
        "solvable": game_content.get("solvable", True),
        "num_expert_steps": len(expert_actions),
    }
    
    # 构建 extra_info
    # Note: interaction_kwargs 需要有至少一个字段，否则 parquet 无法保存空 struct
    extra_info = {
        "task_id": task_id,
        "task_type": task_type,
        "split": split_name,
        "index": row_index,
        "need_tools_kwargs": True,
        "tools_kwargs": tools_kwargs,
        "interaction_kwargs": {"_placeholder": True},  # 不使用 interaction，但需要非空
        "game_file_path": row.get("game_file_path", ""),
        "pddl_goal": pddl_goal,
        "walkthrough": walkthrough,
    }
    
    return pd.Series({
        "data_source": f"alfworld_{split_name}",
        "agent_name": "tool_agent",
        "prompt": prompt,
        "ability": "embodied_ai",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": extra_info,
    })


def process_split(dataset, split_name: str, output_dir: str):
    """
    处理数据集的一个划分。
    
    Args:
        dataset: HuggingFace 数据集
        split_name: 划分名称
        output_dir: 输出目录
        
    Returns:
        str: 输出文件路径
    """
    logger.info(f"Processing {split_name} split with {len(dataset)} samples...")
    
    processed_rows = []
    for idx, item in enumerate(dataset):
        try:
            game_content = json.loads(item["game_content"])
            processed_row = process_single_row(item, game_content, split_name, idx)
            processed_rows.append(processed_row)
        except Exception as e:
            logger.warning(f"Error processing row {idx} in {split_name}: {e}")
            continue
    
    df_processed = pd.DataFrame(processed_rows)
    
    # 保存
    output_file = os.path.join(output_dir, f"{split_name}.parquet")
    df_processed.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(df_processed)} processed rows to {output_file}")
    
    return output_file


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    processed_files = []
    
    # 加载数据集
    logger.info(f"Loading dataset from {args.hf_repo_id}...")
    
    # 定义 split 映射
    split_mapping = {
        "train": "train",
        "eval_in_distribution": "eval_id",
        "eval_out_of_distribution": "eval_ood",
    }
    
    for hf_split, output_name in split_mapping.items():
        try:
            logger.info(f"Loading {hf_split} split...")
            dataset = load_dataset(args.hf_repo_id, split=hf_split)
            logger.info(f"Loaded {len(dataset)} samples from {hf_split}")
            
            # 处理数据
            output_file = process_split(dataset, output_name, local_save_dir)
            processed_files.append(output_file)
            
        except Exception as e:
            logger.error(f"Error processing {hf_split}: {e}")
            if hf_split == "train":
                raise  # train split 是必须的
    
    if not processed_files:
        logger.warning("No data was processed or saved")
        return
    
    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")
    
    # 打印统计信息
    print("\n=== Dataset Statistics ===")
    for file_path in processed_files:
        df = pd.read_parquet(file_path)
        split_name = os.path.basename(file_path).replace(".parquet", "")
        print(f"{split_name}: {len(df)} samples")
        
        # 打印任务类型分布
        task_types = df["extra_info"].apply(lambda x: x["task_type"]).value_counts()
        print(f"  Task type distribution:")
        for tt, count in task_types.items():
            print(f"    {tt}: {count}")
    
    # 复制到 HDFS（如果指定）
    if args.hdfs_dir:
        if not HAS_HDFS:
            logger.warning("HDFS support not available (verl not installed). Skipping HDFS copy.")
        else:
            try:
                hdfs_makedirs(args.hdfs_dir)
                hdfs_copy(src=local_save_dir, dst=args.hdfs_dir)
                logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
            except Exception as e:
                logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ALFWorld data from HuggingFace, process, and save to Parquet."
    )
    parser.add_argument(
        "--hf_repo_id",
        default="awawa-agi/alfworld-raw",
        help="HuggingFace dataset repository ID.",
    )
    parser.add_argument(
        "--local_dir",
        default="data/alfworld_train",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the Parquet files to.",
    )
    
    args = parser.parse_args()
    main()
