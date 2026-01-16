#!/usr/bin/env python3
"""
从两个 LogProb 文件计算不一致分数。

读取学生模型和教师模型的 logprob JSONL 文件，计算不一致分数，
并保存为与原始 batch_calculate_logprob.py 相同格式的输出文件。
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Set

from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DisagreementFromFilesCalculator:
    """
    从两个 LogProb 文件计算不一致分数的计算器。
    """
    
    def __init__(self, student_file: str, teacher_file: str):
        """
        初始化计算器。
        
        Args:
            student_file: 学生模型 logprob JSONL 文件路径
            teacher_file: 教师模型 logprob JSONL 文件路径
        """
        self.student_file = student_file
        self.teacher_file = teacher_file
    
    def load_logprob_data(self, file_path: str) -> Dict[int, Dict]:
        """
        加载 logprob JSONL 文件，按 index 索引。
        
        Args:
            file_path: JSONL 文件路径
            
        Returns:
            字典，key 为样本 index，value 为样本数据
        """
        data_dict = {}
        logger.info(f"Loading logprob data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    idx = data.get("index", -1)
                    if idx >= 0:
                        data_dict[idx] = data
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(data_dict)} samples from {file_path}")
        return data_dict
    
    def calculate_disagreement_for_sample(
        self,
        student_data: Dict,
        teacher_data: Dict
    ) -> Optional[Dict]:
        """
        计算单个样本的不一致分数。
        
        Args:
            student_data: 学生模型的 logprob 数据
            teacher_data: 教师模型的 logprob 数据
            
        Returns:
            包含不一致分数的结果字典，或 None（如果计算失败）
        """
        # 检查 turn_logprobs 是否存在
        stu_turns = student_data.get("turn_logprobs", [])
        tea_turns = teacher_data.get("turn_logprobs", [])
        
        if not stu_turns or not tea_turns:
            logger.warning(f"Sample {student_data.get('index', -1)}: missing turn_logprobs")
            return None
        
        if len(stu_turns) != len(tea_turns):
            logger.warning(
                f"Sample {student_data.get('index', -1)}: "
                f"turn count mismatch (student: {len(stu_turns)}, teacher: {len(tea_turns)})"
            )
            return None
        
        # 处理每个 turn
        turn_logprobs = []
        
        for turn_idx, (stu_turn, tea_turn) in enumerate(zip(stu_turns, tea_turns)):
            try:
                # 检查是否有错误
                if "error" in stu_turn or "error" in tea_turn:
                    logger.warning(
                        f"Sample {student_data.get('index', -1)}, turn {turn_idx}: "
                        f"has error in logprob data"
                    )
                    turn_data = {
                        "turn_idx": turn_idx,
                        "raw_content": stu_turn.get("raw_content", tea_turn.get("raw_content", "")),
                        "error": "Error in logprob calculation"
                    }
                    turn_logprobs.append(turn_data)
                    continue
                
                # 获取 logprob 数据
                stu_log_sum = stu_turn.get("log_sum", 0.0)
                tea_log_sum = tea_turn.get("log_sum", 0.0)
                stu_tokens = stu_turn.get("token_count", 0)
                tea_tokens = tea_turn.get("token_count", 0)
                
                # 获取 raw_content（优先使用学生模型的）
                raw_content = stu_turn.get("raw_content", tea_turn.get("raw_content", ""))
                
                # 获取字节长度（优先使用学生模型的）
                byte_length = stu_turn.get("byte_len", tea_turn.get("byte_len", 0))
                
                # 极短文本平滑，防止除以极小数
                if byte_length < 2:
                    turn_data = {
                        "turn_idx": turn_idx,
                        "raw_content": raw_content,
                        "disagreement_score": 0.0,
                        "stu_log_sum": stu_log_sum,
                        "tea_log_sum": tea_log_sum,
                        "stu_tokens": stu_tokens,
                        "tea_tokens": tea_tokens,
                        "byte_len": byte_length,
                        "raw_diff": 0.0
                    }
                    turn_logprobs.append(turn_data)
                    continue
                
                # 计算差异
                # Diff = Student (High Prob, near 0) - Teacher (Low Prob, very negative)
                # Example: -2.0 - (-10.0) = 8.0 (High Disagreement)
                raw_diff = stu_log_sum - tea_log_sum
                
                # 字节级归一化 (Bits per Byte Gap)
                normalized_score = raw_diff / byte_length
                
                turn_data = {
                    "turn_idx": turn_idx,
                    "raw_content": raw_content,
                    "disagreement_score": normalized_score,
                    "stu_log_sum": stu_log_sum,
                    "tea_log_sum": tea_log_sum,
                    "stu_tokens": stu_tokens,
                    "tea_tokens": tea_tokens,
                    "byte_len": byte_length,
                    "raw_diff": raw_diff
                }
                turn_logprobs.append(turn_data)
                
            except Exception as e:
                logger.error(
                    f"Error processing sample {student_data.get('index', -1)}, "
                    f"turn {turn_idx}: {e}"
                )
                turn_data = {
                    "turn_idx": turn_idx,
                    "raw_content": stu_turn.get("raw_content", ""),
                    "error": str(e)
                }
                turn_logprobs.append(turn_data)
        
        # 构建结果（使用学生模型的数据作为基础）
        result = {
            "index": student_data.get("index", -1),
            "question": student_data.get("question", ""),
            "answer": student_data.get("answer", ""),
            "data_source": student_data.get("data_source", ""),
            "assistant_turns": len(turn_logprobs),
            "turn_logprobs": turn_logprobs
        }
        
        return result
    
    def process_files(
        self,
        output_file: str,
        max_samples: int = -1
    ) -> Dict:
        """
        处理两个 logprob 文件，计算不一致分数。
        
        Args:
            output_file: 输出 JSONL 文件路径
            max_samples: 最大处理样本数（-1 表示全部）
            
        Returns:
            处理统计信息
        """
        # 加载数据
        logger.info("Loading student logprob data...")
        student_data_dict = self.load_logprob_data(self.student_file)
        
        logger.info("Loading teacher logprob data...")
        teacher_data_dict = self.load_logprob_data(self.teacher_file)
        
        # 找到共同的索引
        common_indices = set(student_data_dict.keys()) & set(teacher_data_dict.keys())
        logger.info(f"Found {len(common_indices)} common samples")
        
        if len(common_indices) == 0:
            logger.error("No common samples found between student and teacher files!")
            return {
                "total": 0,
                "processed": 0,
                "success": 0,
                "failed": 0,
                "total_turns": 0
            }
        
        # 限制样本数
        if max_samples > 0:
            common_indices = sorted(common_indices)[:max_samples]
            logger.info(f"Processing first {max_samples} samples")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        # 统计信息
        stats = {
            "total": len(common_indices),
            "processed": 0,
            "success": 0,
            "failed": 0,
            "total_turns": 0
        }
        
        # 处理并写入结果
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for idx in tqdm(sorted(common_indices), desc="Calculating disagreement"):
                student_data = student_data_dict[idx]
                teacher_data = teacher_data_dict[idx]
                
                # 计算不一致分数
                result = self.calculate_disagreement_for_sample(student_data, teacher_data)
                stats["processed"] += 1
                
                if result:
                    # 写入结果
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stats["success"] += 1
                    stats["total_turns"] += result.get("assistant_turns", 0)
                else:
                    stats["failed"] += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Complete!")
        logger.info(f"  Total samples: {stats['total']}")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Total turns: {stats['total_turns']}")
        logger.info(f"{'='*60}\n")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="从两个 LogProb 文件计算不一致分数"
    )
    
    # 输入输出参数
    parser.add_argument(
        "--student_file",
        type=str,
        required=True,
        help="学生模型 logprob JSONL 文件路径"
    )
    parser.add_argument(
        "--teacher_file",
        type=str,
        required=True,
        help="教师模型 logprob JSONL 文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="输出 JSONL 文件路径（默认为 student_file 的目录/disagreement.jsonl）"
    )
    
    # 处理参数
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="最大处理样本数（-1 表示全部）"
    )
    
    args = parser.parse_args()
    
    # 设置默认输出文件名
    if not args.output_file:
        student_dir = os.path.dirname(args.student_file) or "."
        args.output_file = os.path.join(student_dir, "disagreement.jsonl")
    
    logger.info(f"Student file: {args.student_file}")
    logger.info(f"Teacher file: {args.teacher_file}")
    logger.info(f"Output file: {args.output_file}")
    
    # 初始化计算器
    calculator = DisagreementFromFilesCalculator(
        student_file=args.student_file,
        teacher_file=args.teacher_file
    )
    
    # 执行处理
    stats = calculator.process_files(
        output_file=args.output_file,
        max_samples=args.max_samples
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
