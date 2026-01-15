#!/usr/bin/env python3
"""
批量计算 LogProb 并缓存结果。

基于 gen_student.py 生成的多轮对话数据，分别计算学生模型和教师模型
对每个 assistant turn 的 logprob，并保存到独立的 JSONL 文件中。
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from calculate_logprob import DisagreementCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LogProbBatchProcessor:
    """
    批量处理 LogProb 计算的处理器。
    
    处理 gen_student.py 生成的 parquet 数据，为每个 assistant turn 
    计算学生模型和教师模型的 logprob 差异。
    """
    
    def __init__(self, calculator: DisagreementCalculator):
        """
        初始化批量处理器。
        
        Args:
            calculator: 已初始化的 DisagreementCalculator 实例
        """
        self.calculator = calculator
    
    def extract_context_messages(
        self, 
        messages: List[Dict], 
        until_turn: int
    ) -> List[Dict[str, str]]:
        """
        提取到指定 assistant turn 之前的所有消息作为 context。
        
        对于第 N 个 assistant turn，context 包含：
        - system message (使用 content)
        - user message
        - 第 0 到 N-1 个 assistant messages (使用 content)
        - 相应的 tool messages
        
        Args:
            messages: 完整的消息列表
            until_turn: 目标 assistant turn 的索引（0-based）
            
        Returns:
            context 消息列表，仅包含 role 和 content
        """
        context = []
        assistant_count = 0
        
        for msg in messages:
            role = msg.get("role", "")
            
            if role == "assistant":
                if assistant_count >= until_turn:
                    # 达到目标 turn，停止收集 context
                    break
                assistant_count += 1
                # 使用 content 而不是 raw_content（content 不含 tool_call 标签）
                content = msg.get("content", "")
                context.append({"role": "assistant", "content": content})
            else:
                # system, user, tool 消息正常添加
                content = msg.get("content", "")
                context.append({"role": role, "content": content})
        
        return context
    
    def get_assistant_turns_info(
        self, 
        messages: List[Dict]
    ) -> List[Tuple[int, str, int]]:
        """
        获取所有 assistant turn 的信息。
        
        Args:
            messages: 完整的消息列表
            
        Returns:
            列表，每个元素为 (turn_idx, raw_content, message_idx)
            - turn_idx: assistant turn 的索引（0-based）
            - raw_content: 该 turn 的原始内容
            - message_idx: 在 messages 列表中的索引
        """
        assistant_turns = []
        turn_idx = 0
        
        for msg_idx, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                # 优先使用 raw_content，因为它包含完整的模型输出
                raw_content = msg.get("raw_content", msg.get("content", ""))
                assistant_turns.append((turn_idx, raw_content, msg_idx))
                turn_idx += 1
        
        return assistant_turns
    
    def process_single_sample(self, sample: Dict) -> Optional[Dict]:
        """
        处理单个样本，计算所有 assistant turn 的 logprob。
        
        Args:
            sample: 包含 messages 和 extra_info 的样本数据
            
        Returns:
            包含所有 turn logprob 数据的字典，或 None（如果处理失败）
        """
        # 检查是否有错误
        if sample.get("error"):
            logger.warning(f"Sample has error, skipping: {sample.get('error')}")
            return None
        
        messages = sample.get("messages", [])
        extra_info = sample.get("extra_info", {})
        
        # 处理 pandas/numpy 数组的情况
        if hasattr(messages, 'tolist'):
            messages = messages.tolist()
        if hasattr(extra_info, 'tolist'):
            extra_info = extra_info.tolist() if not isinstance(extra_info, dict) else extra_info
        
        if messages is None or len(messages) == 0:
            logger.warning("Sample has no messages, skipping")
            return None
        
        # 获取所有 assistant turns
        assistant_turns = self.get_assistant_turns_info(messages)
        
        if not assistant_turns:
            logger.warning("Sample has no assistant turns, skipping")
            return None
        
        # 处理每个 assistant turn
        turn_logprobs = []
        
        for turn_idx, raw_content, msg_idx in assistant_turns:
            try:
                # 构建 context（该 turn 之前的所有消息）
                context = self.extract_context_messages(messages, turn_idx)
                
                # 计算 disagreement
                score, details = self.calculator.calculate_disagreement(
                    context_messages=context,
                    action_text=raw_content
                )
                
                turn_data = {
                    "turn_idx": turn_idx,
                    "raw_content": raw_content,
                    "disagreement_score": score,
                    **details  # 包含 stu_log_sum, tea_log_sum, stu_tokens, tea_tokens, byte_len, raw_diff
                }
                turn_logprobs.append(turn_data)
                
            except Exception as e:
                logger.error(f"Error processing turn {turn_idx}: {e}")
                # 记录错误但继续处理下一个 turn
                turn_data = {
                    "turn_idx": turn_idx,
                    "raw_content": raw_content,
                    "error": str(e)
                }
                turn_logprobs.append(turn_data)
        
        # 构建结果
        result = {
            "index": extra_info.get("index", -1),
            "question": extra_info.get("question", ""),
            "answer": extra_info.get("answer", ""),
            "data_source": extra_info.get("data_source", ""),
            "assistant_turns": len(assistant_turns),
            "turn_logprobs": turn_logprobs
        }
        
        return result
    
    def load_processed_indices(self, output_file: str) -> Set[int]:
        """
        加载已处理的样本索引（用于断点续传）。
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            已处理的索引集合
        """
        processed = set()
        
        if not os.path.exists(output_file):
            return processed
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        idx = data.get("index", -1)
                        if idx >= 0:
                            processed.add(idx)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Found {len(processed)} already processed samples")
        except Exception as e:
            logger.warning(f"Error reading existing output file: {e}")
        
        return processed
    
    def process_batch(
        self,
        input_file: str,
        output_file: str,
        max_samples: int = -1,
        resume: bool = False,
        save_interval: int = 10
    ) -> Dict:
        """
        批量处理样本。
        
        Args:
            input_file: 输入 parquet 文件路径
            output_file: 输出 JSONL 文件路径
            max_samples: 最大处理样本数（-1 表示全部）
            resume: 是否启用断点续传
            save_interval: 保存间隔（每处理多少样本后 flush 一次）
            
        Returns:
            处理统计信息
        """
        # 加载数据
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)
        total_samples = len(df)
        logger.info(f"Total samples in file: {total_samples}")
        
        # 限制样本数
        if max_samples > 0:
            df = df.head(max_samples)
            logger.info(f"Processing first {max_samples} samples")
        
        # 断点续传
        processed_indices = set()
        if resume:
            processed_indices = self.load_processed_indices(output_file)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        # 统计信息
        stats = {
            "total": len(df),
            "processed": 0,
            "skipped": len(processed_indices),
            "success": 0,
            "failed": 0,
            "total_turns": 0,
            "start_time": time.time()
        }
        
        # 打开输出文件（追加模式）
        mode = 'a' if resume else 'w'
        with open(output_file, mode, encoding='utf-8') as f_out:
            # 处理每个样本
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
                # 获取样本索引
                extra_info_temp = row.get('extra_info', {})
                if hasattr(extra_info_temp, 'tolist'):
                    extra_info_temp = extra_info_temp.tolist()
                if not isinstance(extra_info_temp, dict):
                    extra_info_temp = {}
                sample_index = extra_info_temp.get('index', idx)
                if isinstance(sample_index, dict):
                    sample_index = sample_index.get('index', idx)
                
                # 跳过已处理的样本
                if resume and sample_index in processed_indices:
                    continue
                
                # 构建样本字典，处理 pandas/numpy 类型
                messages_raw = row.get('messages', [])
                extra_info_raw = row.get('extra_info', {})
                error_raw = row.get('error')
                
                # 转换 numpy 数组为 Python 列表
                if hasattr(messages_raw, 'tolist'):
                    messages_raw = messages_raw.tolist()
                if hasattr(extra_info_raw, 'tolist'):
                    extra_info_raw = extra_info_raw.tolist()
                if not isinstance(extra_info_raw, dict):
                    extra_info_raw = {}
                
                sample = {
                    "messages": messages_raw,
                    "extra_info": extra_info_raw,
                    "error": error_raw
                }
                
                # 处理样本
                result = self.process_single_sample(sample)
                stats["processed"] += 1
                
                if result:
                    # 写入结果
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stats["success"] += 1
                    stats["total_turns"] += result.get("assistant_turns", 0)
                    
                    # 定期 flush
                    if stats["processed"] % save_interval == 0:
                        f_out.flush()
                        elapsed = time.time() - stats["start_time"]
                        speed = stats["processed"] / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Progress: {stats['processed']}/{len(df)}, "
                            f"Success: {stats['success']}, "
                            f"Speed: {speed:.2f} samples/s"
                        )
                else:
                    stats["failed"] += 1
        
        # 最终统计
        stats["end_time"] = time.time()
        stats["duration"] = stats["end_time"] - stats["start_time"]
        stats["avg_speed"] = stats["processed"] / stats["duration"] if stats["duration"] > 0 else 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Complete!")
        logger.info(f"  Total samples: {stats['total']}")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Skipped (resume): {stats['skipped']}")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Total turns: {stats['total_turns']}")
        logger.info(f"  Duration: {stats['duration']:.2f}s")
        logger.info(f"  Avg speed: {stats['avg_speed']:.2f} samples/s")
        logger.info(f"{'='*60}\n")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="批量计算 LogProb 并缓存结果"
    )
    
    # 输入输出参数
    parser.add_argument(
        "--input_file",
        type=str,
        default="baseline/preliminary/data/train_student.parquet",
        help="输入 parquet 文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="baseline/preliminary/data/train_student_42step_32B_teacher_logprobs.jsonl",
        help="输出 JSONL 文件路径（默认为输入文件名_logprobs.jsonl）"
    )
    
    # 学生模型参数
    parser.add_argument(
        "--student_base_url",
        type=str,
        default="http://10.244.247.213:8100/v1",
        help="学生模型 vLLM 服务地址"
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="student_model",
        help="学生模型名称"
    )
    parser.add_argument(
        "--student_tokenizer",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="学生模型 tokenizer 路径"
    )
    
    # 教师模型参数
    parser.add_argument(
        "--teacher_base_url",
        type=str,
        default="http://10.244.247.213:8101/v1",
        help="教师模型 vLLM 服务地址"
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="teacher_model",
        help="教师模型名称"
    )
    parser.add_argument(
        "--teacher_tokenizer",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="教师模型 tokenizer 路径"
    )
    
    # 处理参数
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="最大处理样本数（-1 表示全部）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="启用断点续传"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="保存间隔（每处理多少样本后 flush）"
    )
    
    args = parser.parse_args()
    
    # 设置默认输出文件名
    if not args.output_file:
        input_basename = os.path.splitext(args.input_file)[0]
        args.output_file = f"{input_basename}_logprobs.jsonl"
    
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    
    # 初始化计算器
    logger.info("Initializing DisagreementCalculator...")
    calculator = DisagreementCalculator(
        student_base_url=args.student_base_url,
        student_model=args.student_model,
        student_tokenizer_path=args.student_tokenizer,
        teacher_base_url=args.teacher_base_url,
        teacher_model=args.teacher_model,
        teacher_tokenizer_path=args.teacher_tokenizer
    )
    
    # 初始化处理器
    processor = LogProbBatchProcessor(calculator)
    
    # 执行批量处理
    stats = processor.process_batch(
        input_file=args.input_file,
        output_file=args.output_file,
        max_samples=args.max_samples,
        resume=args.resume,
        save_interval=args.save_interval
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
