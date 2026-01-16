#!/usr/bin/env python3
"""
计算单个模型的 LogProb 并保存为文件。

基于 gen_student.py 生成的多轮对话数据，计算指定模型对每个 assistant turn 的 logprob，
并保存到 JSONL 文件中。
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import openai
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SingleModelLogProbCalculator:
    """
    单个模型的 LogProb 计算器。
    """
    
    def __init__(
        self,
        base_url: str,
        model: str,
        tokenizer_path: str,
        api_key: str = "EMPTY"
    ):
        """
        初始化计算器。
        
        Args:
            base_url: vLLM 服务地址，如 "http://localhost:8000/v1"
            model: 模型在 vLLM 中的名称
            tokenizer_path: tokenizer 路径（用于 chat template 和特殊 token）
            api_key: API 密钥，vLLM 默认使用 "EMPTY"
        """
        logger.info(f"Connecting to vLLM: {base_url} (model: {model})")
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        
        logger.info(f"Loading Tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        logger.info(f"  Special tokens: {self.special_tokens}")
    
    def get_action_log_prob_sum(
        self,
        context_messages: List[Dict[str, str]],
        action_text: str
    ) -> Tuple[float, int]:
        """
        计算特定动作文本在给定上下文下的 LogProb 总和。
        
        Args:
            context_messages: 上下文消息列表
            action_text: 需要计算 logprob 的动作文本
            
        Returns:
            (log_prob_sum, token_count): logprob 总和和有效 token 数量
        """
        # 1. 构建 Context 文本并计算 token 数量
        context_str = self.tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        context_token_count = len(self.tokenizer.encode(context_str, add_special_tokens=False))
        
        # 2. 构建完整文本
        full_text = context_str + action_text
        
        # 3. 调用 vLLM API 获取 logprobs
        response = self.client.completions.create(
            model=self.model,
            prompt=full_text,
            max_tokens=0,       # 不生成新 token
            echo=True,          # 返回输入 token 的 logprob
            logprobs=1,         # 返回 top-1 logprob
            temperature=0
        )
        
        # 4. 提取 logprobs 数据
        logprobs_data = response.choices[0].logprobs
        tokens = logprobs_data.tokens           # token 文本列表
        token_logprobs = logprobs_data.token_logprobs  # 每个 token 的 logprob
        
        # 5. 计算 action 部分的 logprob 总和（跳过 context 和特殊 token）
        total_logprob = 0.0
        valid_token_count = 0
        
        for i in range(context_token_count, len(tokens)):
            token_text = tokens[i]
            logprob = token_logprobs[i]
            
            # 跳过特殊 token
            if token_text in self.special_tokens:
                continue
            
            # 跳过 None（通常是首 token）
            if logprob is None:
                continue
                
            total_logprob += logprob
            valid_token_count += 1
        
        return total_logprob, valid_token_count


class SingleModelLogProbProcessor:
    """
    批量处理单个模型 LogProb 计算的处理器。
    """
    
    def __init__(self, calculator: SingleModelLogProbCalculator):
        """
        初始化批量处理器。
        
        Args:
            calculator: 已初始化的 SingleModelLogProbCalculator 实例
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
                
                # 计算 logprob
                log_sum, token_count = self.calculator.get_action_log_prob_sum(
                    context_messages=context,
                    action_text=raw_content
                )
                
                # 计算字节长度
                byte_length = len(raw_content.encode('utf-8'))
                
                turn_data = {
                    "turn_idx": turn_idx,
                    "raw_content": raw_content,
                    "log_sum": log_sum,
                    "token_count": token_count,
                    "byte_len": byte_length
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
        description="计算单个模型的 LogProb 并保存为文件"
    )
    
    # 输入输出参数
    parser.add_argument(
        "--input_file",
        type=str,
        default="baseline/preliminary/data/train_student_4epoch.parquet",
        help="输入 parquet 文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="输出 JSONL 文件路径（默认为输入文件名_<model_name>_logprobs.jsonl）"
    )
    
    # 模型参数
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://10.244.247.213:8100/v1",
        help="模型 vLLM 服务地址"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="student_model",
        help="模型名称"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="模型 tokenizer 路径"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API 密钥"
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
        default=100,
        help="保存间隔（每处理多少样本后 flush）"
    )
    
    args = parser.parse_args()
    
    # 设置默认输出文件名
    if not args.output_file:
        input_basename = os.path.splitext(args.input_file)[0]
        model_name_safe = args.model.replace("/", "_")
        args.output_file = f"{input_basename}_{model_name_safe}_logprobs.jsonl"
    
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    
    # 初始化计算器
    logger.info("Initializing SingleModelLogProbCalculator...")
    calculator = SingleModelLogProbCalculator(
        base_url=args.base_url,
        model=args.model,
        tokenizer_path=args.tokenizer,
        api_key=args.api_key
    )
    
    # 初始化处理器
    processor = SingleModelLogProbProcessor(calculator)
    
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
