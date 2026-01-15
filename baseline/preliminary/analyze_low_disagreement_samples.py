#!/usr/bin/env python3
"""
分析 logprobs JSONL 文件，找出所有存在不一致分数大于阈值的step的样本。

对于每个样本，检查其所有 assistant turns 的 disagreement_score，
如果至少有一个 turn 的 disagreement_score > 阈值，则该样本符合条件（保留）。
"""

import argparse
import json
import logging
from typing import Dict, List

from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_samples(
    input_file: str,
    threshold: float = 0.2,
    output_file: str = None
) -> Dict:
    """
    分析样本，找出所有存在不一致分数大于阈值的step的样本（保留这些样本）。
    
    Args:
        input_file: 输入的 JSONL 文件路径
        threshold: 不一致分数阈值（默认 0.2）
        output_file: 输出文件路径（可选，如果提供则保存符合条件的样本）
        
    Returns:
        统计信息字典
    """
    logger.info(f"开始分析文件: {input_file}")
    logger.info(f"阈值: {threshold}")
    
    # 统计信息
    stats = {
        "total_samples": 0,
        "valid_samples": 0,  # 有有效 turn_logprobs 的样本
        "qualified_samples": 0,  # 符合条件的样本（至少有一个 turn 的 disagreement_score > threshold）
        "qualified_indices": [],  # 符合条件的样本索引
        "max_disagreement_scores": [],  # 每个样本的最大 disagreement_score
        "samples_with_low_disagreement": 0,  # 所有 turn 的 disagreement_score <= threshold 的样本数
    }
    
    qualified_samples_data = []  # 符合条件的样本完整数据
    
    # 读取并分析文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="分析样本"), 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    stats["total_samples"] += 1
                    
                    # 获取样本信息
                    sample_index = data.get("index", -1)
                    turn_logprobs = data.get("turn_logprobs", [])
                    
                    if not turn_logprobs:
                        logger.warning(f"样本 {sample_index} (行 {line_num}) 没有 turn_logprobs，跳过")
                        continue
                    
                    stats["valid_samples"] += 1
                    
                    # 检查所有 turns 的 disagreement_score
                    max_disagreement = float('-inf')
                    has_high_disagreement = False
                    
                    for turn_data in turn_logprobs:
                        # 跳过有错误的 turn
                        if "error" in turn_data:
                            continue
                        
                        disagreement_score = turn_data.get("disagreement_score")
                        if disagreement_score is None:
                            continue
                        
                        # 更新最大 disagreement_score
                        if disagreement_score > max_disagreement:
                            max_disagreement = disagreement_score
                        
                        # 检查是否超过阈值
                        if disagreement_score > threshold:
                            has_high_disagreement = True
                            break  # 找到一个超过阈值的就可以停止
                    
                    stats["max_disagreement_scores"].append({
                        "index": sample_index,
                        "max_disagreement": max_disagreement if max_disagreement != float('-inf') else None
                    })
                    
                    if has_high_disagreement:
                        # 符合条件的样本：至少有一个 turn 的 disagreement_score > 阈值
                        stats["qualified_samples"] += 1
                        stats["qualified_indices"].append(sample_index)
                        qualified_samples_data.append(data)
                    else:
                        stats["samples_with_low_disagreement"] += 1
                        
                except json.JSONDecodeError as e:
                    logger.error(f"行 {line_num} JSON 解析错误: {e}")
                    continue
                except Exception as e:
                    logger.error(f"行 {line_num} 处理错误: {e}")
                    continue
    
    except FileNotFoundError:
        logger.error(f"文件不存在: {input_file}")
        return stats
    except Exception as e:
        logger.error(f"读取文件时出错: {e}")
        return stats
    
    # 输出统计信息
    logger.info(f"\n{'='*60}")
    logger.info(f"分析完成！")
    logger.info(f"  总样本数: {stats['total_samples']}")
    logger.info(f"  有效样本数（有 turn_logprobs）: {stats['valid_samples']}")
    logger.info(f"  符合条件的样本数（至少有一个 turn > {threshold}): {stats['qualified_samples']}")
    logger.info(f"  低不一致分数的样本数（所有 turn <= {threshold}): {stats['samples_with_low_disagreement']}")
    logger.info(f"{'='*60}\n")
    
    # 如果有符合条件的样本，显示前10个的索引
    if stats["qualified_indices"]:
        logger.info(f"符合条件的样本索引（前20个）: {stats['qualified_indices'][:20]}")
        if len(stats["qualified_indices"]) > 20:
            logger.info(f"  ... 共 {len(stats['qualified_indices'])} 个样本")
    
    # 保存符合条件的样本到文件
    if output_file and qualified_samples_data:
        logger.info(f"保存符合条件的样本到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for sample in qualified_samples_data:
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"已保存 {len(qualified_samples_data)} 个样本")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="分析 logprobs JSONL 文件，找出所有存在不一致分数大于阈值的step的样本（保留这些样本）"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="baseline/preliminary/data/train_student_42step_32B_teacher_logprobs.jsonl",
        help="输入的 JSONL 文件路径"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="不一致分数阈值（默认 0.2）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出文件路径（可选，如果提供则保存符合条件的样本）"
    )
    
    args = parser.parse_args()
    
    # 执行分析
    stats = analyze_samples(
        input_file=args.input_file,
        threshold=args.threshold,
        output_file=args.output_file
    )
    
    # 如果未指定输出文件，自动生成一个
    if not args.output_file and stats["qualified_samples"] > 0:
        import os
        input_basename = os.path.splitext(args.input_file)[0]
        auto_output_file = f"{input_basename}_high_disagreement_samples.jsonl"
        logger.info(f"自动保存到: {auto_output_file}")
        with open(auto_output_file, 'w', encoding='utf-8') as f_out:
            # 重新读取并保存符合条件的样本
            with open(args.input_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        sample_index = data.get("index", -1)
                        if sample_index in stats["qualified_indices"]:
                            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    except:
                        continue
        logger.info(f"已自动保存 {stats['qualified_samples']} 个符合条件的样本")
    
    logger.info("完成！")


if __name__ == "__main__":
    main()
