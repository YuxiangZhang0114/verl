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
FSDP-compatible distillation loss functions for GKD training.

This module provides efficient KL divergence computation using sparse teacher
top-k log probabilities for knowledge distillation.
"""

import torch
import torch.nn.functional as F


def compute_fsdp_kl_divergence(
    student_logits: torch.Tensor,
    teacher_topk_logps: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 1.0,
    loss_type: str = "forward_kl",
) -> torch.Tensor:
    """
    Compute KL divergence loss between student and teacher distributions.
    
    Uses sparse teacher top-k log probabilities for efficient computation.
    
    Args:
        student_logits: Student model output logits, shape [batch_size, seq_len, vocab_size]
        teacher_topk_logps: Teacher's top-k log probabilities, shape [batch_size, seq_len, topk]
        teacher_topk_indices: Teacher's top-k token indices, shape [batch_size, seq_len, topk]
        response_mask: Mask for response tokens, shape [batch_size, seq_len]
        temperature: Temperature for softmax scaling
        loss_type: Type of KL divergence - "forward_kl" (KL(P||Q)) or "reverse_kl" (KL(Q||P))
                   where P=teacher, Q=student
    
    Returns:
        Scalar loss tensor
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    topk = teacher_topk_logps.shape[-1]
    
    # Apply temperature scaling to student logits
    student_logits_scaled = student_logits / temperature
    
    # Compute student log probabilities (full softmax)
    student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)  # [batch_size, seq_len, vocab_size]
    
    # Gather student log probs at teacher's top-k indices
    # teacher_topk_indices: [batch_size, seq_len, topk]
    student_topk_log_probs = torch.gather(
        student_log_probs, 
        dim=-1, 
        index=teacher_topk_indices.long()
    )  # [batch_size, seq_len, topk]
    
    # Convert teacher log probs to probs for weighting
    teacher_topk_probs = torch.exp(teacher_topk_logps)  # [batch_size, seq_len, topk]
    
    # Normalize teacher probs to sum to 1 (in case they don't due to top-k truncation)
    teacher_topk_probs_normalized = teacher_topk_probs / (teacher_topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
    
    if loss_type == "forward_kl":
        # Forward KL: KL(P||Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
        # This encourages student to cover all modes of teacher distribution
        # P = teacher, Q = student
        teacher_entropy = torch.sum(
            teacher_topk_probs_normalized * teacher_topk_logps, 
            dim=-1
        )  # [batch_size, seq_len]
        
        cross_entropy = torch.sum(
            teacher_topk_probs_normalized * student_topk_log_probs, 
            dim=-1
        )  # [batch_size, seq_len]
        
        per_token_kl = teacher_entropy - cross_entropy  # [batch_size, seq_len]
        
    elif loss_type == "reverse_kl":
        # Reverse KL: KL(Q||P) = sum(Q * log(Q/P))
        # This encourages student to focus on teacher's main modes
        # Q = student (at teacher's top-k positions), P = teacher
        student_topk_probs = torch.exp(student_topk_log_probs)
        student_topk_probs_normalized = student_topk_probs / (student_topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        per_token_kl = torch.sum(
            student_topk_probs_normalized * (student_topk_log_probs - teacher_topk_logps),
            dim=-1
        )  # [batch_size, seq_len]
        
    elif loss_type == "jsd":
        # Jensen-Shannon Divergence: 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5 * (P + Q)
        student_topk_probs = torch.exp(student_topk_log_probs)
        m_probs = 0.5 * (teacher_topk_probs_normalized + student_topk_probs)
        m_log_probs = torch.log(m_probs + 1e-10)
        
        kl_p_m = torch.sum(teacher_topk_probs_normalized * (teacher_topk_logps - m_log_probs), dim=-1)
        kl_q_m = torch.sum(student_topk_probs * (student_topk_log_probs - m_log_probs), dim=-1)
        
        per_token_kl = 0.5 * (kl_p_m + kl_q_m)
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Supported: forward_kl, reverse_kl, jsd")
    
    # Apply response mask and compute mean loss
    # response_mask: [batch_size, seq_len]
    masked_kl = per_token_kl * response_mask
    
    # Compute mean over valid tokens
    num_valid_tokens = response_mask.sum()
    if num_valid_tokens > 0:
        loss = masked_kl.sum() / num_valid_tokens
    else:
        loss = masked_kl.sum() * 0.0  # Return zero with grad
    
    return loss


def compute_distill_loss_with_logits(
    student_logits: torch.Tensor,
    teacher_topk_logps: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    config,
) -> tuple[torch.Tensor, dict]:
    """
    Wrapper function to compute distillation loss compatible with actor update.
    
    Args:
        student_logits: Student model logits [batch_size, seq_len, vocab_size]
        teacher_topk_logps: Teacher's sparse top-k log probs [batch_size, seq_len, topk]
        teacher_topk_indices: Teacher's top-k indices [batch_size, seq_len, topk]
        attention_mask: Full attention mask [batch_size, seq_len]
        response_mask: Response-only mask [batch_size, seq_len]
        config: GKD config containing loss_type, temperature, etc.
    
    Returns:
        Tuple of (loss tensor, metrics dict)
    """
    temperature = getattr(config, "distill_temperature", 1.0)
    loss_type = getattr(config, "distill_loss_type", "forward_kl")
    
    distill_loss = compute_fsdp_kl_divergence(
        student_logits=student_logits,
        teacher_topk_logps=teacher_topk_logps,
        teacher_topk_indices=teacher_topk_indices,
        response_mask=response_mask,
        temperature=temperature,
        loss_type=loss_type,
    )
    
    metrics = {
        "actor/distill_loss": distill_loss.detach().item(),
    }
    
    return distill_loss, metrics
