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

# Simplified GKD Training Script for Qwen3-4B
# 
# This script uses the simplified PPO-style GKD trainer that:
# - Uses FSDP backend (simpler than Megatron)
# - Synchronous teacher knowledge retrieval
# - Supports agent multi-turn interaction via sglang
# - Flexible RL + distillation loss combination
# - Hybrid engine mode (actor and rollout share GPU pool)

set -x

# Make sure your current working directory is the root of the project
ulimit -n 65535

export RAY_NUM_CPUS=24
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_FLASH_ATTN=1
export NVTE_DEBUG=1 
export NVTE_DEBUG_LEVEL=2

PROJECT_DIR="$(pwd)"

# Model path - Qwen3-4B-Instruct (Student)
HF_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
export WANDB_API_KEY="0559d52399bc5d3fd8e373bb4b8b6e8db054b9f7"

# Data paths
TRAIN_DATA="$PROJECT_DIR/data/asearcher_train/train.parquet"
VAL_DATA="$PROJECT_DIR/data/hotpotqa_hard_train/validation.parquet"

# Tool config path (for agent interaction)
TOOL_CONFIG="$PROJECT_DIR/baseline/GRPO/tool_config/search_tool_simple_config.yaml"

# Save path
save_path="/mnt/workspace/checkpoints/qwen3_4b_gkd_simple"

# GPU configuration - using hybrid engine mode (actor and rollout share GPU pool)
NODES=1
N_GPUS_PER_NODE=4  # All 6 GPUs in shared pool

# Sequence parallel size (IMPORTANT: GKD currently only supports sp_size=1)
sp_size=1

# Teacher server configuration
TEACHER_SERVER_HOST=127.0.0.1
TEACHER_SERVER_PORT=15555

# Helper function to check if teacher server is ready
function check_server_ready() {
    local server=$1
    local ip=$2
    local port=$3
    
    echo "Checking $server server ready at $ip:$port..."
    result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
    if [ $result -ne 1 ]; then
        echo "ERROR: Server $server is not ready at $ip:$port"
        echo "Please start the teacher server first using:"
        echo "  cd baseline/GKD/teacher && bash start_server.sh"
        exit 1
    fi
    echo "✓ Teacher server is ready"
}

# Check teacher server is running
check_server_ready teacher $TEACHER_SERVER_HOST $TEACHER_SERVER_PORT

function now() {
    date '+%Y-%m-%d-%H-%M'
}

USE_FUSED_KERNELS=False
sp_size=1
# Run GKD training with hybrid engine mode
python3 -m verl.trainer.main_gkd \
    --config-name=gkd_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.prompt_key=prompt \
    data.train_batch_size=128 \
    data.val_batch_size=400 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=23000 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=10000 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    gkd.enable_teacher=True \
    gkd.teacher_ip=$TEACHER_SERVER_HOST \
    gkd.teacher_port=$TEACHER_SERVER_PORT \
    gkd.teacher_topk=256 \
    gkd.teacher_n_server_workers=1 \
    gkd.distill_loss_coef=1.0 \
    gkd.distill_loss_type=forward_kl \
    gkd.distill_temperature=1.0 \
    gkd.use_rl_loss=false \
    gkd.rl_loss_coef=0.0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='qwen3_4b_gkd_simple' \
    trainer.experiment_name="qwen3-4b-gkd-simple-$(now)" \
    trainer.nnodes=$NODES \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.default_local_dir=$save_path \
    trainer.save_freq=80 \
    trainer.test_freq=10 \
    trainer.log_val_generations=40 \
    trainer.val_before_train=False \
    trainer.total_epochs=2 "$@"
