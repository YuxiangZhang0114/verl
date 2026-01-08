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

set -x

# make sure your current working directory is the root of the project
ulimit -n 65535

export RAY_NUM_CPUS=24
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

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
# Tool config path
TOOL_CONFIG="$PROJECT_DIR/baseline/GRPO/tool_config/search_tool_simple_config.yaml"

# Save path
save_path="/mnt/workspace/checkpoints/qwen3_4b_gkd_distill_asearcher"

# Parallel configuration for Qwen3 4B
NODES=1
PP=1  # Pipeline parallel size
TP=1  # Tensor parallel size
INFER_TP=1  # Inference tensor parallel size
SP=False  # Sequence parallel

# Teacher server configuration
TEACHER_SERVER_HOST=127.0.0.1
TEACHER_SERVER_PORT=15555

function check_server_ready() {
    local server=$1
    local ip=$2
    local port=$3
    
    echo "check $server server ready at $ip:$port..."
    result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
    if [ $result -ne 1 ]; then
        echo "server $server is not ready at $ip:$port, exit..."
        echo "Please start the teacher server first using:"
        echo "  cd baseline/GKD/teacher && bash start_server.sh"
        exit 1
    fi
}

check_server_ready teacher $TEACHER_SERVER_HOST $TEACHER_SERVER_PORT

function now() {
    date '+%Y-%m-%d-%H-%M'
}

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GKD_CONFIG_PATH="$PROJECT_DIR/recipe/gkd/config"
RUNTIME_ENV=${RUNTIME_ENV:-"$GKD_CONFIG_PATH/runtime_env.yaml"}

python3 -m recipe.gkd.main_gkd \
    --config-path="$GKD_CONFIG_PATH" \
    --config-name=on_policy_distill_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.prompt_key=prompt \
    data.train_batch_size=128 \
    data.val_batch_size=400 \
    data.max_prompt_length=4096 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.megatron.sequence_parallel=$SP \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.router_replay.mode=disabled \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.micro_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.max_token_len=23000 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra'] \
    actor_rollout_ref.actor.checkpoint.load_contents=[] \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.load_format='dummy_megatron' \
    +actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=10000 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.teacher.server_ip=$TEACHER_SERVER_HOST \
    actor_rollout_ref.teacher.server_port=$TEACHER_SERVER_PORT \
    actor_rollout_ref.teacher.n_server_workers=1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='qwen3_4b_gkd_distill' \
    trainer.experiment_name="qwen3-4b-instruct-gkd-asearcher-$(now)" \
    trainer.nnodes=$NODES \
    trainer.n_gpus_per_node=4 \
    rollout.nnodes=$NODES \
    rollout.n_gpus_per_node=2 \
    trainer.scheduler="one_step_off" \
    trainer.default_local_dir=$save_path \
    trainer.save_freq=80 \
    trainer.test_freq=10 \
    trainer.log_val_generations=40 \
    trainer.val_before_train=True \
    trainer.total_epochs=2 "$@"
