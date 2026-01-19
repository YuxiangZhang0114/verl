#!/bin/bash
# ALFWorld Multi-turn GRPO Training Script
# Run on 8xH100 or similar GPU setup
# Make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
DATA_DIR="$PROJECT_DIR/data/alfworld_train"

function now() {
    date '+%d-%H-%M'
}

EXPERIMENT_NAME="alfworld_grpo_$(now)"

# Check if data exists, if not, generate it
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ALFWorld training data not found. Generating..."
    python3 baseline/data_process/prepare_data_alfworld.py --local_dir "$DATA_DIR"
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='alfworld_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='alfworld-grpo' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.val_before_train=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/eval_id.parquet" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/alfworld_tool_config.yaml" \
    trainer.total_epochs=10 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 $@
