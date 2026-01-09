#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_NUM_CPUS=24
export OMP_NUM_THREADS=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"

# Aligning with GRPO experiment settings
# Data paths
TRAIN_DATA="$PROJECT_DIR/data/asearcher_train/train.parquet"
VAL_DATA="$PROJECT_DIR/data/hotpotqa_hard_train/validation.parquet"
TOOL_CONFIG="$PROJECT_DIR/baseline/GRPO/tool_config/search_tool_simple_config.yaml"

# Teacher Configuration
TEACHER_IP="127.0.0.1"
TEACHER_PORT=15555

# Save path
save_path="/mnt/workspace/checkpoints/search_r1_like_gkd_qwen3-4b-instruct_asearcher"
export WANDB_API_KEY="0559d52399bc5d3fd8e373bb4b8b6e8db054b9f7"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=128 \
    data.val_batch_size=400 \
    data.max_prompt_length=2048 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    trainer.default_local_dir=$save_path \
    actor_rollout_ref.model.path="Qwen/Qwen3-4B-Instruct-2507" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=23000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=10000 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="Qwen/Qwen3-4B-Instruct-2507" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_gkd_example' \
    trainer.experiment_name='qwen3_4b_gkd_aligned_grpo_asearcher' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=80 \
    trainer.test_freq=10 \
    trainer.log_val_generations=40 \
    trainer.total_epochs=2 \
    algorithm.policy_loss=gkd \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.teacher.server_ip=$TEACHER_IP \
    actor_rollout_ref.teacher.server_port=$TEACHER_PORT \
    actor_rollout_ref.teacher.num_microbatches=1 \
    actor_rollout_ref.teacher.n_server_workers=1 \
    actor_rollout_ref.teacher.temperature=1.0 \
    actor_rollout_ref.teacher.only_response=False \
    $@
