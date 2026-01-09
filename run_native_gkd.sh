set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo     algorithm.adv_estimator=gae     data.train_files=$DATA_DIR/train.parquet     data.val_files=$DATA_DIR/test.parquet     data.train_batch_size=128     data.val_batch_size=128     data.max_prompt_length=1024     data.max_response_length=512     actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507     actor_rollout_ref.actor.optim.lr=1e-6     actor_rollout_ref.model.use_remove_padding=True     actor_rollout_ref.actor.ppo_mini_batch_size=64     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4     actor_rollout_ref.actor.use_kl_loss=True     actor_rollout_ref.actor.kl_loss_coef=0.001     actor_rollout_ref.actor.kl_loss_type=low_var_kl     actor_rollout_ref.model.enable_gradient_checkpointing=True     actor_rollout_ref.actor.fsdp_config.param_offload=False     actor_rollout_ref.actor.fsdp_config.grad_offload=False     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8     actor_rollout_ref.rollout.tensor_model_parallel_size=1     actor_rollout_ref.rollout.name=vllm     actor_rollout_ref.rollout.gpu_memory_utilization=0.4     actor_rollout_ref.rollout.n=1     actor_rollout_ref.ref.fsdp_config.param_offload=True     critic.optim.lr=1e-5     critic.model.use_remove_padding=True     critic.model.path=Qwen/Qwen3-4B-Instruct-2507     critic.model.enable_gradient_checkpointing=True     critic.ppo_micro_batch_size_per_gpu=4     critic.model.fsdp_config.param_offload=False     critic.model.fsdp_config.grad_offload=False     critic.model.fsdp_config.optimizer_offload=False     reward_model.enable=False     trainer.critic_warmup=0     trainer.logger=['console']     trainer.project_name='verl_gkd_example'     trainer.experiment_name='qwen3_4b_gkd'     trainer.n_gpus_per_node=8     trainer.nnodes=1     trainer.save_freq=-1     trainer.test_freq=10     trainer.total_epochs=1     algorithm.policy_loss=gkd     actor_rollout_ref.teacher.server_ip=127.0.0.1     actor_rollout_ref.teacher.server_port=15555     actor_rollout_ref.teacher.num_microbatches=1     actor_rollout_ref.teacher.n_server_workers=1     actor_rollout_ref.teacher.temperature=1.0     actor_rollout_ref.teacher.only_response=False     cat > run_native_gkd.sh <<EOF
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=\$DATA_DIR/train.parquet \
    data.val_files=\$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_gkd_example' \
    trainer.experiment_name='qwen3_4b_gkd' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    algorithm.policy_loss=gkd \
    actor_rollout_ref.teacher.server_ip=127.0.0.1 \
    actor_rollout_ref.teacher.server_port=15555 \
    actor_rollout_ref.teacher.num_microbatches=1 \
    actor_rollout_ref.teacher.n_server_workers=1 \
    actor_rollout_ref.teacher.temperature=1.0 \
    actor_rollout_ref.teacher.only_response=False \
    $@
EOF

