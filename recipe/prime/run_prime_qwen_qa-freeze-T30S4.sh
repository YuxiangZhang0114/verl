set -x


# gsm8k_train_path=$HOME/data/gsm8k/train.parquet
# gsm8k_test_path=$HOME/data/gsm8k/test.parquet
export WANDB_API_KEY="0559d52399bc5d3fd8e373bb4b8b6e8db054b9f7"

# # download from https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data
# math_train_path=$HOME/data/math/train.parquet
# math_test_path=$HOME/data/math/test.parquet

# train_files="['$gsm8k_train_path', '$math_train_path']"
# test_files="['$gsm8k_test_path', '$math_test_path']"

# model_path=PRIME-RL/Eurus-2-7B-SFT
model_path=Qwen/Qwen3-4B-Instruct-2507
reward_model_path=Qwen/Qwen3-30B-A3B-Instruct-2507
PROJECT_DIR="$(pwd)"
TRAIN_DATA="$PROJECT_DIR/data/asearcher_train/train.parquet"
VAL_DATA="$PROJECT_DIR/data/hotpotqa_hard_train/validation.parquet"

train_files="['$TRAIN_DATA']"
test_files="['$VAL_DATA']"
# Tool config path

TOOL_CONFIG="$PROJECT_DIR/baseline/GRPO/tool_config/search_tool_simple_config.yaml"

save_path="/mnt/workspace/checkpoints/search_r1_like_grpo_sglang_qwen3-4b-instruct_asearcher"


python3 -m recipe.prime.main_prime \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=200 \
    data.max_prompt_length=512 \
    data.max_response_length=12800 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.0 \
    data.accuracy_upper_bound=1.0 \
    data.oversample_factor=1 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.model.path=$reward_model_path \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.update=after \
    reward_model.model.beta_train=0.05 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=1 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='search_r1_like_grpo_sglang' \
    trainer.experiment_name='qwen3-4b-instruct-grpo-sglang-async-toolagent-n5-8gpu-prime-T4S4' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=80 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 $@
