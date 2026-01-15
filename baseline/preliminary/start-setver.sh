CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
    --model /root/actor/huggingface \
    --served_model_name student_model \
    --port 8100 --tensor_parallel_size 2