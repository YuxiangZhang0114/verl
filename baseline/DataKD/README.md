# Data Distillation for Search Tool Tasks

Generate multi-turn conversation data with search tool calls using Tongyi-DeepResearch-30B model.

## Quick Start

### Prerequisites

1. **vLLM Server**: Start vLLM server with the Tongyi-DeepResearch model
2. **Retrieval Server**: Ensure your retrieval server is running

### Basic Usage

```bash
# Use default settings
bash baseline/DataKD/run_generate_search_data.sh

# Or customize parameters via environment variables
VLLM_URL="http://localhost:8000/v1" \
RETRIEVAL_URL="http://127.0.0.1:8000/retrieve" \
MAX_ASSISTANT_TURNS=5 \
MAX_SAMPLES=1000 \
bash baseline/DataKD/run_generate_search_data.sh
```

### Configuration Parameters

All parameters can be set via environment variables:

- `VLLM_URL`: vLLM server endpoint (default: `http://localhost:8000/v1`)
- `RETRIEVAL_URL`: Retrieval server endpoint (default: `http://127.0.0.1:8000/retrieve`)
- `MAX_ASSISTANT_TURNS`: Maximum assistant turns per conversation (default: `3`)
- `MAX_SAMPLES`: Number of samples to process, `-1` for all (default: `-1`)
- `INPUT_FILE`: Input parquet file (default: `data/hotpotqa_hard_train/train.parquet`)
- `OUTPUT_FILE`: Output parquet file (default: `baseline/DataKD/train_distilled.parquet`)
- `BATCH_SIZE`: Async batch size (default: `16`)
- `TEMPERATURE`: Sampling temperature (default: `0.7`)
- `MAX_TOKENS`: Max tokens per generation (default: `2048`)

### Direct Python Usage

```bash
python3 baseline/DataKD/generate_search_data.py \
    --vllm_url "http://localhost:8000/v1" \
    --retrieval_url "http://127.0.0.1:8000/retrieve" \
    --max_assistant_turns 3 \
    --max_samples 1000 \
    --input_file "data/hotpotqa_hard_train/train.parquet" \
    --output_file "baseline/DataKD/train_distilled.parquet" \
    --batch_size 16 \
    --temperature 0.7 \
    --max_tokens 2048
```

## Output Format

The generated parquet file is compatible with `verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset`.

Each row contains:
- `messages`: List of conversation messages (system, user, assistant, tool)
- `tools`: Tool schema definitions
- `question`: Original question
- `extra_info`: Metadata from input dataset
- `assistant_turns`: Number of assistant turns in conversation

### Example Message Structure

```python
{
    "messages": [
        {"role": "system", "content": "You are a deep research assistant..."},
        {"role": "user", "content": "Question: What is the capital of France?"},
        {
            "role": "assistant",
            "content": "Let me search for information.",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query_list": ["capital of France"]}'
                }
            }]
        },
        {"role": "tool", "content": "Query: capital of France\n[1] Paris..."},
        {"role": "assistant", "content": "<answer>Paris</answer>"}
    ],
    "tools": [{"type": "function", "function": {...}}]
}
```

## Use with MultiTurnSFTDataset

```python
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-model")

dataset = MultiTurnSFTDataset(
    parquet_files=["baseline/DataKD/train_distilled.parquet"],
    tokenizer=tokenizer,
    config={
        "messages_key": "messages",
        "tools_key": "tools",
        "max_length": 4096,
        "pad_mode": "right",
    }
)
```

