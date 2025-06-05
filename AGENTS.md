# Guidelines for verl contributors

This repository contains the open-source implementation of the HybridFlow RL framework for post-training large language models (LLMs). Below is a quick orientation for newcomers and some basic contribution rules.

## Project overview

- `verl` package: main library code.
  - `trainer/main_ppo.py` – entry point for PPO training.
  - `trainer/ppo/ray_trainer.py` – training loop for RL algorithms.
  - `trainer/fsdp_sft_trainer.py` – SFT trainer using FSDP backend.
  - `config/` – example configuration files.
  - `workers/` – worker implementations for rollout, critic, reward model, etc.
  - `utils/` – helper modules, datasets and reward functions.
  - `models/` – model back ends (Megatron, Transformers) and weight loaders.
  - `third_party/vllm/` – adapters for using vLLM with RL.
- `examples/` – example scripts.
- `tests/` – integration and unit tests.
- `docs/` – documentation, including the HybridFlow programming guide and quickstart tutorials.

See `docs/hybrid_flow.rst` for an overview of the framework design and repository organization.

## Development setup

1. Install the required dependencies, e.g. `pip install -r requirements.txt`.
2. We use **pre-commit** for linting and formatting. After cloning, run:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

   Before committing, run `pre-commit run --files <changed files>` to fix style issues locally.

3. When adding new features, consider adding tests under `tests/` and update CI patterns as described in `docs/index.rst`.

Happy hacking!
