# GKD Baseline - Qwen3-4B 知识蒸馏训练

使用 Qwen3-30B 作为 Teacher 模型，对 Qwen3-4B 进行在线知识蒸馏训练。

## 快速开始

### 1. 启动 Teacher Server

```bash
cd baseline/GKD/teacher
# 确保 GPU 0-1 可用（Teacher 使用 TP=2）
export CUDA_VISIBLE_DEVICES=0,1
bash start_server.sh
```

验证 server 就绪：
```bash
telnet localhost 15555
```

### 2. 运行训练

```bash
cd /root/yuxiang/verl
# Student 使用 GPU 2-7（Actor 4卡 + Rollout 2卡）
bash baseline/GKD/run_gkd_qwen3_4b.sh
```

## GPU 分配

- **Teacher Server**: GPU 0-1 (2卡, TP=2)
- **Student Training**: GPU 2-7
  - Actor: GPU 2-5 (4卡)
  - Rollout: GPU 6-7 (2卡)

## 配置说明

- **Teacher Model**: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **Student Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Dataset**: `data/asearcher_train/train.parquet`
- **Scheduler**: `one_step_off`
