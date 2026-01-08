# 简化版 GKD 训练指南

这是一个基于 PPO-style 的简化 GKD (Generalized Knowledge Distillation) 训练实现，相比原有的 `recipe/gkd` 实现更加简单易用。

## 🎯 主要特性

- **简化架构**：使用 FSDP 后端而非 Megatron，配置更简单
- **同步调用**：Teacher 知识获取采用同步方式，实现简单清晰
- **灵活损失组合**：支持纯蒸馏或 RL+蒸馏混合训练
- **复用 GRPO 工具**：直接使用 GRPO 的 agent 交互和 rollout 配置

## 📁 文件结构

```
verl/
├── trainer/
│   ├── gkd/                          # GKD 训练模块
│   │   ├── __init__.py
│   │   ├── ray_trainer.py            # RayGKDTrainer (继承 RayPPOTrainer)
│   │   └── distill_loss.py           # FSDP 兼容的蒸馏损失函数
│   ├── main_gkd.py                   # GKD 训练入口点
│   └── config/
│       └── gkd_trainer.yaml          # GKD 配置文件
├── workers/
│   └── actor/
│       └── dp_actor.py               # 扩展支持蒸馏损失
└── baseline/
    └── GKD/
        ├── run_gkd_simple.sh         # 简化启动脚本
        ├── teacher/                   # Teacher server (复用原有)
        └── README_SIMPLE.md          # 本文件
```

## 🚀 快速开始

### 1. 启动 Teacher Server

首先启动 teacher model server（使用 GPU 0-1）：

```bash
cd baseline/GKD/teacher
export CUDA_VISIBLE_DEVICES=0,1
bash start_server.sh
```

验证 server 已启动：

```bash
telnet localhost 15555
```

### 2. 运行 GKD 训练

在另一个终端中，启动 student 训练（使用 GPU 2-7）：

```bash
cd /root/yuxiang/verl
bash baseline/GKD/run_gkd_simple.sh
```

## ⚙️ 配置说明

### GKD 核心配置

在 `verl/trainer/config/gkd_trainer.yaml` 中：

```yaml
gkd:
  # Teacher 配置
  enable_teacher: true
  teacher_ip: "127.0.0.1"
  teacher_port: 15555
  teacher_topk: 256
  
  # 蒸馏损失配置
  distill_loss_coef: 1.0
  distill_loss_type: "forward_kl"  # forward_kl, reverse_kl, jsd
  distill_temperature: 1.0
  
  # RL 损失配置（可选）
  use_rl_loss: false
  rl_loss_coef: 0.0
```

### 损失类型说明

- **forward_kl**: `KL(teacher||student)` - 鼓励 student 覆盖 teacher 的所有模式
- **reverse_kl**: `KL(student||teacher)` - 鼓励 student 聚焦于 teacher 的主要模式
- **jsd**: Jensen-Shannon Divergence - 对称的平衡方法

### 训练模式

#### 模式 1: 纯蒸馏（推荐）

```bash
gkd.use_rl_loss=false \
gkd.distill_loss_coef=1.0
```

#### 模式 2: RL + 蒸馏混合

```bash
gkd.use_rl_loss=true \
gkd.rl_loss_coef=0.1 \
gkd.distill_loss_coef=1.0
```

## 🔍 与原有实现的对比

| 特性 | 原有 `recipe/gkd` | 简化版 `baseline/GKD` |
|------|-------------------|----------------------|
| 后端 | Megatron | FSDP |
| Teacher 调用 | 异步 (one_step_off) | 同步 |
| 配置复杂度 | 高 | 低 |
| 代码复杂度 | 高（自定义 worker） | 低（复用 PPO） |
| 性能优化 | 高（流水线优化） | 中（同步调用） |
| 易用性 | 低 | 高 |
| 适用场景 | 生产环境大规模训练 | 研究、实验、小规模训练 |

## 📊 监控指标

训练过程中会记录以下指标：

- `actor/distill_loss`: 蒸馏损失值
- `actor/distill_coef`: 蒸馏损失系数
- `actor/pg_loss`: 策略梯度损失（如果启用 RL）
- `actor/entropy`: 策略熵
- `training/global_step`: 全局训练步数

## 🛠️ 自定义修改

### 1. 修改蒸馏损失

编辑 `verl/trainer/gkd/distill_loss.py` 中的 `compute_fsdp_kl_divergence` 函数。

### 2. 修改训练流程

编辑 `verl/trainer/gkd/ray_trainer.py` 中的 `RayGKDTrainer.fit()` 方法。

### 3. 添加自定义 reward

在启动脚本中设置：

```bash
custom_reward_function.path=path/to/your/reward.py \
custom_reward_function.name=your_reward_function
```

## 🐛 故障排查

### 问题 1: Teacher server 连接失败

**症状**: `Teacher request failed` 错误

**解决方案**:
1. 确认 teacher server 已启动：`telnet localhost 15555`
2. 检查 GPU 分配是否冲突
3. 查看 teacher server 日志

### 问题 2: OOM 错误

**解决方案**:
1. 减小 `ppo_micro_batch_size_per_gpu`
2. 启用 `param_offload` 和 `optimizer_offload`
3. 减小 `ppo_max_token_len_per_gpu`

### 问题 3: 蒸馏损失为 0

**解决方案**:
1. 检查 `has_teacher_knowledge` 是否为 True
2. 确认 teacher_topk_logps 在 batch 中
3. 检查 distill_loss_coef 是否为 0

## 📝 引用

如果使用本实现，请引用：

```bibtex
@software{verl_gkd_simple,
  title = {Simplified GKD Training for veRL},
  author = {veRL Team},
  year = {2025},
  url = {https://github.com/volcengine/verl}
}
```

## 📄 许可证

Apache License 2.0

---

**注意**: 本实现优先考虑简洁性和易用性。如需生产级性能优化，请使用原有的 `recipe/gkd` 实现。
