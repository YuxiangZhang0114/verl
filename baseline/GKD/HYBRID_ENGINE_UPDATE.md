# GKD Hybrid Engine 模式更新说明

## 📋 更新概述

本次更新将 GKD 训练器修改为使用标准 PPO 的 hybrid engine 模式，让 actor 和 rollout 共享同一个 GPU 池，支持完整的 offload 能力。

## ✅ 已完成的修改

### 1. 简化 `verl/trainer/gkd/ray_trainer.py`

**修改前**：
- 重写了完整的 `fit()` 方法（~200 行）
- 自定义训练循环，没有复用 PPO 的逻辑
- 手动管理 worker 调用

**修改后**：
- 只重写 `_update_actor()` 方法（~30 行）
- 完全复用 PPO 的 `fit()` 训练循环
- 自动使用 PPO 的 hybrid engine、offload 等特性

```python
def _update_actor(self, batch: DataProto) -> DataProto:
    # Get teacher knowledge if enabled
    if self.use_teacher:
        teacher_output = self._get_teacher_knowledge(batch)
        if teacher_output.non_tensor_batch:
            batch = batch.union(teacher_output)
    # Call parent's update_actor
    return super()._update_actor(batch)
```

**优势**：
- ✅ 代码量减少 85%（从 ~200 行到 ~30 行）
- ✅ 自动支持所有 PPO 特性（validation、checkpoint、profiling 等）
- ✅ 维护成本大幅降低

### 2. 简化 `verl/trainer/main_gkd.py`

**修改前**：
- 自定义 `GKDTaskRunner.run()` 方法
- 重复实现 PPO 的 dataset/worker 初始化逻辑

**修改后**：
- 完全复用 `PPOTaskRunner.run()` 方法
- 只在创建 trainer 时替换为 `RayGKDTrainer`

```python
class GKDTaskRunner(PPOTaskRunner):
    def run(self, config):
        # ... 复用 PPO 的完整逻辑 ...
        trainer = RayGKDTrainer(...)  # 唯一区别
        trainer.init_workers()
        trainer.fit()
```

**优势**：
- ✅ 与 PPO 保持一致的行为
- ✅ 自动获得 PPO 的所有 bug 修复和功能更新

### 3. 修改 `baseline/GKD/run_gkd_simple.sh`

**修改前**：
```bash
N_GPUS_PER_NODE=4  # Actor uses 4 GPUs
ROLLOUT_GPUS=2     # Rollout uses 2 GPUs (separate from actor)
```

**修改后**：
```bash
N_GPUS_PER_NODE=6  # All 6 GPUs in shared pool
actor_rollout_ref.hybrid_engine=true
```

**关键变化**：
- ✅ 移除了 `ROLLOUT_GPUS` 变量（不再分离配置）
- ✅ 明确设置 `hybrid_engine=true`
- ✅ actor 和 rollout 共享同一个 GPU 池

### 4. 更新 `verl/trainer/config/gkd_trainer.yaml`

**新增配置**：
```yaml
actor_rollout_ref:
  # Use hybrid engine mode (actor and rollout share GPU pool)
  hybrid_engine: true
```

## 🔄 架构对比

### 修改前（分离模式）

```
┌─────────────┐     ┌─────────────┐
│   Actor     │     │   Rollout   │
│  GPU 2-5    │     │   GPU 6-7   │
│  (4 GPUs)   │     │   (2 GPUs)  │
└─────────────┘     └─────────────┘
      ↓                    ↓
   训练模型            生成序列
      ↑                    ↓
      └────── NCCL ────────┘
```

**问题**：
- ❌ 需要手动管理两个独立的 GPU 池
- ❌ 无法使用 PPO 的 offload 优化
- ❌ 复杂的自定义训练循环

### 修改后（Hybrid Engine）

```
┌─────────────────────────────────┐
│      Shared GPU Pool            │
│       GPU 2-7 (6 GPUs)          │
│  ┌──────────┐   ┌──────────┐   │
│  │  Actor   │◄─►│ Rollout  │   │
│  └──────────┘   └──────────┘   │
│       Automatic Offload         │
└─────────────────────────────────┘
```

**优势**：
- ✅ Actor 和 rollout 共享 GPU 池
- ✅ 自动支持 param_offload 和 optimizer_offload
- ✅ 完全复用 PPO 的训练流程和优化

## 🚀 使用方式

### 1. 启动 Teacher Server

```bash
cd baseline/GKD/teacher
export CUDA_VISIBLE_DEVICES=0,1
bash start_server.sh
```

### 2. 运行 GKD 训练

```bash
cd /root/yuxiang/verl
bash baseline/GKD/run_gkd_simple.sh
```

## 📊 特性对比

| 特性 | 修改前 | 修改后 |
|------|--------|--------|
| **代码行数** | ~380 行 | ~240 行 |
| **训练循环** | 自定义 | 复用 PPO |
| **Hybrid Engine** | ❌ 不支持 | ✅ 支持 |
| **Offload** | ❌ 不支持 | ✅ 支持 |
| **GPU 管理** | 手动分离 | 自动共享 |
| **Validation** | 自定义 | 复用 PPO |
| **Checkpoint** | 自定义 | 复用 PPO |
| **Profiling** | 部分支持 | 完全支持 |
| **维护成本** | 高 | 低 |

## 🔧 配置说明

### Hybrid Engine 相关配置

```yaml
# 在 gkd_trainer.yaml 或启动脚本中
actor_rollout_ref:
  hybrid_engine: true  # 启用 hybrid engine
  
  actor:
    fsdp_config:
      param_offload: true      # 参数 offload
      optimizer_offload: true  # 优化器 offload
```

### GPU 分配

```bash
# Teacher: GPU 0-1
export CUDA_VISIBLE_DEVICES=0,1

# Student: GPU 2-7 (shared pool)
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
trainer.n_gpus_per_node=6
```

## ⚠️ 注意事项

1. **Hybrid Engine 是默认模式**：`ppo_trainer.yaml` 中默认 `hybrid_engine: true`

2. **不要手动分离 rollout GPU**：让 hybrid engine 自动管理

3. **Offload 配置**：
   - `param_offload=True` - 训练时将参数 offload 到 CPU
   - `optimizer_offload=True` - 将优化器状态 offload 到 CPU

4. **与原实现的兼容性**：
   - 原有的 `recipe/gkd` 实现（Megatron 版本）仍然可用
   - 新的 `baseline/GKD` 实现（FSDP 版本）现在更简单

## 📝 总结

本次更新通过：
1. **大幅简化代码**（减少 ~140 行）
2. **完全复用 PPO 架构**
3. **启用 hybrid engine 模式**
4. **支持完整的 offload 能力**

使得 GKD 训练器与标准 PPO 训练器保持一致，降低维护成本，同时获得更好的性能和功能。

---

**更新日期**: 2026-01-08  
**版本**: 2.0.0  
**状态**: ✅ 已完成并测试
