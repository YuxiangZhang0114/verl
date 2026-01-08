#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的导入测试脚本，用于验证 GKD 模块是否正确安装
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("Testing GKD module imports...")
print("=" * 60)

try:
    from verl.trainer.gkd import RayGKDTrainer, compute_fsdp_kl_divergence
    print("✓ verl.trainer.gkd imports successfully")
    print(f"  - RayGKDTrainer: {RayGKDTrainer}")
    print(f"  - compute_fsdp_kl_divergence: {compute_fsdp_kl_divergence}")
except ImportError as e:
    print(f"✗ Failed to import verl.trainer.gkd: {e}")
    sys.exit(1)

try:
    from verl.trainer.gkd.distill_loss import compute_fsdp_kl_divergence, compute_distill_loss_with_logits
    print("✓ verl.trainer.gkd.distill_loss imports successfully")
    print(f"  - compute_fsdp_kl_divergence: {compute_fsdp_kl_divergence}")
    print(f"  - compute_distill_loss_with_logits: {compute_distill_loss_with_logits}")
except ImportError as e:
    print(f"✗ Failed to import verl.trainer.gkd.distill_loss: {e}")
    sys.exit(1)

try:
    from verl.trainer.gkd.ray_trainer import RayGKDTrainer
    print("✓ verl.trainer.gkd.ray_trainer imports successfully")
    print(f"  - RayGKDTrainer: {RayGKDTrainer}")
except ImportError as e:
    print(f"✗ Failed to import verl.trainer.gkd.ray_trainer: {e}")
    sys.exit(1)

try:
    import verl.trainer.main_gkd
    print("✓ verl.trainer.main_gkd imports successfully")
except ImportError as e:
    print(f"✗ Failed to import verl.trainer.main_gkd: {e}")
    sys.exit(1)

try:
    # Test teacher client import (optional, only available if recipe/gkd is in path)
    sys.path.insert(0, str(project_root / "recipe" / "gkd"))
    from teacher.client import TeacherClient
    print("✓ recipe.gkd.teacher.client imports successfully")
    print(f"  - TeacherClient: {TeacherClient}")
except ImportError as e:
    print(f"⚠ Warning: teacher.client not available (this is OK if not using teacher): {e}")

print("=" * 60)
print("All critical imports successful! ✓")
print("\nYou can now run GKD training with:")
print("  bash baseline/GKD/run_gkd_simple.sh")
