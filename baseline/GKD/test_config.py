#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 GKD 配置是否正确
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("Testing GKD configuration...")
print("=" * 60)

try:
    import hydra
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    
    # Initialize Hydra
    with initialize(version_base=None, config_path="../../verl/trainer/config"):
        # Compose config
        cfg = compose(config_name="gkd_trainer")
        
        print("✓ GKD configuration loaded successfully")
        print("\n--- GKD Configuration ---")
        print(OmegaConf.to_yaml(cfg.gkd))
        
        print("\n--- Key Configurations ---")
        print(f"  hybrid_engine: {cfg.actor_rollout_ref.hybrid_engine}")
        print(f"  rollout.n: {cfg.actor_rollout_ref.rollout.n}")
        print(f"  adv_estimator: {cfg.algorithm.adv_estimator}")
        print(f"  use_rl_loss: {cfg.gkd.use_rl_loss}")
        print(f"  distill_loss_coef: {cfg.gkd.distill_loss_coef}")
        
        # Validate critical settings
        assert cfg.actor_rollout_ref.hybrid_engine == True, "hybrid_engine should be True"
        assert cfg.actor_rollout_ref.rollout.n == 1, "rollout.n should be 1 for pure distillation"
        assert cfg.gkd.use_rl_loss == False, "use_rl_loss should be False for pure distillation"
        assert cfg.gkd.enable_teacher == True, "enable_teacher should be True"
        
        print("\n✓ All validations passed!")
        
        # Print training mode
        if cfg.gkd.use_rl_loss:
            print("\n🔄 Mode: RL + Distillation Hybrid")
            print(f"   Loss = {cfg.gkd.rl_loss_coef} * pg_loss + {cfg.gkd.distill_loss_coef} * distill_loss")
        else:
            print("\n🎯 Mode: Pure Distillation")
            print(f"   Loss = {cfg.gkd.distill_loss_coef} * distill_loss")
        
except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("Configuration test completed successfully! ✓")
