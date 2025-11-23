@echo off
REM ============================================================================
REM Adaptive Subspace Allocation Experiments - ICML 2025 Project
REM ============================================================================
REM Total: 30 training runs (10 configs × 3 seeds)
REM Model: GPT-2 Small (124M), LoRA rank r=8
REM Budgets: d ∈ {1000, 2000}
REM Allocations: uniform, fixed (0.2, 0.5, 0.8), learned
REM ============================================================================

REM Set paths (ADJUST THESE)
set DATA_DIR=data/openwebtext
set OUT_DIR=out/adaptive_experiments

echo ============================================================================
echo BUDGET d=1000 EXPERIMENTS (15 runs)
echo ============================================================================

REM ---------- d=1000: Baseline Uniform ----------
echo [1/30] d=1000 Uniform seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_uniform_seed42 --login.wandb_run_name=d1000_uniform_s42 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=uniform --sublora.allocation_ratio=0.5

echo [2/30] d=1000 Uniform seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_uniform_seed123 --login.wandb_run_name=d1000_uniform_s123 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=uniform --sublora.allocation_ratio=0.5

echo [3/30] d=1000 Uniform seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_uniform_seed999 --login.wandb_run_name=d1000_uniform_s999 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=uniform --sublora.allocation_ratio=0.5

REM ---------- d=1000: Fixed B-heavy (ratio=0.8, d_B=800, d_A=200) ----------
echo [4/30] d=1000 Fixed B-heavy seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_fixed_bheavy_seed42 --login.wandb_run_name=d1000_bheavy_s42 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.8

echo [5/30] d=1000 Fixed B-heavy seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_fixed_bheavy_seed123 --login.wandb_run_name=d1000_bheavy_s123 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.8

echo [6/30] d=1000 Fixed B-heavy seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_fixed_bheavy_seed999 --login.wandb_run_name=d1000_bheavy_s999 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.8

REM ---------- d=1000: Fixed Equal (ratio=0.5, d_B=500, d_A=500) ----------
echo [7/30] d=1000 Fixed Equal seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_fixed_equal_seed42 --login.wandb_run_name=d1000_equal_s42 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.5

echo [8/30] d=1000 Fixed Equal seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_fixed_equal_seed123 --login.wandb_run_name=d1000_equal_s123 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.5

echo [9/30] d=1000 Fixed Equal seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_fixed_equal_seed999 --login.wandb_run_name=d1000_equal_s999 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.5

REM ---------- d=1000: Fixed A-heavy (ratio=0.2, d_B=200, d_A=800) ----------
echo [10/30] d=1000 Fixed A-heavy seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_fixed_aheavy_seed42 --login.wandb_run_name=d1000_aheavy_s42 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.2

echo [11/30] d=1000 Fixed A-heavy seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_fixed_aheavy_seed123 --login.wandb_run_name=d1000_aheavy_s123 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.2

echo [12/30] d=1000 Fixed A-heavy seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_fixed_aheavy_seed999 --login.wandb_run_name=d1000_aheavy_s999 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.2

REM ---------- d=1000: Learned Gating ----------
echo [13/30] d=1000 Learned seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_learned_seed42 --login.wandb_run_name=d1000_learned_s42 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=learned --sublora.allocation_ratio=0.5

echo [14/30] d=1000 Learned seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_learned_seed123 --login.wandb_run_name=d1000_learned_s123 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=learned --sublora.allocation_ratio=0.5

echo [15/30] d=1000 Learned seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d1000_learned_seed999 --login.wandb_run_name=d1000_learned_s999 --sublora.intrinsic_dim=1000 --sublora.allocation_mode=learned --sublora.allocation_ratio=0.5

echo ============================================================================
echo BUDGET d=2000 EXPERIMENTS (15 runs)
echo ============================================================================

REM ---------- d=2000: Baseline Uniform ----------
echo [16/30] d=2000 Uniform seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_uniform_seed42 --login.wandb_run_name=d2000_uniform_s42 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=uniform --sublora.allocation_ratio=0.5

echo [17/30] d=2000 Uniform seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_uniform_seed123 --login.wandb_run_name=d2000_uniform_s123 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=uniform --sublora.allocation_ratio=0.5

echo [18/30] d=2000 Uniform seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_uniform_seed999 --login.wandb_run_name=d2000_uniform_s999 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=uniform --sublora.allocation_ratio=0.5

REM ---------- d=2000: Fixed B-heavy (ratio=0.8, d_B=1600, d_A=400) ----------
echo [19/30] d=2000 Fixed B-heavy seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_fixed_bheavy_seed42 --login.wandb_run_name=d2000_bheavy_s42 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.8

echo [20/30] d=2000 Fixed B-heavy seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_fixed_bheavy_seed123 --login.wandb_run_name=d2000_bheavy_s123 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.8

echo [21/30] d=2000 Fixed B-heavy seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_fixed_bheavy_seed999 --login.wandb_run_name=d2000_bheavy_s999 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.8

REM ---------- d=2000: Fixed Equal (ratio=0.5, d_B=1000, d_A=1000) ----------
echo [22/30] d=2000 Fixed Equal seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_fixed_equal_seed42 --login.wandb_run_name=d2000_equal_s42 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.5

echo [23/30] d=2000 Fixed Equal seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_fixed_equal_seed123 --login.wandb_run_name=d2000_equal_s123 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.5

echo [24/30] d=2000 Fixed Equal seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_fixed_equal_seed999 --login.wandb_run_name=d2000_equal_s999 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.5

REM ---------- d=2000: Fixed A-heavy (ratio=0.2, d_B=400, d_A=1600) ----------
echo [25/30] d=2000 Fixed A-heavy seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_fixed_aheavy_seed42 --login.wandb_run_name=d2000_aheavy_s42 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.2

echo [26/30] d=2000 Fixed A-heavy seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_fixed_aheavy_seed123 --login.wandb_run_name=d2000_aheavy_s123 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.2

echo [27/30] d=2000 Fixed A-heavy seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_fixed_aheavy_seed999 --login.wandb_run_name=d2000_aheavy_s999 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.2

REM ---------- d=2000: Learned Gating ----------
echo [28/30] d=2000 Learned seed=42
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_learned_seed42 --login.wandb_run_name=d2000_learned_s42 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=learned --sublora.allocation_ratio=0.5

echo [29/30] d=2000 Learned seed=123
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_learned_seed123 --login.wandb_run_name=d2000_learned_s123 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=learned --sublora.allocation_ratio=0.5

echo [30/30] d=2000 Learned seed=999
python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=%DATA_DIR% --login.out_dir=%OUT_DIR%/d2000_learned_seed999 --login.wandb_run_name=d2000_learned_s999 --sublora.intrinsic_dim=2000 --sublora.allocation_mode=learned --sublora.allocation_ratio=0.5

echo ============================================================================
echo ALL 30 EXPERIMENTS COMPLETE!
echo ============================================================================
