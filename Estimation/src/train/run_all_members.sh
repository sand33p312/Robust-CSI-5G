#!/bin/bash
# run_all_members.sh
# Train all 5 ensemble members sequentially.
# On Kaggle: open 5 separate notebooks and run each seed in parallel.
# On HPC: submit 5 PBS jobs using train/train_member.py --seed 0..4

set -e

for SEED in 0 1 2 3 4; do
    echo "========================================"
    echo " Training ensemble member SEED=${SEED}"
    echo "========================================"
    python train/train_member.py --seed $SEED
done

echo ""
echo "All 5 members trained."
echo "Checkpoints in: $(python -c 'import config; print(config.SAVE_DIR)')"