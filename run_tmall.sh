#!/bin/bash

echo "Running Tmall experiments..."

# Full model with different training ratios
for train_size in 0.4 0.6 0.8; do
    echo "=== Train size: $train_size ==="
    python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size $train_size \
        --use_gs_delay --use_tsr --use_learnable_p --learnable_threshold \
        --delay_groups 16 --alpha_warmup --lambda_spike 5e-5 --lambda_temp 5e-5 \
        --epochs 100 --seed 2022
done
