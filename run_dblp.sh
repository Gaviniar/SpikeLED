#!/bin/bash

echo "Running DBLP experiments..."

# Baseline (原始SpikeNet)
echo "=== Baseline ==="
python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4 \
    --use_gs_delay --use_tsr --use_learnable_p \
    --epochs 100 --seed 2022

# Ablation studies
echo "=== +L2S Gate only ==="
python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4 \
    --no-use_gs_delay --no-use_tsr --use_learnable_p \
    --epochs 100 --seed 2022

echo "=== +TSR only ==="
python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4 \
    --no-use_gs_delay --use_tsr --no-use_learnable_p \
    --epochs 100 --seed 2022

echo "=== +GS-Delay only ==="
python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4 \
    --use_gs_delay --no-use_tsr --no-use_learnable_p \
    --epochs 100 --seed 2022

echo "=== Full model (SpikeNet-LED) ==="
python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4 \
    --use_gs_delay --use_tsr --use_learnable_p --learnable_threshold \
    --alpha_warmup --lambda_spike 1e-4 --lambda_temp 1e-4 \
    --epochs 100 --seed 2022
