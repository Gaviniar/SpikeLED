# SpikeNet-LED: Learnable Efficient Delays

> Enhanced SpikeNet with **L**earnable **E**fficient **D**elays for scalable dynamic graph representation learning.

## New Features

### Core Components

1. **GS-Delay** (Group-Shared Sparse Delay Kernel): Efficient approximation of Gaussian delay kernels with group-shared sparse 1D convolution.
2. **L2S-Gate** (Learnable Local 2-source Sampling Gate): Adaptive gating mechanism for historical vs. current graph sampling allocation.
3. **TSR-Readout** (Temporal Separable Readout): Parameter-efficient temporal readout with depthwise separable convolutions.

### Training Enhancements (Optional)

- **SCAT**: Self-adaptive threshold with surrogate gradient curriculum
- **Temporal Consistency**: Lightweight temporal smoothness regularization

## New Usage

```bash
# Run with all enhancements
python main.py --dataset dblp --hids 128 10 --batch_size 1024 \
    --use_gs_delay --use_tsr --use_learnable_p --learnable_threshold \
    --alpha_warmup --lambda_spike 1e-4 --lambda_temp 1e-4

# Ablation studies
python main.py --dataset dblp --use_gs_delay  # Only delay kernel
python main.py --dataset dblp --use_tsr       # Only separable readout  
python main.py --dataset dblp --use_learnable_p  # Only adaptive gating
```
