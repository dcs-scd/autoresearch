#!/bin/bash
# Launch autoresearch training.
# Usage:
#   ./run.sh          # auto-detect all GPUs
#   ./run.sh 3        # use 3 GPUs
#   ./run.sh 1        # single GPU (no DDP)

NUM_GPUS=${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}

if [ "$NUM_GPUS" -le 1 ]; then
    echo "Running single-GPU (no DDP)"
    uv run train.py
else
    echo "Running with $NUM_GPUS GPUs (DDP)"
    uv run torchrun --nproc_per_node="$NUM_GPUS" train.py
fi
