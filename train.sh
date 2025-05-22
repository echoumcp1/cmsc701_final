#!/usr/bin/env bash
#
# Launches train.py on 4 GPUs using torchrun + HF Trainer (DDP under the hood).

# — Select GPUs 0 and 1 (change if you want different IDs)
module load cuda
module load gcc/14.2.0

eval "$(micromamba shell hook --shell bash)"
micromamba activate genomics
cd /fs/nexus-scratch/zli12321/nucleotide-transformer/workspace


export CUDA_VISIBLE_DEVICES=0,1,2,3

# — Number of processes = number of GPUs
GPUS=4

# — (Optional) choose a specific master port if you have conflicts
# export MASTER_PORT=12345

# — Launch
torchrun \
  --nproc_per_node=$GPUS \
  train.py \
  "$@"
