#!/bin/bash
#SBATCH --job-name=gemma3-12b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h200

set -e

echo "=========================================="
echo "Gemma 3 12B Multi-Node Training (SLURM)"
echo "=========================================="
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))"
echo "=========================================="

# Get node list
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Set environment variables for multi-node training
export OUMI_NUM_NODES=$SLURM_JOB_NUM_NODES
export OUMI_TOTAL_NUM_GPUS=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))
export OUMI_MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# NCCL Infiniband optimizations
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=136
export NCCL_IB_SL=3

echo "Master address: $OUMI_MASTER_ADDR"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate oumi

# Run training
torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node-rank=$SLURM_PROCID \
    --nproc-per-node=$SLURM_GPUS_ON_NODE \
    --master-addr=$OUMI_MASTER_ADDR \
    --master-port=29500 \
    -m oumi train \
    -c configs/recipes/gemma3/sft/12b_multinode/train.yaml

echo "Training complete!"
