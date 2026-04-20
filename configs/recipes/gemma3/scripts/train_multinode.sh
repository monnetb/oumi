#!/bin/bash
# Multi-node training launch script for Gemma 3 12B on 2x H200 nodes
# Usage: ./train_multinode.sh [master_ip] [node_rank]
# If no args provided, runs in single-node mode

set -e

# Configuration
CONFIG="configs/recipes/gemma3/sft/12b_multinode/train.yaml"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}

# Get master IP from argument or default
MASTER_ADDR=${1:-"127.0.0.1"}

# NCCL optimizations for Infiniband
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=136
export NCCL_IB_SL=3
export NCCL_SOCKET_IFNAME=eth0

# DeepSpeed and distributed training env vars
export OUMI_MASTER_ADDR=$MASTER_ADDR
export OUMI_MASTER_PORT=$MASTER_PORT
export OUMI_NUM_NODES=$NNODES
export OUMI_TOTAL_NUM_GPUS=$((NNODES * GPUS_PER_NODE))

# Get node rank from argument or environment
if [ -n "$2" ]; then
    NODE_RANK=$2
fi

echo "=============================================="
echo "Gemma 3 12B Multi-Node Training"
echo "=============================================="
echo "Config: $CONFIG"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $NNODES"
echo "Node Rank: $NODE_RANK"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $((NNODES * GPUS_PER_NODE))"
echo "=============================================="

# Build torchrun command
TORCHRUN_CMD="torchrun"
if ! command -v torchrun &> /dev/null; then
    TORCHRUN_CMD="python -m torch.distributed.run"
fi

# Run training
$TORCHRUN_CMD \
    --nnodes=$NNODES \
    --node-rank=$NODE_RANK \
    --nproc-per-node=$GPUS_PER_NODE \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    -m oumi train -c $CONFIG

echo "Training completed!"
