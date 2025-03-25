#!/bin/bash

export OUMI_ROOT=$home
export OUMI_GPU=NVIDIA

# Activate the virtual environment
source ${OUMI_ROOT}/oumi.venv.${OUMI_GPU}/bin/activate

# Set MASTER_ADDR to the first allocated node and MASTER_PORT to 8000
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR
export MASTER_PORT=8000

# count number of core online , excluding the hyperthreading cores
export NUM_CORES=$(lscpu | grep 'Core(s) per socket' | awk '{print $4}')
export OMP_NUM_THREADS=$((NUM_CORES/2))
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

#export TORCH_LOGS="+dynamo" 
#export TORCH_LOGS="+dynamo,guards,bytecode"
#export TORCHDYNAMO_VERBOSE=1
#export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

# Run the training command
#oumi train -c ./train.yaml
oumi distributed torchrun -m oumi train  -c $PWD/configs/recipes/llama3_3/sft/70b_full/train.yaml
