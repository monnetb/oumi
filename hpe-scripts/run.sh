#!/bin/bash

export OUMI_ROOT=${OUMI_ROOT:-$HOME/oumi}
export OUMI_GPU=${OUMI_GPU:-NVIDIA}
export TGT=oumi.venv.${OUMI_GPU}

#check if a yaml file is provided with --yaml option. use argument as the yaml file
if [ "$1" == "--yaml" ]; then
    shift
    export YAML_FILE=$1
    shift
else
    export YAML_FILE="train-llama3.1-8b-h100.yaml"
fi

#check if ${OUMI_ROOT}/${TGT} exists, if not exit and ask for OUMI_ROOT 
if [ ! -d ${OUMI_ROOT}/${TGT} ]; then
    echo "Please set OUMI_ROOT to the directory where the virtual environment is installed"
    exit 1
fi


# Activate the virtual environment
source ${OUMI_ROOT}/${TGT}/bin/activate

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
oumi distributed torchrun --log-level INFO  -m oumi train  -c $PWD/${YAML_FILE}
