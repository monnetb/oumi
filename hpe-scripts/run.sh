#!/bin/bash

#set -eux 

export OUMI_ROOT=${OUMI_ROOT:-$HOME/oumi}
export OUMI_GPU=${OUMI_GPU:-NVIDIA}
export TGT=oumi.venv.${OUMI_GPU}

#check if a yaml file is provided with --yaml option. use argument as the yaml file
# Parse arguments for --yaml option
YAML_FILE="train-llama3.1-8b-h200.yaml"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaml)
            shift
            if [[ $# -gt 0 ]]; then
                YAML_FILE="$1"
                shift
            else
                echo "Error: --yaml requires a filename argument."
                exit 1
            fi
            ;;
        *)
            shift
            ;;
    esac
done
export YAML_FILE

#check if ${OUMI_ROOT}/${TGT} exists, if not exit and ask for OUMI_ROOT 
if [ ! -d ${OUMI_ROOT}/${TGT} ]; then
    echo "Please set OUMI_ROOT to the directory where the virtual environment is installed"
    exit 1
fi


# Activate the virtual environment
source ${OUMI_ROOT}/${TGT}/bin/activate

# Set MASTER_ADDR to the first allocated node and MASTER_PORT to 8000
#MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#export MASTER_ADDR="`hostname`"
#export MASTER_PORT=8000

# count number of core online , excluding the hyperthreading cores
export NUM_CORES=$(lscpu | grep 'Core(s) per socket' | awk '{print $4}')
export OMP_NUM_THREADS=$((NUM_CORES/2))
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

set -x
#export OUMI_NUM_NODES=${SLURM_NNODES}
#export OUMI_MASTER_ADDR=${MASTER_ADDR}

#export TORCH_LOGS="+dynamo" 
#export TORCH_LOGS="+dynamo,guards,bytecode"
#export TORCHDYNAMO_VERBOSE=1
#export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

export LOGLEVEL=DEBUG
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL 
#export TORCH_CPP_LOG_LEVEL=INFO
#export TORCH_DISTRIBUTED_TIMEOUT=100
#export TORCH_LOGS="+dynamo"
#export TORCHDYNAMO_VERBOSE=1
# Run the training command
#oumi distributed torchrun --log-level DEBUG --nnodes=${OUMI_NUM_NODES} --master-addr=${OUMI_MASTER_ADDR} --master-port=${MASTER_PORT}  -m oumi train  -c $PWD/${YAML_FILE}
srun --ntasks=2 --ntasks-per-node=1 --nodes=2 oumi distributed torchrun --log-level DEBUG -m oumi train  -c $PWD/${YAML_FILE}
