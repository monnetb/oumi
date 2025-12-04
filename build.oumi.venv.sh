#!/bin/bash
#
#
set -eux 
export OUMI_ROOT=${OUMI_ROOT:-$HOME/oumi}
export OUMI_GPU=${OUMI_GPU:-NVIDIA}
export TGT=oumi.venv.`uname -m`.${OUMI_GPU}

mkdir -p ${OUMI_ROOT}
cd ${OUMI_ROOT}
rm -rf ${TGT}
python -m venv ${TGT}

source ${OUMI_ROOT}/${TGT}/bin/activate

pip install --upgrade  pip

#pip3 install oumi[gpu] vllm
pip3 install  -e /nfs/bruno/APPLICATIONS/LLM/oumi  #  ".[dev]"
pip3 install liger-kernel
#pip3 install vllm
