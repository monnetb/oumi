#!/bin/bash
#
#
set -eux 
export OUMI_ROOT=${OUMI_ROOT:-$HOME/oumi}
export OUMI_GPU=${OUMI_GPU:-NVIDIA}
export TGT=oumi.venv.${OUMI_GPU}

mkdir -p ${OUMI_ROOT}
pushd ${OUMI_ROOT}
rm -rf ${TGT}
python -m venv ${TGT}
popd

source ${OUMI_ROOT}/${TGT}/bin/activate

pip install --upgrade  pip

pip3 install -e .  vllm

