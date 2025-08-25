#!/bin/bash

set -x

export PYTHONPATH=$(pwd)

# 设置CUDA环境变量
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /data/code/Oprover
source /data/anaconda3/bin/activate oprover

python infer/infer.py \
  --config config/config_goedel.yaml \
  --split lean_statement_part_01 \
  --mode proof_cot-bon \
  --model_name Goedel-Prover-V2-8B \
  --output_dir Goedel-Prover-V2-8B_results \
  --batch_size 1500 \
  --use_accel \
  --index 0 \
  --world_size 1
