#!/bin/bash

set -x



export PYTHONPATH=$(pwd)


python infer/infer.py \
  --config config/config_default.yaml \
  --split lean_statement_part_01 \
  --mode proof_cot-bon \
  --model_name DeepSeek-Prover-V2-7B \
  --output_dir DeepSeek-Prover-V2-7B_results \
  --batch_size 1000 \
  --use_accel \
  --index 0 \
  --world_size 1


python infer/infer.py \
  --config config/config_goedel.yaml \
  --split lean_statement_part_01 \
  --mode proof_cot-bon \
  --model_name Goedel-Prover-V2-32B \
  --output_dir Goedel-Prover-V2-32B_results \
  --batch_size 1000 \
  --use_accel \
  --index 0 \
  --world_size 1


python infer/infer.py \
  --config config/config_kimina.yaml \
  --split lean_statement_part_01 \
  --mode proof_kimina-bon \
  --model_name Kimina-Prover-72B \
  --output_dir Kimina-Prover-72B_results \
  --batch_size 1000 \
  --use_accel \
  --index 0 \
  --world_size 1
