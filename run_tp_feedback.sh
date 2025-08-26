#!/bin/bash

# Multi-Round Feedback Inference Script (Simplified)
# 简化的多轮反馈推理脚本

set -e

# 参数解析
SPLIT_NUM=${1:-"00"}
MODEL_NAME=${2:-"DeepSeek-Prover-V2-7B"}
DATASET_NAME=${3:-"FineLeanCorpus"}
PROMPT_CONFIG=${4:-"proof_cot_feedback"}
MAX_ROUNDS=${5:-"32"}

# 配置
cd /data/code/Oprover
export PYTHONPATH=$(pwd)
LOG_DIR="/madehua/data/oprover/generated_data/${DATASET_NAME}/${MODEL_NAME}_results/logs"
mkdir -p "$LOG_DIR"

echo "====================================="
echo "Multi-Round Feedback Inference"
echo "Model: $MODEL_NAME, Dataset: $DATASET_NAME, Split: $SPLIT_NUM"
echo "====================================="

# ==============================================================================

# 1. 启动验证服务（后台）
echo "Starting validation service..."
source /data/anaconda3/bin/activate proof
cd /data/code/kimina-lean-server/server/proof
nohup bash start_servers.sh "01" > "$LOG_DIR/validation_server.log" 2>&1 &
VALIDATION_PID=$!
echo "Validation service PID: $VALIDATION_PID"

# 等待验证服务启动
echo "Waiting for validation service to start..."
sleep 15
echo "Validation service should be running now"

# 2. 启动推理脚本（后台）
echo "Starting inference script..."
cd /data/code/Oprover
source /data/anaconda3/bin/activate oprover
python run_tp_feedback.py \
    --split_num "$SPLIT_NUM" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --prompt_config "$PROMPT_CONFIG" \
    --max_rounds "$MAX_ROUNDS" \
    > "$LOG_DIR/inference_${MODEL_NAME}_${SPLIT_NUM}.log" 2>&1 &
INFERENCE_PID=$!
echo "Inference script PID: $INFERENCE_PID"

# 3. 等待推理完成
echo "Waiting for inference to complete..."
wait $INFERENCE_PID

# 4. 清理验证服务
echo "Cleaning up validation service..."
pkill -f "start_servers.sh 01" || true
pkill -f "python -m server" || true

echo "====================================="
echo "Process completed!"
echo "Logs available in: $LOG_DIR"
echo "====================================="

