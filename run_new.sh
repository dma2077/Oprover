#!/bin/bash
# ========== 用户设置 ==========
PART_ID="15" # 可根据需要修改
PART_NAME="lean_statement_part_${PART_ID}"
MODEL_NAME="DeepSeek-Prover-V2-7B"

# ========== 设置通用环境变量 ==========
export HF_ENDPOINT=https://hf-mirror.com

# ========== 路径配置 ==========
TMP_FILE="results/${MODEL_NAME}_${PART_NAME}_proof-bon.jsonl.tmp"
JSONL_FILE="results/${MODEL_NAME}_${PART_NAME}_proof-bon.jsonl"
SNAPSHOT_DIR="results/${MODEL_NAME}/part${PART_ID}"

# ========== 创建目录 ==========
mkdir -p results
mkdir -p "$SNAPSHOT_DIR"

# ========== 恢复快照 ==========
LATEST_SNAPSHOT=$(ls -t "${SNAPSHOT_DIR}/snapshot_"*.jsonl 2>/dev/null | head -n 1)
if [ -n "$LATEST_SNAPSHOT" ]; then
    cp "$LATEST_SNAPSHOT" "$TMP_FILE"
    echo "恢复快照：$LATEST_SNAPSHOT"
fi  
export PYTHONPATH="/llm_reco/dehua/code/Oprover"
# ========== 启动推理 ==========
python infer/infer.py \
--config config/config_default.yaml \
--split "$PART_NAME" \
--mode proof-bon \
--model_name "$MODEL_NAME" \
--output_dir results \
--batch_size 1000 \
--use_accel \
--index 0 \
--world_size 1 