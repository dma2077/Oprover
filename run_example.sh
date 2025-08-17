#!/bin/bash

# ========== 用户设置 ==========
PART_ID="09"  # 可根据需要修改
PART_NAME="lean_statement_part_${PART_ID}"
MODEL_NAME="DeepSeek-Prover-V2-7B"  

# ========== 设置通用环境变量 ==========
export HF_ENDPOINT=https://hf-mirror.com 
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118 
export no_proxy=.byted.org

# ========== 路径配置 ==========
BASE_DIR="/opt/tiger/rh2"
PROJECT_DIR="${BASE_DIR}/LeanProof"
RESULTS_DIR="${MODEL_NAME}_results"
TMP_FILE="${RESULTS_DIR}/${MODEL_NAME}_${PART_NAME}_proof_cot-bon.jsonl.tmp"  
JSONL_FILE="${RESULTS_DIR}/${MODEL_NAME}_${PART_NAME}_proof_cot-bon.jsonl" 
SNAPSHOT_DIR="${RESULTS_DIR}/${MODEL_NAME}_cot/part${PART_ID}"  
HDFS_PROJECT_PATH="hdfs://haruna/home/byte_data_seed/ssd_hldy/user/makaijing.69/LeanProof"
HDFS_RESULT_DIR="${HDFS_PROJECT_PATH}/${RESULTS_DIR}/${MODEL_NAME}_cot/part${PART_ID}/"  

# ========== 拉取项目代码 ==========
echo "? 下载 LeanProof 项目中..."
hdfs dfs -get "$HDFS_PROJECT_PATH" "$BASE_DIR/"

# ========== 等待 HDFS 下载完成 ==========
while [ ! -f "${PROJECT_DIR}/requirements.txt" ]; do
  echo "⏳ 等待 HDFS 下载完成..."
  sleep 5
done

cd "$PROJECT_DIR"

# ========== 安装依赖 ==========
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
pip3 uninstall -y vllm
pip3 install --no-cache-dir --force-reinstall vllm

# ========== 恢复最新快照 ==========
echo "? 恢复上次推理进度..."
LATEST_SNAPSHOT=$(ls -t "${SNAPSHOT_DIR}/snapshot_"*.jsonl 2>/dev/null | head -n 1)

if [ -z "$LATEST_SNAPSHOT" ]; then
    echo "⚠️  没有找到任何快照文件在 ${SNAPSHOT_DIR}，将从头开始运行"
else
    mkdir -p "$(dirname "$TMP_FILE")"
    cp "$LATEST_SNAPSHOT" "$TMP_FILE"
    echo "✅ 恢复完成：$LATEST_SNAPSHOT → $TMP_FILE"
fi

# ========== 启动监控上传 ==========
echo "? 启动后台监控上传 $TMP_FILE 到 $HDFS_RESULT_DIR"
nohup bash monitor.sh "$TMP_FILE" "$HDFS_RESULT_DIR" > "watch_${PART_ID}.log" 2>&1 &

# ========== 启动推理 ==========
echo "? 启动模型推理任务：$PART_NAME 使用模型 $MODEL_NAME"
python infer/infer.py \
  --config config/config_default.yaml \
  --split "$PART_NAME" \
  --mode proof_cot-bon \
  --model_name "$MODEL_NAME" \
  --output_dir "$RESULTS_DIR" \
  --batch_size 1000 \
  --use_accel \
  --index 0 \
  --world_size 1

# ========== 可选：推理结束后手动上传一次结果 ==========
echo "? 推理完成，手动上传最终结果到 HDFS..."
hdfs dfs -mkdir -p "$HDFS_RESULT_DIR"
hdfs dfs -put -f "$JSONL_FILE" "$HDFS_RESULT_DIR"

echo "✅ 脚本执行完毕 ✅"