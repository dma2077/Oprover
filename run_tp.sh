#!/bin/bash
set -x

# 检查参数数量
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name> <dataset_name> [tp_value] [split_num]"
    echo "Example: $0 DeepSeek-Prover-V2-7B FineLeanCorpus 8 00"
    echo "Example: $0 Goedel-Prover-V2-8B FineLeanCorpus 4 01"
    echo "Example: $0 Kimina-Prover-72B FineLeanCorpus 8 05"
    exit 1
fi

# 基本配置
cd /data/code/Oprover
export PYTHONPATH=$(pwd)
source /data/anaconda3/bin/activate oprover

# 数据输出根目录配置 (使用绝对路径)
DATA_ROOT_DIR="/madehua/data/oprover/generated_data"  # 修改为您想要的数据输出根目录

# 参数解析
MODEL_NAME=$1
DATASET_NAME=$2
TP_VALUE=${3:-"8"}
SPLIT_NUM=${4:-"00"}

# 格式化数字为两位数
SPLIT_NUM=$(printf "%02d" $SPLIT_NUM)

# 根据模型名称生成配置映射
get_model_config() {
    local model_name=$1
    case $model_name in
        "DeepSeek-Prover-V2-7B")
            echo "config/config_default.yaml"
            ;;
        "Goedel-Prover-V2-8B"|"Goedel-Prover-V2-32B")
            echo "config/config_goedel.yaml"
            ;;
        "Kimina-Prover-72B")
            echo "config/config_kimina.yaml"
            ;;
        *)
            echo "config/config_default.yaml"
            ;;
    esac
}

# 根据模型名称生成日志前缀
get_log_prefix() {
    local model_name=$1
    case $model_name in
        "DeepSeek-Prover-V2-7B")
            echo "dpsk"
            ;;
        "Goedel-Prover-V2-8B")
            echo "g8"
            ;;
        "Goedel-Prover-V2-32B")
            echo "g32"
            ;;
        "Kimina-Prover-72B")
            echo "k72"
            ;;
        *)
            # 默认使用模型名称的简化版本
            echo "${model_name,,}" | sed 's/[^a-z0-9]//g' | cut -c1-10
            ;;
    esac
}

# 根据模型名称生成批次大小
get_batch_size() {
    local model_name=$1
    case $model_name in
        "DeepSeek-Prover-V2-7B")
            echo "1500"
            ;;
        "Goedel-Prover-V2-8B"|"Goedel-Prover-V2-32B"|"Kimina-Prover-72B")
            echo "3000"
            ;;
        *)
            echo "1500"
            ;;
    esac
}

# 设置变量
CONFIG_FILE=$(get_model_config "$MODEL_NAME")
LOG_PREFIX=$(get_log_prefix "$MODEL_NAME")
BATCH_SIZE=$(get_batch_size "$MODEL_NAME")

# 构建数据集路径
SPLIT="${DATASET_NAME}/lean_statement_part_${SPLIT_NUM}"

# 生成输出目录名称（基于模型名称，使用数据根目录）
OUTPUT_DIR="${DATA_ROOT_DIR}/${MODEL_NAME}_results"

# 检查并创建输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 创建日志目录（也使用数据根目录）
LOG_DIR="${DATA_ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

# 根据tp值计算world_size
case $TP_VALUE in
    1)
        WORLD_SIZE=8
        ;;
    2)
        WORLD_SIZE=4
        ;;
    4)
        WORLD_SIZE=2
        ;;
    8)
        WORLD_SIZE=1
        ;;
    *)
        echo "Error: Invalid tp value. Must be 1, 2, 4, or 8."
        exit 1
        ;;
esac

echo "=== Configuration Summary ==="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Split: $SPLIT"
echo "Config: $CONFIG_FILE"
echo "Data Root Dir: $DATA_ROOT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Log Dir: $LOG_DIR"
echo "Log Prefix: $LOG_PREFIX"
echo "Batch Size: $BATCH_SIZE"
echo "TP Value: $TP_VALUE"
echo "World Size: $WORLD_SIZE"
echo "============================"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file $CONFIG_FILE not found, using config/config_default.yaml"
    CONFIG_FILE="config/config_default.yaml"
fi

# 临时修改模型配置中的tp值
echo "Temporarily updating model configuration tp value to $TP_VALUE..."
sed -i "s/'tp': [0-9]*/'tp': $TP_VALUE/" infer/models/__init__.py

# 启动进程
echo "Starting $WORLD_SIZE process(es) for split: $SPLIT (using tp=$TP_VALUE)"

# 存储进程的PID
PIDS=()
for i in $(seq 0 $((WORLD_SIZE-1))); do
    echo "Starting worker $i/$WORLD_SIZE"
    nohup python infer/infer.py \
      --config "$CONFIG_FILE" \
      --split "$SPLIT" \
      --mode proof_cot-bon \
      --model_name "$MODEL_NAME" \
      --output_dir "$OUTPUT_DIR" \
      --batch_size "$BATCH_SIZE" \
      --use_accel \
      --index "$i" \
      --world_size "$WORLD_SIZE" > "$LOG_DIR/${LOG_PREFIX}_worker_${SPLIT_NUM}_${i}_${WORLD_SIZE}_tp${TP_VALUE}.log" 2>&1 &
    
    # 保存进程PID
    PIDS+=($!)
    echo "Worker $i started with PID: ${PIDS[-1]}"
done

echo "All $WORLD_SIZE workers started in background."
echo "Check logs in $LOG_DIR/"
echo "Use 'ps aux | grep infer.py' to check running processes"
echo "Use 'tail -f $LOG_DIR/${LOG_PREFIX}_worker_${SPLIT_NUM}_*_${WORLD_SIZE}_tp${TP_VALUE}.log' to monitor logs"

# 等待所有进程完成
echo "Waiting for all workers to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All $WORLD_SIZE workers completed for split: $SPLIT (tp=$TP_VALUE)"

# 恢复原始的tp值（假设为8）
echo "Restoring original tp value to 8..."
sed -i "s/'tp': [0-9]*/'tp': 8/" infer/models/__init__.py

echo "Script execution completed successfully!"
