#!/bin/bash
set -x


cd /data/code/Oprover
export PYTHONPATH=$(pwd)
source /data/anaconda3/bin/activate oprover

# 参数：tp值和文件编号
TP_VALUE=${1:-"8"}
SPLIT_NUM=${2:-"00"}

# 格式化数字为两位数
SPLIT_NUM=$(printf "%02d" $SPLIT_NUM)

# 拼接完整的文件名
SPLIT="FineLeanCorpus/lean_statement_part_${SPLIT_NUM}"

# 输出目录
OUTPUT_DIR="Kimina-Prover-72B_results"

# 检查并创建输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 创建日志目录
LOG_DIR="logs"
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

echo "Configuration: tp=$TP_VALUE, world_size=$WORLD_SIZE, split=$SPLIT"

# 临时修改模型配置中的tp值
echo "Temporarily updating model configuration tp value to $TP_VALUE..."
sed -i "s/'tp': [0-9]*/'tp': $TP_VALUE/" infer/models/__init__.py

# 启动进程
echo "Starting $WORLD_SIZE process(es) for split: $SPLIT (using tp=$TP_VALUE)"

# 存储进程的PID
PIDS=()
PIDS=()
for i in $(seq 0 $((WORLD_SIZE-1))); do
    echo "Starting worker $i/$WORLD_SIZE"
    nohup python infer/infer.py \
      --config config/config_kimina.yaml \
      --split "$SPLIT" \
      --mode proof_cot-bon \
      --model_name Kimina-Prover-72B \
      --output_dir "$OUTPUT_DIR" \
      --batch_size 3000 \
      --use_accel \
      --index "$i" \
      --world_size "$WORLD_SIZE" > "$LOG_DIR/k72_worker_${SPLIT_NUM}_${i}_${WORLD_SIZE}_tp${TP_VALUE}.log" 2>&1 &
    
    # 保存进程PID
    PIDS+=($!)
    echo "Worker $i started with PID: ${PIDS[-1]}"
done

echo "All $WORLD_SIZE workers started in background."
echo "Check logs in $LOG_DIR/"
echo "Use 'ps aux | grep infer.py' to check running processes"
echo "Use 'tail -f $LOG_DIR/k72_worker_${SPLIT_NUM}_*_${WORLD_SIZE}_tp${TP_VALUE}.log' to monitor logs"

# 等待所有进程完成
echo "Waiting for all workers to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All $WORLD_SIZE workers completed for split: $SPLIT (tp=$TP_VALUE)"

# 恢复原始的tp值（假设为8）
echo "Restoring original tp value to 8..."
sed -i "s/'tp': [0-9]*/'tp': 8/" infer/models/__init__.py
