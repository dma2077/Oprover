#!/bin/bash

# 难度评估推理引擎运行脚本

# 默认参数
INPUT_FILE="/madehua/data/Leanabell-Prover-Formal-Statement/formal_statement_with_uuid.parquet"
OUTPUT_FILE="/madehua/data/Leanabell-Prover-Formal-Statement/formal_statement_with_uuid_difficulty.jsonl"
WORKERS=32
BATCH_SIZE=1000

echo "🚀 启动难度评估推理引擎"
echo "📁 输入文件: $INPUT_FILE"
echo "📁 输出文件: $OUTPUT_FILE"
echo "🔧 工作线程: $WORKERS"
echo "📦 批次大小: $BATCH_SIZE"
echo ""

# 检查输入文件
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ 错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 运行Python脚本
echo "开始执行..."
python inference.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --workers $WORKERS \
    --batch-size $BATCH_SIZE

if [[ $? -eq 0 ]]; then
    echo "✅ 任务完成!"
else
    echo "❌ 任务失败!"
fi
