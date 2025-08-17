#!/bin/bash

# ========== 动态路径配置 ==========
PART_ID="01"  # 可根据需要修改
PART_NAME="lean_statement_part_${PART_ID}"

# 自动拼接目标路径
TARGET_TMP_FILE="results/DeepSeek-Prover-V2-7B_${PART_NAME}_proof-bon.jsonl.tmp"

# 拼接快照目录路径
SNAPSHOT_DIR="results/part${PART_ID}"

# ========== 找到最新的快照文件 ==========
LATEST_SNAPSHOT=$(ls -t ${SNAPSHOT_DIR}/snapshot_*.jsonl 2>/dev/null | head -n 1)

if [ -z "$LATEST_SNAPSHOT" ]; then
    echo "❌ 没有找到任何快照文件在 $SNAPSHOT_DIR"
    exit 1
fi

# ========== 恢复 ==========
mkdir -p "$(dirname "$TARGET_TMP_FILE")"

cp "$LATEST_SNAPSHOT" "$TARGET_TMP_FILE"

echo "✅ 恢复完成：$LATEST_SNAPSHOT → $TARGET_TMP_FILE"

