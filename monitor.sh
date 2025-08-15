#!/bin/bash

# ====== 参数检查与默认值 ======
if [ $# -lt 2 ]; then
    echo "用法: $0 <SOURCE_FILE> <HDFS_DIR>"
    echo "示例: $0 /path/to/file.jsonl hdfs://path/to/hdfs/dir/"
    exit 1
fi

SOURCE_FILE="$1"
HDFS_DIR="$2"

# ====== 用户自定义配置 ======
BACKUP_DIR="/tmp/model_jsonl_snapshots"
BATCH_SIZE=1000       # 每新增多少行触发上传
SLEEP_INTERVAL=300    # 检查间隔（秒）
MAX_BACKUP_COUNT=2    # 保留的最大快照数量
# ===========================

mkdir -p "$BACKUP_DIR"

echo "📡 开始监控文件: $SOURCE_FILE"
echo "📦 快照将保存在: $BACKUP_DIR"
echo "🚀 上传目标 HDFS 路径: $HDFS_DIR"

LAST_LINE_COUNT=0
SNAPSHOT_INDEX=0

while true; do
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "[等待] 文件未找到: $SOURCE_FILE"
        sleep $SLEEP_INTERVAL
        continue
    fi

    CURRENT_LINE_COUNT=$(wc -l < "$SOURCE_FILE")
    DELTA=$((CURRENT_LINE_COUNT - LAST_LINE_COUNT))

    if [ "$DELTA" -ge "$BATCH_SIZE" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        SNAPSHOT_FILE="$BACKUP_DIR/snapshot_${TIMESTAMP}.jsonl"

        cp "$SOURCE_FILE" "$SNAPSHOT_FILE"

        hdfs dfs -mkdir -p "$HDFS_DIR"
        hdfs dfs -put -f "$SNAPSHOT_FILE" "$HDFS_DIR"

        echo "[上传成功] $SNAPSHOT_FILE -> $HDFS_DIR"
        
        # 删除旧的快照文件（最多保留 MAX_BACKUP_COUNT 个快照）
        BACKUP_FILES=($(ls -t "$BACKUP_DIR/snapshot_*.jsonl"))
        BACKUP_COUNT=${#BACKUP_FILES[@]}

        if [ "$BACKUP_COUNT" -gt "$MAX_BACKUP_COUNT" ]; then
            FILES_TO_DELETE=$((BACKUP_COUNT - MAX_BACKUP_COUNT))
            for i in $(seq 0 $((FILES_TO_DELETE - 1))); do
                rm -f "${BACKUP_FILES[$i]}"
                echo "[删除] 旧快照: ${BACKUP_FILES[$i]}"
            done
        fi

        LAST_LINE_COUNT=$CURRENT_LINE_COUNT
        SNAPSHOT_INDEX=$((SNAPSHOT_INDEX + 1))
    else
        echo "[跳过] 当前行数 $CURRENT_LINE_COUNT，未增加 $BATCH_SIZE 行"
    fi

    sleep $SLEEP_INTERVAL
done
