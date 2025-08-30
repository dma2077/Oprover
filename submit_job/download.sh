for i in {0..0}; do
    echo "Submitting job for group id: $i"
    
    # 提交任务，使用tp=8处理文件编号i，后台运行
    nohup bash /data/code/submit_job/submit_job_cpu.sh "bash /data/code/Oprover/fasttext/download.sh $i" project > /data/code/Oprover/submit_job/logs/fasttext_submit_job_${i}.log 2>&1 &
    
    echo "Job submitted for group id $i (PID: $!)"
    echo "----------------------------------------"
    
done