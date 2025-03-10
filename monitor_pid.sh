#!/bin/bash
# 設定要搜尋的進程名稱（pattern）
PATTERN1="python pytorch_speed.py"
PATTERN2="python tvm_speed.py"
PATTERN3="resnet"

LOGFILE1="log/mem_usage_pytorch_speed.log"
LOGFILE2="log/mem_usage_tvm_speed.log"
LOGFILE3="log/resnet.log"

# 輸出標題行
echo "Time,PID,%MEM,RSS(KB)" > "$LOGFILE1"
echo "Time,PID,%MEM,RSS(KB)" > "$LOGFILE2"
echo "Time,PID,%MEM,RSS(KB)" > "$LOGFILE3"

while true; do
    TIMESTAMP=$(date +"%s")
    
    # 追蹤所有符合第一個 pattern 的 PID
    PIDS1=($(pgrep -f "$PATTERN1"))
    for pid in "${PIDS1[@]}"; do
        MEM_USAGE=$(ps -p "$pid" -o %mem=)
        RSS=$(ps -p "$pid" -o rss=)
        echo "$TIMESTAMP,$pid,$MEM_USAGE,$RSS" >> "$LOGFILE1"
    done

    # 追蹤所有符合第二個 pattern 的 PID
    PIDS2=($(pgrep -f "$PATTERN2"))
    for pid in "${PIDS2[@]}"; do
        MEM_USAGE=$(ps -p "$pid" -o %mem=)
        RSS=$(ps -p "$pid" -o rss=)
        echo "$TIMESTAMP,$pid,$MEM_USAGE,$RSS" >> "$LOGFILE2"
    done

    # 追蹤所有符合第三個 pattern 的 PID
    PIDS3=($(pgrep -f "$PATTERN3"))
    for pid in "${PIDS3[@]}"; do
        MEM_USAGE=$(ps -p "$pid" -o %mem=)
        RSS=$(ps -p "$pid" -o rss=)
        echo "$TIMESTAMP,$pid,$MEM_USAGE,$RSS" >> "$LOGFILE3"
    done

    sleep 1
done

