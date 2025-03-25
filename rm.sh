#!/bin/bash

# 目录列表
dirs=("default" "ncdown" "ncddown" "ncupload" "nc" "ncd" "agr" "agrw" "ncagr" "ncagrw" "ncdagrw" "dncagr" "dncdagr" "hier")

# 循环遍历每个目录
for dir in "${dirs[@]}"; do
    # 删除目录及其内容
    rm -r "./log/$dir/"
    # 创建新目录
    mkdir "./log/$dir/"
    echo "Directory ./log/$dir/ has been removed and recreated."
done
