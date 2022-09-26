#!/bin/bash

##################################################
# run.sh
# 使用方法：$ bash run.sh
# created by LuYF-Lemon-love <luyanfeng@qq.com>
##################################################

# 生成临时数据文件
python3 n-n.py

# train
g++ transE.cpp -o transE -pthread -O3 -march=native
./transE
 
# test
g++ test_transE.cpp -o test_transE -pthread -O3 -march=native
./test_transE
