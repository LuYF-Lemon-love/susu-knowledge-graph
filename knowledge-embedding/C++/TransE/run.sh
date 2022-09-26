#!/bin/bash

##################################################
# run.sh
# 使用方法：$ bash run.sh
# created by LuYF-Lemon-love <luyanfeng@qq.com>
##################################################

# 生成临时数据文件
python3 data_preprocessing.py

# 创建 build 目录
echo "##################################################"
echo ""
mkdir build
echo "./build 目录创建成功."
echo ""

# train
g++ transE.cpp -o ./build/transE -pthread -O3 -march=native
./build/transE
 
# test
g++ test_transE.cpp -o ./build/test_transE -pthread -O3 -march=native
./build/test_transE
