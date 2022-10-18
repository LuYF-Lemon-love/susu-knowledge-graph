#!/bin/bash

##################################################
# run.sh
# 使用方法：$ bash run.sh
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
#
# 该 Shell 脚本用于模型训练和模型测试
##################################################

# 创建 build 目录
echo ""
echo "##################################################"
echo ""
mkdir -p build
mkdir -p output
echo "./build 和 ./output 目录创建成功."

# compile
g++ train.cpp -o ./build/train -pthread -O3 -march=native
g++ test.cpp -o ./build/test -pthread -O3 -march=native

# train
./build/train
