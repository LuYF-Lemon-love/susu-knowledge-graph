#!/bin/bash

##################################################
# run.sh
# 使用方法：$ bash run.sh
# created by LuYF-Lemon-love <luyanfeng@qq.com>
##################################################

# train
g++ transE.cpp -o transE -pthread -O3 -march=native
./transE -size 50 -input ../data/FB15K/ -output ./ -thread 32 -epochs 1000 -nbatches 1 -alpha 0.01 -margin 1 -note 01
 
# test
g++ test_transE.cpp -o test_transE -pthread -O3 -march=native
./test_transE -size 50 -sizeR 50 -input ../data/FB15K/ -init ./ -thread 32 -note 01
