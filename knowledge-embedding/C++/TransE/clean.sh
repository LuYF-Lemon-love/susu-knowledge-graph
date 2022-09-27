#!/bin/bash

##################################################
# clean.sh
# 使用方法：$ bash clean.sh
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
##################################################

# 删除目标文件和嵌入文件
echo ""
echo "##################################################"
echo ""
rm -rf ./build
echo "./build 目录递归删除成功."
echo ""

# 删除临时的数据文件
rm -f ../data/FB15K/1-1.txt ../data/FB15K/1-n.txt ../data/FB15K/n-1.txt ../data/FB15K/n-n.txt ../data/FB15K/test2id_all.txt ../data/FB15K/type_constrain.txt
echo "已删除 ../data/FB15K/1-1.txt ../data/FB15K/1-n.txt ../data/FB15K/n-1.txt ../data/FB15K/n-n.txt ../data/FB15K/test2id_all.txt ../data/FB15K/type_constrain.txt."
echo ""
echo "##################################################"
echo ""
