#!/bin/bash

##################################################
# clean.sh
# 使用方法：$ bash clean.sh
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
##################################################

# 删除目标文件和嵌入文件
rm -rf *.vec *.vec test_transE transE

# 删除临时的数据文件
rm -f ../data/FB15K/1-1.txt ../data/FB15K/1-n.txt ../data/FB15K/n-1.txt ../data/FB15K/n-n.txt ../data/FB15K/test2id_all.txt ../data/FB15K/type_constrain.txt
