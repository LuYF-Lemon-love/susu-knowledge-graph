## 介绍

一些知识表示学习 (knowledge representation learning, KRL) 的 C++ 实现, 利用多线程加速模型的训练和测试.

## 数据

### FB15K

该数据集是 Wikilinks database 的子集, 该子集中的实体和关系在 Freebase 至少出现了 100 次. 并且移除了 ’!/people/person/nationality’ 的关系, 因为它是关系 ’/people/person/nationality’ head 和 tail 的颠倒. 一共 592,213 个三元组, 14,951 种实体, 1,345 种关系, 被随机分成了训练集 (483,142 个), 验证集 (50,000 个), 测试集 (59,071 个).

- entity2id.txt: 第一行是实体种类数. 其余行是实体名和对应的实体 ID, 每行一个.

- relation2id.txt: 第一行是关系种类数. 其余行是关系名和对应的关系 ID, 每行一个.

- train2id.txt: 训练文件. 第一行是训练集三元组的个数. 其余行是 (e1, e2, rel) 格式的三元组, 每行一个. e1, e2 是实体 ID, rel 是关系 ID.

- valid2id.txt: 验证文件. 第一行是验证集三元组的个数. 其余行是 (e1, e2, rel) 格式的三元组, 每行一个. e1, e2 是实体 ID, rel 是关系 ID.

- test2id.txt: 测试文件. 第一行是测试集三元组的个数. 其余行是 (e1, e2, rel) 格式的三元组, 每行一个. e1, e2 是实体 ID, rel 是关系 ID.

## TransE

**TransE** 是一个基于能量 *(energy-based)* 的学习**实体低维度嵌入向量**的模型, **关系**被表示**嵌入空间**的**平移**: 如果 *$(h, \ell, t)$* 成立, *t* 的嵌入应该接近于 *h* 的嵌入加上某个**向量**, 某个向量就是**关系的嵌入**.

**TransE** 原论文链接：[Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)

### 使用

```bash
$ ls
clean.sh  data_preprocessing.py  run.sh  test_transE.cpp  transE.cpp
$ bash run.sh 

##################################################

数据预处理开始...

../data/FB15K/type_constrain.txt 创建成功.

../data/FB15K/1-1.txt ../data/FB15K/1-n.txt ../data/FB15K/n-1.txt ../data/FB15K/n-n.txt ../data/FB15K/test2id_all.txt 创建成功.

数据预处理结束.

##################################################

./build 目录创建成功.

##################################################

训练开始:

relation_total: 1345
entity_total: 14951
train_triple_total: 483142

Epoch 50/1000 - loss: 4254.677734
Epoch 100/1000 - loss: 3452.187988
Epoch 150/1000 - loss: 3018.641113
Epoch 200/1000 - loss: 2960.243408
Epoch 250/1000 - loss: 2882.189209
Epoch 300/1000 - loss: 2870.555420
Epoch 350/1000 - loss: 2682.759277
Epoch 400/1000 - loss: 2600.209229
Epoch 450/1000 - loss: 2621.255371
Epoch 500/1000 - loss: 2612.536133
Epoch 550/1000 - loss: 2447.342529
Epoch 600/1000 - loss: 2606.887695
Epoch 650/1000 - loss: 2414.228760
Epoch 700/1000 - loss: 2550.244141
Epoch 750/1000 - loss: 2453.343018
Epoch 800/1000 - loss: 2515.363037
Epoch 850/1000 - loss: 2455.265137
Epoch 900/1000 - loss: 2484.796631
Epoch 950/1000 - loss: 2418.574219
Epoch 1000/1000 - loss: 2373.793945

输出预训练实体嵌入 (./build/entity2vec.vec) 成功.
输出预训练关系嵌入 (./build/relation2vec.vec) 成功.

训练结束, 用时 50.864228 秒.

##################################################

测试开始:

加载预训练实体嵌入 (./build/entity2vec.vec) 成功.
加载预训练关系嵌入 (./build/relation2vec.vec) 成功.

总体结果：

heads(raw) 		平均排名: 306.326965, 	Hits@10: 0.372924
heads(filter) 		平均排名: 191.934326, 	Hits@10: 0.495946
tails(raw) 		平均排名: 222.083633, 	Hits@10: 0.446649
tails(filter) 		平均排名: 150.873444, 	Hits@10: 0.563000

通过 type_constrain.txt 限制的总体结果：

heads(raw) 		平均排名: 202.505310, 	Hits@10: 0.399011
heads(filter) 		平均排名: 88.112625, 	Hits@10: 0.560614
tails(raw) 		平均排名: 138.922943, 	Hits@10: 0.473650
tails(filter) 		平均排名: 67.712753, 	Hits@10: 0.606389

(关系: 1-1, 1-n, n-1, n-n) 测试三元组的结果：

关系: 1-1:

heads(raw) 		平均排名: 124.536644, 	Hits@10: 0.712766
heads(filter) 		平均排名: 124.309692, 	Hits@10: 0.718676
tails(raw) 		平均排名: 147.830963, 	Hits@10: 0.687943
tails(filter) 		平均排名: 147.565018, 	Hits@10: 0.693853

关系: 1-n:

heads(raw) 		平均排名: 22.789383, 	Hits@10: 0.836019
heads(filter) 		平均排名: 22.579716, 	Hits@10: 0.840000
tails(raw) 		平均排名: 1221.926880, 	Hits@10: 0.188057
tails(filter) 		平均排名: 831.402466, 	Hits@10: 0.241517

关系: n-1:

heads(raw) 		平均排名: 1148.632812, 	Hits@10: 0.133086
heads(filter) 		平均排名: 701.605835, 	Hits@10: 0.197778
tails(raw) 		平均排名: 31.964703, 	Hits@10: 0.850365
tails(filter) 		平均排名: 31.795856, 	Hits@10: 0.852795

关系: n-n:

heads(raw) 		平均排名: 179.289429, 	Hits@10: 0.358076
heads(filter) 		平均排名: 113.992714, 	Hits@10: 0.508881
tails(raw) 		平均排名: 141.546097, 	Hits@10: 0.394096
tails(filter) 		平均排名: 93.141548, 	Hits@10: 0.542260

测试结束, 用时 12.604605 秒.

##################################################

$ tree
.
├── build
│   ├── entity2vec.vec
│   ├── relation2vec.vec
│   ├── test_transE
│   └── transE
├── clean.sh
├── data_preprocessing.py
├── run.sh
├── test_transE.cpp
└── transE.cpp

1 directory, 9 files
$ bash clean.sh 

##################################################

./build 目录递归删除成功.

已删除 ../data/FB15K/1-1.txt ../data/FB15K/1-n.txt ../data/FB15K/n-1.txt ../data/FB15K/n-n.txt ../data/FB15K/test2id_all.txt ../data/FB15K/type_constrain.txt.

##################################################

$ ls
clean.sh  data_preprocessing.py  run.sh  test_transE.cpp  transE.cpp
```
