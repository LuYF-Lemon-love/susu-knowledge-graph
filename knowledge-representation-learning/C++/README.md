## 介绍

一些知识表示学习 (knowledge representation learning, KRL) 的 C++ 实现, 利用多线程加速模型的训练和测试.

## 数据

### [FB15K](./data/FB15K/)

该数据集是 Wikilinks database 的实体子集, 该子集中的实体和关系在 Freebase 至少出现了 100 次. 并且移除了像 ’!/people/person/nationality’ 这样的关系, 因为它是关系 ’/people/person/nationality’ head 和 tail 的颠倒. 一共 592,213 个三元组, 14,951 个实体, 1,345 个关系, 被随机地分成了训练集 (483,142 个), 验证集 (50,000 个), 测试集 (59,071 个).

- [entity2id.txt](./data/FB15K/entity2id.txt): 第一行是实体个数. 其余行是实体名和对应的实体 ID, 每行一个.

- [relation2id.txt](./data/FB15K/relation2id.txt): 第一行是关系个数. 其余行是关系名和对应的关系 ID, 每行一个.

- [train2id.txt](./data/FB15K/train2id.txt): 训练文件. 第一行是训练集三元组的个数. 其余行是 (e1, e2, rel) 格式的三元组, 每行一个. e1, e2 是实体 ID, rel 是关系 ID.

- [valid2id.txt](./data/FB15K/valid2id.txt): 验证文件. 第一行是验证集三元组的个数. 其余行是 (e1, e2, rel) 格式的三元组, 每行一个. e1, e2 是实体 ID, rel 是关系 ID.

- [test2id.txt](./data/FB15K/test2id.txt): 测试文件. 第一行是测试集三元组的个数. 其余行是 (e1, e2, rel) 格式的三元组, 每行一个. e1, e2 是实体 ID, rel 是关系 ID.

## TransE

**TransE** 是一个基于能量 *(energy-based)* 的学习**实体低维度嵌入向量**的模型, **关系**被表示**嵌入空间**的**平移**: 如果 *(h, r, t)* 成立, *t* 的嵌入应该接近于 *h* 的嵌入向量加上某个**向量**, 某个向量就是**关系的嵌入向量**.

对应的博客: https://www.luyf-lemon-love.space/1560426839/ .

**TransE** 原论文链接：[Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf).

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

### 训练和测试的参数

#### transE.cpp

```
./transE [-bern 0/1] [-load-binary 0/1] [-out-binary 0/1]
         [-size SIZE] [-alpha ALPHA] [-margin MARGIN]
         [-nbatches NBATCHES] [-epochs EPOCHS]
         [-threads THREAD] [-input INPUT] [-output OUTPUT]
         [-load LOAD] [-note NOTE]

optional arguments:
-bern [0/1]          [1] 使用 bern 算法进行负采样，默认值为 [1]
-load-binary [0/1]   [1] 以二进制形式加载预训练嵌入，默认值为 [0]
-out-binary [0/1]    [1] 以二进制形式输出嵌入，默认值为 [0]
-size SIZE           实体和关系嵌入维度，默认值为 [50]
-alpha ALPHA         学习率，默认值为 0.01
-margin MARGIN       margin in max-margin loss for pairwise training，默认值为 1.0
-nbatches NBATCHES   number of batches for each epoch. if unspecified, nbatches will default to 1
-epochs EPOCHS       number of epochs. if unspecified, epochs will default to 1000
-threads THREAD      number of worker threads. if unspecified, threads will default to 32
-input INPUT         folder of training data. if unspecified, in_path will default to "../data/FB15K/"
-output OUTPUT       folder of outputing results. if unspecified, out_path will default to "./build/"
-load LOAD           folder of pretrained data. if unspecified, load_path will default to ""
-note NOTE           information you want to add to the filename. if unspecified, note will default to ""
```

#### test_transE.cpp

```
./test_transE [-load-binary 0/1] [-size SIZE]
         [-threads THREAD] [-input INPUT]
         [-load LOAD] [-note NOTE]

optional arguments:
-load-binary [0/1]   [1] 以二进制形式加载预训练嵌入，默认值为 [0]
-size SIZE           实体和关系嵌入维度，默认值为 [50]
-threads THREAD      number of worker threads. if unspecified, threads will default to 32
-input INPUT         folder of training data. if unspecified, in_path will default to "../data/FB15K/"
-load LOAD           folder of pretrained data. if unspecified, load_path will default to "./build/"
-note NOTE           information you want to add to the filename. if unspecified, note will default to ""
```

### 文件

- [data_preprocessing.py](./TransE/data_preprocessing.py): 该 Python 脚本用于创建下面这些临时数据文件.

```
../data/FB15K/1-1.txt ../data/FB15K/1-n.txt ../data/FB15K/n-1.txt ../data/FB15K/n-n.txt ../data/FB15K/test2id_all.txt ../data/FB15K/type_constrain.txt
```

- [transE.cpp](./TransE/transE.cpp): 该 C++ 文件用于模型训练.

- [test_transE.cpp](./TransE/test_transE.cpp): 该 C++ 文件用于模型测试.

- [run.sh](./TransE/run.sh): 该 Shell 脚本用于模型训练和模型测试.

- [clean.sh](./TransE/clean.sh): 该 Shell 脚本用于清理临时文件.

## Reference

[1] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. Proceedings of NIPS, 2013.

[2] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu. Learning Entity and Relation Embeddings for Knowledge Graph Completion. The 29th AAAI Conference on Artificial Intelligence, 2015.
