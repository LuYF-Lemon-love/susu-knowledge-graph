# susu-knowledge-graph

一些知识图谱的算法实现。

## 知识表示学习 (knowledge representation learning, KRL)

### C++

#### [TransE](./knowledge-representation-learning/C%2B%2B/TransE/)

**TransE** 是一个基于能量 *(energy-based)* 的学习**实体低维度嵌入向量**的模型, **关系**被表示**嵌入空间**的**平移**: 如果 *(h, r, t)* 成立, *t* 的嵌入应该接近于 *h* 的嵌入向量加上某个**向量**, 某个向量就是**关系的嵌入向量**.

对应的博客: https://www.luyf-lemon-love.space/1560426839/ .

**TransE** 原论文链接：[Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf).

## 神经关系抽取 (Neural Relation Extraction, NRE)

### C++

#### [CNN+ATT](./neural-relation-extraction/C%2B%2B/CNN%2BATT/)

**CNN+ATT** 是一种**基于语句级别选择性注意力机制**的神经网络模型, 用于构建**基于远程监督**的**关系抽取系统**.

对应的博客: https://www.luyf-lemon-love.space/4249978267/ .

**CNN+ATT** 原论文链接: [Neural Relation Extraction with Selective Attention over Instances](https://aclanthology.org/P16-1200v2.pdf).

## Reference

[1] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. Proceedings of NIPS, 2013.

[2] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu. Learning entity and relation embeddings for knowledge graph completion. The 29th AAAI Conference on Artificial Intelligence, 2015.

[3] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Neural Relation Extraction with Selective Attention over Instances. In Proceedings of ACL, 2016.