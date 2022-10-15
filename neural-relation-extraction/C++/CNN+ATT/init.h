#ifndef INIT_H
#define INIT_H
#include <cstring>
#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <float.h>          // FLT_MAX
#include <cmath>
#include <algorithm>
#include <pthread.h>
#include <iostream>

#define INT int
#define REAL float


using namespace std;

std::string version = "";

INT output_model = 0;
INT num_threads = 32;
INT train_times = 25;
REAL reduce = 0.98;

// dimension_c: sentence embedding size
// dimension_pos: position dimension
// window: window size
// batch: batch size
// alpha: learning rate
// dropout_probability: dropout probability
INT dimension_c = 230;
INT dimension_pos = 5;
INT window = 3;
INT batch = 160;
REAL alpha = 0.02;
REAL dropout_probability = 0.5;

// limit: 限制句子中 (头, 尾) 实体相对每个单词的最大距离
INT limit = 30;

INT npoch;
INT len;
REAL rate = 1;

// word_total: 词汇总数, 包括 "UNK"
// dimension: 词嵌入维度
// word_vec (word_total * dimension): 词嵌入矩阵
// id2word (word_total): id2word[id] -> id 对应的词汇名
// word2id (word_total): word2id[name] -> name 对应的词汇 id
INT word_total, dimension;
REAL *word_vec;
std::vector<std::string> id2word;
std::map<std::string, INT> word2id;

// relation_total: 关系总数
// id2relation (relation_total): id2relation[id] -> id 对应的关系名
// relation2id (relation_total): relation2id[name] -> name 对应的关系 id
INT relation_total;
std::vector<std::string> id2relation;
std::map<std::string, INT> relation2id;

// position_min_head: 保存数据集 (训练集, 测试集) 句子中头实体相对每个单词的最小距离, 理论上取值范围为 -limit
// position_max_head: 保存数据集 (训练集, 测试集) 句子中头实体相对每个单词的最大距离, 理论上取值范围为 limit
// position_min_tail: 保存数据集 (训练集, 测试集) 句子中尾实体相对每个单词的最小距离, 理论上取值范围为 -limit
// position_max_tail: 保存数据集 (训练集, 测试集) 句子中尾实体相对每个单词的最大距离, 理论上取值范围为 limit
// position_total_head = position_max_head - position_min_head + 1
// position_total_tail = position_max_tail - position_min_tail + 1
INT position_min_head, position_max_head, position_min_tail, position_max_tail, position_total_head,position_total_tail;

// bags_train: key -> (头实体 + "\t" + 尾实体 + "\t" + 关系名), value -> 句子索引 (训练文件中该句子的位置)
// train_head_list: 保存训练集每个句子的头实体 id, 按照训练文件句子的读取顺序排列
// train_tail_list: 保存训练集每个句子的尾实体 id, 按照训练文件句子的读取顺序排列
// train_relation_list: 保存训练集每个句子的关系 id, 按照训练文件句子的读取顺序排列
// train_length: 保存训练集每个句子的单词个数, 按照训练文件句子的读取顺序排列
// train_sentence_list: 保存训练集中的句子, 按照训练文件句子的读取顺序排列
// train_position_head: 保存训练集每个句子的头实体相对每个单词的距离, 理论上取值范围为 [0, 2 * limit], 其中头实体对应单词的取值为 limit
// train_position_tail: 保存训练集每个句子的尾实体相对每个单词的距离, 理论上取值范围为 [0, 2 * limit], 其中尾实体对应单词的取值为 limit
std::map<std::string, std::vector<INT> > bags_train;
std::vector<INT> train_head_list, train_tail_list, train_relation_list;
std::vector<INT> train_length;
std::vector<INT *> train_sentence_list, train_position_head, train_position_tail;

// bags_test: key -> (头实体 + "\t" + 尾实体), value -> 句子索引 (测试文件中该句子的位置)
// test_head_list: 保存测试集每个句子的头实体 id, 按照测试文件句子的读取顺序排列
// test_tail_list: 保存测试集每个句子的尾实体 id, 按照测试文件句子的读取顺序排列
// test_relation_list: 保存测试集每个句子的关系 id, 按照测试文件句子的读取顺序排列
// test_length: 保存测试集每个句子的单词个数, 按照测试文件句子的读取顺序排列
// test_sentence_list: 保存测试集中的句子, 按照测试文件句子的读取顺序排列
// test_position_head: 保存测试集每个句子的头实体相对每个单词的距离, 理论上取值范围为 [0, 2 * limit], 其中头实体对应单词的取值为 limit
// test_position_tail: 保存测试集每个句子的尾实体相对每个单词的距离, 理论上取值范围为 [0, 2 * limit], 其中尾实体对应单词的取值为 limit
std::map<std::string, std::vector<INT> > bags_test;
std::vector<INT> test_head_list, test_tail_list, test_relation_list;
std::vector<INT> test_length;
std::vector<INT *> test_sentence_list, test_position_head, test_position_tail;

// position_vec_head (position_total_head * dimension_pos): 头实体的位置嵌入矩阵
// position_vec_tail (position_total_tail * dimension_pos): 尾实体的位置嵌入矩阵
REAL *position_vec_head, *position_vec_tail;

// conv_1d_word (dimension_c * window * dimension): 一维卷机的权重矩阵 (词嵌入)
// conv_1d_position_head (dimension_c * window * dimension_pos): 一维卷机的权重矩阵 (头实体的位置嵌入)
// conv_1d_position_tail (dimension_c * window * dimension_pos): 一维卷机的权重矩阵 (尾实体的位置嵌入)
// conv_1d_bias (dimension_c): 一维卷机的偏置向量
REAL *conv_1d_word, *conv_1d_position_head, *conv_1d_position_tail, *conv_1d_bias;

// attention_weights (relation_total * dimension_c * dimension_c): 注意力权重矩阵
// attention_weights_copy (relation_total * dimension_c * dimension_c): 注意力权重矩阵副本, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
std::vector<std::vector<std::vector<REAL> > > attention_weights;

// relation_matrix (relation_total * dimension_c): the representation matrix of relation
// relation_matrix_bias (relation_total): the bias vector of the representation matrix of relation
REAL *relation_matrix, *relation_matrix_bias;

// word_vec_copy (word_total * dimension): 词嵌入矩阵副本, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
// position_vec_head_copy (position_total_head * dimension_pos): 头实体的位置嵌入矩阵副本, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
// position_vec_tail_copy (position_total_tail * dimension_pos): 尾实体的位置嵌入矩阵副本, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
REAL *word_vec_copy, *position_vec_head_copy, *position_vec_tail_copy;

// conv_1d_word_copy (dimension_c * window * dimension): 一维卷机的权重矩阵 (词嵌入) 副本, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
// conv_1d_position_head_copy (dimension_c * window * dimension_pos): 一维卷机的权重矩阵 (头实体的位置嵌入) 副本, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
// conv_1d_position_tail_copy (dimension_c * window * dimension_pos): 一维卷机的权重矩阵 (尾实体的位置嵌入) 副本, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
// conv_1d_bias_copy (dimension_c): 一维卷机的偏置向量副本, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
REAL *conv_1d_word_copy, *conv_1d_position_head_copy, *conv_1d_position_tail_copy, *conv_1d_bias_copy;

// attention_weights_copy (relation_total * dimension_c * dimension_c): 注意力权重矩阵副本, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
std::vector<std::vector<std::vector<REAL> > > attention_weights_copy;

// relation_matrix_copy (relation_total * dimension_c): the copy of the representation matrix of relation, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
// relation_matrix_bias_copy (relation_total): the copy of the bias vector of the representation matrix of relation, 由于使用多线程训练模型, 该副本用于每一训练批次计算损失值
REAL *relation_matrix_copy, *relation_matrix_bias_copy;

void init() {

	INT tmp;

	// 读取预训练词嵌入
	FILE *f = fopen("../data/vec.bin", "rb");
	tmp = fscanf(f, "%d", &word_total);
	tmp = fscanf(f, "%d", &dimension);
	std::cout << "word_total (exclude \"UNK\") = " << word_total << std::endl;
	std::cout << "word dimension = " << dimension << std::endl;
	word_vec = (REAL *)malloc((word_total+1) * dimension * sizeof(REAL));
	id2word.resize(word_total + 1);
	id2word[0] = "UNK";
	word2id["UNK"] = 0;
	for (INT i = 1; i <= word_total; i++) {
		std::string name = "";
		while (1) {
			char ch = fgetc(f);
			if (feof(f) || ch == ' ') break;
			if (ch != '\n') name = name + ch;
		}
		long long last = i * dimension;
		REAL sum = 0;
		for (INT a = 0; a < dimension; a++) {
			tmp = fread(&word_vec[a + last], sizeof(REAL), 1, f);
			sum += word_vec[a + last] * word_vec[a + last];
		}
		sum = sqrt(sum);
		for (INT a = 0; a< dimension; a++)
			word_vec[a+last] = word_vec[a+last] / sum;		
		word2id[name] = i;
		id2word[i] = name;
	}
	word_total+=1;
	fclose(f);

	// 读取 relation2id.txt 文件
	char buffer[1000];
	f = fopen("../data/RE/relation2id.txt", "r");
	while (fscanf(f,"%s",buffer)==1) {
		INT id;
		tmp = fscanf(f, "%d", &id);
		relation2id[(std::string)(buffer)] = id;
		relation_total++;
		id2relation.push_back((std::string)(buffer));
	}
	fclose(f);
	std::cout << "relation_total: " << relation_total << std::endl;
	
	// 读取训练文件 (train.txt)
	position_min_head = 0;
	position_max_head = 0;
	position_min_tail = 0;
	position_max_tail = 0;
	f = fopen("../data/RE/train.txt", "r");
	while (fscanf(f,"%s",buffer)==1)  {
		std::string e1 = buffer;
		tmp = fscanf(f,"%s",buffer);
		std::string e2 = buffer;

		tmp = fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		INT head_id = word2id[head_s];
		tmp = fscanf(f,"%s",buffer);
		string tail_s = (string)(buffer);
		INT tail_id = word2id[tail_s];
			
		tmp = fscanf(f,"%s",buffer);
		bags_train[e1+"\t"+e2+"\t"+(string)(buffer)].push_back(train_head_list.size());
		INT relation_id = relation2id[(string)(buffer)];

		INT len_s = 0, head_pos = 0, tail_pos = 0;
		std::vector<INT> sentence;
		while (fscanf(f,"%s", buffer)==1) {
			std::string word = buffer;
			if (word == "###END###") break;
			INT word_id = word2id[word];
			if (word == head_s) head_pos = len_s;
			if (word == tail_s) tail_pos = len_s;
			len_s++;
			sentence.push_back(word_id);
		}

		train_head_list.push_back(head_id);
		train_tail_list.push_back(tail_id);
		train_relation_list.push_back(relation_id);
		train_length.push_back(len_s);
		
		INT *sentence_ptr = (INT *)calloc(len_s, sizeof(INT));
		INT *sentence_head_pos = (INT *)calloc(len_s, sizeof(INT));
		INT *sentence_tail_pos = (INT *)calloc(len_s, sizeof(INT));
		for (INT i = 0; i < len_s; i++) {
			sentence_ptr[i] = sentence[i];
			sentence_head_pos[i] = head_pos - i;
			sentence_tail_pos[i] = tail_pos - i;
			if (sentence_head_pos[i] >= limit) sentence_head_pos[i] = limit;
			if (sentence_tail_pos[i] >= limit) sentence_tail_pos[i] = limit;
			if (sentence_head_pos[i] <= -limit) sentence_head_pos[i] = -limit;
			if (sentence_tail_pos[i] <= -limit) sentence_tail_pos[i] = -limit;
			if (sentence_head_pos[i] > position_max_head) position_max_head = sentence_head_pos[i];
			if (sentence_tail_pos[i] > position_max_tail) position_max_tail = sentence_tail_pos[i];
			if (sentence_head_pos[i] < position_min_head) position_min_head = sentence_head_pos[i];
			if (sentence_tail_pos[i] < position_min_tail) position_min_tail = sentence_tail_pos[i];
		}

		train_sentence_list.push_back(sentence_ptr);
		train_position_head.push_back(sentence_head_pos);
		train_position_tail.push_back(sentence_tail_pos);
	}
	fclose(f);

	// 读取测试文件 (test.txt)
	f = fopen("../data/RE/test.txt", "r");	
	while (fscanf(f,"%s",buffer)==1)  {
		std::string e1 = buffer;
		tmp = fscanf(f,"%s",buffer);
		std::string e2 = buffer;

		tmp = fscanf(f,"%s",buffer);
		std::string head_s = (string)(buffer);
		INT head_id = word2id[head_s];
		tmp = fscanf(f,"%s",buffer);
		std::string tail_s = (string)(buffer);
		INT tail_id = word2id[tail_s];

		tmp = fscanf(f,"%s",buffer);
		bags_test[e1+"\t"+e2].push_back(test_head_list.size());	
		INT relation_id = relation2id[(string)(buffer)];

		INT len_s = 0 , head_pos = 0, tail_pos = 0;
		std::vector<INT> sentence;
		while (fscanf(f,"%s", buffer)==1) {
			std::string word = buffer;
			if (word=="###END###") break;
			INT word_id = word2id[word];
			if (head_s == word) head_pos = len_s;
			if (tail_s == word) tail_pos = len_s;
			len_s++;
			sentence.push_back(word_id);
		}

		test_head_list.push_back(head_id);
		test_tail_list.push_back(tail_id);
		test_relation_list.push_back(relation_id);
		test_length.push_back(len_s);

		INT *sentence_ptr=(INT *)calloc(len_s, sizeof(INT));
		INT *sentence_head_pos=(INT *)calloc(len_s, sizeof(INT));
		INT *sentence_tail_pos=(INT *)calloc(len_s, sizeof(INT));
		for (INT i = 0; i < len_s; i++) {
			sentence_ptr[i] = sentence[i];
			sentence_head_pos[i] = head_pos - i;
			sentence_tail_pos[i] = tail_pos - i;
			if (sentence_head_pos[i] >= limit) sentence_head_pos[i] = limit;
			if (sentence_tail_pos[i] >= limit) sentence_tail_pos[i] = limit;
			if (sentence_head_pos[i] <= -limit) sentence_head_pos[i] = -limit;
			if (sentence_tail_pos[i] <= -limit) sentence_tail_pos[i] = -limit;
			if (sentence_head_pos[i] > position_max_head) position_max_head = sentence_head_pos[i];
			if (sentence_tail_pos[i] > position_max_tail) position_max_tail = sentence_tail_pos[i];
			if (sentence_head_pos[i] < position_min_head) position_min_head = sentence_head_pos[i];
			if (sentence_tail_pos[i] < position_min_tail) position_min_tail = sentence_tail_pos[i];
		}

		test_sentence_list.push_back(sentence_ptr);
		test_position_head.push_back(sentence_head_pos);
		test_position_tail.push_back(sentence_tail_pos);
	}
	fclose(f);

	std::cout << "position_min_head: " << position_min_head << std::endl 
			  << "position_max_head: " << position_max_head << std::endl
			  << "position_min_tail: " << position_min_tail << std::endl
			  << "position_max_tail: " << position_max_tail << std::endl;

	for (INT i = 0; i < train_position_head.size(); i++) {
		INT len_s = train_length[i];
		INT *position = train_position_head[i];
		for (INT j = 0; j < len_s; j++)
			position[j] = position[j] - position_min_head;
		position = train_position_tail[i];
		for (INT j = 0; j < len_s; j++)
			position[j] = position[j] - position_min_tail;
	}

	for (INT i = 0; i < test_position_head.size(); i++) {
		INT len_s = test_length[i];
		INT *position = test_position_head[i];
		for (INT j = 0; j < len_s; j++)
			position[j] = position[j] - position_min_head;
		position = test_position_tail[i];
		for (INT j = 0; j < len_s; j++)
			position[j] = position[j] - position_min_tail;
	}

	position_total_head = position_max_head - position_min_head + 1;
	position_total_tail = position_max_tail - position_min_tail + 1;

	std::cout << "position_total_head: " << position_total_head << std::endl 
			  << "position_total_tail: " << position_total_tail << std::endl;
}

// 计算双曲正切函数（tanh）
REAL calc_tanh(REAL value) {
	if (value > 20) return 1.0;
	if (value < -20) return -1.0;
	REAL sinhx = exp(value) - exp(-value);
	REAL coshx = exp(value) + exp(-value);
	return sinhx / coshx;
}

INT get_rand(INT l,INT r) {
	INT len = r - l;
	INT res = rand()*rand() % len;
	if (res < 0)
		res += len;
	return res + l;
}

REAL get_rand_u(REAL l, REAL r) {
	REAL len = r - l;
	REAL res = (REAL)(rand()) / RAND_MAX;
	return res * len + l;
}

#endif
