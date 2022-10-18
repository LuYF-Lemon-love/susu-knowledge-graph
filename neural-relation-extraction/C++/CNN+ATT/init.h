// init.h
//
// created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
// 
// 该 C++ 文件用于初始化, 即读取训练数据和测试数据
//
// prerequisites:
//     ../data/vec.bin
//     ../data/re/relation2id.txt
//     ../data/re/train.txt
//     ../data/re/test.txt

// ##################################################
// 包含标准库
// ##################################################

#ifndef INIT_H
#define INIT_H

#include <cstdio>          // FILE, fscanf, fopen, fclose, fgetc, feof, fread
#include <cstdlib>         // malloc, calloc, free, rand, RAND_MAX
#include <cmath>           // exp, fabs
#include <cstring>         // memcpy
#include <cfloat>          // FLT_MAX
#include <cassert>         // assert
#include <pthread.h>       // pthread_create, pthread_join, pthread_mutex_t
#include <sys/time.h>      // timeval, gettimeofday
#include <vector>          // std::vector, std::vector::resize, std::vector::operator[], std::vector::push_back, std::vector::size
#include <map>             // std::map, std::map::operator[], std::map::clear, std::map::size
#include <string>          // std::string, std::string::c_str
#include <algorithm>       // std::sort, std::min
#include <utility>         // std::make_pair

// ##################################################
// 声明和定义超参数变量
// ##################################################

#define INT int
#define REAL float

// batch: batch size
// num_threads: number of threads
// alpha: learning rate
// current_rate: init rate of learning rate
// reduce_epoch: reduce of init rate of learning rate per epoch
// epochs: epochs
// limit: 限制句子中 (头, 尾) 实体相对每个单词的最大距离
// dimension_pos: position dimension
// window: window size
// dimension_c: sentence embedding size
// dropout_probability: dropout probability
// output_model: 是否保存模型, 1: 保存模型, 0: 不保存模型
// note: 保存模型时, 文件名的额外的信息, ("./out/word2vec" + note + ".txt")
// data_path: folder of data
// output_path: folder of outputing results (precion/recall curves) and models
INT batch = 40;
INT num_threads = 32;
REAL alpha = 0.00125;
REAL current_rate = 1.0;
REAL reduce_epoch = 0.98;
INT epochs = 25;
INT limit = 30;
INT dimension_pos = 5;
INT window = 3;
INT dimension_c = 230;
REAL dropout_probability = 0.5;
INT output_model = 0;
std::string note = "";
std::string data_path = "../data/";
std::string output_path = "./output/";

// ##################################################
// 声明和定义保存训练数据和测试数据的变量
// ##################################################

// word_total: 词汇总数, 包括 "UNK"
// dimension: 词嵌入维度
// word_vec (word_total * dimension): 词嵌入矩阵
// word2id (word_total): word2id[name] -> name 对应的词汇 id
INT word_total, dimension;
REAL *word_vec;
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
INT position_min_head, position_max_head, position_min_tail, position_max_tail;
INT position_total_head, position_total_tail;

// bags_train: key -> (头实体 + "\t" + 尾实体 + "\t" + 关系名), value -> 句子索引 (训练文件中该句子的位置)
// train_relation_list: 保存训练集每个句子的关系 id, 按照训练文件句子的读取顺序排列
// train_length: 保存训练集每个句子的单词个数, 按照训练文件句子的读取顺序排列
// train_sentence_list: 保存训练集中的句子, 按照训练文件句子的读取顺序排列
// train_position_head: 保存训练集每个句子的头实体相对每个单词的距离, 理论上取值范围为 [0, 2 * limit], 其中头实体对应单词的取值为 limit
// train_position_tail: 保存训练集每个句子的尾实体相对每个单词的距离, 理论上取值范围为 [0, 2 * limit], 其中尾实体对应单词的取值为 limit
std::map<std::string, std::vector<INT> > bags_train;
std::vector<INT> train_relation_list, train_length;
std::vector<INT *> train_sentence_list, train_position_head, train_position_tail;

// bags_test: key -> (头实体 + "\t" + 尾实体), value -> 句子索引 (测试文件中该句子的位置)
// test_relation_list: 保存测试集每个句子的关系 id, 按照测试文件句子的读取顺序排列
// test_length: 保存测试集每个句子的单词个数, 按照测试文件句子的读取顺序排列
// test_sentence_list: 保存测试集中的句子, 按照测试文件句子的读取顺序排列
// test_position_head: 保存测试集每个句子的头实体相对每个单词的距离, 理论上取值范围为 [0, 2 * limit], 其中头实体对应单词的取值为 limit
// test_position_tail: 保存测试集每个句子的尾实体相对每个单词的距离, 理论上取值范围为 [0, 2 * limit], 其中尾实体对应单词的取值为 limit
std::map<std::string, std::vector<INT> > bags_test;
std::vector<INT> test_relation_list, test_length;
std::vector<INT *> test_sentence_list, test_position_head, test_position_tail;

// ##################################################
// 声明和定义模型的权重矩阵
// ##################################################

// position_vec_head (position_total_head * dimension_pos): 头实体的位置嵌入矩阵
// position_vec_tail (position_total_tail * dimension_pos): 尾实体的位置嵌入矩阵
REAL *position_vec_head, *position_vec_tail;

// conv_1d_word (dimension_c * window * dimension): 一维卷机的权重矩阵 (词嵌入)
// conv_1d_position_head (dimension_c * window * dimension_pos): 一维卷机的权重矩阵 (头实体的位置嵌入)
// conv_1d_position_tail (dimension_c * window * dimension_pos): 一维卷机的权重矩阵 (尾实体的位置嵌入)
// conv_1d_bias (dimension_c): 一维卷机的偏置向量
REAL *conv_1d_word, *conv_1d_position_head, *conv_1d_position_tail, *conv_1d_bias;

// attention_weights (relation_total * dimension_c * dimension_c): 注意力权重矩阵
std::vector<std::vector<std::vector<REAL> > > attention_weights;

// relation_matrix (relation_total * dimension_c): the representation matrix of relation
// relation_matrix_bias (relation_total): the bias vector of the representation matrix of relation
REAL *relation_matrix, *relation_matrix_bias;

// ##################################################
// 声明和定义模型的权重矩阵的副本, 用于每一训练批次计算损失值
// ##################################################

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

// 初始化函数, 即读取训练数据和测试数据
void init() {
	
	printf("\n##################################################\n\nInit start...\n\n");

	INT tmp;

	// 读取预训练词嵌入
	FILE *f = fopen((data_path + "vec.bin").c_str(), "rb");
	tmp = fscanf(f, "%d", &word_total);
	tmp = fscanf(f, "%d", &dimension);
	word_vec = (REAL *)malloc((word_total + 1) * dimension * sizeof(REAL));
	word2id["UNK"] = 0;
	for (INT i = 1; i <= word_total; i++) {
		std::string name = "";
		while (1) {
			char ch = fgetc(f);
			if (feof(f) || ch == ' ') break;
			if (ch != '\n') name = name + ch;
		}
		word2id[name] = i;

		long long last = i * dimension;
		REAL sum = 0;
		for (INT a = 0; a < dimension; a++) {
			tmp = fread(&word_vec[last + a], sizeof(REAL), 1, f);
			sum += word_vec[last + a] * word_vec[last + a];
		}
		sum = sqrt(sum);
		for (INT a = 0; a < dimension; a++)
			word_vec[last + a] = word_vec[last + a] / sum;
	}
	word_total += 1;
	fclose(f);

	// 读取 relation2id.txt 文件
	char buffer[1000];
	f = fopen((data_path + "relation2id.txt").c_str(), "r");
	while (fscanf(f, "%s", buffer) == 1) {
		relation2id[(std::string)(buffer)] = relation_total++;
		id2relation.push_back((std::string)(buffer));
	}
	fclose(f);
	
	// 读取训练文件 (train.txt)
	position_min_head = 0;
	position_max_head = 0;
	position_min_tail = 0;
	position_max_tail = 0;
	f = fopen((data_path + "train.txt").c_str(), "r");
	while (fscanf(f, "%s", buffer) == 1)  {
		std::string e1 = buffer;
		tmp = fscanf(f, "%s", buffer);
		std::string e2 = buffer;

		tmp = fscanf(f, "%s", buffer);
		std::string head_s = (std::string)(buffer);
		tmp = fscanf(f, "%s", buffer);
		std::string tail_s = (std::string)(buffer);
			
		tmp = fscanf(f, "%s", buffer);
		bags_train[e1 + "\t" + e2 + "\t" + (std::string)(buffer)].push_back(train_relation_list.size());
		INT relation_id = relation2id[(std::string)(buffer)];

		INT len_s = 0, head_pos = 0, tail_pos = 0;
		std::vector<INT> sentence;
		while (fscanf(f," %s", buffer) == 1) {
			std::string word = buffer;
			if (word == "###END###") break;
			INT word_id = word2id[word];
			if (word == head_s) head_pos = len_s;
			if (word == tail_s) tail_pos = len_s;
			len_s++;
			sentence.push_back(word_id);
		}

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
	f = fopen((data_path + "test.txt").c_str(), "r");
	while (fscanf(f,"%s",buffer)==1)  {
		std::string e1 = buffer;
		tmp = fscanf(f,"%s",buffer);
		std::string e2 = buffer;

		tmp = fscanf(f,"%s",buffer);
		std::string head_s = (std::string)(buffer);
		tmp = fscanf(f,"%s",buffer);
		std::string tail_s = (std::string)(buffer);

		tmp = fscanf(f, "%s", buffer);
		bags_test[e1 + "\t" + e2].push_back(test_relation_list.size());	
		INT relation_id = relation2id[(std::string)(buffer)];

		INT len_s = 0 , head_pos = 0, tail_pos = 0;
		std::vector<INT> sentence;
		while (fscanf(f,"%s", buffer) == 1) {
			std::string word = buffer;
			if (word=="###END###") break;
			INT word_id = word2id[word];
			if (head_s == word) head_pos = len_s;
			if (tail_s == word) tail_pos = len_s;
			len_s++;
			sentence.push_back(word_id);
		}

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

	// 将 train_position_head, train_position_tail, test_position_head, test_position_tail 的元素值转换到 [0, 2 * limit] 范围内
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
}

// 打印一些重要的信息
void print_information() {
	std::string save_model[] = {"不会保存模型.", "将会保存模型."};

	printf("batch: %d\nnumber of threads: %d\nlearning rate: %.8f\n", batch, num_threads, alpha);
	printf("init_rate: %.2f\nreduce_epoch: %.2f\nepochs: %d\n\n", current_rate, reduce_epoch, epochs);
	printf("word_total: %d\nword dimension: %d\n\n", word_total, dimension);
	printf("limit: %d\nposition_total_head: %d\nposition_total_tail: %d\ndimension_pos: %d\n\n",
		limit, position_total_head, position_total_tail, dimension_pos);
	printf("window: %d\ndimension_c: %d\n\n", window, dimension_c);
	printf("relation_total: %d\ndropout_probability: %.2f\n\n", relation_total, dropout_probability);
	printf("%s\nnote: %s\n\n", save_model[output_model].c_str(), note.c_str());
	printf("folder of data: %s\n", data_path.c_str());
	printf("folder of outputing results (precion/recall curves) and models: %s\n\n", output_path.c_str());

	printf("number of training samples: %7d - average sentence number of per training sample: %.2f\n",
		INT(bags_train.size()), float(float(train_sentence_list.size()) / bags_train.size()));
	printf("number of testing samples:  %7d - average sentence number of per testing sample:  %.2f\n\n",
		INT(bags_test.size()), float(float(test_sentence_list.size()) / bags_test.size()));
	
	printf("Init end.\n\n");
}

// 计算双曲正切函数（tanh）
REAL calc_tanh(REAL value) {
	if (value > 20) return 1.0;
	if (value < -20) return -1.0;
	REAL sinhx = exp(value) - exp(-value);
	REAL coshx = exp(value) + exp(-value);
	return sinhx / coshx;
}

// 返回取值为 [min, max) 的伪随机整数
INT get_rand_i(INT min, INT max) {
	INT d = max - min;
	INT res = rand() % d;
	if (res < 0)
		res += d;
	return res + min;
}

// 返回取值为 [min, max) 的伪随机浮点数 
REAL get_rand_u(REAL min, REAL max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

// 寻找特定参数的位置
INT arg_pos(char *str, INT argc, char **argv) {
	INT a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

#endif
