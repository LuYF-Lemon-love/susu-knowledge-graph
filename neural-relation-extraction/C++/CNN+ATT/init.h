#ifndef INIT_H
#define INIT_H
#include <cstring>
#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <float.h>
#include <cmath>

#define INT int
#define REAL float


using namespace std;

std::string version = "";

INT output_model = 0;
INT num_threads = 32;
INT trainTimes = 1;
REAL alpha = 0.02;
REAL reduce = 0.98;
INT dimensionC = 230;
INT dimensionWPE = 5;
INT window = 3;

// limit: 限制句子中每个到 (头, 尾) 实体的最大距离
INT limit = 30;
REAL *matrixB1, *matrixRelation, *matrixW1, *matrixRelationDao, *matrixRelationPr, *matrixRelationPrDao;
REAL *wordVecDao;
REAL *positionVecE1, *positionVecE2, *matrixW1PositionE1, *matrixW1PositionE2;
REAL *matrixW1PositionE1Dao;
REAL *matrixW1PositionE2Dao;
REAL *positionVecDaoE1;
REAL *positionVecDaoE2;
REAL *matrixW1Dao;
REAL *matrixB1Dao;

std::vector<std::vector<std::vector<REAL> > > att_W, att_W_Dao;
double mx = 0;
INT batch = 16;
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


INT PositionMinE1, PositionMaxE1, PositionTotalE1,PositionMinE2, PositionMaxE2, PositionTotalE2;

// train_head_list: 保存训练集每个句子的头实体 id, 按照训练文件句子的读取顺序排列
// train_tail_list: 保存训练集每个句子的尾实体 id, 按照训练文件句子的读取顺序排列
// train_relation_list: 保存训练集每个句子的关系 id, 按照训练文件句子的读取顺序排列
// train_length: 保存训练集每个句子的单词个数, 按照训练文件句子的读取顺序排列
// train_sentence_list: 保存训练集中的句子, 按照训练文件句子的读取顺序排列
// train_position_head: 保存训练集每个句子的每个单词相对头实体的距离, 理论上取值范围为 [0, 2 * limit], 其中头实体对应单词的取值为 limit
// train_position_tail: 保存训练集每个句子的每个单词相对尾实体的距离, 理论上取值范围为 [0, 2 * limit], 其中尾实体对应单词的取值为 limit
std::vector<INT> train_head_list, train_tail_list, train_relation_list;
std::vector<INT> train_length;
std::vector<INT *> train_sentence_list, train_position_head, train_position_tail;

// test_head_list: 保存测试集每个句子的头实体 id, 按照测试文件句子的读取顺序排列
// test_tail_list: 保存测试集每个句子的尾实体 id, 按照测试文件句子的读取顺序排列
// test_relation_list: 保存测试集每个句子的关系 id, 按照测试文件句子的读取顺序排列
// test_length: 保存测试集每个句子的单词个数, 按照测试文件句子的读取顺序排列
std::vector<INT> test_head_list, test_tail_list, test_relation_list;
std::vector<INT> test_length;
std::vector<INT *> testtrainLists, testPositionE1, testPositionE2;



std::map<std::string, std::vector<INT> > bags_train, bags_test;

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
	PositionMinE1 = 0;
	PositionMaxE1 = 0;
	PositionMinE2 = 0;
	PositionMaxE2 = 0;
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
			if (sentence_head_pos[i] > PositionMaxE1) PositionMaxE1 = sentence_head_pos[i];
			if (sentence_tail_pos[i] > PositionMaxE2) PositionMaxE2 = sentence_tail_pos[i];
			if (sentence_head_pos[i] < PositionMinE1) PositionMinE1 = sentence_head_pos[i];
			if (sentence_tail_pos[i] < PositionMinE2) PositionMinE2 = sentence_tail_pos[i];
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
		INT *sentence_ptr=(INT *)calloc(len_s,sizeof(INT));
		INT *sentence_head_pos=(INT *)calloc(len_s,sizeof(INT));
		INT *sentence_tail_pos=(INT *)calloc(len_s,sizeof(INT));
		for (INT i = 0; i < len_s; i++) {
			sentence_ptr[i] = sentence[i];
			sentence_head_pos[i] = head_pos - i;
			sentence_tail_pos[i] = tail_pos - i;
			if (sentence_head_pos[i] >= limit) sentence_head_pos[i] = limit;
			if (sentence_tail_pos[i] >= limit) sentence_tail_pos[i] = limit;
			if (sentence_head_pos[i] <= -limit) sentence_head_pos[i] = -limit;
			if (sentence_tail_pos[i] <= -limit) sentence_tail_pos[i] = -limit;
			if (sentence_head_pos[i] > PositionMaxE1) PositionMaxE1 = sentence_head_pos[i];
			if (sentence_tail_pos[i] > PositionMaxE2) PositionMaxE2 = sentence_tail_pos[i];
			if (sentence_head_pos[i] < PositionMinE1) PositionMinE1 = sentence_head_pos[i];
			if (sentence_tail_pos[i] < PositionMinE2) PositionMinE2 = sentence_tail_pos[i];
		}
		testtrainLists.push_back(sentence_ptr);
		testPositionE1.push_back(sentence_head_pos);
		testPositionE2.push_back(sentence_tail_pos);
	}
	fclose(f);
	std::cout<<PositionMinE1<<' '<<PositionMaxE1<<' '<<PositionMinE2<<' '<<PositionMaxE2<<std::endl;

	for (INT i = 0; i < train_position_head.size(); i++) {
		INT len = train_length[i];
		INT *work1 = train_position_head[i];
		for (INT j = 0; j < len; j++)
			work1[j] = work1[j] - PositionMinE1;
		INT *work2 = train_position_tail[i];
		for (INT j = 0; j < len; j++)
			work2[j] = work2[j] - PositionMinE2;
	}

	for (INT i = 0; i < testPositionE1.size(); i++) {
		INT len = test_length[i];
		INT *work1 = testPositionE1[i];
		for (INT j = 0; j < len; j++)
			work1[j] = work1[j] - PositionMinE1;
		INT *work2 = testPositionE2[i];
		for (INT j = 0; j < len; j++)
			work2[j] = work2[j] - PositionMinE2;
	}
	PositionTotalE1 = PositionMaxE1 - PositionMinE1 + 1;
	PositionTotalE2 = PositionMaxE2 - PositionMinE2 + 1;
}

REAL calc_tanh(REAL con) {
	if (con > 20) return 1.0;
	if (con < -20) return -1.0;
	REAL sinhx = exp(con) - exp(-con);
	REAL coshx = exp(con) + exp(-con);
	return sinhx / coshx;
}

INT get_rand(INT l,INT r) {
	INT len = r - l;
	INT res = rand()*rand() % len;
	if (res < 0)
		res+=len;
	return res + l;
}

REAL get_rand_u(REAL l, REAL r) {
	REAL len = r - l;
	REAL res = (REAL)(rand()) / RAND_MAX;
	return res * len + l;
}

#endif
