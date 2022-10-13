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




std::vector<INT *> trainLists, trainPositionE1, trainPositionE2;
std::vector<INT> trainLength;
std::vector<INT> headList, tailList, relationList;
std::vector<INT *> testtrainLists, testPositionE1, testPositionE2;
std::vector<INT> testtrainLength;
std::vector<INT> testheadList, testtailList, testrelationList;

std::map<std::string, std::vector<INT> > bags_train, bags_test;

void init() {

	INT tmp;

	// 读取预训练词嵌入
	FILE *f = fopen("../data/vec.bin", "rb");
	tmp = fscanf(f, "%d", &word_total);
	tmp = fscanf(f, "%d", &dimension);
	std::cout << "word_total (exclude \"UNK\") = " << word_total << std::endl;
	std::cout << "word dimension = " << dimension << std::endl;

	PositionMinE1 = 0;
	PositionMaxE1 = 0;
	PositionMinE2 = 0;
	PositionMaxE2 = 0;

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
	f = fopen("../data/RE/train.txt", "r");
	while (fscanf(f,"%s",buffer)==1)  {
		fscanf(f,"%s",buffer);
		fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		INT head = word2id[(string)(buffer)];
		string e1 = buffer;
		fscanf(f,"%s",buffer);
		INT tail = word2id[(string)(buffer)];
		string e2 = buffer;
		string tail_s = (string)(buffer);
		fscanf(f,"%s",buffer);
		bags_train[e1+"\t"+e2+"\t"+(string)(buffer)].push_back(headList.size());
		INT num = relation2id[(string)(buffer)];
		INT len = 0, lefnum = 0, rignum = 0;
		std::vector<INT> tmpp;
		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
			INT gg = word2id[con];
			if (con == head_s) lefnum = len;
			if (con == tail_s) rignum = len;
			len++;
			tmpp.push_back(gg);
		}
		headList.push_back(head);
		tailList.push_back(tail);
		relationList.push_back(num);
		trainLength.push_back(len);
		INT *con=(INT *)calloc(len,sizeof(INT));
		INT *conl=(INT *)calloc(len,sizeof(INT));
		INT *conr=(INT *)calloc(len,sizeof(INT));
		for (INT i = 0; i < len; i++) {
			con[i] = tmpp[i];
			conl[i] = lefnum - i;
			conr[i] = rignum - i;
			if (conl[i] >= limit) conl[i] = limit;
			if (conr[i] >= limit) conr[i] = limit;
			if (conl[i] <= -limit) conl[i] = -limit;
			if (conr[i] <= -limit) conr[i] = -limit;
			if (conl[i] > PositionMaxE1) PositionMaxE1 = conl[i];
			if (conr[i] > PositionMaxE2) PositionMaxE2 = conr[i];
			if (conl[i] < PositionMinE1) PositionMinE1 = conl[i];
			if (conr[i] < PositionMinE2) PositionMinE2 = conr[i];
		}
		trainLists.push_back(con);
		trainPositionE1.push_back(conl);
		trainPositionE2.push_back(conr);
	}
	fclose(f);

	f = fopen("../data/RE/test.txt", "r");	
	while (fscanf(f,"%s",buffer)==1)  {
		fscanf(f,"%s",buffer);
		fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		INT head = word2id[(string)(buffer)];
		string e1 = buffer;
		fscanf(f,"%s",buffer);
		string tail_s = (string)(buffer);
		string e2 = buffer;
		bags_test[e1+"\t"+e2].push_back(testheadList.size());
		INT tail = word2id[(string)(buffer)];
		fscanf(f,"%s",buffer);
		INT num = relation2id[(string)(buffer)];
		INT len = 0 , lefnum = 0, rignum = 0;
		std::vector<INT> tmpp;
		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
			INT gg = word2id[con];
			if (head_s == con) lefnum = len;
			if (tail_s == con) rignum = len;
			len++;
			tmpp.push_back(gg);
		}
		testheadList.push_back(head);
		testtailList.push_back(tail);
		testrelationList.push_back(num);
		testtrainLength.push_back(len);
		INT *con=(INT *)calloc(len,sizeof(INT));
		INT *conl=(INT *)calloc(len,sizeof(INT));
		INT *conr=(INT *)calloc(len,sizeof(INT));
		for (INT i = 0; i < len; i++) {
			con[i] = tmpp[i];
			conl[i] = lefnum - i;
			conr[i] = rignum - i;
			if (conl[i] >= limit) conl[i] = limit;
			if (conr[i] >= limit) conr[i] = limit;
			if (conl[i] <= -limit) conl[i] = -limit;
			if (conr[i] <= -limit) conr[i] = -limit;
			if (conl[i] > PositionMaxE1) PositionMaxE1 = conl[i];
			if (conr[i] > PositionMaxE2) PositionMaxE2 = conr[i];
			if (conl[i] < PositionMinE1) PositionMinE1 = conl[i];
			if (conr[i] < PositionMinE2) PositionMinE2 = conr[i];
		}
		testtrainLists.push_back(con);
		testPositionE1.push_back(conl);
		testPositionE2.push_back(conr);
	}
	fclose(f);
	std::cout<<PositionMinE1<<' '<<PositionMaxE1<<' '<<PositionMinE2<<' '<<PositionMaxE2<<std::endl;

	for (INT i = 0; i < trainPositionE1.size(); i++) {
		INT len = trainLength[i];
		INT *work1 = trainPositionE1[i];
		for (INT j = 0; j < len; j++)
			work1[j] = work1[j] - PositionMinE1;
		INT *work2 = trainPositionE2[i];
		for (INT j = 0; j < len; j++)
			work2[j] = work2[j] - PositionMinE2;
	}

	for (INT i = 0; i < testPositionE1.size(); i++) {
		INT len = testtrainLength[i];
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

REAL CalcTanh(REAL con) {
	if (con > 20) return 1.0;
	if (con < -20) return -1.0;
	REAL sinhx = exp(con) - exp(-con);
	REAL coshx = exp(con) + exp(-con);
	return sinhx / coshx;
}

REAL tanhDao(REAL con) {
	REAL res = CalcTanh(con);
	return 1 - res * res;
}

REAL sigmod(REAL con) {
	if (con > 20) return 1.0;
	if (con < -20) return 0.0;
	con = exp(con);
	return con / (1 + con);
}

INT getRand(INT l,INT r) {
	INT len = r - l;
	INT res = rand()*rand() % len;
	if (res < 0)
		res+=len;
	return res + l;
}

REAL getRandU(REAL l, REAL r) {
	REAL len = r - l;
	REAL res = (REAL)(rand()) / RAND_MAX;
	return res * len + l;
}

void norm(REAL* a, INT ll, INT rr)
{
	REAL tmp = 0;
	for (INT i=ll; i<rr; i++)
		tmp+=a[i]*a[i];
	if (tmp>1)
	{
		tmp = sqrt(tmp);
		for (INT i=ll; i<rr; i++)
			a[i]/=tmp;
	}
}

void norm(vector<double> &a)
{
	double tmp = 0;
	for (INT i=0; i<a.size(); i++)
		tmp+=a[i];
	//if (tmp>1)
	{
	//	tmp = sqrt(tmp);
		for (INT i=0; i<a.size(); i++)
			a[i]/=tmp;
	}
}


#endif
