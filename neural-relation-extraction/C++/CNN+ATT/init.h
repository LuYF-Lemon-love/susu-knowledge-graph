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

string version = "";

int output_model = 0;
int num_threads = 32;
int trainTimes = 1;
float alpha = 0.02;
float reduce = 0.98;
int dimensionC = 230;
int dimensionWPE = 5;
int window = 3;
int limit = 30;
float *matrixB1, *matrixRelation, *matrixW1, *matrixRelationDao, *matrixRelationPr, *matrixRelationPrDao;
float *wordVecDao;
float *positionVecE1, *positionVecE2, *matrixW1PositionE1, *matrixW1PositionE2;
float *matrixW1PositionE1Dao;
float *matrixW1PositionE2Dao;
float *positionVecDaoE1;
float *positionVecDaoE2;
float *matrixW1Dao;
float *matrixB1Dao;

vector<vector<vector<float> > > att_W, att_W_Dao;
double mx = 0;
int batch = 16;
int npoch;
int len;
float rate = 1;

// word_total: 词汇总数, 包括 "UNK"
// dimension: 词嵌入维度
INT word_total, dimension, relationTotal;

// word_vec (word_total * dimension): 词嵌入矩阵
REAL *word_vec;

// id2word (word_total): id2word[id] -> id 对应的词汇名
// word2id (word_total): word2id[name] -> name 对应的词汇 id
vector<std::string> id2word;
std::map<std::string, INT> word2id;

int  PositionMinE1, PositionMaxE1, PositionTotalE1,PositionMinE2, PositionMaxE2, PositionTotalE2;



map<string,int> relationMapping;
vector<int *> trainLists, trainPositionE1, trainPositionE2;
vector<int> trainLength;
vector<int> headList, tailList, relationList;
vector<int *> testtrainLists, testPositionE1, testPositionE2;
vector<int> testtrainLength;
vector<int> testheadList, testtailList, testrelationList;
vector<std::string> nam;

map<string,vector<int> > bags_train, bags_test;

void init() {

	INT tmp;

	FILE *f = fopen("../data/vec.bin", "rb");

	tmp = fscanf(f, "%d", &word_total);
	tmp = fscanf(f, "%d", &dimension);

	std::cout << "word_total (exclude \"UNK\") =\t" << word_total << endl;
	std::cout << "word dimension =\t" << dimension << endl;

	PositionMinE1 = 0;
	PositionMaxE1 = 0;
	PositionMinE2 = 0;
	PositionMaxE2 = 0;

	word_vec = (float *)malloc((word_total+1) * dimension * sizeof(float));
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
			fread(&word_vec[a + last], sizeof(float), 1, f);
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

	char buffer[1000];
	f = fopen("../data/RE/relation2id.txt", "r");
	while (fscanf(f,"%s",buffer)==1) {
		int id;
		fscanf(f,"%d",&id);
		relationMapping[(string)(buffer)] = id;
		relationTotal++;
		nam.push_back((std::string)(buffer));
	}
	fclose(f);
	cout<<"relationTotal:\t"<<relationTotal<<endl;
	
	f = fopen("../data/RE/train.txt", "r");
	while (fscanf(f,"%s",buffer)==1)  {
		fscanf(f,"%s",buffer);
		fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		int head = word2id[(string)(buffer)];
		string e1 = buffer;
		fscanf(f,"%s",buffer);
		int tail = word2id[(string)(buffer)];
		string e2 = buffer;
		string tail_s = (string)(buffer);
		fscanf(f,"%s",buffer);
		bags_train[e1+"\t"+e2+"\t"+(string)(buffer)].push_back(headList.size());
		int num = relationMapping[(string)(buffer)];
		int len = 0, lefnum = 0, rignum = 0;
		std::vector<int> tmpp;
		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
			int gg = word2id[con];
			if (con == head_s) lefnum = len;
			if (con == tail_s) rignum = len;
			len++;
			tmpp.push_back(gg);
		}
		headList.push_back(head);
		tailList.push_back(tail);
		relationList.push_back(num);
		trainLength.push_back(len);
		int *con=(int *)calloc(len,sizeof(int));
		int *conl=(int *)calloc(len,sizeof(int));
		int *conr=(int *)calloc(len,sizeof(int));
		for (int i = 0; i < len; i++) {
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
		int head = word2id[(string)(buffer)];
		string e1 = buffer;
		fscanf(f,"%s",buffer);
		string tail_s = (string)(buffer);
		string e2 = buffer;
		bags_test[e1+"\t"+e2].push_back(testheadList.size());
		int tail = word2id[(string)(buffer)];
		fscanf(f,"%s",buffer);
		int num = relationMapping[(string)(buffer)];
		int len = 0 , lefnum = 0, rignum = 0;
		std::vector<int> tmpp;
		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
			int gg = word2id[con];
			if (head_s == con) lefnum = len;
			if (tail_s == con) rignum = len;
			len++;
			tmpp.push_back(gg);
		}
		testheadList.push_back(head);
		testtailList.push_back(tail);
		testrelationList.push_back(num);
		testtrainLength.push_back(len);
		int *con=(int *)calloc(len,sizeof(int));
		int *conl=(int *)calloc(len,sizeof(int));
		int *conr=(int *)calloc(len,sizeof(int));
		for (int i = 0; i < len; i++) {
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
	cout<<PositionMinE1<<' '<<PositionMaxE1<<' '<<PositionMinE2<<' '<<PositionMaxE2<<endl;

	for (int i = 0; i < trainPositionE1.size(); i++) {
		int len = trainLength[i];
		int *work1 = trainPositionE1[i];
		for (int j = 0; j < len; j++)
			work1[j] = work1[j] - PositionMinE1;
		int *work2 = trainPositionE2[i];
		for (int j = 0; j < len; j++)
			work2[j] = work2[j] - PositionMinE2;
	}

	for (int i = 0; i < testPositionE1.size(); i++) {
		int len = testtrainLength[i];
		int *work1 = testPositionE1[i];
		for (int j = 0; j < len; j++)
			work1[j] = work1[j] - PositionMinE1;
		int *work2 = testPositionE2[i];
		for (int j = 0; j < len; j++)
			work2[j] = work2[j] - PositionMinE2;
	}
	PositionTotalE1 = PositionMaxE1 - PositionMinE1 + 1;
	PositionTotalE2 = PositionMaxE2 - PositionMinE2 + 1;
}

float CalcTanh(float con) {
	if (con > 20) return 1.0;
	if (con < -20) return -1.0;
	float sinhx = exp(con) - exp(-con);
	float coshx = exp(con) + exp(-con);
	return sinhx / coshx;
}

float tanhDao(float con) {
	float res = CalcTanh(con);
	return 1 - res * res;
}

float sigmod(float con) {
	if (con > 20) return 1.0;
	if (con < -20) return 0.0;
	con = exp(con);
	return con / (1 + con);
}

int getRand(int l,int r) {
	int len = r - l;
	int res = rand()*rand() % len;
	if (res < 0)
		res+=len;
	return res + l;
}

float getRandU(float l, float r) {
	float len = r - l;
	float res = (float)(rand()) / RAND_MAX;
	return res * len + l;
}

void norm(float* a, int ll, int rr)
{
	float tmp = 0;
	for (int i=ll; i<rr; i++)
		tmp+=a[i]*a[i];
	if (tmp>1)
	{
		tmp = sqrt(tmp);
		for (int i=ll; i<rr; i++)
			a[i]/=tmp;
	}
}

void norm(vector<double> &a)
{
	double tmp = 0;
	for (int i=0; i<a.size(); i++)
		tmp+=a[i];
	//if (tmp>1)
	{
	//	tmp = sqrt(tmp);
		for (int i=0; i<a.size(); i++)
			a[i]/=tmp;
	}
}


#endif
