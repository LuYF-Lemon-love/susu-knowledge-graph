#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <map>
#include <cmath>
#include <pthread.h>
#include <iostream>

#include<assert.h>
#include<ctime>
#include<sys/time.h>

#include "init.h"
#include "test.h"

using namespace std;

double score = 0;
REAL alpha1;

struct timeval t_start,t_end; 
long start,end;

void time_begin()
{
  
  gettimeofday(&t_start, NULL);
}
void time_end()
{
  gettimeofday(&t_end, NULL);
  std::cout<<"time(s):\t"<<(double(((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000)-double(((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000))/1000<<std::endl;
}



vector<REAL> train(INT *sentence, INT *trainPositionE1, INT *trainPositionE2, INT len, vector<INT> &tip) {
	vector<REAL> r;
	r.resize(dimensionC);
	for (INT i = 0; i < dimensionC; i++) {
		r[i] = 0;
		INT last = i * dimension * window;
		INT lastt = i * dimensionWPE * window;
		REAL mx = -FLT_MAX;
		for (INT i1 = 0; i1 <= len - window; i1++) {
			REAL res = 0;
			INT tot = 0;
			INT tot1 = 0;
			for (INT j = i1; j < i1 + window; j++)  {
				INT last1 = sentence[j] * dimension;
			 	for (INT k = 0; k < dimension; k++) {
			 		res += matrixW1Dao[last + tot] * wordVecDao[last1+k];
			 		tot++;
			 	}
			 	INT last2 = trainPositionE1[j] * dimensionWPE;
			 	INT last3 = trainPositionE2[j] * dimensionWPE;
			 	for (INT k = 0; k < dimensionWPE; k++) {
			 		res += matrixW1PositionE1Dao[lastt + tot1] * positionVecDaoE1[last2+k];
			 		res += matrixW1PositionE2Dao[lastt + tot1] * positionVecDaoE2[last3+k];
			 		tot1++;
			 	}
			}
			if (res > mx) {
				mx = res;
				tip[i] = i1;
			}
		}
		r[i] = mx + matrixB1Dao[i];
	}

	for (INT i = 0; i < dimensionC; i++) {
		r[i] = CalcTanh(r[i]);
	}
	return r;
}

void train_gradient(INT *sentence, INT *trainPositionE1, INT *trainPositionE2, INT len, INT e1, INT e2, INT r1, REAL alpha, vector<REAL> &r,vector<INT> &tip, vector<REAL> &grad)
{
	for (INT i = 0; i < dimensionC; i++) {
		if (fabs(grad[i])<1e-8)
			continue;
		INT last = i * dimension * window;
		INT tot = 0;
		INT lastt = i * dimensionWPE * window;
		INT tot1 = 0;
		REAL g1 = grad[i] * (1 -  r[i] * r[i]);
		for (INT j = 0; j < window; j++)  {
			INT last1 = sentence[tip[i] + j] * dimension;
			for (INT k = 0; k < dimension; k++) {
				matrixW1[last + tot] -= g1 * wordVecDao[last1+k];
				word_vec[last1 + k] -= g1 * matrixW1Dao[last + tot];
				tot++;
			}
			INT last2 = trainPositionE1[tip[i] + j] * dimensionWPE;
			INT last3 = trainPositionE2[tip[i] + j] * dimensionWPE;
			for (INT k = 0; k < dimensionWPE; k++) {
				matrixW1PositionE1[lastt + tot1] -= g1 * positionVecDaoE1[last2 + k];
				matrixW1PositionE2[lastt + tot1] -= g1 * positionVecDaoE2[last3 + k];
				positionVecE1[last2 + k] -= g1 * matrixW1PositionE1Dao[lastt + tot1];
				positionVecE2[last3 + k] -= g1 * matrixW1PositionE2Dao[lastt + tot1];
				tot1++;
			}
		}
		matrixB1[i] -= g1;
	}
}

REAL train_bags(string bags_name)
{
	INT bags_size = bags_train[bags_name].size();
	double bags_rate = max(1.0,1.0*bags_size/2);
	vector<vector<REAL> > rList;
	vector<vector<INT> > tipList;
	tipList.resize(bags_size);
	INT r1 = -1;
	for (INT k=0; k<bags_size; k++)
	{
		tipList[k].resize(dimensionC);
		INT i = bags_train[bags_name][k];
		if (r1==-1)
			r1 = relationList[i];
		else
			assert(r1==relationList[i]);
		rList.push_back(train(trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], tipList[k]));
	}
	
	vector<REAL> f_r;	
	
	vector<INT> dropout;
	for (INT i = 0; i < dimensionC; i++) 
		//dropout.push_back(1);
		dropout.push_back(rand()%2);
	
	vector<REAL> weight;
	REAL weight_sum = 0;
	for (INT k=0; k<bags_size; k++)
	{
		REAL s = 0;
		for (INT i = 0; i < dimensionC; i++) 
		{
			REAL tmp = 0;
			for (INT j = 0; j < dimensionC; j++)
				tmp+=rList[k][j]*att_W_Dao[r1][j][i];
			s += tmp * matrixRelationDao[r1 * dimensionC + i];
		}
		s = exp(s); 
		weight.push_back(s);
		weight_sum += s;
	}
	for (INT k=0; k<bags_size; k++)
		weight[k] /=weight_sum;
	
	REAL sum = 0;
	for (INT j = 0; j < relation_total; j++) {	
		vector<REAL> r;
		r.resize(dimensionC);
		for (INT i = 0; i < dimensionC; i++) 
			for (INT k=0; k<bags_size; k++)
				r[i] += rList[k][i] * weight[k];
	
		REAL ss = 0;
		for (INT i = 0; i < dimensionC; i++) {
			ss += dropout[i] * r[i] * matrixRelationDao[j * dimensionC + i];
		}
		ss += matrixRelationPrDao[j];
		f_r.push_back(exp(ss));
		sum+=f_r[j];
	}
	
	double rt = (log(f_r[r1]) - log(sum));
	
	vector<vector<REAL> > grad;
	grad.resize(bags_size);
	for (INT k=0; k<bags_size; k++)
		grad[k].resize(dimensionC);
	vector<REAL> g1_tmp;
	g1_tmp.resize(dimensionC);
	for (INT r2 = 0; r2<relation_total; r2++)
	{	
		vector<REAL> r;
		r.resize(dimensionC);
		for (INT i = 0; i < dimensionC; i++) 
			for (INT k=0; k<bags_size; k++)
				r[i] += rList[k][i] * weight[k];
		
		REAL g = f_r[r2]/sum*alpha1;
		if (r2 == r1)
			g -= alpha1;
		for (INT i = 0; i < dimensionC; i++) 
		{
			REAL g1 = 0;
			if (dropout[i]!=0)
			{
				g1 += g * matrixRelationDao[r2 * dimensionC + i];
				matrixRelation[r2 * dimensionC + i] -= g * r[i];
			}
			g1_tmp[i]+=g1;
		}
		matrixRelationPr[r2] -= g;
	}
		for (INT i = 0; i < dimensionC; i++) 
		{
			REAL g1 = g1_tmp[i];
			double tmp_sum = 0; //for rList[k][i]*weight[k]
			for (INT k=0; k<bags_size; k++)
			{
				grad[k][i]+=g1*weight[k];
				for (INT j = 0; j < dimensionC; j++)
				{
					grad[k][j]+=g1*rList[k][i]*weight[k]*matrixRelationDao[r1 * dimensionC + i]*att_W_Dao[r1][j][i];
					matrixRelation[r1 * dimensionC + i] += g1*rList[k][i]*weight[k]*rList[k][j]*att_W_Dao[r1][j][i];
					if (i==j)
					  att_W[r1][j][i] += g1*rList[k][i]*weight[k]*rList[k][j]*matrixRelationDao[r1 * dimensionC + i];
				}
				tmp_sum += rList[k][i]*weight[k];
			}	
			for (INT k1=0; k1<bags_size; k1++)
			{
				for (INT j = 0; j < dimensionC; j++)
				{
					grad[k1][j]-=g1*tmp_sum*weight[k1]*matrixRelationDao[r1 * dimensionC + i]*att_W_Dao[r1][j][i];
					matrixRelation[r1 * dimensionC + i] -= g1*tmp_sum*weight[k1]*rList[k1][j]*att_W_Dao[r1][j][i];
					if (i==j)
					  att_W[r1][j][i] -= g1*tmp_sum*weight[k1]*rList[k1][j]*matrixRelationDao[r1 * dimensionC + i];
				}
			}
		}
	for (INT k=0; k<bags_size; k++)
	{
		INT i = bags_train[bags_name][k];
		train_gradient(trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], headList[i], tailList[i], relationList[i], alpha1,rList[k], tipList[k], grad[k]);
		
	}
	return rt;
}

INT turn;

INT test_tmp = 0;

vector<string> b_train;
vector<INT> c_train;
double score_tmp = 0, score_max = 0;
pthread_mutex_t mutex1;

INT tot_batch;
void* trainMode(void *id ) {
		unsigned long long next_random = (long long)id;
		test_tmp = 0;
	//	for (INT k1 = batch; k1 > 0; k1--)
		while (true)
		{

			pthread_mutex_lock (&mutex1);
			if (score_tmp>=score_max)
			{
				pthread_mutex_unlock (&mutex1);
				break;
			}
			score_tmp+=1;
		//	std::cout<<score_tmp<<' '<<score_max<<std::endl;
			pthread_mutex_unlock (&mutex1);
			INT j = getRand(0, c_train.size());
			//std::cout<<j<<'|';
			j = c_train[j];
			//std::cout<<j<<'|';
			//test_tmp+=bags_train[b_train[j]].size();
			//std::cout<<test_tmp<<' ';
			score += train_bags(b_train[j]);
		}
		//std::cout<<std::endl;
}

void train() {
	INT tmp = 0;
	b_train.clear();
	c_train.clear();
	for (map<string,vector<INT> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	{
		INT max_size = 1;//it->second.size()/2;
		for (INT i=0; i<max(1,max_size); i++)
			c_train.push_back(b_train.size());
		b_train.push_back(it->first);
		tmp+=it->second.size();
	}
	std::cout<<c_train.size()<<std::endl;
	
	att_W.resize(relation_total);
	for (INT i=0; i<relation_total; i++)
	{
		att_W[i].resize(dimensionC);
		for (INT j=0; j<dimensionC; j++)
		{
			att_W[i][j].resize(dimensionC);
			att_W[i][j][j] = 1.00;//1;
		}
	}
	att_W_Dao = att_W;

	REAL con = sqrt(6.0/(dimensionC+relation_total));
	REAL con1 = sqrt(6.0/((dimensionWPE+dimension)*window));
	matrixRelation = (REAL *)calloc(dimensionC * relation_total, sizeof(REAL));
	matrixRelationPr = (REAL *)calloc(relation_total, sizeof(REAL));
	matrixRelationPrDao = (REAL *)calloc(relation_total, sizeof(REAL));
	wordVecDao = (REAL *)calloc(dimension * word_total, sizeof(REAL));
	positionVecE1 = (REAL *)calloc(PositionTotalE1 * dimensionWPE, sizeof(REAL));
	positionVecE2 = (REAL *)calloc(PositionTotalE2 * dimensionWPE, sizeof(REAL));
	
	matrixW1 = (REAL*)calloc(dimensionC * dimension * window, sizeof(REAL));
	matrixW1PositionE1 = (REAL *)calloc(dimensionC * dimensionWPE * window, sizeof(REAL));
	matrixW1PositionE2 = (REAL *)calloc(dimensionC * dimensionWPE * window, sizeof(REAL));
	matrixB1 = (REAL*)calloc(dimensionC, sizeof(REAL));

	for (INT i = 0; i < dimensionC; i++) {
		INT last = i * window * dimension;
		for (INT j = dimension * window - 1; j >=0; j--)
			matrixW1[last + j] = getRandU(-con1, con1);
		last = i * window * dimensionWPE;
		REAL tmp1 = 0;
		REAL tmp2 = 0;
		for (INT j = dimensionWPE * window - 1; j >=0; j--) {
			matrixW1PositionE1[last + j] = getRandU(-con1, con1);
			tmp1 += matrixW1PositionE1[last + j]  * matrixW1PositionE1[last + j] ;
			matrixW1PositionE2[last + j] = getRandU(-con1, con1);
			tmp2 += matrixW1PositionE2[last + j]  * matrixW1PositionE2[last + j] ;
		}
		matrixB1[i] = getRandU(-con1, con1);
	}

	for (INT i = 0; i < relation_total; i++) 
	{
		matrixRelationPr[i] = getRandU(-con, con);				//add
		for (INT j = 0; j < dimensionC; j++)
			matrixRelation[i * dimensionC + j] = getRandU(-con, con);
	}

	for (INT i = 0; i < PositionTotalE1; i++) {
		REAL tmp = 0;
		for (INT j = 0; j < dimensionWPE; j++) {
			positionVecE1[i * dimensionWPE + j] = getRandU(-con1, con1);
			tmp += positionVecE1[i * dimensionWPE + j] * positionVecE1[i * dimensionWPE + j];
		}
	}

	for (INT i = 0; i < PositionTotalE2; i++) {
		REAL tmp = 0;
		for (INT j = 0; j < dimensionWPE; j++) {
			positionVecE2[i * dimensionWPE + j] = getRandU(-con1, con1);
			tmp += positionVecE2[i * dimensionWPE + j] * positionVecE2[i * dimensionWPE + j];
		}
	}

	matrixRelationDao = (REAL *)calloc(dimensionC*relation_total, sizeof(REAL));
	matrixW1Dao =  (REAL*)calloc(dimensionC * dimension * window, sizeof(REAL));
	matrixB1Dao =  (REAL*)calloc(dimensionC, sizeof(REAL));
	
	positionVecDaoE1 = (REAL *)calloc(PositionTotalE1 * dimensionWPE, sizeof(REAL));
	positionVecDaoE2 = (REAL *)calloc(PositionTotalE2 * dimensionWPE, sizeof(REAL));
	matrixW1PositionE1Dao = (REAL *)calloc(dimensionC * dimensionWPE * window, sizeof(REAL));
	matrixW1PositionE2Dao = (REAL *)calloc(dimensionC * dimensionWPE * window, sizeof(REAL));
	/*time_begin();
	test();
	time_end();*/
//	return;
	for (turn = 0; turn < trainTimes; turn ++) {

	//	len = trainLists.size();
		len = c_train.size();
		npoch  =  len / (batch * num_threads);
		alpha1 = alpha*rate/batch;

		score = 0;
		score_max = 0;
		score_tmp = 0;
		double score1 = score;
		time_begin();
		for (INT k = 1; k <= npoch; k++) {
			score_max += batch * num_threads;
		//	std::cout<<k<<std::endl;
			memcpy(positionVecDaoE1, positionVecE1, PositionTotalE1 * dimensionWPE* sizeof(REAL));
			memcpy(positionVecDaoE2, positionVecE2, PositionTotalE2 * dimensionWPE* sizeof(REAL));
			memcpy(matrixW1PositionE1Dao, matrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(REAL));
			memcpy(matrixW1PositionE2Dao, matrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(REAL));
			memcpy(wordVecDao, word_vec, dimension * word_total * sizeof(REAL));

			memcpy(matrixW1Dao, matrixW1, sizeof(REAL) * dimensionC * dimension * window);
			memcpy(matrixB1Dao, matrixB1, sizeof(REAL) * dimensionC);
			memcpy(matrixRelationPrDao, matrixRelationPr, relation_total * sizeof(REAL));				//add
			memcpy(matrixRelationDao, matrixRelation, dimensionC*relation_total * sizeof(REAL));
			att_W_Dao = att_W;
			pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
			for (INT a = 0; a < num_threads; a++)
				pthread_create(&pt[a], NULL, trainMode,  (void *)a);
			for (INT a = 0; a < num_threads; a++)
			pthread_join(pt[a], NULL);
			free(pt);
			if (k%(npoch/5)==0)
			{
				std::cout<<"npoch:\t"<<k<<'/'<<npoch<<std::endl;
				time_end();
				time_begin();
				std::cout<<"score:\t"<<score-score1<<' '<<score_tmp<<std::endl;
				score1 = score;
			}
		}
		printf("Total Score:\t%f\n",score);
		printf("test\n");
		test();
		//if ((turn+1)%1==0) 
		//	rate=rate*reduce;
	}
	test();
	std::cout<<"Train End"<<std::endl;
}

INT main(INT argc, char ** argv) {
	output_model = 1;
	std::cout<<"Init Begin."<<std::endl;
	init();
	//for (map<string,vector<INT> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	//	std::cout<<it->first<<std::endl;
	std::cout<<bags_train.size()<<' '<<bags_test.size()<<std::endl;
	std::cout<<"Init End."<<std::endl;
	train();
}
