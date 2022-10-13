#ifndef TEST_H
#define TEST_H
#include "init.h"
#include <algorithm>
#include <map>

INT tipp = 0;
REAL ress = 0;

vector<double> test(INT *sentence, INT *test_position_head, INT *test_position_tail, INT len, REAL *r) {
	INT tip[dimensionC];
		
	for (INT i = 0; i < dimensionC; i++) {
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
			 		res += matrixW1[last + tot] * word_vec[last1+k];
			 		tot++;
			 	}
			 	INT last2 = test_position_head[j] * dimensionWPE;
			 	INT last3 = test_position_tail[j] * dimensionWPE;
			 	for (INT k = 0; k < dimensionWPE; k++) {
			 		res += matrixW1PositionE1[lastt + tot1] * positionVecE1[last2+k];
			 		res += matrixW1PositionE2[lastt + tot1] * positionVecE2[last3+k];
			 		tot1++;
			 	}
			}
			if (res > mx) mx = res;
		}
		r[i] = mx + matrixB1[i];
	}

	for (INT i = 0; i < dimensionC; i++)
		r[i] = calc_tanh(r[i]);
	vector<double> res;
	double tmp = 0;
	for (INT j = 0; j < relation_total; j++) {
		REAL s = 0;
		for (INT i = 0; i < dimensionC; i++)
			s +=  0.5 * matrixRelation[j * dimensionC + i] * r[i];
		s += matrixRelationPr[j];
		s = exp(s);
		tmp+=s;
		res.push_back(s);
	}
	for (INT j = 0; j < relation_total; j++) 
		res[j]/=tmp;
	return res;
}


bool cmp(pair<string, pair<INT,double> > a,pair<string, pair<INT,double> >b)
{
    return a.second.second>b.second.second;
}

vector<string> b;
double tot;
vector<pair<string, pair<INT,double> > >aa;

pthread_mutex_t mutex;
vector<INT> ll_test;


bool cmp1(pair<double,INT> a, pair<double,INT> b)
{
	return a.first<b.first;
}
void output(pair<double,INT> a)
{
	std::cout<<"weight:\t"<<a.first<<' ';
	INT i = a.second;
	for (INT j=0; j<test_length[i]; j++)
		std::cout<<id2word[test_sentence_list[i][j]]<<' ';
	std::cout<<std::endl;
}


void* testMode(void *id ) 
{
	INT ll = ll_test[(long long)id];
	INT rr;
	if ((long long)id==num_threads-1)
		rr = b.size();
	else
		rr = ll_test[(long long)id+1];
	//std::cout<<ll<<' '<<rr<<' '<<((long long)id)<<std::endl;
	REAL *r = (REAL *)calloc(dimensionC, sizeof(REAL));
	double eps = 0.1;
	for (INT ii = ll; ii < rr; ii++)
	{
		vector<double> sum;
		vector<double> r_sum;
		r_sum.resize(dimensionC);
		for (INT j = 0; j < relation_total; j++)
			sum.push_back(0.0);
		map<INT,INT> ok;
		ok.clear();
		vector<vector<double> > rList;
		INT bags_size = bags_test[b[ii]].size();
		INT used = 0;
		for (INT k=0; k<bags_size; k++)
		{
			INT i = bags_test[b[ii]][k];
			ok[test_relation_list[i]]=1;
			{
				vector<double> score = test(test_sentence_list[i],  test_position_head[i], test_position_tail[i], test_length[i], r);
				vector<double> r_tmp;
				for (INT j = 0; j < dimensionC; j++)
					r_tmp.push_back(r[j]);
				rList.push_back(r_tmp);
			}
		}
		for (INT j = 0; j < relation_total; j++) {
			vector<REAL> weight;
			REAL weight_sum = 0;
			for (INT k=0; k<bags_size; k++)
			{
				REAL s = 0;
				for (INT i = 0; i < dimensionC; i++) 
				{
					REAL tmp = 0;
					for (INT jj = 0; jj < dimensionC; jj++)
					//	if (i==jj)
						tmp+=rList[k][jj]*att_W[j][jj][i];
					s += tmp * matrixRelation[j * dimensionC + i];
				}
				s = exp(s); 
				weight.push_back(s);
				weight_sum += s;
			}
			for (INT k=0; k<bags_size; k++)
				weight[k]/=weight_sum;
			
			vector<REAL> r;
			r.resize(dimensionC);
			for (INT i = 0; i < dimensionC; i++) 
				for (INT k=0; k<bags_size; k++)
					r[i] += rList[k][i] * weight[k];
			vector<REAL> res;
			double tmp = 0;
			for (INT j1 = 0; j1 < relation_total; j1++) {
				REAL s = 0;
				for (INT i1 = 0; i1 < dimensionC; i1++)
					s +=  0.5 * matrixRelation[j1 * dimensionC + i1] * r[i1];
				s += matrixRelationPr[j1];
				s = exp(s);
				tmp+=s;
				res.push_back(s);
			}
			sum[j] = max(sum[j],res[j]/tmp);
		}
		pthread_mutex_lock (&mutex);
		for (INT j = 1; j < relation_total; j++) 
		{
			INT i = bags_test[b[ii]][0];
			aa.push_back(make_pair(b[ii]+"\t"+id2relation[j],make_pair(ok.count(j),sum[j])));
		}
		pthread_mutex_unlock(&mutex);
	}

	free(r);
}

double max_pre = 0;

void test() {
	std::cout<<std::endl;
	aa.clear();
	b.clear();
	tot = 0;
	ll_test.clear();
	vector<INT> b_sum;
	b_sum.clear();
	//for (map<string,vector<INT> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	for (map<string,vector<INT> >:: iterator it = bags_test.begin(); it!=bags_test.end(); it++)
	{
		
		map<INT,INT> ok;
		ok.clear();
		for (INT k=0; k<it->second.size(); k++)
		{
			INT i = it->second[k];
			if (test_relation_list[i]>0)
				ok[test_relation_list[i]]=1;
			//if (train_relation_list[i]>0)
			//	ok[train_relation_list[i]]=1;
		}
		tot+=ok.size();
		{
			b.push_back(it->first);
			b_sum.push_back(it->second.size());
		}
	}
	for (INT i=1; i<b_sum.size(); i++)
		b_sum[i] += b_sum[i-1];
	INT now = 0;
	ll_test.resize(num_threads+1);
	for (INT i=0; i<b_sum.size(); i++)
		if (b_sum[i]>=b_sum[b_sum.size()-1]/num_threads*now)
		{
			ll_test[now] = i;
			now+=1;
		}
	std::cout<<"tot:\t"<<tot<<std::endl;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	for (INT a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, testMode,  (void *)a);
	for (INT a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);
	//std::cout<<"begin sort"<<std::endl;
	free(pt);
	sort(aa.begin(),aa.end(),cmp);
	double correct=0;
	REAL correct1 = 0;
	for (INT i=0; i<min(2000,INT(aa.size())); i++)
	{
		if (aa[i].second.first!=0)
			correct1++;	
		REAL precision = correct1/(i+1);
		REAL recall = correct1/tot;
		if (i%100==0)
			std::cout<<"precision:\t"<<correct1/(i+1)<<'\t'<<"recall:\t"<<correct1/tot<<std::endl;	
	}
	//assert(version!="");
	{
		FILE* f = fopen(("out/pr"+version+".txt").c_str(), "w");
		for (INT i=0; i<2000; i++)
		{
			if (aa[i].second.first!=0)
				correct++;	
			fprintf(f,"%lf\t%lf\t%lf\t%s\n",correct/(i+1), correct/tot,aa[i].second.second, aa[i].first.c_str());
		}
		fclose(f);
		if (!output_model)return;
		FILE *fout = fopen(("./out/matrixW1+B1.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\t%d\t%d\n", dimensionC, dimension, window, dimensionWPE);
		for (INT i = 0; i < dimensionC; i++) {
			for (INT j = 0; j < dimension * window; j++)
				fprintf(fout, "%f\t",matrixW1[i* dimension*window+j]);
			for (INT j = 0; j < dimensionWPE * window; j++)
				fprintf(fout, "%f\t",matrixW1PositionE1[i* dimensionWPE*window+j]);
			for (INT j = 0; j < dimensionWPE * window; j++)
				fprintf(fout, "%f\t",matrixW1PositionE2[i* dimensionWPE*window+j]);
			fprintf(fout, "%f\n", matrixB1[i]);
		}
		fclose(fout);

		fout = fopen(("./out/matrixRl.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\n", relation_total, dimensionC);
		for (INT i = 0; i < relation_total; i++) {
			for (INT j = 0; j < dimensionC; j++)
				fprintf(fout, "%f\t", matrixRelation[i * dimensionC + j]);
			fprintf(fout, "\n");
		}
		for (INT i = 0; i < relation_total; i++) 
			fprintf(fout, "%f\t",matrixRelationPr[i]);
		fprintf(fout, "\n");
		fclose(fout);

		fout = fopen(("./out/matrixPosition.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\t%d\n", position_total_head, position_total_tail, dimensionWPE);
		for (INT i = 0; i < position_total_head; i++) {
			for (INT j = 0; j < dimensionWPE; j++)
				fprintf(fout, "%f\t", positionVecE1[i * dimensionWPE + j]);
			fprintf(fout, "\n");
		}
		for (INT i = 0; i < position_total_tail; i++) {
			for (INT j = 0; j < dimensionWPE; j++)
				fprintf(fout, "%f\t", positionVecE2[i * dimensionWPE + j]);
			fprintf(fout, "\n");
		}
		fclose(fout);
	
		fout = fopen(("./out/word2vec.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\n",word_total,dimension);
		for (INT i = 0; i < word_total; i++)
		{
			for (INT j=0; j<dimension; j++)
				fprintf(fout,"%f\t",word_vec[i*dimension+j]);
			fprintf(fout,"\n");
		}
		fclose(fout);
		fout = fopen(("./out/att_W.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\n", relation_total, dimensionC);
		for (INT r1 = 0; r1 < relation_total; r1++) {
			for (INT i = 0; i < dimensionC; i++)
			{
				for (INT j = 0; j < dimensionC; j++)
					fprintf(fout, "%f\t", att_W[r1][i][j]);
				fprintf(fout, "\n");
			}
		}
		fclose(fout);
	}
}

#endif
