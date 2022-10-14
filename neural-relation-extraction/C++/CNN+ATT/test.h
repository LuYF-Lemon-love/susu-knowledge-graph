#ifndef TEST_H
#define TEST_H
#include "init.h"

bool cmp(pair<string, pair<INT,double> > a,pair<string, pair<INT,double> >b)
{
    return a.second.second>b.second.second;
}

// total: 计算测试集中样本数 (每个样本包含 n 个句子, 每个句子包含相同的 head, relation (label), tail)
// bags_test_key: 保存 bags_test 的 key (头实体 + "\t" + 尾实体), 按照 bags_test 的迭代顺序
// thread_first_bags_test (num_threads + 1): 保存每个线程第一个样本在 bags_test_key 中的位置
double total;
std::vector<std::string> bags_test_key;
std::vector<INT> thread_first_bags_test;

std::vector<std::pair<std::string, std::pair<INT,double> > >aa;

pthread_mutex_t mutex;


vector<double> test(INT *sentence, INT *test_position_head, INT *test_position_tail, INT len, REAL *r) {

	INT tip[dimension_c];
	for (INT i = 0; i < dimension_c; i++) {
		INT last = i * dimension * window;
		INT lastt = i * dimension_pos * window;
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
			 	INT last2 = test_position_head[j] * dimension_pos;
			 	INT last3 = test_position_tail[j] * dimension_pos;
			 	for (INT k = 0; k < dimension_pos; k++) {
			 		res += matrixW1PositionE1[lastt + tot1] * positionVecE1[last2+k];
			 		res += matrixW1PositionE2[lastt + tot1] * positionVecE2[last3+k];
			 		tot1++;
			 	}
			}
			if (res > mx) mx = res;
		}
		r[i] = mx + matrixB1[i];
	}

	for (INT i = 0; i < dimension_c; i++)
		r[i] = calc_tanh(r[i]);
	vector<double> res;
	double tmp = 0;
	for (INT j = 0; j < relation_total; j++) {
		REAL s = 0;
		for (INT i = 0; i < dimension_c; i++)
			s +=  0.5 * matrixRelation[j * dimension_c + i] * r[i];
		s += matrixRelationPr[j];
		s = exp(s);
		tmp+=s;
		res.push_back(s);
	}
	for (INT j = 0; j < relation_total; j++) 
		res[j]/=tmp;
	return res;
}

void* test_mode(void *thread_id) 
{
	INT id;
	id = (unsigned long long)(thread_id);
	INT ll = thread_first_bags_test[id];
	INT rr;
	if (id==num_threads-1)
		rr = bags_test_key.size();
	else
		rr = thread_first_bags_test[id + 1];
	REAL *r = (REAL *)calloc(dimension_c, sizeof(REAL));

	double eps = 0.1;
	for (INT ii = ll; ii < rr; ii++)
	{
		vector<double> sum;
		vector<double> r_sum;
		r_sum.resize(dimension_c);
		for (INT j = 0; j < relation_total; j++)
			sum.push_back(0.0);
		map<INT,INT> ok;
		ok.clear();
		vector<vector<double> > rList;
		INT bags_size = bags_test[bags_test_key[ii]].size();
		INT used = 0;
		for (INT k=0; k<bags_size; k++)
		{
			INT i = bags_test[bags_test_key[ii]][k];
			ok[test_relation_list[i]]=1;
			{
				vector<double> score = test(test_sentence_list[i],  test_position_head[i], test_position_tail[i], test_length[i], r);
				vector<double> r_tmp;
				for (INT j = 0; j < dimension_c; j++)
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
				for (INT i = 0; i < dimension_c; i++) 
				{
					REAL tmp = 0;
					for (INT jj = 0; jj < dimension_c; jj++)
					//	if (i==jj)
						tmp+=rList[k][jj]*att_W[j][jj][i];
					s += tmp * matrixRelation[j * dimension_c + i];
				}
				s = exp(s); 
				weight.push_back(s);
				weight_sum += s;
			}
			for (INT k=0; k<bags_size; k++)
				weight[k]/=weight_sum;
			
			vector<REAL> r;
			r.resize(dimension_c);
			for (INT i = 0; i < dimension_c; i++) 
				for (INT k=0; k<bags_size; k++)
					r[i] += rList[k][i] * weight[k];
			vector<REAL> res;
			double tmp = 0;
			for (INT j1 = 0; j1 < relation_total; j1++) {
				REAL s = 0;
				for (INT i1 = 0; i1 < dimension_c; i1++)
					s +=  0.5 * matrixRelation[j1 * dimension_c + i1] * r[i1];
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
			INT i = bags_test[bags_test_key[ii]][0];
			aa.push_back(make_pair(bags_test_key[ii]+"\t"+id2relation[j],make_pair(ok.count(j),sum[j])));
		}
		pthread_mutex_unlock(&mutex);
	}

	free(r);
}

// 测试函数
void test() {
	aa.clear();
	total = 0;
	bags_test_key.clear();
	thread_first_bags_test.clear();

	std::vector<INT> sample_sum;
	sample_sum.clear();
	for (std::map<std::string, std::vector<INT> >::iterator it = bags_test.begin();
		it != bags_test.end(); it++)
	{
		
		std::map<INT, INT> sample_relation_list;
		sample_relation_list.clear();
		for (INT i = 0; i < it->second.size(); i++)
		{
			INT pos = it->second[i];
			if (test_relation_list[pos] > 0)
				sample_relation_list[test_relation_list[pos]]=1;
		}
		total += sample_relation_list.size();
		{
			bags_test_key.push_back(it->first);
			sample_sum.push_back(it->second.size());
		}
	}

	for (INT i = 1; i < sample_sum.size(); i++)
		sample_sum[i] += sample_sum[i-1];
	
	INT thread_id = 0;
	thread_first_bags_test.resize(num_threads+1);
	for (INT i = 0; i < sample_sum.size(); i++)
		if (sample_sum[i] >= (sample_sum[sample_sum.size()-1] / num_threads) * thread_id)
		{
			thread_first_bags_test[thread_id] = i;
			thread_id += 1;
		}
		
	std::cout << "total: " << total << std::endl;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	for (long a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, test_mode,  (void *)a);
	for (long a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);
	free(pt);

	sort(aa.begin(),aa.end(),cmp);
	double correct=0;
	REAL correct1 = 0;
	for (INT i=0; i<min(2000,INT(aa.size())); i++)
	{
		if (aa[i].second.first!=0)
			correct1++;	
		REAL precision = correct1/(i+1);
		REAL recall = correct1/total;
		if (i%100==0)
			std::cout<<"precision:\t"<<correct1/(i+1)<<'\t'<<"recall:\t"<<correct1/total<<std::endl;	
	}
	{
		FILE* f = fopen(("out/pr"+version+".txt").c_str(), "w");
		for (INT i=0; i<2000; i++)
		{
			if (aa[i].second.first!=0)
				correct++;	
			fprintf(f,"%lf\t%lf\t%lf\t%s\n",correct/(i+1), correct/total,aa[i].second.second, aa[i].first.c_str());
		}
		fclose(f);
		if (!output_model)return;
		FILE *fout = fopen(("./out/matrixW1+B1.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\t%d\t%d\n", dimension_c, dimension, window, dimension_pos);
		for (INT i = 0; i < dimension_c; i++) {
			for (INT j = 0; j < dimension * window; j++)
				fprintf(fout, "%f\t",matrixW1[i* dimension*window+j]);
			for (INT j = 0; j < dimension_pos * window; j++)
				fprintf(fout, "%f\t",matrixW1PositionE1[i* dimension_pos*window+j]);
			for (INT j = 0; j < dimension_pos * window; j++)
				fprintf(fout, "%f\t",matrixW1PositionE2[i* dimension_pos*window+j]);
			fprintf(fout, "%f\n", matrixB1[i]);
		}
		fclose(fout);

		fout = fopen(("./out/matrixRl.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\n", relation_total, dimension_c);
		for (INT i = 0; i < relation_total; i++) {
			for (INT j = 0; j < dimension_c; j++)
				fprintf(fout, "%f\t", matrixRelation[i * dimension_c + j]);
			fprintf(fout, "\n");
		}
		for (INT i = 0; i < relation_total; i++) 
			fprintf(fout, "%f\t",matrixRelationPr[i]);
		fprintf(fout, "\n");
		fclose(fout);

		fout = fopen(("./out/matrixPosition.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\t%d\n", position_total_head, position_total_tail, dimension_pos);
		for (INT i = 0; i < position_total_head; i++) {
			for (INT j = 0; j < dimension_pos; j++)
				fprintf(fout, "%f\t", positionVecE1[i * dimension_pos + j]);
			fprintf(fout, "\n");
		}
		for (INT i = 0; i < position_total_tail; i++) {
			for (INT j = 0; j < dimension_pos; j++)
				fprintf(fout, "%f\t", positionVecE2[i * dimension_pos + j]);
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
		fprintf(fout,"%d\t%d\n", relation_total, dimension_c);
		for (INT r1 = 0; r1 < relation_total; r1++) {
			for (INT i = 0; i < dimension_c; i++)
			{
				for (INT j = 0; j < dimension_c; j++)
					fprintf(fout, "%f\t", att_W[r1][i][j]);
				fprintf(fout, "\n");
			}
		}
		fclose(fout);
	}
}

#endif
