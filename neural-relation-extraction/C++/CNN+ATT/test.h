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

// 互斥锁
pthread_mutex_t mutex;

// 计算句子的一维卷机
void calc_conv_1d(INT *sentence, INT *test_position_head,
		INT *test_position_tail, INT len, REAL *result) {
	for (INT i = 0; i < dimension_c; i++) {
		INT last_word = i * window * dimension;
		INT last_pos = i * window * dimension_pos;
		REAL max_pool_1d = -FLT_MAX;
		for (INT last_window = 0; last_window <= len - window; last_window++) {
			REAL sum = 0;
			INT tot_word = 0;
			INT tot_pos = 0;
			for (INT j = last_window; j < last_window + window; j++)  {
				INT last_word_vec = sentence[j] * dimension;
			 	for (INT k = 0; k < dimension; k++) {
			 		sum += conv_1d_word[last_word + tot_word] * word_vec[last_word_vec+k];
			 		tot_word++;
			 	}
			 	INT last_pos_head = test_position_head[j] * dimension_pos;
			 	INT last_pos_tail = test_position_tail[j] * dimension_pos;
			 	for (INT k = 0; k < dimension_pos; k++) {
			 		sum += conv_1d_position_head[last_pos + tot_pos] * position_vec_head[last_pos_head+k];
			 		sum += conv_1d_position_tail[last_pos + tot_pos] * position_vec_tail[last_pos_tail+k];
			 		tot_pos++;
			 	}
			}
			if (sum > max_pool_1d) max_pool_1d = sum;
		}
		result[i] = max_pool_1d + conv_1d_bias[i];
	}

	for (INT i = 0; i < dimension_c; i++)
		result[i] = calc_tanh(result[i]);
}

// 单个线程内运行的任务
void* test_mode(void *thread_id) 
{
	INT id;
	id = (unsigned long long)(thread_id);
	INT left = thread_first_bags_test[id];
	INT right;
	if (id == num_threads-1)
		right = bags_test_key.size();
	else
		right = thread_first_bags_test[id + 1];
	REAL *result = (REAL *)calloc(dimension_c, sizeof(REAL));

	for (INT i_sample = left; i_sample < right; i_sample++)
	{
		std::vector<double> result_final;
		for (INT j = 0; j < relation_total; j++)
			result_final.push_back(0.0);

		std::map<INT,INT> sample_relation_list;
		sample_relation_list.clear();
		std::vector<std::vector<double> > result_list;

		INT bags_size = bags_test[bags_test_key[i_sample]].size();
		for (INT k = 0; k < bags_size; k++)
		{
			INT i = bags_test[bags_test_key[i_sample]][k];
			sample_relation_list[test_relation_list[i]]=1;

			calc_conv_1d(test_sentence_list[i],  test_position_head[i],
				test_position_tail[i], test_length[i], result);
			vector<double> result_temp;
			for (INT j = 0; j < dimension_c; j++)
				result_temp.push_back(result[j]);
			result_list.push_back(result_temp);
		}

		for (INT index_r = 0; index_r < relation_total; index_r++) {
			vector<REAL> weight;
			REAL weight_sum = 0;
			for (INT k = 0; k < bags_size; k++)
			{
				REAL s = 0;
				for (INT i_r = 0; i_r < dimension_c; i_r++) 
				{
					REAL temp = 0;
					for (INT i_x = 0; i_x < dimension_c; i_x++)
						temp += result_list[k][i_x] * attention_weights[index_r][i_x][i_r];
					s += temp * relation_matrix[index_r * dimension_c + i_r];
				}
				s = exp(s);
				weight.push_back(s);
				weight_sum += s;
			}

			for (INT k = 0; k < bags_size; k++)
				weight[k] /= weight_sum;
			
			vector<REAL> result_sentence;
			result_sentence.resize(dimension_c);
			for (INT i = 0; i < dimension_c; i++) 
				for (INT k = 0; k < bags_size; k++)
					result_sentence[i] += result_list[k][i] * weight[k];

			vector<REAL> result_final_r;
			double temp = 0;
			for (INT i_r = 0; i_r < relation_total; i_r++) {
				REAL s = 0;
				for (INT i_s = 0; i_s < dimension_c; i_s++)
					s +=  dropout * relation_matrix[i_r * dimension_c + i_s] * result_sentence[i_s];
				s += relation_matrix_bias[i_r];
				s = exp(s);
				temp += s;
				result_final_r.push_back(s);
			}
			result_final[index_r] = max(result_final[index_r], result_final_r[index_r]/temp);
		}

		pthread_mutex_lock (&mutex);
		for (INT i_r = 1; i_r < relation_total; i_r++) 
		{
			aa.push_back(make_pair(bags_test_key[i_sample] + "\t" + id2relation[i_r],
				make_pair(sample_relation_list.count(i_r), result_final[i_r])));
		}
		pthread_mutex_unlock(&mutex);
	}

	free(result);
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
		bags_test_key.push_back(it->first);
		sample_sum.push_back(it->second.size());
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
				fprintf(fout, "%f\t",conv_1d_word[i* dimension*window+j]);
			for (INT j = 0; j < dimension_pos * window; j++)
				fprintf(fout, "%f\t",conv_1d_position_head[i* dimension_pos*window+j]);
			for (INT j = 0; j < dimension_pos * window; j++)
				fprintf(fout, "%f\t",conv_1d_position_tail[i* dimension_pos*window+j]);
			fprintf(fout, "%f\n", conv_1d_bias[i]);
		}
		fclose(fout);

		fout = fopen(("./out/matrixRl.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\n", relation_total, dimension_c);
		for (INT i = 0; i < relation_total; i++) {
			for (INT j = 0; j < dimension_c; j++)
				fprintf(fout, "%f\t", relation_matrix[i * dimension_c + j]);
			fprintf(fout, "\n");
		}
		for (INT i = 0; i < relation_total; i++) 
			fprintf(fout, "%f\t",relation_matrix_bias[i]);
		fprintf(fout, "\n");
		fclose(fout);

		fout = fopen(("./out/matrixPosition.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\t%d\n", position_total_head, position_total_tail, dimension_pos);
		for (INT i = 0; i < position_total_head; i++) {
			for (INT j = 0; j < dimension_pos; j++)
				fprintf(fout, "%f\t", position_vec_head[i * dimension_pos + j]);
			fprintf(fout, "\n");
		}
		for (INT i = 0; i < position_total_tail; i++) {
			for (INT j = 0; j < dimension_pos; j++)
				fprintf(fout, "%f\t", position_vec_tail[i * dimension_pos + j]);
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
					fprintf(fout, "%f\t", attention_weights[r1][i][j]);
				fprintf(fout, "\n");
			}
		}
		fclose(fout);
	}
}

#endif
