#ifndef TEST_H
#define TEST_H
#include "init.h"



// total: 计算测试集中样本数 (其中 relation 非 NA,每个样本包含 n 个句子, 每个句子包含相同的 head, relation (label), tail)
// bags_test_key: 保存 bags_test 的 key (头实体 + "\t" + 尾实体), 按照 bags_test 的迭代顺序
// thread_first_bags_test (num_threads + 1): 保存每个线程第一个样本在 bags_test_key 中的位置
// test_mutex: 互斥锁
INT total;
std::vector<std::string> bags_test_key;
std::vector<INT> thread_first_bags_test;
pthread_mutex_t test_mutex;

// predict_relation_vector: 每一个元素的 key -> (头实体 + "\t" + 尾实体 + "\t" + 预测关系名)
// value 的 key -> (0 或 1, 0 表示关系预测错误, 1 表示关系预测正确)
// value 的 value -> 模型给出的该关系成立的概率
// 以模型给出的关系成立的概率降序排列
std::vector<std::pair<std::string, std::pair<INT,double> > > predict_relation_vector;

// 为 std::sort() 定义比较函数
// 以模型给出的关系成立的概率降序排列
bool cmp_predict_probability(pair<string, pair<INT,double> > a,pair<string, pair<INT,double> >b)
{
    return a.second.second > b.second.second;
}

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
					s +=  dropout_probability * relation_matrix[i_r * dimension_c + i_s] * result_sentence[i_s];
				s += relation_matrix_bias[i_r];
				s = exp(s);
				temp += s;
				result_final_r.push_back(s);
			}
			result_final[index_r] = max(result_final[index_r], result_final_r[index_r]/temp);
		}

		pthread_mutex_lock (&test_mutex);
		for (INT i_r = 1; i_r < relation_total; i_r++) 
		{
			predict_relation_vector.push_back(make_pair(bags_test_key[i_sample] + "\t" + id2relation[i_r],
				make_pair(sample_relation_list.count(i_r), result_final[i_r])));
		}
		pthread_mutex_unlock(&test_mutex);
	}

	free(result);
}

// 测试函数
void test() {

	printf("##################################################\n\nTest start...\n\n");

	total = 0;
	bags_test_key.clear();
	thread_first_bags_test.clear();
	predict_relation_vector.clear();

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
	printf("Number of test samples for non NA relation: %d\n\n", total);

	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	for (long a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, test_mode,  (void *)a);
	for (long a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);
	free(pt);

	std::sort(predict_relation_vector.begin(),predict_relation_vector.end(), cmp_predict_probability);

	REAL correct = 0;
	FILE* f = fopen(("out/pr" + version + ".txt").c_str(), "w");
	INT top_2000 = min(2000, INT(predict_relation_vector.size()));
	for (INT i = 0; i < top_2000; i++)
	{
		if (predict_relation_vector[i].second.first != 0)
			correct++;	
		REAL precision = correct / (i+1);
		REAL recall = correct / total;
		if ((i+1) % 50 == 0)
			printf("precion/recall curves %4d / %4d - precision: %.3lf - recall: %.3lf\n", (i + 1), top_2000, precision, recall);
		fprintf(f, "precision: %.3lf  recall: %.3lf  correct: %d  predict_probability: %.2lf  predict_triplet: %s\n",
			precision, recall, predict_relation_vector[i].second.first, predict_relation_vector[i].second.second,
			predict_relation_vector[i].first.c_str());	
	}
	fclose(f);

	printf("\nTest end.\n\n##################################################\n\n");

	if (!output_model)return;

	FILE *fout = fopen(("./out/word2vec" + version + ".txt").c_str(), "w");
	fprintf(fout, "%d\t%d\n", word_total, dimension);
	for (INT i = 0; i < word_total; i++)
	{
		for (INT j = 0; j < dimension; j++)
			fprintf(fout,"%f\t", word_vec[i * dimension + j]);
		fprintf(fout, "\n");
	}
	fclose(fout);

	fout = fopen(("./out/position_vec" + version + ".txt").c_str(), "w");
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

	fout = fopen(("./out/conv_1d" + version + ".txt").c_str(), "w");
	fprintf(fout,"%d\t%d\t%d\t%d\n", dimension_c, dimension, window, dimension_pos);
	for (INT i = 0; i < dimension_c; i++) {
		for (INT j = 0; j < dimension * window; j++)
			fprintf(fout, "%f\t", conv_1d_word[i * dimension * window + j]);
		for (INT j = 0; j < dimension_pos * window; j++)
			fprintf(fout, "%f\t", conv_1d_position_head[i * dimension_pos * window + j]);
		for (INT j = 0; j < dimension_pos * window; j++)
			fprintf(fout, "%f\t", conv_1d_position_tail[i * dimension_pos * window + j]);
		fprintf(fout, "%f\n", conv_1d_bias[i]);
	}
	fclose(fout);

	fout = fopen(("./out/attention_weights" + version + ".txt").c_str(), "w");
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

	fout = fopen(("./out/relation_matrix" + version + ".txt").c_str(), "w");
	fprintf(fout, "%d\t%d\n", relation_total, dimension_c);
	for (INT i = 0; i < relation_total; i++) {
		for (INT j = 0; j < dimension_c; j++)
			fprintf(fout, "%f\t", relation_matrix[i * dimension_c + j]);
		fprintf(fout, "\n");
	}
	for (INT i = 0; i < relation_total; i++) 
		fprintf(fout, "%f\t", relation_matrix_bias[i]);
	fprintf(fout, "\n");
	fclose(fout);
	
}

#endif
