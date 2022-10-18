// test.h
//
// created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
//
// 该 C++ 文件用于模型测试
//
// 输出 precion/recall curves
// output:
//     ./out/pr + note + .txt
//
// 输出模型 (可选)
// output:
//     ./out/word2vec + note + .txt
//     ./out/position_vec + note + .txt
//     ./out/conv_1d + note + .txt
//     ./out/attention_weights + note + .txt
//     ./out/relation_matrix + note + .txt

// ##################################################
// 包含标准库和头文件
// ##################################################

#ifndef TEST_H
#define TEST_H
#include "init.h"

// ##################################################
// 声明和定义变量
// ##################################################

// predict_relation_vector: 每一个元素的 key -> (头实体 + "\t" + 尾实体 + "\t" + 预测关系名)
// value 的 key -> (0 或 1, 0 表示关系预测错误, 1 表示关系预测正确)
// value 的 value -> 模型给出的该关系成立的概率
// 以模型给出的关系成立的概率降序排列
std::vector<std::pair<std::string, std::pair<INT,double> > > predict_relation_vector;

// num_test_non_NA: 计算测试集中样本数 (其中 relation 非 NA,每个样本包含 n 个句子, 每个句子包含相同的 head, relation (label), tail)
// bags_test_key: 保存 bags_test 的 key (头实体 + "\t" + 尾实体), 按照 bags_test 的迭代顺序
// thread_first_bags_test (num_threads + 1): 保存每个线程第一个样本在 bags_test_key 中的位置
// test_mutex: 互斥锁, 线程同步 predict_relation_vector 变量
INT num_test_non_NA;
std::vector<std::string> bags_test_key;
std::vector<INT> thread_first_bags_test;
pthread_mutex_t test_mutex;

// 为 std::sort() 定义比较函数
// 以模型给出的关系成立的概率降序排列, 用于 predict_relation_vector 变量
bool cmp_predict_probability(std::pair<std::string, std::pair<INT,double> > a,
	std::pair<std::string, std::pair<INT,double> >b)
{
    return a.second.second > b.second.second;
}

// 计算句子的一维卷机
std::vector<REAL> calc_conv_1d(INT *sentence, INT *test_position_head,
	INT *test_position_tail, INT sentence_length) {
	
	std::vector<REAL> conv_1d_result_k;
	conv_1d_result_k.resize(dimension_c, 0);
	
	for (INT i = 0; i < dimension_c; i++) {
		INT last_word = i * window * dimension;
		INT last_pos = i * window * dimension_pos;
		REAL max_pool_1d = -FLT_MAX;
		for (INT last_window = 0; last_window <= sentence_length - window; last_window++) {
			REAL sum = 0;
			INT total_word = 0;
			INT total_pos = 0;
			for (INT j = last_window; j < last_window + window; j++)  {
				INT last_word_vec = sentence[j] * dimension;
			 	for (INT k = 0; k < dimension; k++) {
			 		sum += conv_1d_word[last_word + total_word] * word_vec[last_word_vec + k];
			 		total_word++;
			 	}
			 	INT last_pos_head = test_position_head[j] * dimension_pos;
			 	INT last_pos_tail = test_position_tail[j] * dimension_pos;
			 	for (INT k = 0; k < dimension_pos; k++) {
			 		sum += conv_1d_position_head[last_pos + total_pos] * position_vec_head[last_pos_head + k];
			 		sum += conv_1d_position_tail[last_pos + total_pos] * position_vec_tail[last_pos_tail + k];
			 		total_pos++;
			 	}
			}

			// 对应于论文中的公式 (3), [x]_i = max(p_i), 其中 x \in R^{d^c}
			if (sum > max_pool_1d) max_pool_1d = sum;
		}
		conv_1d_result_k[i] = max_pool_1d + conv_1d_bias[i];
	}

	for (INT i = 0; i < dimension_c; i++)
		conv_1d_result_k[i] = calc_tanh(conv_1d_result_k[i]);
	return conv_1d_result_k;
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

	// 保存样本的正确标签 (关系)
	std::map<INT,INT> sample_relation_list;

	for (INT i_sample = left; i_sample < right; i_sample++)
	{
		// 一维卷机部分
		sample_relation_list.clear();
		std::vector<std::vector<REAL> > conv_1d_result;
		INT bags_size = bags_test[bags_test_key[i_sample]].size();
		for (INT k = 0; k < bags_size; k++)
		{
			INT i = bags_test[bags_test_key[i_sample]][k];
			sample_relation_list[test_relation_list[i]] = 1;

			conv_1d_result.push_back(calc_conv_1d(test_sentence_list[i],
				test_position_head[i], test_position_tail[i], test_length[i]));
		}

		// 对应于论文中的公式 (8), e_i = x_iAr, 其中 r is the query vector associated with relation r which
		// indicates the representation of relation r, 也就是 predict 时, 需要用每一个关系依次查询.
		std::vector<float> result_final;
		result_final.resize(relation_total, 0.0);
		for (INT index_r = 0; index_r < relation_total; index_r++) {
			
			// 获取每一个句子的权重
			std::vector<REAL> weight;
			REAL weight_sum = 0;
			for (INT k = 0; k < bags_size; k++)
			{
				REAL s = 0;
				for (INT i_r = 0; i_r < dimension_c; i_r++) 
				{
					REAL temp = 0;
					for (INT i_x = 0; i_x < dimension_c; i_x++)
						temp += conv_1d_result[k][i_x] * attention_weights[index_r][i_x][i_r];
					s += temp * relation_matrix[index_r * dimension_c + i_r];
				}
				s = exp(s);
				weight.push_back(s);
				weight_sum += s;
			}

			for (INT k = 0; k < bags_size; k++)
				weight[k] /= weight_sum;
			
			// 获取 s, i.e., s indicates the representation of the sentence set
			std::vector<REAL> result_sentence;
			result_sentence.resize(dimension_c);
			for (INT i = 0; i < dimension_c; i++) 
				for (INT k = 0; k < bags_size; k++)
					result_sentence[i] += conv_1d_result[k][i] * weight[k];

			// 获取关系 (id 为 index_r) 成立的概率
			std::vector<REAL> result_final_r;
			double temp = 0;
			for (INT i_r = 0; i_r < relation_total; i_r++) {
				REAL s = 0;
				for (INT i_s = 0; i_s < dimension_c; i_s++)
					s +=  dropout_probability * result_sentence[i_s] *
						relation_matrix[i_r * dimension_c + i_s];
				s += relation_matrix_bias[i_r];
				s = exp(s);
				temp += s;
				result_final_r.push_back(s);
			}
			result_final[index_r] = result_final_r[index_r]/temp;
		}

		// 保存该测试样本各个关系 (非 NA) 成立的概率, 使用线程同步
		pthread_mutex_lock (&test_mutex);
		for (INT i_r = 1; i_r < relation_total; i_r++) 
		{
			predict_relation_vector.push_back(std::make_pair(bags_test_key[i_sample] + "\t" + id2relation[i_r],
				std::make_pair(sample_relation_list.count(i_r), result_final[i_r])));
		}
		pthread_mutex_unlock(&test_mutex);
	}
}

// 测试函数
void test() {

	printf("##################################################\n\nTest start...\n\n");

	num_test_non_NA = 0;
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
				sample_relation_list[test_relation_list[pos]] = 1;
		}
		num_test_non_NA += sample_relation_list.size();
		bags_test_key.push_back(it->first);
		sample_sum.push_back(it->second.size());
	}

	for (INT i = 1; i < sample_sum.size(); i++)
		sample_sum[i] += sample_sum[i - 1];
	
	INT thread_id = 0;
	thread_first_bags_test.resize(num_threads + 1);
	for (INT i = 0; i < sample_sum.size(); i++)
		if (sample_sum[i] >= (sample_sum[sample_sum.size()-1] / num_threads) * thread_id)
		{
			thread_first_bags_test[thread_id] = i;
			thread_id += 1;
		}
	printf("Number of test samples for non NA relation: %d\n\n", num_test_non_NA);

	// 多线程模型测试
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	for (long a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, test_mode,  (void *)a);
	for (long a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);
	free(pt);

	// 以模型给出的关系成立的概率降序排列
	std::sort(predict_relation_vector.begin(),predict_relation_vector.end(), cmp_predict_probability);

	// 输出 precion/recall curves
	REAL correct = 0;
	FILE* f = fopen(("./out/pr" + note + ".txt").c_str(), "w");
	INT top_2000 = std::min(2000, INT(predict_relation_vector.size()));
	for (INT i = 0; i < top_2000; i++)
	{
		if (predict_relation_vector[i].second.first != 0)
			correct++;	
		REAL precision = correct / (i + 1);
		REAL recall = correct / num_test_non_NA;
		if ((i+1) % 50 == 0)
			printf("precion/recall curves %4d / %4d - precision: %.3lf - recall: %.3lf\n", (i + 1), top_2000, precision, recall);
		fprintf(f, "precision: %.3lf  recall: %.3lf  correct: %d  predict_probability: %.2lf  predict_triplet: %s\n",
			precision, recall, predict_relation_vector[i].second.first, predict_relation_vector[i].second.second,
			predict_relation_vector[i].first.c_str());	
	}
	fclose(f);

	printf("\nTest end.\n\n##################################################\n\n");

	if (!output_model)return;

	// 输出词嵌入
	FILE *fout = fopen(("./out/word2vec" + note + ".txt").c_str(), "w");
	fprintf(fout, "%d\t%d\n", word_total, dimension);
	for (INT i = 0; i < word_total; i++)
	{
		for (INT j = 0; j < dimension; j++)
			fprintf(fout, "%f\t", word_vec[i * dimension + j]);
		fprintf(fout, "\n");
	}
	fclose(fout);

	// 输出位置嵌入
	fout = fopen(("./out/position_vec" + note + ".txt").c_str(), "w");
	fprintf(fout, "%d\t%d\t%d\n", position_total_head, position_total_tail, dimension_pos);
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

	// 输出一维卷机权重矩阵和对应的偏置向量
	fout = fopen(("./out/conv_1d" + note + ".txt").c_str(), "w");
	fprintf(fout,"%d\t%d\t%d\t%d\n", dimension_c, window, dimension, dimension_pos);
	for (INT i = 0; i < dimension_c; i++) {
		for (INT j = 0; j < window * dimension; j++)
			fprintf(fout, "%f\t", conv_1d_word[i * window * dimension + j]);
		for (INT j = 0; j < window * dimension_pos; j++)
			fprintf(fout, "%f\t", conv_1d_position_head[i * window * dimension_pos + j]);
		for (INT j = 0; j < window * dimension_pos; j++)
			fprintf(fout, "%f\t", conv_1d_position_tail[i * window * dimension_pos + j]);
		fprintf(fout, "%f\n", conv_1d_bias[i]);
	}
	fclose(fout);

	// 输出注意力权重矩阵
	fout = fopen(("./out/attention_weights" + note + ".txt").c_str(), "w");
	fprintf(fout,"%d\t%d\n", relation_total, dimension_c);
	for (INT r = 0; r < relation_total; r++) {
		for (INT i_x = 0; i_x < dimension_c; i_x++)
		{
			for (INT i_r = 0; i_r < dimension_c; i_r++)
				fprintf(fout, "%f\t", attention_weights[r][i_x][i_r]);
			fprintf(fout, "\n");
		}
	}
	fclose(fout);

	// 输出 relation_matrix 和对应的偏置向量
	fout = fopen(("./out/relation_matrix" + note + ".txt").c_str(), "w");
	fprintf(fout, "%d\t%d\t%f\n", relation_total, dimension_c, dropout_probability);
	for (INT i_r = 0; i_r < relation_total; i_r++) {
		for (INT i_s = 0; i_s < dimension_c; i_s++)
			fprintf(fout, "%f\t", relation_matrix[i_r * dimension_c + i_s]);
		fprintf(fout, "\n");
	}
	for (INT i_r = 0; i_r < relation_total; i_r++) 
		fprintf(fout, "%f\t", relation_matrix_bias[i_r]);
	fprintf(fout, "\n");
	fclose(fout);
}

#endif
