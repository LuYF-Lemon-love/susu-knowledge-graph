#include "init.h"
#include "test.h"

// bags_test_key: 保存 bags_train 的 key (头实体 + "\t" + 尾实体 + "\t" + 关系名), 按照 bags_train 的迭代顺序
// total_loss: 每一轮次的总损失
// current_alpha: 当前轮次的学习率
// current_sample, final_sample: 由于使用多线程训练模型, 这两个变量用于确定当前训练批次是否完成, 进而更新各种权重矩阵的副本, 如 word_vec_copy
// train_mutex: 互斥锁
std::vector<std::string> bags_train_key;
double total_loss = 0;
REAL current_alpha;
double current_sample = 0, final_sample = 0;
pthread_mutex_t train_mutex;

INT nbatches;
INT len;

struct timeval t_start, t_end;

std::vector<REAL> calc_conv_1d(INT *sentence, INT *train_position_head,
	INT *train_position_tail, INT len, std::vector<INT> &max_pool_window_k) {
	std::vector<REAL> conv_1d_result_k;
	conv_1d_result_k.resize(dimension_c, 0);

	for (INT i = 0; i < dimension_c; i++) {
		INT last_word = i * window * dimension;
		INT last_pos = i * window * dimension_pos;
		REAL max_pool_1d = -FLT_MAX;
		for (INT last_window = 0; last_window <= len - window; last_window++) {
			REAL sum = 0;
			INT total_word = 0;
			INT total_pos = 0;
			for (INT j = last_window; j < last_window + window; j++)  {
				INT last_word_vec = sentence[j] * dimension;
			 	for (INT k = 0; k < dimension; k++) {
			 		sum += conv_1d_word_copy[last_word + total_word] * word_vec_copy[last_word_vec + k];
			 		total_word++;
			 	}
			 	INT last_pos_head = train_position_head[j] * dimension_pos;
			 	INT last_pos_tail = train_position_tail[j] * dimension_pos;
			 	for (INT k = 0; k < dimension_pos; k++) {
			 		sum += conv_1d_position_head_copy[last_pos + total_pos] * position_vec_head_copy[last_pos_head+k];
			 		sum += conv_1d_position_tail_copy[last_pos + total_pos] * position_vec_tail_copy[last_pos_tail+k];
			 		total_pos++;
			 	}
			}
			if (sum > max_pool_1d) {
				max_pool_1d = sum;
				max_pool_window_k[i] = last_window;
			}
		}
		conv_1d_result_k[i] = max_pool_1d + conv_1d_bias_copy[i];
	}

	for (INT i = 0; i < dimension_c; i++) {
		conv_1d_result_k[i] = calc_tanh(conv_1d_result_k[i]);
	}
	return conv_1d_result_k;
}

void gradient_conv_1d(INT *sentence, INT *train_position_head, INT *train_position_tail,
	INT len, std::vector<REAL> &conv_1d_result_k,
	std::vector<INT> &max_pool_window_k, std::vector<REAL> &grad_x_k)
{
	for (INT i = 0; i < dimension_c; i++) {
		if (fabs(grad_x_k[i]) < 1e-8)
			continue;
		INT last_word = i * window * dimension;
		INT last_pos = i * window * dimension_pos;
		INT total_word = 0;
		INT total_pos = 0;
		REAL grad_word_pos = grad_x_k[i] * (1 -  conv_1d_result_k[i] * conv_1d_result_k[i]);
		for (INT j = 0; j < window; j++)  {
			INT last_word_vec = sentence[max_pool_window_k[i] + j] * dimension;
			for (INT k = 0; k < dimension; k++) {
				conv_1d_word[last_word + total_word] -= grad_word_pos * word_vec_copy[last_word_vec + k];
				word_vec[last_word_vec + k] -= grad_word_pos * conv_1d_word_copy[last_word + total_word];
				total_word++;
			}
			INT last_pos_head = train_position_head[max_pool_window_k[i] + j] * dimension_pos;
			INT last_pos_tail = train_position_tail[max_pool_window_k[i] + j] * dimension_pos;
			for (INT k = 0; k < dimension_pos; k++) {
				conv_1d_position_head[last_pos + total_pos] -= grad_word_pos * position_vec_head_copy[last_pos_head + k];
				conv_1d_position_tail[last_pos + total_pos] -= grad_word_pos * position_vec_tail_copy[last_pos_tail + k];
				position_vec_head[last_pos_head + k] -= grad_word_pos * conv_1d_position_head_copy[last_pos + total_pos];
				position_vec_tail[last_pos_tail + k] -= grad_word_pos * conv_1d_position_tail_copy[last_pos + total_pos];
				total_pos++;
			}
		}
		conv_1d_bias[i] -= grad_word_pos;
	}
}

REAL train_bags(std::string bags_name)
{
	INT relation = -1;
	INT bags_size = bags_train[bags_name].size();
	std::vector<std::vector<INT> > max_pool_window;
	max_pool_window.resize(bags_size);
	std::vector<std::vector<REAL> > conv_1d_result;
	
	for (INT k = 0; k < bags_size; k++)
	{
		max_pool_window[k].resize(dimension_c);
		INT pos = bags_train[bags_name][k];
		if (relation == -1)
			relation = train_relation_list[pos];
		else
			assert(relation == train_relation_list[pos]);
		conv_1d_result.push_back(calc_conv_1d(train_sentence_list[pos], train_position_head[pos],
			train_position_tail[pos], train_length[pos], max_pool_window[k]));
	}	
	
	std::vector<REAL> weight;
	REAL weight_sum = 0;
	for (INT k = 0; k < bags_size; k++)
	{
		REAL s = 0;
		for (INT i_r = 0; i_r < dimension_c; i_r++) 
		{
			REAL temp = 0;
			for (INT i_x = 0; i_x < dimension_c; i_x++)
				temp += conv_1d_result[k][i_x] * attention_weights_copy[relation][i_x][i_r];
			s += temp * relation_matrix_copy[relation * dimension_c + i_r];
		}
		s = exp(s); 
		weight.push_back(s);
		weight_sum += s;
	}

	for (INT k = 0; k < bags_size; k++)
		weight[k] /= weight_sum;
	
	std::vector<REAL> result_sentence;
	result_sentence.resize(dimension_c);
	for (INT i = 0; i < dimension_c; i++) 
		for (INT k = 0; k < bags_size; k++)
			result_sentence[i] += conv_1d_result[k][i] * weight[k];

	std::vector<REAL> result_final;

	std::vector<INT> dropout;
	for (INT i_s = 0; i_s < dimension_c; i_s++)
		dropout.push_back((double)(rand()) / RAND_MAX < dropout_probability);

	REAL sum = 0;
	for (INT i_r = 0; i_r < relation_total; i_r++) {
		REAL s = 0;
		for (INT i_s = 0; i_s < dimension_c; i_s++) {
			s += dropout[i_s] * result_sentence[i_s] * relation_matrix_copy[i_r * dimension_c + i_s];
		}
		s += relation_matrix_bias_copy[i_r];
		s = exp(s);
		sum += s;
		result_final.push_back(s);
	}
	
	double loss = -(log(result_final[relation]) - log(sum));
	
	std::vector<REAL> grad_s;
	grad_s.resize(dimension_c);

	for (INT i_r = 0; i_r < relation_total; i_r++)
	{	
		REAL grad_final = result_final[i_r] / sum * current_alpha;
		if (i_r == relation)
			grad_final -= current_alpha;

		for (INT i_s = 0; i_s < dimension_c; i_s++) 
		{
			REAL grad_i_s = 0;
			if (dropout[i_s] != 0)
			{
				grad_i_s += grad_final * relation_matrix_copy[i_r * dimension_c + i_s];
				relation_matrix[i_r * dimension_c + i_s] -= grad_final * result_sentence[i_s];
			}
			grad_s[i_s] += grad_i_s;
		}
		relation_matrix_bias[i_r] -= grad_final;
	}

	std::vector<std::vector<REAL> > grad_x;
	grad_x.resize(bags_size);

	for (INT k = 0; k < bags_size; k++)
		grad_x[k].resize(dimension_c);

	for (INT i_r = 0; i_r < dimension_c; i_r++) 
	{
		REAL grad_i_s = grad_s[i_r];
		double a_denominator_sum_exp = 0;

		for (INT k = 0; k < bags_size; k++)
		{
			grad_x[k][i_r] += grad_i_s * weight[k];
			for (INT i_x = 0; i_x < dimension_c; i_x++)
			{
				grad_x[k][i_x] += grad_i_s * conv_1d_result[k][i_r] * weight[k] *
					relation_matrix_copy[relation * dimension_c + i_r] *
					attention_weights_copy[relation][i_x][i_r];

				relation_matrix[relation * dimension_c + i_r] -= grad_i_s *
					conv_1d_result[k][i_r] * weight[k] * conv_1d_result[k][i_x] *
					attention_weights_copy[relation][i_x][i_r];

				if (i_r == i_x)
					attention_weights[relation][i_x][i_r] -= grad_i_s * conv_1d_result[k][i_r] *
						weight[k] * conv_1d_result[k][i_x] *
						relation_matrix_copy[relation * dimension_c + i_r];
			}
			a_denominator_sum_exp += conv_1d_result[k][i_r] * weight[k];
		}	
		for (INT k = 0; k < bags_size; k++)
		{
			for (INT i_x = 0; i_x < dimension_c; i_x++)
			{
				grad_x[k][i_x]-= grad_i_s * a_denominator_sum_exp * weight[k] *
					relation_matrix_copy[relation * dimension_c + i_r] *
					attention_weights_copy[relation][i_x][i_r];

				relation_matrix[relation * dimension_c + i_r] += grad_i_s *
					a_denominator_sum_exp * weight[k] * conv_1d_result[k][i_x] *
					attention_weights_copy[relation][i_x][i_r];

				if (i_r == i_x)
					attention_weights[relation][i_x][i_r] += grad_i_s * a_denominator_sum_exp *
						weight[k] * conv_1d_result[k][i_x] *
						relation_matrix_copy[relation * dimension_c + i_r];
			}
		}
	}

	for (INT k = 0; k < bags_size; k++)
	{
		INT pos = bags_train[bags_name][k];
		gradient_conv_1d(train_sentence_list[pos], train_position_head[pos], train_position_tail[pos],
			train_length[pos], conv_1d_result[k], max_pool_window[k], grad_x[k]);
	}
	return loss;
}

void* train_mode(void *id) {
	while (true)
	{
		pthread_mutex_lock (&train_mutex);
		if (current_sample >= final_sample)
		{
			pthread_mutex_unlock (&train_mutex);
			break;
		}
		current_sample += 1;
		pthread_mutex_unlock (&train_mutex);
		INT i = get_rand_i(0, len);
		total_loss += train_bags(bags_train_key[i]);
	}
}

void train() {

	bags_train_key.clear();
	for (std::map<std::string, std::vector<INT> >:: iterator it = bags_train.begin();
		it != bags_train.end(); it++)
	{
		bags_train_key.push_back(it->first);
	}

	position_vec_head = (REAL *)calloc(position_total_head * dimension_pos, sizeof(REAL));
	position_vec_tail = (REAL *)calloc(position_total_tail * dimension_pos, sizeof(REAL));
	conv_1d_word = (REAL*)calloc(dimension_c * window * dimension, sizeof(REAL));
	conv_1d_position_head = (REAL *)calloc(dimension_c * window * dimension_pos, sizeof(REAL));
	conv_1d_position_tail = (REAL *)calloc(dimension_c * window * dimension_pos, sizeof(REAL));
	conv_1d_bias = (REAL*)calloc(dimension_c, sizeof(REAL));
	attention_weights.resize(relation_total);
	for (INT i = 0; i < relation_total; i++)
	{
		attention_weights[i].resize(dimension_c);
		for (INT j = 0; j < dimension_c; j++)
		{
			attention_weights[i][j].resize(dimension_c);
			attention_weights[i][j][j] = 1.00;
		}
	}
	relation_matrix = (REAL *)calloc(relation_total * dimension_c, sizeof(REAL));
	relation_matrix_bias = (REAL *)calloc(relation_total, sizeof(REAL));

	word_vec_copy = (REAL *)calloc(dimension * word_total, sizeof(REAL));
	position_vec_head_copy = (REAL *)calloc(position_total_head * dimension_pos, sizeof(REAL));
	position_vec_tail_copy = (REAL *)calloc(position_total_tail * dimension_pos, sizeof(REAL));
	conv_1d_word_copy =  (REAL*)calloc(dimension_c * window * dimension, sizeof(REAL));
	conv_1d_position_head_copy = (REAL *)calloc(dimension_c * window * dimension_pos, sizeof(REAL));
	conv_1d_position_tail_copy = (REAL *)calloc(dimension_c * window * dimension_pos, sizeof(REAL));
	conv_1d_bias_copy =  (REAL*)calloc(dimension_c, sizeof(REAL));
	attention_weights_copy = attention_weights;
	relation_matrix_copy = (REAL *)calloc(relation_total * dimension_c, sizeof(REAL));
	relation_matrix_bias_copy = (REAL *)calloc(relation_total, sizeof(REAL));

	REAL relation_matrix_init = sqrt(6.0 / (relation_total + dimension_c));
	REAL conv_1d_position_vec_init = sqrt(6.0 / ((dimension + dimension_pos) * window));

	for (INT i = 0; i < position_total_head; i++) {
		for (INT j = 0; j < dimension_pos; j++) {
			position_vec_head[i * dimension_pos + j] = get_rand_u(-conv_1d_position_vec_init,
				conv_1d_position_vec_init);
		}
	}
	for (INT i = 0; i < position_total_tail; i++) {
		for (INT j = 0; j < dimension_pos; j++) {
			position_vec_tail[i * dimension_pos + j] = get_rand_u(-conv_1d_position_vec_init,
				conv_1d_position_vec_init);
		}
	}
	for (INT i = 0; i < dimension_c; i++) {
		INT last = i * window * dimension;
		for (INT j = 0; j < window * dimension; j++)
			conv_1d_word[last + j] = get_rand_u(-conv_1d_position_vec_init, conv_1d_position_vec_init);
		last = i * window * dimension_pos;
		for (INT j = dimension_pos * window - 1; j >=0; j--) {
			conv_1d_position_head[last + j] = get_rand_u(-conv_1d_position_vec_init, conv_1d_position_vec_init);
			conv_1d_position_tail[last + j] = get_rand_u(-conv_1d_position_vec_init, conv_1d_position_vec_init);
		}
		conv_1d_bias[i] = get_rand_u(-conv_1d_position_vec_init, conv_1d_position_vec_init);
	}
	for (INT i = 0; i < relation_total; i++) 
	{
		for (INT j = 0; j < dimension_c; j++)
			relation_matrix[i * dimension_c + j] = get_rand_u(-relation_matrix_init, relation_matrix_init);
		relation_matrix_bias[i] = get_rand_u(-relation_matrix_init, relation_matrix_init);
	}

	printf("##################################################\n\nTrain start...\n\n");

	for (INT epoch = 1; epoch <= epochs; epoch++) {
		
		len = bags_train.size();
		nbatches  =  len / (batch * num_threads);
		current_alpha = alpha * current_rate;

		current_sample = 0;
		final_sample = 0;
		total_loss = 0;

		gettimeofday(&t_start, NULL);

		for (INT i = 1; i <= nbatches; i++) {
			final_sample += batch * num_threads;
			
			memcpy(word_vec_copy, word_vec, word_total * dimension * sizeof(REAL));
			memcpy(position_vec_head_copy, position_vec_head, position_total_head * dimension_pos * sizeof(REAL));
			memcpy(position_vec_tail_copy, position_vec_tail, position_total_tail * dimension_pos * sizeof(REAL));
			memcpy(conv_1d_word_copy, conv_1d_word, dimension_c * window * dimension * sizeof(REAL));
			memcpy(conv_1d_position_head_copy, conv_1d_position_head, dimension_c * window * dimension_pos * sizeof(REAL));
			memcpy(conv_1d_position_tail_copy, conv_1d_position_tail, dimension_c * window * dimension_pos * sizeof(REAL));
			memcpy(conv_1d_bias_copy, conv_1d_bias, dimension_c * sizeof(REAL));
			attention_weights_copy = attention_weights;
			memcpy(relation_matrix_copy, relation_matrix, relation_total * dimension_c * sizeof(REAL));
			memcpy(relation_matrix_bias_copy, relation_matrix_bias, relation_total * sizeof(REAL));
			
			pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
			for (long a = 0; a < num_threads; a++)
				pthread_create(&pt[a], NULL, train_mode,  (void *)a);
			for (long a = 0; a < num_threads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}

		gettimeofday(&t_end, NULL);
		long double time_use = 1000000 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec - t_start.tv_usec;

		printf("Epoch %d/%d - current_alpha: %.8f - loss: %f - %.2Lfs\n\n", epoch, epochs,
			current_alpha, total_loss / final_sample, time_use / 1000000.0);
		test();

		current_rate = current_rate * reduce_epoch;
	}
	printf("Train end.\n\n##################################################\n\n");
}

INT main(INT argc, char ** argv) {
	//output_model = 1;
	init();
	train();
}
