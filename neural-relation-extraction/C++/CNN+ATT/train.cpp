// train.cpp
//
// 使用方法:
//     编译:
//           $ g++ train.cpp -o ./build/train -pthread -O3 -march=native
//     运行:
//           $ ./build/train
//
// created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
//
// 该 C++ 文件用于模型训练

// ##################################################
// 包含标准库和头文件
// ##################################################

#include "init.h"
#include "test.h"

// bags_test_key: 保存 bags_train 的 key (头实体 + "\t" + 尾实体 + "\t" + 关系名), 按照 bags_train 的迭代顺序
// total_loss: 每一轮次的总损失
// current_alpha: 当前轮次的学习率
// current_sample, final_sample: 由于使用多线程训练模型, 这两个变量用于确定当前训练批次是否完成, 进而更新各种权重矩阵的副本, 如 word_vec_copy
// train_mutex: 互斥锁, 线程同步 current_sample 变量
// len = bags_train.size()
// nbatches  =  len / (batch * num_threads)
std::vector<std::string> bags_train_key;
double total_loss = 0;
REAL current_alpha;
double current_sample = 0, final_sample = 0;
pthread_mutex_t train_mutex;
INT len;
INT nbatches;

struct timeval train_start, train_end;

// 计算句子的一维卷积
std::vector<REAL> calc_conv_1d(INT *sentence, INT *train_position_head,
	INT *train_position_tail, INT sentence_length, std::vector<INT> &max_pool_window_k) {
	
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
			 		sum += conv_1d_word_copy[last_word + total_word] * word_vec_copy[last_word_vec + k];
			 		total_word++;
			 	}
			 	INT last_pos_head = train_position_head[j] * dimension_pos;
			 	INT last_pos_tail = train_position_tail[j] * dimension_pos;
			 	for (INT k = 0; k < dimension_pos; k++) {
			 		sum += conv_1d_position_head_copy[last_pos + total_pos] * position_vec_head_copy[last_pos_head + k];
			 		sum += conv_1d_position_tail_copy[last_pos + total_pos] * position_vec_tail_copy[last_pos_tail + k];
			 		total_pos++;
			 	}
			}

			// 对应于论文中的公式 (3), [x]_i = max(p_i), 其中 x \in R^{d^c}
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

// 根据梯度更新一维卷积的权重矩阵, 位置嵌入矩阵, 词嵌入矩阵
void gradient_conv_1d(INT *sentence, INT *train_position_head, INT *train_position_tail,
	std::vector<REAL> &conv_1d_result_k, std::vector<INT> &max_pool_window_k, std::vector<REAL> &grad_x_k)
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

// 训练一个样本
REAL train_bags(std::string bags_name)
{
	// ##################################################
	// 正向传播
	// ##################################################

	// 一维卷积部分
	// relation: 该训练样本的正确标签 (关系)
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
				temp += conv_1d_result[k][i_x] * attention_weights_copy[relation][i_x][i_r];
			s += temp * relation_matrix_copy[relation * dimension_c + i_r];
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
	
	// 计算各种关系成立的概率
	std::vector<REAL> result_final;
	std::vector<INT> dropout;
	for (INT i_s = 0; i_s < dimension_c; i_s++)
		dropout.push_back((REAL)(rand()) / (RAND_MAX + 1.0) < dropout_probability);

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

	// 计算损失值
	double loss = -(log(result_final[relation]) - log(sum));

	// ##################################################
	// 反向传播
	// ##################################################
	
	// 更新 relation_matrix, 对应于论文中的公式 (12), o = M(s \circ h) + d
	std::vector<REAL> grad_s;
	grad_s.resize(dimension_c);

	for (INT i_r = 0; i_r < relation_total; i_r++)
	{
		// 由于损失函数是 cross-entropy, 负标签是 0
		// 对于负标签 (关系) 的梯度是计算的概率, 即 result_final[i_r] / sum
		// 这样做, 能省略一层 softmax
		REAL grad_final = result_final[i_r] / sum * current_alpha;
		
		// 正标签是 0, 对于正标签 (关系) 的梯度是计算的概率 - 1, 即 result_final[i_r] / sum - 1
		// 这样做, 能省略一层 softmax
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

	// 更新注意力权重矩阵和 relation_matrix, 对应于论文中的公式 (5), (7), (8)
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
			// grad_i_s * weight[k] 对应于论文中的公式 5
			grad_x[k][i_r] += grad_i_s * weight[k];
			for (INT i_x = 0; i_x < dimension_c; i_x++)
			{
				// 对应于论文中的公式 7 中分子 (exp(e_i)) 的公式 8 中的 x_i
				grad_x[k][i_x] += grad_i_s * conv_1d_result[k][i_r] * weight[k] *
					relation_matrix_copy[relation * dimension_c + i_r] *
					attention_weights_copy[relation][i_x][i_r];

				// 对应于论文中的公式 7 中分子 (exp(e_i)) 的公式 8 中的 r
				relation_matrix[relation * dimension_c + i_r] -= grad_i_s *
					conv_1d_result[k][i_r] * weight[k] * conv_1d_result[k][i_x] *
					attention_weights_copy[relation][i_x][i_r];
				
				// 对应于论文中的公式 7 中分子 (exp(e_i)) 的公式 8 中的 A
				if (i_r == i_x)
					attention_weights[relation][i_x][i_r] -= grad_i_s * conv_1d_result[k][i_r] *
						weight[k] * conv_1d_result[k][i_x] *
						relation_matrix_copy[relation * dimension_c + i_r];
			}

			// 由于 1/x 的导数是 -1/x^2, exp(x) 的导数是 exp(x)
			// 所以论文中的公式 (7) 中分母 (exp(e_i)) 的公式 8 的求导需要一个和 (exp(x_1), exp(x_2) ,...)
			// 并且需要多乘一次 weight[k]
			a_denominator_sum_exp += conv_1d_result[k][i_r] * weight[k];
		}	
		for (INT k = 0; k < bags_size; k++)
		{
			for (INT i_x = 0; i_x < dimension_c; i_x++)
			{
				// 对应于论文中的公式 7 中分母 (exp(e_i)) 的公式 8 中的 x_i
				grad_x[k][i_x]-= grad_i_s * a_denominator_sum_exp * weight[k] *
					relation_matrix_copy[relation * dimension_c + i_r] *
					attention_weights_copy[relation][i_x][i_r];
				
				// 对应于论文中的公式 7 中分母 (exp(e_i)) 的公式 8 中的 r
				relation_matrix[relation * dimension_c + i_r] += grad_i_s *
					a_denominator_sum_exp * weight[k] * conv_1d_result[k][i_x] *
					attention_weights_copy[relation][i_x][i_r];
				
				// 对应于论文中的公式 7 中分母 (exp(e_i)) 的公式 8 中的 A
				if (i_r == i_x)
					attention_weights[relation][i_x][i_r] += grad_i_s * a_denominator_sum_exp *
						weight[k] * conv_1d_result[k][i_x] *
						relation_matrix_copy[relation * dimension_c + i_r];
			}
		}
	}

	// 根据梯度更新一维卷积的权重矩阵, 位置嵌入矩阵, 词嵌入矩阵
	for (INT k = 0; k < bags_size; k++)
	{
		INT pos = bags_train[bags_name][k];
		gradient_conv_1d(train_sentence_list[pos], train_position_head[pos], train_position_tail[pos],
			conv_1d_result[k], max_pool_window[k], grad_x[k]);
	}
	return loss;
}

// 单个线程内运行的任务
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

// 训练函数
void train() {

	len = bags_train.size();
	nbatches  =  len / (batch * num_threads);

	bags_train_key.clear();
	for (std::map<std::string, std::vector<INT> >:: iterator it = bags_train.begin();
		it != bags_train.end(); it++)
	{
		bags_train_key.push_back(it->first);
	}

	// 为模型的权重矩阵分配内存空间
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

	// 为模型的权重矩阵的副本分配内存空间
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

	// 模型的权重矩阵初始化
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

		// 更新当前 epoch 的学习率		
		current_alpha = alpha * current_rate;

		current_sample = 0;
		final_sample = 0;
		total_loss = 0;

		gettimeofday(&train_start, NULL);

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

		gettimeofday(&train_end, NULL);
		long double time_use = (1000000 * (train_end.tv_sec - train_start.tv_sec)
			+ train_end.tv_usec - train_start.tv_usec) / 1000000.0;

		printf("Epoch %d/%d - current_alpha: %.8f - loss: %f - %02d:%02d:%02d\n\n", epoch, epochs,
			current_alpha, total_loss / final_sample, INT(time_use / 3600.0),
			INT(time_use) % 3600 / 60, INT(time_use) % 60);
		test();
		printf("Test end.\n\n##################################################\n\n");

		current_rate = current_rate * reduce_epoch;
	}
	printf("Train end.\n\n##################################################\n\n");
}

// ##################################################
// Main function
// ##################################################

void setparameters(INT argc, char **argv) {
	INT i;
	if ((i = arg_pos((char *)"-batch", argc, argv)) > 0) batch = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = arg_pos((char *)"-init_rate", argc, argv)) > 0) current_rate = atof(argv[i + 1]);
	if ((i = arg_pos((char *)"-reduce_epoch", argc, argv)) > 0) reduce_epoch = atof(argv[i + 1]);
	if ((i = arg_pos((char *)"-epochs", argc, argv)) > 0) epochs = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-limit", argc, argv)) > 0) limit = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-dimension_pos", argc, argv)) > 0) dimension_pos = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-dimension_c", argc, argv)) > 0) dimension_c = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-dropout", argc, argv)) > 0) dropout_probability = atof(argv[i + 1]);
	if ((i = arg_pos((char *)"-output_model", argc, argv)) > 0) output_model = atoi(argv[i + 1]);	
	if ((i = arg_pos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
	if ((i = arg_pos((char *)"-data_path", argc, argv)) > 0) data_path = argv[i + 1];
	if ((i = arg_pos((char *)"-output_path", argc, argv)) > 0) output_path = argv[i + 1];
}

void print_train_help() {
	std::string str = R"(
// ##################################################
// ./train [-batch BATCH] [-threads THREAD] [-alpha ALPHA]
//         [-init_rate INIT_RATE] [-reduce_epoch REDUCE_EPOCH]
//         [-epochs EPOCHS] [-limit LIMIT] [-dimension_pos DIMENSION_POS]
//         [-window WINDOW] [-dimension_c DIMENSION_C]
//         [-dropout DROPOUT] [-output_model 0/1]
//         [-note NOTE] [-data_path DATA_PATH]
//         [-output_path OUTPUT_PATH] [--help]

// optional arguments:
// -batch BATCH                   batch size. if unspecified, batch will default to [40]
// -threads THREAD                number of worker threads. if unspecified, num_threads will default to [32]
// -alpha ALPHA                   learning rate. if unspecified, alpha will default to [0.00125]
// -init_rate INIT_RATE           init rate of learning rate. if unspecified, current_rate will default to [1.0]
// -reduce_epoch REDUCE_EPOCH     reduce of init rate of learning rate per epoch. if unspecified, reduce_epoch will default to [0.98]
// -epochs EPOCHS                 number of epochs. if unspecified, epochs will default to [25]
// -limit LIMIT                   限制句子中 (头, 尾) 实体相对每个单词的最大距离. 默认值为 [30]
// -dimension_pos DIMENSION_POS   位置嵌入维度，默认值为 [5]
// -window WINDOW                 一维卷积的 window 大小. 默认值为 [3]
// -dimension_c DIMENSION_C       sentence embedding size, if unspecified, dimension_c will default to [230]
// -dropout DROPOUT               dropout probability. if unspecified, dropout_probability will default to [0.5]
// -output_model 0/1              [1] 保存模型, [0] 不保存模型. 默认值为 [1]
// -note NOTE                     information you want to add to the filename, like ("./output/word2vec" + note + ".txt"). if unspecified, note will default to ""
// -data_path DATA_PATH           folder of data. if unspecified, data_path will default to "../data/"
// -output_path OUTPUT_PATH       folder of outputing results (precion/recall curves) and models. if unspecified, output_path will default to "./output/"
// --help                         print help information of ./train
// ##################################################
)";

	printf("%s\n", str.c_str());
}

// ##################################################
// ./train [-batch BATCH] [-threads THREAD] [-alpha ALPHA]
//         [-init_rate INIT_RATE] [-reduce_epoch REDUCE_EPOCH]
//         [-epochs EPOCHS] [-limit LIMIT] [-dimension_pos DIMENSION_POS]
//         [-window WINDOW] [-dimension_c DIMENSION_C]
//         [-dropout DROPOUT] [-output_model 0/1]
//         [-note NOTE] [-data_path DATA_PATH]
//         [-output_path OUTPUT_PATH] [--help]

// optional arguments:
// -batch BATCH                   batch size. if unspecified, batch will default to [40]
// -threads THREAD                number of worker threads. if unspecified, num_threads will default to [32]
// -alpha ALPHA                   learning rate. if unspecified, alpha will default to [0.00125]
// -init_rate INIT_RATE           init rate of learning rate. if unspecified, current_rate will default to [1.0]
// -reduce_epoch REDUCE_EPOCH     reduce of init rate of learning rate per epoch. if unspecified, reduce_epoch will default to [0.98]
// -epochs EPOCHS                 number of epochs. if unspecified, epochs will default to [25]
// -limit LIMIT                   限制句子中 (头, 尾) 实体相对每个单词的最大距离. 默认值为 [30]
// -dimension_pos DIMENSION_POS   位置嵌入维度，默认值为 [5]
// -window WINDOW                 一维卷积的 window 大小. 默认值为 [3]
// -dimension_c DIMENSION_C       sentence embedding size, if unspecified, dimension_c will default to [230]
// -dropout DROPOUT               dropout probability. if unspecified, dropout_probability will default to [0.5]
// -output_model 0/1              [1] 保存模型, [0] 不保存模型. 默认值为 [1]
// -note NOTE                     information you want to add to the filename, like ("./output/word2vec" + note + ".txt"). if unspecified, note will default to ""
// -data_path DATA_PATH           folder of data. if unspecified, data_path will default to "../data/"
// -output_path OUTPUT_PATH       folder of outputing results (precion/recall curves) and models. if unspecified, output_path will default to "./output/"
// --help                         print help information of ./train
// ##################################################

INT main(INT argc, char **argv) {
	for (INT a = 1; a < argc; a++) if (!strcmp((char *)"--help", argv[a])) {
		print_train_help();
		return 0;
	}
	output_model = 1;
	setparameters(argc, argv);
	init();
	print_information();
	train();
	return 0;
}
