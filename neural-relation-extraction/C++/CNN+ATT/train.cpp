#include<assert.h>
#include<sys/time.h>

#include "init.h"
#include "test.h"

using namespace std;

// bags_test_key: 保存 bags_train 的 key (头实体 + "\t" + 尾实体 + "\t" + 关系名), 按照 bags_train 的迭代顺序
// total_loss: 每一轮次的总损失
// current_alpha: 当前轮次的学习率
// current_sample, final_sample: 由于使用多线程训练模型, 这两个变量用于确定当前训练批次是否完成, 进而更新各种权重矩阵的副本, 如 word_vec_copy
// train_mutex: 互斥锁
std::vector<string> bags_train_key;
double total_loss = 0;
REAL current_alpha;
double current_sample = 0, final_sample = 0;
pthread_mutex_t train_mutex;

struct timeval t_start, t_end;

void time_begin()
{
	gettimeofday(&t_start, NULL);
}
void time_end()
{
	gettimeofday(&t_end, NULL);
	long double time_use = 1000000 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec - t_start.tv_usec;
	printf("time(s): %.2Lf.\n", time_use/1000000.0);
}

vector<REAL> train(INT *sentence, INT *train_position_head, INT *train_position_tail, INT len, vector<INT> &tip) {
	vector<REAL> r;
	r.resize(dimension_c);
	for (INT i = 0; i < dimension_c; i++) {
		r[i] = 0;
		INT last = i * dimension * window;
		INT lastt = i * dimension_pos * window;
		REAL max_pool_1d = -FLT_MAX;
		for (INT i1 = 0; i1 <= len - window; i1++) {
			REAL res = 0;
			INT tot = 0;
			INT tot1 = 0;
			for (INT j = i1; j < i1 + window; j++)  {
				INT last1 = sentence[j] * dimension;
			 	for (INT k = 0; k < dimension; k++) {
			 		res += conv_1d_word_copy[last + tot] * word_vec_copy[last1+k];
			 		tot++;
			 	}
			 	INT last2 = train_position_head[j] * dimension_pos;
			 	INT last3 = train_position_tail[j] * dimension_pos;
			 	for (INT k = 0; k < dimension_pos; k++) {
			 		res += conv_1d_position_head_copy[lastt + tot1] * position_vec_head_copy[last2+k];
			 		res += conv_1d_position_tail_copy[lastt + tot1] * position_vec_tail_copy[last3+k];
			 		tot1++;
			 	}
			}
			if (res > max_pool_1d) {
				max_pool_1d = res;
				tip[i] = i1;
			}
		}
		r[i] = max_pool_1d + conv_1d_bias_copy[i];
	}

	for (INT i = 0; i < dimension_c; i++) {
		r[i] = calc_tanh(r[i]);
	}
	return r;
}

void train_gradient(INT *sentence, INT *train_position_head, INT *train_position_tail, INT len, INT e1, INT e2, INT r1, REAL alpha, vector<REAL> &r,vector<INT> &tip, vector<REAL> &grad)
{
	for (INT i = 0; i < dimension_c; i++) {
		if (fabs(grad[i])<1e-8)
			continue;
		INT last = i * dimension * window;
		INT tot = 0;
		INT lastt = i * dimension_pos * window;
		INT tot1 = 0;
		REAL g1 = grad[i] * (1 -  r[i] * r[i]);
		for (INT j = 0; j < window; j++)  {
			INT last1 = sentence[tip[i] + j] * dimension;
			for (INT k = 0; k < dimension; k++) {
				conv_1d_word[last + tot] -= g1 * word_vec_copy[last1+k];
				word_vec[last1 + k] -= g1 * conv_1d_word_copy[last + tot];
				tot++;
			}
			INT last2 = train_position_head[tip[i] + j] * dimension_pos;
			INT last3 = train_position_tail[tip[i] + j] * dimension_pos;
			for (INT k = 0; k < dimension_pos; k++) {
				conv_1d_position_head[lastt + tot1] -= g1 * position_vec_head_copy[last2 + k];
				conv_1d_position_tail[lastt + tot1] -= g1 * position_vec_tail_copy[last3 + k];
				position_vec_head[last2 + k] -= g1 * conv_1d_position_head_copy[lastt + tot1];
				position_vec_tail[last3 + k] -= g1 * conv_1d_position_tail_copy[lastt + tot1];
				tot1++;
			}
		}
		conv_1d_bias[i] -= g1;
	}
}

REAL train_bags(std::string bags_name)
{
	INT bags_size = bags_train[bags_name].size();
	double bags_rate = max(1.0,1.0*bags_size/2);
	vector<vector<REAL> > rList;
	vector<vector<INT> > tipList;
	tipList.resize(bags_size);
	INT r1 = -1;
	for (INT k=0; k<bags_size; k++)
	{
		tipList[k].resize(dimension_c);
		INT i = bags_train[bags_name][k];
		if (r1==-1)
			r1 = train_relation_list[i];
		else
			assert(r1==train_relation_list[i]);
		rList.push_back(train(train_sentence_list[i], train_position_head[i], train_position_tail[i], train_length[i], tipList[k]));
	}
	
	vector<REAL> f_r;	
	
	vector<INT> dropout;
	for (INT i = 0; i < dimension_c; i++)
		dropout.push_back((double)(rand()) / RAND_MAX < dropout_probability);
		
	
	std::vector<REAL> weight;
	REAL weight_sum = 0;
	for (INT k=0; k<bags_size; k++)
	{
		REAL s = 0;
		for (INT i = 0; i < dimension_c; i++) 
		{
			REAL tmp = 0;
			for (INT j = 0; j < dimension_c; j++)
				tmp+=rList[k][j]*attention_weights_copy[r1][j][i];
			s += tmp * relation_matrix_copy[r1 * dimension_c + i];
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
		r.resize(dimension_c);
		for (INT i = 0; i < dimension_c; i++) 
			for (INT k=0; k<bags_size; k++)
				r[i] += rList[k][i] * weight[k];
	
		REAL ss = 0;
		for (INT i = 0; i < dimension_c; i++) {
			ss += dropout[i] * r[i] * relation_matrix_copy[j * dimension_c + i];
		}
		ss += relation_matrix_bias_copy[j];
		f_r.push_back(exp(ss));
		sum+=f_r[j];
	}
	
	double loss = -(log(f_r[r1]) - log(sum));
	
	vector<vector<REAL> > grad;
	grad.resize(bags_size);
	for (INT k=0; k<bags_size; k++)
		grad[k].resize(dimension_c);
	vector<REAL> g1_tmp;
	g1_tmp.resize(dimension_c);
	for (INT r2 = 0; r2<relation_total; r2++)
	{	
		vector<REAL> r;
		r.resize(dimension_c);
		for (INT i = 0; i < dimension_c; i++) 
			for (INT k=0; k<bags_size; k++)
				r[i] += rList[k][i] * weight[k];
		
		REAL g = f_r[r2]/sum*current_alpha;
		if (r2 == r1)
			g -= current_alpha;
		for (INT i = 0; i < dimension_c; i++) 
		{
			REAL g1 = 0;
			if (dropout[i]!=0)
			{
				g1 += g * relation_matrix_copy[r2 * dimension_c + i];
				relation_matrix[r2 * dimension_c + i] -= g * r[i];
			}
			g1_tmp[i]+=g1;
		}
		relation_matrix_bias[r2] -= g;
	}
	for (INT i = 0; i < dimension_c; i++) 
	{
		REAL g1 = g1_tmp[i];
		double tmp_sum = 0; //for rList[k][i]*weight[k]
		for (INT k=0; k<bags_size; k++)
		{
			grad[k][i]+=g1*weight[k];
			for (INT j = 0; j < dimension_c; j++)
			{
				grad[k][j]+=g1*rList[k][i]*weight[k]*relation_matrix_copy[r1 * dimension_c + i]*attention_weights_copy[r1][j][i];
				relation_matrix[r1 * dimension_c + i] -= g1*rList[k][i]*weight[k]*rList[k][j]*attention_weights_copy[r1][j][i];
				if (i==j)
				  attention_weights[r1][j][i] -= g1*rList[k][i]*weight[k]*rList[k][j]*relation_matrix_copy[r1 * dimension_c + i];
			}
			tmp_sum += rList[k][i]*weight[k];
		}	
		for (INT k1=0; k1<bags_size; k1++)
		{
			for (INT j = 0; j < dimension_c; j++)
			{
				grad[k1][j]-=g1*tmp_sum*weight[k1]*relation_matrix_copy[r1 * dimension_c + i]*attention_weights_copy[r1][j][i];
				relation_matrix[r1 * dimension_c + i] += g1*tmp_sum*weight[k1]*rList[k1][j]*attention_weights_copy[r1][j][i];
				if (i==j)
				  attention_weights[r1][j][i] += g1*tmp_sum*weight[k1]*rList[k1][j]*relation_matrix_copy[r1 * dimension_c + i];
			}
		}
	}
	for (INT k=0; k<bags_size; k++)
	{
		INT i = bags_train[bags_name][k];
		train_gradient(train_sentence_list[i], train_position_head[i], train_position_tail[i], train_length[i], train_head_list[i], train_tail_list[i], train_relation_list[i], current_alpha,rList[k], tipList[k], grad[k]);
		
	}
	return loss;
}

void* train_mode(void *id ) {
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
	

	for (INT epoch = 0; epoch < epochs; epoch ++) {
		
		len = bags_train.size();
		nbatches  =  len / (batch * num_threads);
		current_alpha = alpha * current_rate;

		current_sample = 0;
		final_sample = 0;
		total_loss = 0;

		time_begin();

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

		time_end();
		printf("Epoch %d/%d - loss: %f\tcurrent_alpha: %.8f\ntest:\n", epoch, epochs, total_loss/final_sample, current_alpha);
		test();

		if ((epoch + 1) % 1 == 0) 
			current_rate = current_rate * reduce_epoch;
	}
	printf("Train End\n");
}

INT main(INT argc, char ** argv) {
	//output_model = 1;
	init();
	train();
}
