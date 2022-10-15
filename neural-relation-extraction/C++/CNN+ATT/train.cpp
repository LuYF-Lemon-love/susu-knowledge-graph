#include<assert.h>
#include<sys/time.h>

#include "init.h"
#include "test.h"

using namespace std;

double score = 0;
REAL alpha1;

INT turn;

INT test_tmp = 0;

std::vector<string> b_train;
// std::vector<INT> c_train;
double score_tmp = 0, score_max = 0;
pthread_mutex_t mutex1;
INT tot_batch;

struct timeval t_start, t_end;

void time_begin()
{
	gettimeofday(&t_start, NULL);
}
void time_end()
{
	gettimeofday(&t_end, NULL);
	long double time_use = 1000000 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec - t_start.tv_usec;
	std::cout << "time(s):\t" << time_use/1000000.0 << std::endl;
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
			 		res += matrixW1Dao[last + tot] * wordVecDao[last1+k];
			 		tot++;
			 	}
			 	INT last2 = train_position_head[j] * dimension_pos;
			 	INT last3 = train_position_tail[j] * dimension_pos;
			 	for (INT k = 0; k < dimension_pos; k++) {
			 		res += matrixW1PositionE1Dao[lastt + tot1] * positionVecDaoE1[last2+k];
			 		res += matrixW1PositionE2Dao[lastt + tot1] * positionVecDaoE2[last3+k];
			 		tot1++;
			 	}
			}
			if (res > max_pool_1d) {
				max_pool_1d = res;
				tip[i] = i1;
			}
		}
		r[i] = max_pool_1d + matrixB1Dao[i];
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
				conv_1d_word[last + tot] -= g1 * wordVecDao[last1+k];
				word_vec[last1 + k] -= g1 * matrixW1Dao[last + tot];
				tot++;
			}
			INT last2 = train_position_head[tip[i] + j] * dimension_pos;
			INT last3 = train_position_tail[tip[i] + j] * dimension_pos;
			for (INT k = 0; k < dimension_pos; k++) {
				conv_1d_position_head[lastt + tot1] -= g1 * positionVecDaoE1[last2 + k];
				conv_1d_position_tail[lastt + tot1] -= g1 * positionVecDaoE2[last3 + k];
				position_vec_head[last2 + k] -= g1 * matrixW1PositionE1Dao[lastt + tot1];
				position_vec_tail[last3 + k] -= g1 * matrixW1PositionE2Dao[lastt + tot1];
				tot1++;
			}
		}
		conv_1d_bias[i] -= g1;
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
		//dropout.push_back(1);
		dropout.push_back(rand()%2);
	
	vector<REAL> weight;
	REAL weight_sum = 0;
	for (INT k=0; k<bags_size; k++)
	{
		REAL s = 0;
		for (INT i = 0; i < dimension_c; i++) 
		{
			REAL tmp = 0;
			for (INT j = 0; j < dimension_c; j++)
				tmp+=rList[k][j]*att_W_Dao[r1][j][i];
			s += tmp * matrixRelationDao[r1 * dimension_c + i];
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
			ss += dropout[i] * r[i] * matrixRelationDao[j * dimension_c + i];
		}
		ss += matrixRelationPrDao[j];
		f_r.push_back(exp(ss));
		sum+=f_r[j];
	}
	
	double rt = (log(f_r[r1]) - log(sum));
	
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
		
		REAL g = f_r[r2]/sum*alpha1;
		if (r2 == r1)
			g -= alpha1;
		for (INT i = 0; i < dimension_c; i++) 
		{
			REAL g1 = 0;
			if (dropout[i]!=0)
			{
				g1 += g * matrixRelationDao[r2 * dimension_c + i];
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
					grad[k][j]+=g1*rList[k][i]*weight[k]*matrixRelationDao[r1 * dimension_c + i]*att_W_Dao[r1][j][i];
					relation_matrix[r1 * dimension_c + i] -= g1*rList[k][i]*weight[k]*rList[k][j]*att_W_Dao[r1][j][i];
					if (i==j)
					  attention_weights[r1][j][i] -= g1*rList[k][i]*weight[k]*rList[k][j]*matrixRelationDao[r1 * dimension_c + i];
				}
				tmp_sum += rList[k][i]*weight[k];
			}	
			for (INT k1=0; k1<bags_size; k1++)
			{
				for (INT j = 0; j < dimension_c; j++)
				{
					grad[k1][j]-=g1*tmp_sum*weight[k1]*matrixRelationDao[r1 * dimension_c + i]*att_W_Dao[r1][j][i];
					relation_matrix[r1 * dimension_c + i] += g1*tmp_sum*weight[k1]*rList[k1][j]*att_W_Dao[r1][j][i];
					if (i==j)
					  attention_weights[r1][j][i] += g1*tmp_sum*weight[k1]*rList[k1][j]*matrixRelationDao[r1 * dimension_c + i];
				}
			}
		}
	for (INT k=0; k<bags_size; k++)
	{
		INT i = bags_train[bags_name][k];
		train_gradient(train_sentence_list[i], train_position_head[i], train_position_tail[i], train_length[i], train_head_list[i], train_tail_list[i], train_relation_list[i], alpha1,rList[k], tipList[k], grad[k]);
		
	}
	return rt;
}

void* trainMode(void *id ) {
		unsigned long long next_random = (long long)id;
		test_tmp = 0;
		while (true)
		{

			pthread_mutex_lock (&mutex1);
			if (score_tmp>=score_max)
			{
				pthread_mutex_unlock (&mutex1);
				break;
			}
			score_tmp+=1;
			pthread_mutex_unlock (&mutex1);
			// INT j = get_rand(0, c_train.size());
			// j = c_train[j];
			INT j = get_rand(0, len);
			score += train_bags(b_train[j]);
		}
}

void train() {

	b_train.clear();
	// c_train.clear();
	for (std::map<std::string, std::vector<INT> >:: iterator it = bags_train.begin();
		it != bags_train.end(); it++)
	{
		// INT max_size = 1;
		// for (INT i=0; i<max(1,max_size); i++)
			// c_train.push_back(b_train.size());
		b_train.push_back(it->first);
	}
	// std::cout<<c_train.size()<<std::endl;
	
	attention_weights.resize(relation_total);
	for (INT i=0; i<relation_total; i++)
	{
		attention_weights[i].resize(dimension_c);
		for (INT j=0; j<dimension_c; j++)
		{
			attention_weights[i][j].resize(dimension_c);
			attention_weights[i][j][j] = 1.00;
		}
	}
	att_W_Dao = attention_weights;

	REAL con = sqrt(6.0/(dimension_c+relation_total));
	REAL con1 = sqrt(6.0/((dimension_pos+dimension)*window));
	relation_matrix = (REAL *)calloc(dimension_c * relation_total, sizeof(REAL));
	relation_matrix_bias = (REAL *)calloc(relation_total, sizeof(REAL));
	matrixRelationPrDao = (REAL *)calloc(relation_total, sizeof(REAL));
	wordVecDao = (REAL *)calloc(dimension * word_total, sizeof(REAL));
	position_vec_head = (REAL *)calloc(position_total_head * dimension_pos, sizeof(REAL));
	position_vec_tail = (REAL *)calloc(position_total_tail * dimension_pos, sizeof(REAL));
	
	conv_1d_word = (REAL*)calloc(dimension_c * dimension * window, sizeof(REAL));
	conv_1d_position_head = (REAL *)calloc(dimension_c * dimension_pos * window, sizeof(REAL));
	conv_1d_position_tail = (REAL *)calloc(dimension_c * dimension_pos * window, sizeof(REAL));
	conv_1d_bias = (REAL*)calloc(dimension_c, sizeof(REAL));

	for (INT i = 0; i < dimension_c; i++) {
		INT last = i * window * dimension;
		for (INT j = dimension * window - 1; j >=0; j--)
			conv_1d_word[last + j] = get_rand_u(-con1, con1);
		last = i * window * dimension_pos;
		REAL tmp1 = 0;
		REAL tmp2 = 0;
		for (INT j = dimension_pos * window - 1; j >=0; j--) {
			conv_1d_position_head[last + j] = get_rand_u(-con1, con1);
			tmp1 += conv_1d_position_head[last + j]  * conv_1d_position_head[last + j] ;
			conv_1d_position_tail[last + j] = get_rand_u(-con1, con1);
			tmp2 += conv_1d_position_tail[last + j]  * conv_1d_position_tail[last + j] ;
		}
		conv_1d_bias[i] = get_rand_u(-con1, con1);
	}

	for (INT i = 0; i < relation_total; i++) 
	{
		relation_matrix_bias[i] = get_rand_u(-con, con);				//add
		for (INT j = 0; j < dimension_c; j++)
			relation_matrix[i * dimension_c + j] = get_rand_u(-con, con);
	}

	for (INT i = 0; i < position_total_head; i++) {
		for (INT j = 0; j < dimension_pos; j++) {
			position_vec_head[i * dimension_pos + j] = get_rand_u(-con1, con1);
		}
	}

	for (INT i = 0; i < position_total_tail; i++) {
		for (INT j = 0; j < dimension_pos; j++) {
			position_vec_tail[i * dimension_pos + j] = get_rand_u(-con1, con1);
		}
	}

	matrixRelationDao = (REAL *)calloc(dimension_c*relation_total, sizeof(REAL));
	matrixW1Dao =  (REAL*)calloc(dimension_c * dimension * window, sizeof(REAL));
	matrixB1Dao =  (REAL*)calloc(dimension_c, sizeof(REAL));
	
	positionVecDaoE1 = (REAL *)calloc(position_total_head * dimension_pos, sizeof(REAL));
	positionVecDaoE2 = (REAL *)calloc(position_total_tail * dimension_pos, sizeof(REAL));
	matrixW1PositionE1Dao = (REAL *)calloc(dimension_c * dimension_pos * window, sizeof(REAL));
	matrixW1PositionE2Dao = (REAL *)calloc(dimension_c * dimension_pos * window, sizeof(REAL));

	for (turn = 0; turn < train_times; turn ++) {
		// len = c_train.size();
		len = bags_train.size();
		npoch  =  len / (batch * num_threads);
		alpha1 = alpha*rate/batch;

		score = 0;
		score_max = 0;
		score_tmp = 0;
		double score1 = score;
		time_begin();
		for (INT k = 1; k <= npoch; k++) {
			score_max += batch * num_threads;
			memcpy(positionVecDaoE1, position_vec_head, position_total_head * dimension_pos* sizeof(REAL));
			memcpy(positionVecDaoE2, position_vec_tail, position_total_tail * dimension_pos* sizeof(REAL));
			memcpy(matrixW1PositionE1Dao, conv_1d_position_head, dimension_c * dimension_pos * window* sizeof(REAL));
			memcpy(matrixW1PositionE2Dao, conv_1d_position_tail, dimension_c * dimension_pos * window* sizeof(REAL));
			memcpy(wordVecDao, word_vec, dimension * word_total * sizeof(REAL));

			memcpy(matrixW1Dao, conv_1d_word, sizeof(REAL) * dimension_c * dimension * window);
			memcpy(matrixB1Dao, conv_1d_bias, sizeof(REAL) * dimension_c);
			memcpy(matrixRelationPrDao, relation_matrix_bias, relation_total * sizeof(REAL));				//add
			memcpy(matrixRelationDao, relation_matrix, dimension_c*relation_total * sizeof(REAL));
			att_W_Dao = attention_weights;
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
	//output_model = 1;
	std::cout<<"Init Begin."<<std::endl;
	init();
	std::cout<< "bags_train.size: " << bags_train.size() << '\t' 
	         << "bags_test.size: " << bags_test.size() << std::endl;
	std::cout<<"Init End."<<std::endl;
	train();
}
