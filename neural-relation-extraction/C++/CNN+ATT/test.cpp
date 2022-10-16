#include "init.h"
#include "test.h"

void preprocess()
{
	position_vec_head = (REAL *)calloc(position_total_head * dimension_pos, sizeof(REAL));
	position_vec_tail = (REAL *)calloc(position_total_tail * dimension_pos, sizeof(REAL));

	conv_1d_word = (REAL*)calloc(dimension_c * dimension * window, sizeof(REAL));
	conv_1d_position_head = (REAL *)calloc(dimension_c * dimension_pos * window, sizeof(REAL));
	conv_1d_position_tail = (REAL *)calloc(dimension_c * dimension_pos * window, sizeof(REAL));
	conv_1d_bias = (REAL*)calloc(dimension_c, sizeof(REAL));

	attention_weights.resize(relation_total);
	for (INT i = 0; i < relation_total; i++)
	{
		attention_weights[i].resize(dimension_c);
		for (INT j=0; j < dimension_c; j++)
			attention_weights[i][j].resize(dimension_c);
	}

	relation_matrix = (REAL *)calloc(relation_total * dimension_c, sizeof(REAL));
	relation_matrix_bias = (REAL *)calloc(relation_total, sizeof(REAL));
	
	INT tmp;

	FILE *fout = fopen(("./out/word2vec" + version + ".txt").c_str(), "r");
	tmp = fscanf(fout,"%d%d", &word_total, &dimension);
	for (INT i = 0; i < word_total; i++)
	{
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fout, "%f", &word_vec[i * dimension + j]);
	}
	fclose(fout);

	fout = fopen(("./out/position_vec" + version + ".txt").c_str(), "r");
	tmp = fscanf(fout, "%d%d%d", &position_total_head, &position_total_tail, &dimension_pos);
	for (INT i = 0; i < position_total_head; i++) {
		for (INT j = 0; j < dimension_pos; j++)
			tmp = fscanf(fout, "%f", &position_vec_head[i * dimension_pos + j]);
	}
	for (INT i = 0; i < position_total_tail; i++) {
		for (INT j = 0; j < dimension_pos; j++)
			tmp = fscanf(fout, "%f", &position_vec_tail[i * dimension_pos + j]);
	}
	fclose(fout);

	fout = fopen(("./out/conv_1d" + version + ".txt").c_str(), "r");
	tmp = fscanf(fout, "%d%d%d%d", &dimension_c, &dimension, &window, &dimension_pos);
	for (INT i = 0; i < dimension_c; i++) {
		for (INT j = 0; j < dimension * window; j++)
			tmp = fscanf(fout, "%f", &conv_1d_word[i * dimension * window + j]);
		for (INT j = 0; j < dimension_pos * window; j++)
			tmp = fscanf(fout, "%f", &conv_1d_position_head[i * dimension_pos * window + j]);
		for (INT j = 0; j < dimension_pos * window; j++)
			tmp = fscanf(fout, "%f", &conv_1d_position_tail[i * dimension_pos * window + j]);
		tmp = fscanf(fout, "%f", &conv_1d_bias[i]);
	}
	fclose(fout);

	fout = fopen(("./out/attention_weights" + version + ".txt").c_str(), "r");
	tmp = fscanf(fout,"%d%d", &relation_total, &dimension_c);
	for (INT r1 = 0; r1 < relation_total; r1++) {
		for (INT i = 0; i < dimension_c; i++)
		{
			for (INT j = 0; j < dimension_c; j++)
				tmp = fscanf(fout, "%f", &attention_weights[r1][i][j]);
		}
	}
	fclose(fout);

	fout = fopen(("./out/relation_matrix" + version + ".txt").c_str(), "r");
	tmp = fscanf(fout, "%d%d", &relation_total, &dimension_c);
	for (INT i = 0; i < relation_total; i++) {
		for (INT j = 0; j < dimension_c; j++)
			tmp = fscanf(fout, "%f", &relation_matrix[i * dimension_c + j]);
	}
	for (INT i = 0; i < relation_total; i++) 
		tmp = fscanf(fout, "%f", &relation_matrix_bias[i]);
	fclose(fout);
}

INT main()
{
	init();
	preprocess();
	test();
	return 0;
}