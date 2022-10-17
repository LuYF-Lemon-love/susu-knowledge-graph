// test.cpp
//
// created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
//
// 该 C++ 文件用于模型测试
//
// 加载模型
// prerequisites:
//     ./out/word2vec + note + .txt
//     ./out/position_vec + note + .txt
//     ./out/conv_1d + note + .txt
//     ./out/attention_weights + note + .txt
//     ./out/relation_matrix + note + .txt

// ##################################################
// 包含标准库和头文件
// ##################################################

#include "init.h"
#include "test.h"

// 加载模型
void load_model()
{
	// 为模型的权重矩阵分配内存空间
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
		for (INT j = 0; j < dimension_c; j++)
			attention_weights[i][j].resize(dimension_c);
	}

	relation_matrix = (REAL *)calloc(relation_total * dimension_c, sizeof(REAL));
	relation_matrix_bias = (REAL *)calloc(relation_total, sizeof(REAL));
	
	INT tmp;

	// 加载词嵌入
	FILE *fout = fopen(("./out/word2vec" + note + ".txt").c_str(), "r");
	tmp = fscanf(fout,"%d%d", &word_total, &dimension);
	for (INT i = 0; i < word_total; i++)
	{
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fout, "%f", &word_vec[i * dimension + j]);
	}
	fclose(fout);

	// 加载位置嵌入
	fout = fopen(("./out/position_vec" + note + ".txt").c_str(), "r");
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

	// 加载一维卷机权重矩阵和对应的偏置向量
	fout = fopen(("./out/conv_1d" + note + ".txt").c_str(), "r");
	tmp = fscanf(fout, "%d%d%d%d", &dimension_c, &window, &dimension, &dimension_pos);
	for (INT i = 0; i < dimension_c; i++) {
		for (INT j = 0; j < window * dimension; j++)
			tmp = fscanf(fout, "%f", &conv_1d_word[i * window * dimension + j]);
		for (INT j = 0; j < window * dimension_pos; j++)
			tmp = fscanf(fout, "%f", &conv_1d_position_head[i * window * dimension_pos + j]);
		for (INT j = 0; j < window * dimension_pos; j++)
			tmp = fscanf(fout, "%f", &conv_1d_position_tail[i * window * dimension_pos + j]);
		tmp = fscanf(fout, "%f", &conv_1d_bias[i]);
	}
	fclose(fout);

	// 加载注意力权重矩阵
	fout = fopen(("./out/attention_weights" + note + ".txt").c_str(), "r");
	tmp = fscanf(fout,"%d%d", &relation_total, &dimension_c);
	for (INT r = 0; r < relation_total; r++) {
		for (INT i_x = 0; i_x < dimension_c; i_x++)
		{
			for (INT i_r = 0; i_r < dimension_c; i_r++)
				tmp = fscanf(fout, "%f", &attention_weights[r][i_x][i_r]);
		}
	}
	fclose(fout);

	// 加载 relation_matrix 和对应的偏置向量
	fout = fopen(("./out/relation_matrix" + note + ".txt").c_str(), "r");
	tmp = fscanf(fout, "%d%d", &relation_total, &dimension_c);
	for (INT i_r = 0; i_r < relation_total; i_r++) {
		for (INT i_s = 0; i_s < dimension_c; i_s++)
			tmp = fscanf(fout, "%f", &relation_matrix[i_r * dimension_c + i_s]);
	}
	for (INT i_r = 0; i_r < relation_total; i_r++) 
		tmp = fscanf(fout, "%f", &relation_matrix_bias[i_r]);
	fclose(fout);
}

INT main()
{
	init();
	load_model();
	test();
	return 0;
}