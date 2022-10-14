#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <map>
#include <cmath>
#include <pthread.h>
#include <iostream>

#include<assert.h>
#include<ctime>
#include<sys/time.h>

#include "init.h"
#include "test.h"

void preprocess()
{

	matrixRelation = (REAL *)calloc(dimension_c * relation_total, sizeof(REAL));
	matrixRelationPr = (REAL *)calloc(relation_total, sizeof(REAL));
	matrixRelationPrDao = (REAL *)calloc(relation_total, sizeof(REAL));
	wordVecDao = (REAL *)calloc(dimension * word_total, sizeof(REAL));
	position_vec_head = (REAL *)calloc(position_total_head * dimension_pos, sizeof(REAL));
	position_vec_tail = (REAL *)calloc(position_total_tail * dimension_pos, sizeof(REAL));
	
	conv_1d_word = (REAL*)calloc(dimension_c * dimension * window, sizeof(REAL));
	conv_1d_position_head = (REAL *)calloc(dimension_c * dimension_pos * window, sizeof(REAL));
	conv_1d_position_tail = (REAL *)calloc(dimension_c * dimension_pos * window, sizeof(REAL));
	conv_1d_bias = (REAL*)calloc(dimension_c, sizeof(REAL));
	
	att_W.resize(relation_total);
	for (INT i=0; i<relation_total; i++)
	{
		att_W[i].resize(dimension_c);
		for (INT j=0; j<dimension_c; j++)
			att_W[i][j].resize(dimension_c);
	}
	version = "";
	
	FILE *fout = fopen(("./out/matrixW1+B1.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d%d%d", &dimension_c, &dimension, &window, &dimension_pos);
	for (INT i = 0; i < dimension_c; i++) {
		for (INT j = 0; j < dimension * window; j++)
			fscanf(fout, "%f", &conv_1d_word[i* dimension*window+j]);
		for (INT j = 0; j < dimension_pos * window; j++)
			fscanf(fout, "%f", &conv_1d_position_head[i* dimension_pos*window+j]);
		for (INT j = 0; j < dimension_pos * window; j++)
			fscanf(fout, "%f", &conv_1d_position_tail[i* dimension_pos*window+j]);
		fscanf(fout, "%f", &conv_1d_bias[i]);
	}
	fclose(fout);

	fout = fopen(("./out/matrixRl.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d", &relation_total, &dimension_c);
	for (INT i = 0; i < relation_total; i++) {
		for (INT j = 0; j < dimension_c; j++)
			fscanf(fout, "%f", &matrixRelation[i * dimension_c + j]);
	}
	for (INT i = 0; i < relation_total; i++) 
		fscanf(fout, "%f", &matrixRelationPr[i]);
	fclose(fout);

	fout = fopen(("./out/matrixPosition.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d%d", &position_total_head, &position_total_tail, &dimension_pos);
	for (INT i = 0; i < position_total_head; i++) {
		for (INT j = 0; j < dimension_pos; j++)
			fscanf(fout, "%f", &position_vec_head[i * dimension_pos + j]);
	}
	for (INT i = 0; i < position_total_tail; i++) {
		for (INT j = 0; j < dimension_pos; j++)
			fscanf(fout, "%f", &position_vec_tail[i * dimension_pos + j]);
	}
	fclose(fout);

	fout = fopen(("./out/word2vec.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d",&word_total,&dimension);
	for (INT i = 0; i < word_total; i++)
	{
		for (INT j=0; j<dimension; j++)
			fscanf(fout,"%f", &word_vec[i*dimension+j]);
	}
	fclose(fout);
	fout = fopen(("./out/att_W.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d", &relation_total, &dimension_c);
	for (INT r1 = 0; r1 < relation_total; r1++) {
		for (INT i = 0; i < dimension_c; i++)
		{
			for (INT j = 0; j < dimension_c; j++)
				fscanf(fout, "%f", &att_W[r1][i][j]);
		}
	}
	fclose(fout);
}

INT main()
{
	init();
	preprocess();
	test();
	return 0;
}