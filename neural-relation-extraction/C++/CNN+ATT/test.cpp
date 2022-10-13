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

	matrixRelation = (REAL *)calloc(dimensionC * relation_total, sizeof(REAL));
	matrixRelationPr = (REAL *)calloc(relation_total, sizeof(REAL));
	matrixRelationPrDao = (REAL *)calloc(relation_total, sizeof(REAL));
	wordVecDao = (REAL *)calloc(dimension * word_total, sizeof(REAL));
	positionVecE1 = (REAL *)calloc(position_total_head * dimensionWPE, sizeof(REAL));
	positionVecE2 = (REAL *)calloc(position_total_tail * dimensionWPE, sizeof(REAL));
	
	matrixW1 = (REAL*)calloc(dimensionC * dimension * window, sizeof(REAL));
	matrixW1PositionE1 = (REAL *)calloc(dimensionC * dimensionWPE * window, sizeof(REAL));
	matrixW1PositionE2 = (REAL *)calloc(dimensionC * dimensionWPE * window, sizeof(REAL));
	matrixB1 = (REAL*)calloc(dimensionC, sizeof(REAL));
	
	att_W.resize(relation_total);
	for (INT i=0; i<relation_total; i++)
	{
		att_W[i].resize(dimensionC);
		for (INT j=0; j<dimensionC; j++)
			att_W[i][j].resize(dimensionC);
	}
	version = "";
	
	FILE *fout = fopen(("./out/matrixW1+B1.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d%d%d", &dimensionC, &dimension, &window, &dimensionWPE);
	for (INT i = 0; i < dimensionC; i++) {
		for (INT j = 0; j < dimension * window; j++)
			fscanf(fout, "%f", &matrixW1[i* dimension*window+j]);
		for (INT j = 0; j < dimensionWPE * window; j++)
			fscanf(fout, "%f", &matrixW1PositionE1[i* dimensionWPE*window+j]);
		for (INT j = 0; j < dimensionWPE * window; j++)
			fscanf(fout, "%f", &matrixW1PositionE2[i* dimensionWPE*window+j]);
		fscanf(fout, "%f", &matrixB1[i]);
	}
	fclose(fout);

	fout = fopen(("./out/matrixRl.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d", &relation_total, &dimensionC);
	for (INT i = 0; i < relation_total; i++) {
		for (INT j = 0; j < dimensionC; j++)
			fscanf(fout, "%f", &matrixRelation[i * dimensionC + j]);
	}
	for (INT i = 0; i < relation_total; i++) 
		fscanf(fout, "%f", &matrixRelationPr[i]);
	fclose(fout);

	fout = fopen(("./out/matrixPosition.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d%d", &position_total_head, &position_total_tail, &dimensionWPE);
	for (INT i = 0; i < position_total_head; i++) {
		for (INT j = 0; j < dimensionWPE; j++)
			fscanf(fout, "%f", &positionVecE1[i * dimensionWPE + j]);
	}
	for (INT i = 0; i < position_total_tail; i++) {
		for (INT j = 0; j < dimensionWPE; j++)
			fscanf(fout, "%f", &positionVecE2[i * dimensionWPE + j]);
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
	fscanf(fout,"%d%d", &relation_total, &dimensionC);
	for (INT r1 = 0; r1 < relation_total; r1++) {
		for (INT i = 0; i < dimensionC; i++)
		{
			for (INT j = 0; j < dimensionC; j++)
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