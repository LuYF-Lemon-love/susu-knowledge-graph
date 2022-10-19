// test.cpp
//
// 使用方法:
//     编译:
//           $ g++ test.cpp -o ./build/test -pthread -O3 -march=native
//     运行:
//           $ ./build/test
//
// created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
//
// 该 C++ 文件用于模型测试
//
// 加载模型
// prerequisites:
//     ./output/word2vec + note + .txt
//     ./output/position_vec + note + .txt
//     ./output/conv_1d + note + .txt
//     ./output/attention_weights + note + .txt
//     ./output/relation_matrix + note + .txt

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
	FILE *fout = fopen((output_path + "word2vec" + note + ".txt").c_str(), "r");
	tmp = fscanf(fout,"%d%d", &word_total, &dimension);
	for (INT i = 0; i < word_total; i++)
	{
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fout, "%f", &word_vec[i * dimension + j]);
	}
	fclose(fout);

	// 加载位置嵌入
	fout = fopen((output_path + "position_vec" + note + ".txt").c_str(), "r");
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

	// 加载一维卷积权重矩阵和对应的偏置向量
	fout = fopen((output_path + "conv_1d" + note + ".txt").c_str(), "r");
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
	fout = fopen((output_path + "attention_weights" + note + ".txt").c_str(), "r");
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
	fout = fopen((output_path + "relation_matrix" + note + ".txt").c_str(), "r");
	tmp = fscanf(fout, "%d%d%f", &relation_total, &dimension_c, &dropout_probability);
	for (INT i_r = 0; i_r < relation_total; i_r++) {
		for (INT i_s = 0; i_s < dimension_c; i_s++)
			tmp = fscanf(fout, "%f", &relation_matrix[i_r * dimension_c + i_s]);
	}
	for (INT i_r = 0; i_r < relation_total; i_r++) 
		tmp = fscanf(fout, "%f", &relation_matrix_bias[i_r]);
	fclose(fout);

	printf("模型加载成功!\n\n");
}

// ##################################################
// Main function
// ##################################################

void setparameters(INT argc, char **argv) {
	INT i;
	if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
	if ((i = arg_pos((char *)"-data_path", argc, argv)) > 0) data_path = argv[i + 1];
	if ((i = arg_pos((char *)"-load_path", argc, argv)) > 0) output_path = argv[i + 1];
}

void print_test_help() {
	std::string str = R"(
// ##################################################
// ./test [-threads THREAD] [-dropout DROPOUT]
//        [-note NOTE] [-data_path DATA_PATH]
//        [-load_path LOAD_PATH] [--help]

// optional arguments:
// -threads THREAD                number of worker threads. if unspecified, num_threads will default to [32]
// -note NOTE                     information you want to add to the filename, like ("./output/word2vec" + note + ".txt"). if unspecified, note will default to ""
// -data_path DATA_PATH           folder of data. if unspecified, data_path will default to "../data/"
// -load_path LOAD_PATH           folder of pretrained models. if unspecified, load_path will default to "./output/"
// --help                         print help information of ./test
// ##################################################
)";

	printf("%s\n", str.c_str());
}

// ##################################################
// ./test [-threads THREAD] [-dropout DROPOUT]
//        [-note NOTE] [-data_path DATA_PATH]
//        [-load_path LOAD_PATH] [--help]

// optional arguments:
// -threads THREAD                number of worker threads. if unspecified, num_threads will default to [32]
// -note NOTE                     information you want to add to the filename, like ("./output/word2vec" + note + ".txt"). if unspecified, note will default to ""
// -data_path DATA_PATH           folder of data. if unspecified, data_path will default to "../data/"
// -load_path LOAD_PATH           folder of pretrained models. if unspecified, load_path will default to "./output/"
// --help                         print help information of ./test
// ##################################################

INT main(INT argc, char **argv) {
	for (INT a = 1; a < argc; a++) if (!strcmp((char *)"--help", argv[a])) {
		print_test_help();
		return 0;
	}	
	setparameters(argc, argv);
	init();
	load_model();
	print_information();
	test();
	printf("Test end.\n\n##################################################\n\n");
	return 0;
}