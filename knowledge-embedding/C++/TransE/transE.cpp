// transE.cpp
// created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>

// ##################################################
// 包含标准库
// ##################################################

#include <cstdio>           // FILE, fscanf, fwrite, fopen
#include <cstdlib>          // calloc, free, atoi, atof, rand, RAND_MAX
#include <cmath>            // exp, fabs
#include <cstring>          // memcmp, memcpy, strcmp
#include <fcntl.h>          // open, close, O_RDONLY
#include <unistd.h>         // stat
#include <sys/stat.h>       // stat
#include <sys/mman.h>       // mmap, munmap
#include <pthread.h>        // pthread_create, pthread_exit, pthread_join
#include <string>           // std::string, std::string::c_str
#include <algorithm>        // std::sort

// ##################################################
// 声明和定义变量
// ##################################################

#define REAL float
#define INT int

const REAL pi = 3.141592653589793238462643383;

INT bern_flag = 0;
INT load_binary_flag = 0;
INT out_binary_flag = 0;
INT dimension = 50;
REAL alpha = 0.01;
REAL margin = 1.0;
INT nbatches = 1;
INT epochs = 1000;
INT threads = 32;

std::string in_path = "../data/FB15K/";
std::string out_path = "./";
std::string load_path = "";
std::string note = "";

// 三元组: (head, label, tail)
// h: head
// r: label or relationship
// t: tail
// a relationship of name label between the entities head and tail
struct Triple {
	INT h, r, t;
};

// 为 std::sort() 定义比较仿函数
// 以三元组的 head 进行比较
struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)
			||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

// 为 std::sort() 定义比较仿函数
// 以三元组的 tail 进行比较
struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)
			||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

// ##################################################
// 一些用于程序初始化的数学函数
// ##################################################

// 为每一个线程存储独立的随机种子
unsigned long long *next_random;

// 更新第 id[0, threads) 线程的随机种子
unsigned long long randd(INT id) {
	next_random[id] = next_random[id] 
		* (unsigned long long)25214903917 + 11;
	return next_random[id];
}

// 为第 id[0, threads) 线程返回取值为 [0, x) 的伪随机数
INT rand_max(INT id, INT x) {
	INT res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

// 返回取值为 [min, max) 的伪随机数 
REAL rand(REAL min, REAL max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

// 正态分布函数，X ~ N (miu, sigma)
REAL normal(REAL x, REAL miu,REAL sigma) {
	return 1.0 / sqrt(2 * pi) / sigma
		* exp(-1 * (x-miu) * (x-miu) / (2 * sigma * sigma));
}

// 从正态（高斯）分布中抽取随机样本，取值为 [min, max) 的伪随机数
REAL randn(REAL miu, REAL sigma, REAL min, REAL max) {
	REAL x, y, d_scope;
	do {
		x = rand(min, max);
		y = normal(x, miu, sigma);
		d_scope = rand(0.0, normal(miu, miu, sigma));
	} while (d_scope > y);
	return x;
}

// 归一化函数：使用 l2 范数将输入向量缩放为 unit norm (vector length)
void norm(REAL * vec) {
	REAL x = 0;
	for (INT i = 0; i < dimension; i++)
		x += (*(vec + i)) * (*(vec + i));
	x = sqrt(x);
	if (x > 1)
		for (INT i = 0; i < dimension; i++)
			*(vec + i) /= x;
}

// ##################################################
// 从 train2id.txt 中读取三元组
// ##################################################

// relation_total: 关系总数
// entity_total: 实体总数
// triple_total: 训练集中的三元组总数
INT relation_total, entity_total, triple_total;

// relation_vec: 关系嵌入矩阵 (relation_total * dimension)
// entity_vec: 实体嵌入矩阵 (entity_total * dimension)
REAL *relation_vec, *entity_vec;

// freq_rel: 存储各个关系在训练集中的个数 (relation_total)
// freq_ent: 存储各个实体在训练集中的个数 (entity_total)
INT *freq_rel, *freq_ent;

// train_head (triple_total): 训练集中的三元组集合，以 head 排序
// train_tail (triple_total): 训练集中的三元组集合，以 tail 排序
// train_list (triple_total): 训练集中的三元组集合，未排序
Triple *train_head, *train_tail, *train_list;

// left_head: 
INT *left_head, *right_head;
INT *left_tail, *right_tail;

REAL *left_mean, *right_mean;

void init() {

	FILE *fin;
	INT tmp;

	// 初始化 relation_vec
	fin = fopen((in_path + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relation_total);
	fclose(fin);

	relation_vec = (REAL *)calloc(relation_total * dimension,
			sizeof(REAL));
	for (INT i = 0; i < relation_total; i++) {
		for (INT ii = 0; ii < dimension; ii++)
			relation_vec[i * dimension + ii] =
				randn(0, 1.0 / dimension, -6 / sqrt(dimension),
					6 / sqrt(dimension));
	}

	// 初始化 entity_vec
	fin = fopen((in_path + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entity_total);
	fclose(fin);

	entity_vec = (REAL *)calloc(entity_total * dimension,
			sizeof(REAL));
	for (INT i = 0; i < entity_total; i++) {
		for (INT ii = 0; ii < dimension; ii++)
			entity_vec[i * dimension + ii] =
				randn(0, 1.0 / dimension, -6 / sqrt(dimension),
					6 / sqrt(dimension));
	}

	// 为 freq_rel 和 freq_ent 分配一个内存块，并将其所有位初始化为零
	freq_rel = (INT *)calloc(relation_total + entity_total, sizeof(INT));
	freq_ent = freq_rel + relation_total;

	// 读取训练集中的三元组
	fin = fopen((in_path + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &triple_total);
	train_head = (Triple *)calloc(triple_total, sizeof(Triple));
	train_tail = (Triple *)calloc(triple_total, sizeof(Triple));
	train_list = (Triple *)calloc(triple_total, sizeof(Triple));
	for (INT i = 0; i < triple_total; i++) {
		tmp = fscanf(fin, "%d", &train_list[i].h);
		tmp = fscanf(fin, "%d", &train_list[i].t);
		tmp = fscanf(fin, "%d", &train_list[i].r);
		freq_ent[train_list[i].t]++;
		freq_ent[train_list[i].h]++;
		freq_rel[train_list[i].r]++;
		train_head[i] = train_list[i];
		train_tail[i] = train_list[i];
	}
	fclose(fin);

	// 分别以 head 和 tail 排序
	std::sort(train_head, train_head + triple_total, cmp_head());
	std::sort(train_tail, train_tail + triple_total, cmp_tail());

	left_head = (INT *)calloc(entity_total, sizeof(INT));
	right_head = (INT *)calloc(entity_total, sizeof(INT));
	left_tail = (INT *)calloc(entity_total, sizeof(INT));
	right_tail = (INT *)calloc(entity_total, sizeof(INT));
	for (INT i = 1; i < triple_total; i++) {
		if (train_head[i].h != train_head[i - 1].h) {
			right_head[train_head[i - 1].h] = i - 1;
			left_head[train_head[i].h] = i;
		}
		if (train_tail[i].t != train_tail[i - 1].t) {
			right_tail[train_tail[i - 1].t] = i - 1;
			left_tail[train_tail[i].t] = i;
		}
	}
	right_head[train_head[triple_total - 1].h] = triple_total - 1;
	right_tail[train_tail[triple_total - 1].t] = triple_total - 1;

	left_mean = (REAL *)calloc(relation_total * 2, sizeof(REAL));
	right_mean = left_mean + relation_total;
	for (INT i = 0; i < entity_total; i++) {
		for (INT j = left_head[i] + 1; j <= right_head[i]; j++)
			if (train_head[j].r != train_head[j - 1].r)
				left_mean[train_head[j].r] += 1.0;
		if (left_head[i] <= right_head[i])
			left_mean[train_head[left_head[i]].r] += 1.0;
		for (INT j = left_tail[i] + 1; j <= right_tail[i]; j++)
			if (train_tail[j].r != train_tail[j - 1].r)
				right_mean[train_tail[j].r] += 1.0;
		if (left_tail[i] <= right_tail[i])
			right_mean[train_tail[left_tail[i]].r] += 1.0;
	}

	for (INT i = 0; i < relation_total; i++) {
		left_mean[i] = freq_rel[i] / left_mean[i];
		right_mean[i] = freq_rel[i] / right_mean[i];
	}
}

void load_binary() {
	struct stat statbuf1;
	if (stat((load_path + "entity2vec" + note + ".bin").c_str(), &statbuf1) != -1) {  
		INT fd = open((load_path + "entity2vec" + note + ".bin").c_str(), O_RDONLY);
		REAL* entity_vec_tmp = (REAL*)mmap(NULL, statbuf1.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
		memcpy(entity_vec, entity_vec_tmp, statbuf1.st_size);
		munmap(entity_vec_tmp, statbuf1.st_size);
		close(fd);
	}  
	struct stat statbuf2;
	if (stat((load_path + "relation2vec" + note + ".bin").c_str(), &statbuf2) != -1) {  
		INT fd = open((load_path + "relation2vec" + note + ".bin").c_str(), O_RDONLY);
		REAL* relation_vec_tmp =(REAL*)mmap(NULL, statbuf2.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
		memcpy(relation_vec, relation_vec_tmp, statbuf2.st_size);
		munmap(relation_vec_tmp, statbuf2.st_size);
		close(fd);
	}
}

void load() {
	if (load_binary_flag) {
		load_binary();
		return;
	}
	FILE *fin;
	INT tmp;
	fin = fopen((load_path + "entity2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < entity_total; i++) {
		INT last = i * dimension;
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &entity_vec[last + j]);
	}
	fclose(fin);
	fin = fopen((load_path + "relation2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < relation_total; i++) {
		INT last = i * dimension;
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &relation_vec[last + j]);
	}
	fclose(fin);
}


/*
	Training process of transE.
*/

INT Len;
INT Batch;
REAL res;

REAL calc_sum(INT e1, INT e2, INT rel) {
	REAL sum=0;
	INT last1 = e1 * dimension;
	INT last2 = e2 * dimension;
	INT lastr = rel * dimension;
	for (INT ii=0; ii < dimension; ii++)
		sum += fabs(entity_vec[last2 + ii] - entity_vec[last1 + ii] - relation_vec[lastr + ii]);
	return sum;
}

void gradient(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b) {
	INT lasta1 = e1_a * dimension;
	INT lasta2 = e2_a * dimension;
	INT lastar = rel_a * dimension;
	INT lastb1 = e1_b * dimension;
	INT lastb2 = e2_b * dimension;
	INT lastbr = rel_b * dimension;
	for (INT ii=0; ii  < dimension; ii++) {
		REAL x;
		x = (entity_vec[lasta2 + ii] - entity_vec[lasta1 + ii] - relation_vec[lastar + ii]);
		if (x > 0)
			x = -alpha;
		else
			x = alpha;
		relation_vec[lastar + ii] -= x;
		entity_vec[lasta1 + ii] -= x;
		entity_vec[lasta2 + ii] += x;
		x = (entity_vec[lastb2 + ii] - entity_vec[lastb1 + ii] - relation_vec[lastbr + ii]);
		if (x > 0)
			x = alpha;
		else
			x = -alpha;
		relation_vec[lastbr + ii] -=  x;
		entity_vec[lastb1 + ii] -= x;
		entity_vec[lastb2 + ii] += x;
	}
}

void train_kb(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b) {
	REAL sum1 = calc_sum(e1_a, e2_a, rel_a);
	REAL sum2 = calc_sum(e1_b, e2_b, rel_b);
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
	}
}

INT corrupt_head(INT id, INT h, INT r) {
	INT lef, rig, mid, ll, rr;
	lef = left_head[h] - 1;
	rig = right_head[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (train_head[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = left_head[h];
	rig = right_head[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (train_head[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entity_total - (rr - ll + 1));
	if (tmp < train_head[ll].t) return tmp;
	if (tmp > train_head[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (train_head[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

INT corrupt_tail(INT id, INT t, INT r) {
	INT lef, rig, mid, ll, rr;
	lef = left_tail[t] - 1;
	rig = right_tail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (train_tail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = left_tail[t];
	rig = right_tail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (train_tail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entity_total - (rr - ll + 1));
	if (tmp < train_tail[ll].h) return tmp;
	if (tmp > train_tail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (train_tail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* trainMode(void *con) {
	INT id, pr, i, j;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	for (INT k = Batch / threads; k >= 0; k--) {
		i = rand_max(id, Len);
		if (bern_flag)
			pr = 1000 * right_mean[train_list[i].r] / (right_mean[train_list[i].r] + left_mean[train_list[i].r]);
		else
			pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, train_list[i].h, train_list[i].r);
			train_kb(train_list[i].h, train_list[i].t, train_list[i].r, train_list[i].h, j, train_list[i].r);
		} else {
			j = corrupt_tail(id, train_list[i].t, train_list[i].r);
			train_kb(train_list[i].h, train_list[i].t, train_list[i].r, j, train_list[i].t, train_list[i].r);
		}
		norm(relation_vec + dimension * train_list[i].r);
		norm(entity_vec + dimension * train_list[i].h);
		norm(entity_vec + dimension * train_list[i].t);
		norm(entity_vec + dimension * j);
	}
	pthread_exit(NULL);
}

void* train(void *con) {
	Len = triple_total;
	Batch = Len / nbatches;
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));
	for (INT epoch = 0; epoch < epochs; epoch++) {
		res = 0;
		for (INT batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
			for (long a = 0; a < threads; a++)
				pthread_create(&pt[a], NULL, trainMode,  (void*)a);
			for (long a = 0; a < threads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

/*
	Get the results of transE.
*/

void out_binary() {
		INT len, tot;
		REAL *head;		
		FILE* f2 = fopen((out_path + "relation2vec" + note + ".bin").c_str(), "wb");
		FILE* f3 = fopen((out_path + "entity2vec" + note + ".bin").c_str(), "wb");
		len = relation_total * dimension; tot = 0;
		head = relation_vec;
		while (tot < len) {
			INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f2);
			tot = tot + sum;
		}
		len = entity_total * dimension; tot = 0;
		head = entity_vec;
		while (tot < len) {
			INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f3);
			tot = tot + sum;
		}	
		fclose(f2);
		fclose(f3);
}

void out() {
		if (out_binary_flag) {
			out_binary(); 
			return;
		}
		FILE* f2 = fopen((out_path + "relation2vec" + note + ".vec").c_str(), "w");
		FILE* f3 = fopen((out_path + "entity2vec" + note + ".vec").c_str(), "w");
		for (INT i=0; i < relation_total; i++) {
			INT last = dimension * i;
			for (INT ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relation_vec[last + ii]);
			fprintf(f2,"\n");
		}
		for (INT  i = 0; i < entity_total; i++) {
			INT last = i * dimension;
			for (INT ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entity_vec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
}

/*
	Main function
*/

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}


void setparameters(int argc, char **argv) {
	int i;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) in_path = argv[i + 1];
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) out_path = argv[i + 1];
	if ((i = ArgPos((char *)"-load", argc, argv)) > 0) load_path = argv[i + 1];
	if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) epochs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-nbatches", argc, argv)) > 0) nbatches = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-load-binary", argc, argv)) > 0) load_binary_flag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-out-binary", argc, argv)) > 0) out_binary_flag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
}

int main(int argc, char **argv) {
	setparameters(argc, argv);
	init();
	if (load_path != "") load();
	train(NULL);
	if (out_path != "") out();
	return 0;
}
