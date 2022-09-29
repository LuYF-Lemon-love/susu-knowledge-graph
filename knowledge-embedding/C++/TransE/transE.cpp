// transE.cpp
// 使用方法:
//     编译:
//           $ g++ transE.cpp -o ./build/transE -pthread -O3 -march=native
//     运行:
//           $ ./build/transE
//           
// created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
//
// 该 C++ 文件用于模型训练
//
// prerequisites:
//     relation2id.txt, entity2id.txt, train2id.txt
//
// 加载 Pretrained Embeddings (可选)
// prerequisites: 
//     entity2vec + note + .bin
//     relation2vec + note + .bin
//     
//     or
//
//     entity2vec + note + .vec
//     relation2vec + note + .vec
//
// 输出实体嵌入和关系嵌入
// output: 
//     entity2vec + note + .bin
//     relation2vec + note + .bin
//     
//     or
//
//     entity2vec + note + .vec
//     relation2vec + note + .vec

// ##################################################
// 包含标准库
// ##################################################

#include <cstdio>           // FILE, fscanf, fwrite, fopen, fclose
#include <cstdlib>          // calloc, free, atoi, atof, rand, RAND_MAX
#include <cmath>            // exp, fabs
#include <cstring>          // memcmp, memcpy, strcmp
#include <fcntl.h>          // open, close, O_RDONLY
#include <unistd.h>         // stat
#include <sys/stat.h>       // stat
#include <sys/mman.h>       // mmap, munmap
#include <sys/time.h>       // timeval, gettimeofday
#include <pthread.h>        // pthread_create, pthread_exit, pthread_join
#include <string>           // std::string, std::string::c_str
#include <algorithm>        // std::sort

// ##################################################
// 声明和定义变量
// ##################################################

#define REAL float
#define INT int

const REAL pi = 3.141592653589793238462643383;

INT bern_flag = 1;
INT load_binary_flag = 0;
INT out_binary_flag = 0;
INT dimension = 50;
REAL alpha = 0.01;
REAL margin = 1.0;
INT nbatches = 1;
INT epochs = 1000;
INT threads = 32;

std::string in_path = "../data/FB15K/";
std::string out_path = "./build/";
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
REAL normal(REAL x, REAL miu, REAL sigma) {
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

// 归一化函数：使用 L2 范数将输入向量缩放为 unit norm (vector length)
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
// prerequisites: 
//     relation2id.txt, entity2id.txt, train2id.txt
// ##################################################

// relation_total: 关系总数
// entity_total: 实体总数
// train_triple_total: 训练集中的三元组总数
INT relation_total, entity_total, train_triple_total;

// relation_vec (relation_total * dimension): 关系嵌入矩阵
// entity_vec (entity_total * dimension): 实体嵌入矩阵
REAL *relation_vec, *entity_vec;

// train_head (train_triple_total): 训练集中的三元组集合，以 head 排序
// train_tail (train_triple_total): 训练集中的三元组集合，以 tail 排序
// train_list (train_triple_total): 训练集中的三元组集合，未排序
Triple *train_head, *train_tail, *train_list;

// left_head (entity_total): 存储每种实体 (head) 在 train_head 中第一次出现的位置
// right_head (entity_total): 存储每种实体 (head) 在 train_head 中最后一次出现的位置
// left_tail (entity_total): 存储每种实体 (tail) 在 train_tail 中第一次出现的位置
// right_tail (entity_total): 存储每种实体 (tail) 在 train_tail 中最后一次出现的位置
INT *left_head, *right_head;
INT *left_tail, *right_tail;

// left_mean (relation_total): 记录每种关系 head 的种类数
// right_mean (relation_total): 记录每种关系 tail 的种类数
REAL *left_mean, *right_mean;

void init() {

	FILE *fin;
	INT tmp;

	// 初始化 relation_vec
	fin = fopen((in_path + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relation_total);
	fclose(fin);
	printf("relation_total: %d\n", relation_total);

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
	printf("entity_total: %d\n", entity_total);

	entity_vec = (REAL *)calloc(entity_total * dimension,
			sizeof(REAL));
	for (INT i = 0; i < entity_total; i++) {
		for (INT ii = 0; ii < dimension; ii++)
			entity_vec[i * dimension + ii] =
				randn(0, 1.0 / dimension, -6 / sqrt(dimension),
					6 / sqrt(dimension));
	}

	// 读取训练集中的三元组
	fin = fopen((in_path + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &train_triple_total);
	train_head = (Triple *)calloc(train_triple_total, sizeof(Triple));
	train_tail = (Triple *)calloc(train_triple_total, sizeof(Triple));
	train_list = (Triple *)calloc(train_triple_total, sizeof(Triple));
	for (INT i = 0; i < train_triple_total; i++) {
		tmp = fscanf(fin, "%d", &train_list[i].h);
		tmp = fscanf(fin, "%d", &train_list[i].t);
		tmp = fscanf(fin, "%d", &train_list[i].r);
		train_head[i] = train_list[i];
		train_tail[i] = train_list[i];
	}
	fclose(fin);
	printf("train_triple_total: %d\n\n", train_triple_total);

	// train_head 和 train_tail 分别以 head 和 tail 排序
	std::sort(train_head, train_head + train_triple_total, cmp_head());
	std::sort(train_tail, train_tail + train_triple_total, cmp_tail());

	// 获得 left_head, right_head, left_tail, right_tail
	left_head = (INT *)calloc(entity_total, sizeof(INT));
	right_head = (INT *)calloc(entity_total, sizeof(INT));
	left_tail = (INT *)calloc(entity_total, sizeof(INT));
	right_tail = (INT *)calloc(entity_total, sizeof(INT));
	for (INT i = 1; i < train_triple_total; i++) {
		if (train_head[i].h != train_head[i - 1].h) {
			right_head[train_head[i - 1].h] = i - 1;
			left_head[train_head[i].h] = i;
		}
		if (train_tail[i].t != train_tail[i - 1].t) {
			right_tail[train_tail[i - 1].t] = i - 1;
			left_tail[train_tail[i].t] = i;
		}
	}
	right_head[train_head[train_triple_total - 1].h] = train_triple_total - 1;
	right_tail[train_tail[train_triple_total - 1].t] = train_triple_total - 1;

	// 获得 left_mean、right_mean，为 train_mode 中的 bern_flag 做准备
	// 在训练过程中，我们能够构建负三元组进行负采样
	// bern 算法能根据特定关系的 head 和 tail 种类的比值，选择构建适当的负三元组
	// train_mode 中的 bern_flag: pr = left_mean / (left_mean + right_mean)
	// 因此为训练而构建的负三元组比 = tail / (tail + head)
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
}

// ##################################################
// 加载 Pretrained Embeddings
// prerequisites: 
//     entity2vec + note + .bin
//     relation2vec + note + .bin
//     
//     or
//
//     entity2vec + note + .vec
//     relation2vec + note + .vec
// ##################################################

void load_binary() {

	// 以二进制形式加载预训练实体嵌入
	struct stat statbuf1;
	if (stat((load_path + "entity2vec" + note + ".bin").c_str(),
			&statbuf1) != -1) {  
		INT fd = open((load_path + "entity2vec" + note + ".bin").c_str(),
			O_RDONLY);
		REAL* entity_vec_tmp = (REAL*)mmap(NULL, statbuf1.st_size,
			PROT_READ, MAP_PRIVATE, fd, 0); 
		memcpy(entity_vec, entity_vec_tmp, statbuf1.st_size);
		munmap(entity_vec_tmp, statbuf1.st_size);
		close(fd);
		printf("%s", ("以二进制形式加载预训练实体嵌入 (" + load_path + "entity2vec" + note + ".bin" + ") 成功.\n").c_str());
	}

	// 以二进制形式加载预训练关系嵌入
	struct stat statbuf2;
	if (stat((load_path + "relation2vec" + note + ".bin").c_str(),
			&statbuf2) != -1) {  
		INT fd = open((load_path + "relation2vec" + note + ".bin").c_str(),
			O_RDONLY);
		REAL* relation_vec_tmp =(REAL*)mmap(NULL, statbuf2.st_size,
			PROT_READ, MAP_PRIVATE, fd, 0); 
		memcpy(relation_vec, relation_vec_tmp, statbuf2.st_size);
		munmap(relation_vec_tmp, statbuf2.st_size);
		close(fd);
		printf("%s", ("以二进制形式加载预训练关系嵌入 (" + load_path + "relation2vec" + note + ".bin" + ") 成功.\n\n").c_str());
	}
}

void load() {
	
	if (load_binary_flag) {
		load_binary();
		return;
	}
	FILE *fin;
	INT tmp;

	// 加载预训练实体嵌入
	fin = fopen((load_path + "entity2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < entity_total; i++) {
		INT last = i * dimension;
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &entity_vec[last + j]);
	}
	fclose(fin);
	printf("%s", ("加载预训练实体嵌入 (" + load_path + "entity2vec" + note + ".vec" + ") 成功.\n").c_str());

	// 加载预训练关系嵌入
	fin = fopen((load_path + "relation2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < relation_total; i++) {
		INT last = i * dimension;
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &relation_vec[last + j]);
	}
	fclose(fin);
	printf("%s", ("加载预训练关系嵌入 (" + load_path + "relation2vec" + note + ".vec" + ") 成功.\n\n").c_str());
}

// ##################################################
// Update embeddings
// ##################################################

INT Len;
INT Batch;

// 由于没有使用互斥锁、读写锁、条件变量和信号量等手段进行线程同步
// 所以 res 可能被不同线程同时访问并修改，因此 res 会比真实值略小
// 但由于 res 只是为了直观地看到损失值的变化趋势，因此不需要通过
// 线程同步（降低程序性能）获得精确结果
REAL res;

// 使用 L1 范数计算能量 d(h + l, t)
REAL calc_sum(INT e1, INT e2, INT rel) {
	REAL sum = 0;
	INT last1 = e1 * dimension;
	INT last2 = e2 * dimension;
	INT lastr = rel * dimension;
	for (INT i = 0; i < dimension; i++)
		sum += fabs(entity_vec[last2 + i] -
			entity_vec[last1 + i] - relation_vec[lastr + i]);
	return sum;
}

// 根据 d(h + l, t) 更新实体和关系嵌入
// (e1_a, rel_a, e2_a): 正三元组
// (e1_b, rel_b, e2_b): 负三元组
void gradient(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b) {
	INT lasta1 = e1_a * dimension;
	INT lasta2 = e2_a * dimension;
	INT lastar = rel_a * dimension;
	INT lastb1 = e1_b * dimension;
	INT lastb2 = e2_b * dimension;
	INT lastbr = rel_b * dimension;

	for (INT i = 0; i  < dimension; i++) {
		REAL x;

		// 尽可能让 d(e1_a, rel_a, e2_a) 接近 0
		x = (entity_vec[lasta2 + i] -
			entity_vec[lasta1 + i] - relation_vec[lastar + i]);
		if (x > 0)
			x = -alpha;
		else
			x = alpha;
		relation_vec[lastar + i] -= x;
		entity_vec[lasta1 + i] -= x;
		entity_vec[lasta2 + i] += x;

		// 尽可能让 d(e1_b, rel_b, e2_b) 远离 0
		x = (entity_vec[lastb2 + i] -
			entity_vec[lastb1 + i] - relation_vec[lastbr + i]);
		if (x > 0)
			x = alpha;
		else
			x = -alpha;
		relation_vec[lastbr + i] -=  x;
		entity_vec[lastb1 + i] -= x;
		entity_vec[lastb2 + i] += x;
	}
}

// 损失函数 L = [margin + d(e1_a, rel_a, e2_a) - d(e1_b, rel_b, e2_b)]+
// 当 L > 0，说明 (d(e1_b, rel_b, e2_b) - d(e1_a, rel_a, e2_a)) < margin，
// 进而，正负三元组的实体和关系嵌入需要 update
void train_kb(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b) {
	REAL sum1 = calc_sum(e1_a, e2_a, rel_a);
	REAL sum2 = calc_sum(e1_b, e2_b, rel_b);
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
	}
}

// ##################################################
// 构建负三元组
// ##################################################

// 用 head 和 relationship 构建负三元组，即替换 tail
// 该函数返回负三元组的 tail
INT corrupt_with_head(INT id, INT h, INT r) {
	INT lef, rig, mid, ll, rr;

	// lef: head(h) 在 train_head 中第一次出现的前一个位置
	// rig: head(h) 在 train_head 中最后一次出现的位置
	lef = left_head[h] - 1;
	rig = right_head[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 >= -> rig，所以 rig 最终在第一个 r 的位置
		if (train_head[mid].r >= r) rig = mid; else lef = mid;
	}
	ll = rig;

	lef = left_head[h];
	rig = right_head[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 <= -> lef，所以 lef 最终在最后一个 r 的位置
		if (train_head[mid].r <= r) lef = mid; else rig = mid;
	}
	rr = lef;

	// 只能产生 (entity_total - (rr - ll + 1)) 种实体，即去掉训练集中已有的三元组
	INT tmp = rand_max(id, entity_total - (rr - ll + 1));

	// 第一种：tmp 小于第一个 r 对应的 tail
	if (tmp < train_head[ll].t) return tmp;

	// 第二种：tmp 大于最后一个 r 对应的 tail
	if (tmp > train_head[rr].t - rr + ll - 1) return tmp + rr - ll + 1;

	// 第三种：由于 (>= -> rig), (lef + 1 < rig), (tmp + lef - ll + 1)
	// 因此最终返回取值为 (train_head[lef].t, train_head[rig].t) 的 tail
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

// 用 tail 和 relationship 构建负三元组，即替换 head
// 该函数返回负三元组的 head
INT corrupt_with_tail(INT id, INT t, INT r) {
	INT lef, rig, mid, ll, rr;

	// lef: tail(t) 在 train_tail 中第一次出现的前一个位置
	// rig: tail(t) 在 train_tail 中最后一次出现的位置
	lef = left_tail[t] - 1;
	rig = right_tail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 >= -> rig，所以 rig 最终在第一个 r 的位置
		if (train_tail[mid].r >= r) rig = mid; else lef = mid;
	}
	ll = rig;
	lef = left_tail[t];
	rig = right_tail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 <= -> lef，所以 lef 最终在最后一个 r 的位置
		if (train_tail[mid].r <= r) lef = mid; else rig = mid;
	}
	rr = lef;

	// 只能产生 (entity_total - (rr - ll + 1)) 种实体，即去掉训练集中已有的三元组
	INT tmp = rand_max(id, entity_total - (rr - ll + 1));

	// 第一种：tmp 小于第一个 r 对应的 head
	if (tmp < train_tail[ll].h) return tmp;

	// 第二种：tmp 大于最后一个 r 对应的 head
	if (tmp > train_tail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;

	// 第三种：由于 (>= -> rig), (lef + 1 < rig), (tmp + lef - ll + 1)
	// 因此最终返回取值为 (train_tail[lef].h, train_tail[rig].h) 的 head
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

// ##################################################
// 多个线程训练
// ##################################################

// 单个线程内运行的任务
void* train_mode(void *thread_id) {
	INT id, pr, i, j;

	// id: 线程 ID
	id = (unsigned long long)(thread_id);
	next_random[id] = rand();

	// 每一个 Batch 被多个线程同时训练
	for (INT k = Batch / threads; k >= 0; k--) {
		i = rand_max(id, Len);
		if (bern_flag)
			pr = 1000 * left_mean[train_list[i].r] /
				(left_mean[train_list[i].r] + right_mean[train_list[i].r]);
		else
			pr = 500;
		if (randd(id) % 1000 < pr) {
			
			// 通过 h, r 构造出负三元组
			j = corrupt_with_head(id, train_list[i].h, train_list[i].r);
			train_kb(train_list[i].h, train_list[i].t, train_list[i].r,
				train_list[i].h, j, train_list[i].r);
		} else {

			// 通过 t, r 构造出负三元组
			j = corrupt_with_tail(id, train_list[i].t, train_list[i].r);
			train_kb(train_list[i].h, train_list[i].t, train_list[i].r,
				j, train_list[i].t, train_list[i].r);
		}

		// 对于 entity_vec 和 relation_vec 进行归一化
		norm(relation_vec + dimension * train_list[i].r);
		norm(entity_vec + dimension * train_list[i].h);
		norm(entity_vec + dimension * train_list[i].t);
		norm(entity_vec + dimension * j);
	}

	pthread_exit(NULL);
}

// 训练函数
void* train() {
	Len = train_triple_total;
	Batch = Len / nbatches;
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));

	for (INT epoch = 1; epoch <= epochs; epoch++) {
		res = 0;
		for (INT batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
			for (long a = 0; a < threads; a++)
				pthread_create(&pt[a], NULL, train_mode,  (void*)a);
			for (long a = 0; a < threads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}
		
		if (epoch % 50 == 0)
			printf("Epoch %d/%d - loss: %f\n", epoch, epochs, res);
	}
}

// ##################################################
// 输出实体嵌入和关系嵌入
// output: 
//     entity2vec + note + .bin
//     relation2vec + note + .bin
//     
//     or
//
//     entity2vec + note + .vec
//     relation2vec + note + .vec
// ##################################################

void out_binary() {
		
	INT len, tot;
	REAL *head;	
	FILE* f1 = fopen((out_path + "entity2vec" + note + ".bin").c_str(), "wb");
	FILE* f2 = fopen((out_path + "relation2vec" + note + ".bin").c_str(), "wb");

	// 以二进制形式输出实体嵌入
	len = entity_total * dimension; tot = 0;
	head = entity_vec;
	while (tot < len) {
		INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f1);
		tot = tot + sum;
	}
	printf("%s", ("\n以二进制形式输出实体嵌入 (" + out_path + "entity2vec" + note + ".bin" + ") 成功.\n").c_str());

	// 以二进制形式输出关系嵌入
	len = relation_total * dimension; tot = 0;
	head = relation_vec;
	while (tot < len) {
		INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f2);
		tot = tot + sum;
	}
	printf("%s", ("以二进制形式输出关系嵌入 (" + out_path + "relation2vec" + note + ".bin" + ") 成功.\n").c_str());
		
	fclose(f1);
	fclose(f2);
}

void out() {

	if (out_binary_flag) {
		out_binary(); 
		return;
	}

	FILE* f1 = fopen((out_path + "entity2vec" + note + ".vec").c_str(), "w");
	FILE* f2 = fopen((out_path + "relation2vec" + note + ".vec").c_str(), "w");

	// 输出预训练实体嵌入
	for (INT  i = 0; i < entity_total; i++) {
		INT last = i * dimension;
		for (INT j = 0; j < dimension; j++)
			fprintf(f1, "%.6f\t", entity_vec[last + j] );
		fprintf(f1,"\n");
	}
	printf("%s", ("\n输出预训练实体嵌入 (" + out_path + "entity2vec" + note + ".vec" + ") 成功.\n").c_str());

	// 输出预训练关系嵌入
	for (INT i = 0; i < relation_total; i++) {
		INT last = dimension * i;
		for (INT j = 0; j < dimension; j++)
			fprintf(f2, "%.6f\t", relation_vec[last + j]);
		fprintf(f2,"\n");
	}
	printf("%s", ("输出预训练关系嵌入 (" + out_path + "relation2vec" + note + ".vec" + ") 成功.\n").c_str());

	fclose(f1);
	fclose(f2);
}

// ##################################################
// Main function
// ##################################################

// 寻找特定参数的位置
INT arg_pos(char *str, INT argc, char **argv) {
	INT a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

void setparameters(INT argc, char **argv) {
	INT i;
	if ((i = arg_pos((char *)"-bern", argc, argv)) > 0) bern_flag = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-load-binary", argc, argv)) > 0) load_binary_flag = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-out-binary", argc, argv)) > 0) out_binary_flag = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = arg_pos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
	if ((i = arg_pos((char *)"-nbatches", argc, argv)) > 0) nbatches = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-epochs", argc, argv)) > 0) epochs = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) threads = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-input", argc, argv)) > 0) in_path = argv[i + 1];
	if ((i = arg_pos((char *)"-output", argc, argv)) > 0) out_path = argv[i + 1];
	if ((i = arg_pos((char *)"-load", argc, argv)) > 0) load_path = argv[i + 1];	
	if ((i = arg_pos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
}

// ##################################################
// ./transE [-bern 0/1] [-load-binary 0/1] [-out-binary 0/1]
//          [-size SIZE] [-alpha ALPHA] [-margin MARGIN]
//          [-nbatches NBATCHES] [-epochs EPOCHS]
//          [-threads THREAD] [-input INPUT] [-output OUTPUT]
//          [-load LOAD] [-note NOTE]

// optional arguments:
// -bern [0/1]          [1] 使用 bern 算法进行负采样，默认值为 [1]
// -load-binary [0/1]   [1] 以二进制形式加载预训练嵌入，默认值为 [0]
// -out-binary [0/1]    [1] 以二进制形式输出嵌入，默认值为 [0]
// -size SIZE           实体和关系嵌入维度，默认值为 [50]
// -alpha ALPHA         学习率，默认值为 0.01
// -margin MARGIN       margin in max-margin loss for pairwise training，默认值为 1.0
// -nbatches NBATCHES   number of batches for each epoch. if unspecified, nbatches will default to 1
// -epochs EPOCHS       number of epochs. if unspecified, epochs will default to 1000
// -threads THREAD      number of worker threads. if unspecified, threads will default to 32
// -input INPUT         folder of training data. if unspecified, in_path will default to "../data/FB15K/"
// -output OUTPUT       folder of outputing results. if unspecified, out_path will default to "./build/"
// -load LOAD           folder of pretrained data. if unspecified, load_path will default to ""
// -note NOTE           information you want to add to the filename. if unspecified, note will default to ""
// ##################################################

INT main(INT argc, char **argv) {

	printf("##################################################\n\n");
	printf("训练开始:\n\n");

	struct timeval start, end;
	gettimeofday(&start, NULL);

	setparameters(argc, argv);
	init();
	if (load_path != "") load();
	train();
	if (out_path != "") out();
	
	gettimeofday(&end, NULL);
	long double time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

	printf("\n训练结束, 用时 %.6Lf 秒.\n\n", time_use/1000000.0);
	printf("##################################################\n\n");

	return 0;
}
