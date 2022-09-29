// test_transE.cpp
// 使用方法:
//     编译:
//           $ g++ test_transE.cpp -o ./build/test_transE -pthread -O3 -march=native
//     运行:
//           $ ./build/test_transE
//
// created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>
//
// 该 C++ 文件用于模型测试
//
// prerequisites: 
//     relation2id.txt, entity2id.txt, test2id_all.txt
//     train2id.txt、valid2id.txt、type_constrain.txt
//
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
// 包含标准库
// ##################################################

#include <cstdio>           // FILE, fscanf, fopen, fclose
#include <cstdlib>          // calloc, free, atoi
#include <cmath>            // fabs
#include <cstring>          // memcpy, strcmp, memset
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

INT load_binary_flag = 0;
INT dimension = 50;
INT threads = 32;

std::string in_path = "../data/FB15K/";
std::string load_path = "./build/";
std::string note = "";

// 三元组: (head, label, tail)
// h: head
// r: label or relationship
// t: tail
// label(head-tail, relationship type):
//     0: 1-1
//     1: 1-n
//     2: n-1
//     3: n-n
// a relationship of name label between the entities head and tail
struct Triple {
	INT h, r, t;
	INT label;
};

// 为 std::sort() 定义比较仿函数
// 以三元组的 head 进行比较
struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)
		        ||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

// ##################################################
// 从 test2id_all.txt、train2id.txt、valid2id.txt 中读取三元组
// prerequisites: 
//     relation2id.txt, entity2id.txt, test2id_all.txt
//     train2id.txt、valid2id.txt、type_constrain.txt
// ##################################################

// relation_total: 关系总数
// entity_total: 实体总数
INT relation_total;
INT entity_total;

// relation_vec (relation_total * dimension): 关系嵌入矩阵
// entity_vec (entity_total * dimension): 实体嵌入矩阵
REAL *relation_vec, *entity_vec;

// test_total: 测试集中的三元组总数
// train_total: 训练集中的三元组总数
// valid_total: 验证集中的三元组总数
// triple_total: 测试集、训练集、验证集中的三元组总数，以 head 排序
INT test_total, train_total, valid_total, triple_total;

// test_list (test_total): 测试集中的三元组集合
// triple_list (triple_total): 测试集、训练集、验证集中的三元组集合
Triple *test_list, *triple_list;

// 统计测试集中各种三元组 (关系: 1-1, 1-n, n-1, n-n) 的数量
// nntotal[1]: 1-1, nntotal[2]: 1-n, nntotal[3]: n-1, nntotal[4]: n-n
INT nntotal[5];

// head_type: 存储各个关系的 head 类型, 各个关系的 head 类型独立地以升序排列
// tail_type: 存储各个关系的 tail 类型, 各个关系的 tail 类型独立地以升序排列
INT head_type[1000000];
INT tail_type[1000000];

// head_left: 记录各个关系的 head 类型在 head_type 中第一次出现的位置
// head_right: 记录各个关系的 head 类型在 head_type 中最后一次出现的后一个位置
// tail_left: 记录各个关系的 tail 类型在 tail_type 中第一次出现的位置
// tail_right: 记录各个关系的 tail 类型在 tail_type 中最后一次出现的后一个位置
INT head_left[10000];
INT head_right[10000];
INT tail_left[10000];
INT tail_right[10000];

void init() {

	FILE *fin;
	INT tmp, h, r, t, label;

	// 为 relation_vec 分配一个内存块，并将其所有位初始化为零
	fin = fopen((in_path + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relation_total);
	fclose(fin);
	relation_vec = (REAL *)calloc(relation_total * dimension, sizeof(REAL));

	// 为 entity_vec 分配一个内存块，并将其所有位初始化为零
	fin = fopen((in_path + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entity_total);
	fclose(fin);
	entity_vec = (REAL *)calloc(entity_total * dimension, sizeof(REAL));

	// 读取测试集、训练集、验证集中的三元组
	FILE* f_kb1 = fopen((in_path + "test2id_all.txt").c_str(), "r");
	FILE* f_kb2 = fopen((in_path + "train2id.txt").c_str(), "r");
	FILE* f_kb3 = fopen((in_path + "valid2id.txt").c_str(), "r");

	tmp = fscanf(f_kb1, "%d", &test_total);
	tmp = fscanf(f_kb2, "%d", &train_total);
	tmp = fscanf(f_kb3, "%d", &valid_total);
	triple_total = test_total + train_total + valid_total;
	test_list = (Triple *)calloc(test_total, sizeof(Triple));
	triple_list = (Triple *)calloc(triple_total, sizeof(Triple));

	// 将 nntotal 的内存初始化为 0
	memset(nntotal, 0, sizeof(nntotal));

	for (INT i = 0; i < test_total; i++) {
		tmp = fscanf(f_kb1, "%d", &label);
		tmp = fscanf(f_kb1, "%d", &h);
		tmp = fscanf(f_kb1, "%d", &t);
		tmp = fscanf(f_kb1, "%d", &r);
		label++;
		nntotal[label]++;
		test_list[i].label = label;
		test_list[i].h = h;
		test_list[i].t = t;
		test_list[i].r = r;
		triple_list[i].h = h;
		triple_list[i].t = t;
		triple_list[i].r = r;
	}

	for (INT i = 0; i < train_total; i++) {
		tmp = fscanf(f_kb2, "%d", &h);
		tmp = fscanf(f_kb2, "%d", &t);
		tmp = fscanf(f_kb2, "%d", &r);
		triple_list[i + test_total].h = h;
		triple_list[i + test_total].t = t;
		triple_list[i + test_total].r = r;
	}

	for (INT i = 0; i < valid_total; i++) {
		tmp = fscanf(f_kb3, "%d", &h);
		tmp = fscanf(f_kb3, "%d", &t);
		tmp = fscanf(f_kb3, "%d", &r);
		triple_list[i + test_total + train_total].h = h;
		triple_list[i + test_total + train_total].t = t;
		triple_list[i + test_total + train_total].r = r;
	}

	fclose(f_kb1);
	fclose(f_kb2);
	fclose(f_kb3);

	// triple_list 用 head 排序
	std::sort(triple_list, triple_list + triple_total, cmp_head());

	// type_constrain.txt: 类型约束文件, 第一行是关系的个数
	// 下面的行是每个关系的类型限制 (训练集、验证集、测试集中每个关系存在的 head 和 tail 的类型)
	// 每个关系有两行：
	// 第一行：`id of relation` `Number of head types` `head1` `head2` ...
	// 第二行: `id of relation` `number of tail types` `tail1` `tail2` ...
	//
	// For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733
	// The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088
	// 1200	4	3123	1034	58	5733
	// 1200	4	12123	4388	11087	11088
	INT total_left = 0;
	INT total_right = 0;
	FILE* f_type = fopen((in_path + "type_constrain.txt").c_str(), "r");
	tmp = fscanf(f_type, "%d", &relation_total);
	
	for (INT i = 0; i < relation_total; i++) {
		INT rel, tot;
		tmp = fscanf(f_type, "%d%d", &rel, &tot);
		head_left[rel] = total_left;
		for (INT j = 0; j < tot; j++) {
			tmp = fscanf(f_type, "%d", &head_type[total_left]);
			total_left++;
		}
		head_right[rel] = total_left;
		std::sort(head_type + head_left[rel], head_type + head_right[rel]);

		tmp = fscanf(f_type, "%d%d", &rel, &tot);
		tail_left[rel] = total_right;
		for (INT j = 0; j < tot; j++) {
			tmp = fscanf(f_type, "%d", &tail_type[total_right]);
			total_right++;
		}
		tail_right[rel] = total_right;
		std::sort(tail_type + tail_left[rel], tail_type + tail_right[rel]);
	}
	fclose(f_type);
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
		REAL* relation_vec_tmp = (REAL*)mmap(NULL, statbuf2.st_size,
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
// 多个线程测试
// ##################################################

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

// 检查数据集中是否存在 (h, t, r)
bool find(INT h, INT t, INT r) {
	INT lef = 0;
	INT rig = triple_total - 1;
	INT mid;
	while (lef + 1 < rig) {
		INT mid = (lef + rig) >> 1;
		if ((triple_list[mid].h < h) 
			|| (triple_list[mid].h == h && triple_list[mid].r < r) 
			|| (triple_list[mid].h == h && triple_list[mid].r == r 
					&& triple_list[mid].t < t)) 
				lef = mid; else rig = mid;
	}
	if (triple_list[lef].h == h && triple_list[lef].r == r 
		&& triple_list[lef].t == t) return true;
	if (triple_list[rig].h == h && triple_list[rig].r == r 
		&& triple_list[rig].t == t) return true;
	return false;
}

// l_raw_tot, l_filter_tot, r_raw_tot, r_filter_tot 的形状为 [6][threads]
// l_raw_rank, l_filter_rank, r_raw_rank, r_filter_rank 的形状为 [6][threads]
// 第一维度:
// 0: 代表全部测试集的结果
// 1: 代表关系为 1-1 的测试三元组的结果
// 2: 代表关系为 1-n 的测试三元组的结果
// 3: 代表关系为 n-1 的测试三元组的结果
// 4: 代表关系为 n-n 的测试三元组的结果
// 5: 代表全部测试集的结果, 通过 type_constrain.txt 来构造负三元组
// 第二维度:
// 0 ~ (threads - 1): 线程 ID
// l_raw_tot: 记录排名前 10 的 (替换 head 生成负三元组) 测试三元组个数
// l_filter_tot: 记录排名前 10 的 (替换 head 生成负三元组) 测试三元组个数, 且负三元组不在数据集中
// r_raw_tot: 记录排名前 10 的 (替换 tail 生成负三元组) 测试三元组个数
// r_filter_tot: 记录排名前 10 的 (替换 tail 生成负三元组) 测试三元组个数, 且负三元组不在数据集中
// l_raw_rank: 记录 (替换 head 生成负三元组) 测试三元组的排名总和 (排名从 0 开始)
// l_filter_rank: 记录 (替换 head 生成负三元组) 测试三元组的排名总和 (排名从 0 开始), 且负三元组不在数据集中
// r_raw_rank: 记录 (替换 tail 生成负三元组) 测试三元组的排名总和 (排名从 0 开始)
// r_filter_rank: 记录 (替换 tail 生成负三元组) 测试三元组的排名总和 (排名从 0 开始), 且负三元组不在数据集中
REAL *l_raw_tot[6], *l_filter_tot[6], *r_raw_tot[6], *r_filter_tot[6];
REAL *l_raw_rank[6], *l_filter_rank[6], *r_raw_rank[6], *r_filter_rank[6];

// 单个线程内运行的任务
void* test_mode(void *thread_id) {
	INT id;

	// id: 线程 ID
	id = (unsigned long long)(thread_id);
	INT lef = test_total / (threads) * id;
	INT rig = test_total / (threads) * (id + 1) - 1;
	if (id == threads - 1) rig = test_total - 1;

	for (INT i = lef; i <= rig; i++) {

		INT h = test_list[i].h;
		INT t = test_list[i].t;
		INT r = test_list[i].r;
		INT label = test_list[i].label;

		REAL minimal = calc_sum(h, t, r);

		// l_raw: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 head) 负三元组个数
		// l_filter: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 head) 负三元组个数, 且负三元组不在数据集中
		// r_raw: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 tail) 负三元组个数
		// r_filter: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 tail) 负三元组个数, 且负三元组不在数据集中
		INT l_raw = 0;
		INT l_filter = 0;
		INT r_raw = 0;
		INT r_filter = 0;

		// l_raw_constrain: 记录能量 (d(h + l, t)) 小于测试三元组的 (通过 type_constrain.txt 替换 head 构造负三元组) 负三元组个数
		// l_filter_constrain: 记录能量 (d(h + l, t)) 小于测试三元组的 (通过 type_constrain.txt 替换 head 构造负三元组) 负三元组个数, 且负三元组不在数据集中
		// r_raw_constrain: 记录能量 (d(h + l, t)) 小于测试三元组的 (通过 type_constrain.txt 替换 tail 构造负三元组) 负三元组个数
		// r_filter_constrain: 记录能量 (d(h + l, t)) 小于测试三元组的 (通过 type_constrain.txt 替换 tail 构造负三元组) 负三元组个数, 且负三元组不在数据集中
		INT l_raw_constrain = 0;
		INT l_filter_constrain = 0;
		INT r_raw_constrain = 0;
		INT r_filter_constrain = 0;

		// left_head_type: 记录关系 r 的 head 类型在 head_type 中第一次出现的位置
		// left_tail_type: 记录关系 r 的 tail 类型在 tail_type 中第一次出现的位置
		INT left_head_type = head_left[r], left_tail_type = tail_left[r];
		for (INT j = 0; j < entity_total; j++) {

			// 替换 head
			if (j != h) {
				REAL value = calc_sum(j, t, r);
				if (value < minimal) {
					l_raw += 1;
					if (not find(j, t, r))
						l_filter += 1;
				}
				while (left_head_type < head_right[r] && head_type[left_head_type] < j) left_head_type++;
				if (left_head_type < head_right[r] && head_type[left_head_type] == j) {
					if (value < minimal) {
						l_raw_constrain += 1;
						if (not find(j, t, r))
							l_filter_constrain += 1;
					}
				}
			}

			// 替换 tail
			if (j != t) {
				REAL value = calc_sum(h, j, r);
				if (value < minimal) {
					r_raw += 1;
					if (not find(h, j, r))
						r_filter += 1;
				}
				while (left_tail_type < tail_right[r] && tail_type[left_tail_type] < j) left_tail_type++;
				if (left_tail_type < tail_right[r] && tail_type[left_tail_type] == j) {
					if (value < minimal) {
						r_raw_constrain += 1;
						if (not find(h, j, r))
							r_filter_constrain += 1;
					}
				}
			}
		}
		
		// 全部测试集
		if (l_raw < 10) l_raw_tot[0][id] += 1;
		if (l_filter < 10) l_filter_tot[0][id] += 1;
		if (r_raw < 10) r_raw_tot[0][id] += 1;
		if (r_filter < 10) r_filter_tot[0][id] += 1;

		l_raw_rank[0][id] += l_raw;
		l_filter_rank[0][id] += l_filter;
		r_raw_rank[0][id] += r_raw;
		r_filter_rank[0][id] += r_filter;

		// 1-1, 1-n, n-1, n-n
		if (l_raw < 10) l_raw_tot[label][id] += 1;
		if (l_filter < 10) l_filter_tot[label][id] += 1;
		if (r_raw < 10) r_raw_tot[label][id] += 1;
		if (r_filter < 10) r_filter_tot[label][id] += 1;

		l_raw_rank[label][id] += l_raw;
		l_filter_rank[label][id] += l_filter;
		r_raw_rank[label][id] += r_raw;
		r_filter_rank[label][id] += r_filter;

		// 全部测试集的结果, 通过 type_constrain.txt 来构造负三元组
		if (l_raw_constrain < 10) l_raw_tot[5][id] += 1;
		if (l_filter_constrain < 10) l_filter_tot[5][id] += 1;
		if (r_raw_constrain < 10) r_raw_tot[5][id] += 1;
		if (r_filter_constrain < 10) r_filter_tot[5][id] += 1;

		l_raw_rank[5][id] += l_raw_constrain;
		l_filter_rank[5][id] += l_filter_constrain;
		r_raw_rank[5][id] += r_raw_constrain;
		r_filter_rank[5][id] += r_filter_constrain;
	}

	pthread_exit(NULL);
}

// 测试函数
void* test() {

	for (INT i = 0; i <= 5; i++) {

		l_raw_tot[i] = (REAL *)calloc(threads, sizeof(REAL));
		l_filter_tot[i] = (REAL *)calloc(threads, sizeof(REAL));
		r_raw_tot[i] = (REAL *)calloc(threads, sizeof(REAL));
		r_filter_tot[i] = (REAL *)calloc(threads, sizeof(REAL));

		l_raw_rank[i] = (REAL *)calloc(threads, sizeof(REAL));
		l_filter_rank[i] = (REAL *)calloc(threads, sizeof(REAL));
		r_raw_rank[i] = (REAL *)calloc(threads, sizeof(REAL));
		r_filter_rank[i] = (REAL *)calloc(threads, sizeof(REAL));
		
	}

	// 开启多线程测试
	pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
	for (long a = 0; a < threads; a++)
		pthread_create(&pt[a], NULL, test_mode, (void*)a);
	for (long a = 0; a < threads; a++)
		pthread_join(pt[a], NULL);
	free(pt);

	// 将各个线程的结果累加
	for (INT i = 0; i <= 5; i++)
		for (INT a = 1; a < threads; a++) {

			l_raw_tot[i][a] += l_raw_tot[i][a - 1];
			l_filter_tot[i][a] += l_filter_tot[i][a - 1];
			r_raw_tot[i][a] += r_raw_tot[i][a - 1];
			r_filter_tot[i][a] += r_filter_tot[i][a - 1];

			l_raw_rank[i][a] += l_raw_rank[i][a - 1];
			l_filter_rank[i][a] += l_filter_rank[i][a - 1];
			r_raw_rank[i][a] += r_raw_rank[i][a - 1];
			r_filter_rank[i][a] += r_filter_rank[i][a - 1];
			
		}
	
	// 总体结果
	printf("总体结果：\n\n");
	for (INT i = 0; i <= 0; i++) {
		printf("heads(raw) \t\t平均排名: %f, \tHits@10: %f\n", l_raw_rank[i][threads - 1] / test_total,
			l_raw_tot[i][threads - 1] / test_total);
		printf("heads(filter) \t\t平均排名: %f, \tHits@10: %f\n", l_filter_rank[i][threads - 1] / test_total,
			l_filter_tot[i][threads - 1] / test_total);
		printf("tails(raw) \t\t平均排名: %f, \tHits@10: %f\n", r_raw_rank[i][threads - 1] / test_total,
			r_raw_tot[i][threads - 1] / test_total);
		printf("tails(filter) \t\t平均排名: %f, \tHits@10: %f\n", r_filter_rank[i][threads - 1] / test_total,
			r_filter_tot[i][threads - 1] / test_total);
	}

	// 通过 type_constrain.txt 限制的总体结果
	printf("\n通过 type_constrain.txt 限制的总体结果：\n\n");
	for (INT i = 5; i <= 5; i++) {
		printf("heads(raw) \t\t平均排名: %f, \tHits@10: %f\n", l_raw_rank[i][threads - 1] / test_total,
			l_raw_tot[i][threads - 1] / test_total);
		printf("heads(filter) \t\t平均排名: %f, \tHits@10: %f\n", l_filter_rank[i][threads - 1] / test_total,
			l_filter_tot[i][threads - 1] / test_total);
		printf("tails(raw) \t\t平均排名: %f, \tHits@10: %f\n", r_raw_rank[i][threads - 1] / test_total,
			r_raw_tot[i][threads - 1] / test_total);
		printf("tails(filter) \t\t平均排名: %f, \tHits@10: %f\n", r_filter_rank[i][threads - 1] / test_total,
			r_filter_tot[i][threads - 1] / test_total);
	}

	// (关系: 1-1, 1-n, n-1, n-n) 测试三元组的结果
	printf("\n(关系: 1-1, 1-n, n-1, n-n) 测试三元组的结果：\n");

	std::string relation[] = {
		"关系: 1-1",
		"关系: 1-n",
		"关系: n-1",
		"关系: n-n"
	};

	for (INT i = 1; i <= 4; i++) {

		printf("\n%s:\n\n", relation[i - 1].c_str());

		printf("heads(raw) \t\t平均排名: %f, \tHits@10: %f\n", l_raw_rank[i][threads - 1] / nntotal[i],
			l_raw_tot[i][threads - 1] / nntotal[i]);
		printf("heads(filter) \t\t平均排名: %f, \tHits@10: %f\n", l_filter_rank[i][threads - 1] / nntotal[i],
			l_filter_tot[i][threads - 1] / nntotal[i]);
		printf("tails(raw) \t\t平均排名: %f, \tHits@10: %f\n", r_raw_rank[i][threads - 1] / nntotal[i],
			r_raw_tot[i][threads - 1] / nntotal[i]);
		printf("tails(filter) \t\t平均排名: %f, \tHits@10: %f\n", r_filter_rank[i][threads - 1] / nntotal[i],
			r_filter_tot[i][threads - 1] / nntotal[i]);
		
	}
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
	if ((i = arg_pos((char *)"-load-binary", argc, argv)) > 0) load_binary_flag = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) threads = atoi(argv[i + 1]);
	if ((i = arg_pos((char *)"-input", argc, argv)) > 0) in_path = argv[i + 1];
	if ((i = arg_pos((char *)"-load", argc, argv)) > 0) load_path = argv[i + 1];
	if ((i = arg_pos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
}

// ##################################################
// ./test_transE [-load-binary 0/1] [-size SIZE]
//          [-threads THREAD] [-input INPUT]
//          [-load LOAD] [-note NOTE]

// optional arguments:
// -load-binary [0/1]   [1] 以二进制形式加载预训练嵌入，默认值为 [0]
// -size SIZE           实体和关系嵌入维度，默认值为 [50]
// -threads THREAD      number of worker threads. if unspecified, threads will default to 32
// -input INPUT         folder of training data. if unspecified, in_path will default to "../data/FB15K/"
// -load LOAD           folder of pretrained data. if unspecified, load_path will default to "./build/"
// -note NOTE           information you want to add to the filename. if unspecified, note will default to ""
// ##################################################

INT main(INT argc, char **argv) {

	printf("测试开始:\n\n");

	struct timeval start, end;
	gettimeofday(&start, NULL);

	setparameters(argc, argv);
	init();
	load();
	test();

	gettimeofday(&end, NULL);
	long double time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

	printf("\n测试结束, 用时 %.6Lf 秒.\n\n", time_use/1000000.0);
	printf("##################################################\n\n");

	return 0;
}
