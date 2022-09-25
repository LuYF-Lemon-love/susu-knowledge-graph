// test_transE.cpp
// created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>

// ##################################################
// 包含标准库
// ##################################################

#include <cstdio>           // FILE, fscanf, fopen, fclose
#include <cstdlib>          // calloc, free, atoi
#include <cmath>            // fabs
#include <cstring>          // memcpy, strcmp
#include <fcntl.h>          // open, close, O_RDONLY
#include <unistd.h>         // stat
#include <sys/stat.h>       // stat
#include <sys/mman.h>       // mmap, munmap
#include <string>           // std::string, std::string::c_str
#include <pthread.h>        // pthread_create, pthread_exit, pthread_join
#include <algorithm>        // std::sort
#include <iostream>
#include <map>
#include <vector>
#include <ctime>

// ##################################################
// 声明和定义变量
// ##################################################

using namespace std;

#define INT int

INT load_binary_flag = 0;
INT dimension = 50;
INT threads = 32;

string in_path = "../data/FB15K/";
string load_path = "./";
string note = "";

// relation_total: 关系总数
// entity_total: 实体总数
INT relation_total;
INT entity_total;

float *entity_vec, *relation_vec;
INT test_total, train_total, valid_total, triple_total;

struct Triple {
	INT h, r, t;
	INT label;
};

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

Triple *testList, *tripleList;

int nntotal[5];
int head_lef[10000];
int head_rig[10000];
int tail_lef[10000];
int tail_rig[10000];
int head_type[1000000];
int tail_type[1000000];

void init() {
	FILE *fin;
	INT tmp, h, r, t, label;

	fin = fopen((in_path + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relation_total);
	fclose(fin);
	relation_vec = (float *)calloc(relation_total * dimension, sizeof(float));

	fin = fopen((in_path + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entity_total);
	fclose(fin);
	entity_vec = (float *)calloc(entity_total * dimension, sizeof(float));

	FILE* f_kb1 = fopen((in_path + "test2id_all.txt").c_str(), "r");
	FILE* f_kb2 = fopen((in_path + "train2id.txt").c_str(), "r");
	FILE* f_kb3 = fopen((in_path + "valid2id.txt").c_str(), "r");

	tmp = fscanf(f_kb1, "%d", &test_total);
	tmp = fscanf(f_kb2, "%d", &train_total);
	tmp = fscanf(f_kb3, "%d", &valid_total);
	triple_total = test_total + train_total + valid_total;
	testList = (Triple *)calloc(test_total, sizeof(Triple));
	tripleList = (Triple *)calloc(triple_total, sizeof(Triple));

	memset(nntotal, 0, sizeof(nntotal));

	for (INT i = 0; i < test_total; i++) {
		tmp = fscanf(f_kb1, "%d", &label);
		tmp = fscanf(f_kb1, "%d", &h);
		tmp = fscanf(f_kb1, "%d", &t);
		tmp = fscanf(f_kb1, "%d", &r);
		label++;
		nntotal[label]++;
		testList[i].label = label;
		testList[i].h = h;
		testList[i].t = t;
		testList[i].r = r;
		tripleList[i].h = h;
		tripleList[i].t = t;
		tripleList[i].r = r;
	}

	for (INT i = 0; i < train_total; i++) {
		tmp = fscanf(f_kb2, "%d", &h);
		tmp = fscanf(f_kb2, "%d", &t);
		tmp = fscanf(f_kb2, "%d", &r);
		tripleList[i + test_total].h = h;
		tripleList[i + test_total].t = t;
		tripleList[i + test_total].r = r;
	}

	for (INT i = 0; i < valid_total; i++) {
		tmp = fscanf(f_kb3, "%d", &h);
		tmp = fscanf(f_kb3, "%d", &t);
		tmp = fscanf(f_kb3, "%d", &r);
		tripleList[i + test_total + train_total].h = h;
		tripleList[i + test_total + train_total].t = t;
		tripleList[i + test_total + train_total].r = r;
	}

	fclose(f_kb1);
	fclose(f_kb2);
	fclose(f_kb3);

	sort(tripleList, tripleList + triple_total, cmp_head());

	INT total_lef = 0;
	INT total_rig = 0;
	FILE* f_type = fopen((in_path + "type_constrain.txt").c_str(), "r");
	tmp = fscanf(f_type, "%d", &tmp);
	
	for (int i = 0; i < relation_total; i++) {
		int rel, tot;
		tmp = fscanf(f_type, "%d%d", &rel, &tot);
		head_lef[rel] = total_lef;
		for (int j = 0; j < tot; j++) {
			tmp = fscanf(f_type, "%d", &head_type[total_lef]);
			total_lef++;
		}
		head_rig[rel] = total_lef;
		sort(head_type + head_lef[rel], head_type + head_rig[rel]);

		tmp = fscanf(f_type, "%d%d", &rel, &tot);
		tail_lef[rel] = total_rig;
		for (int j = 0; j < tot; j++) {
			tmp = fscanf(f_type, "%d", &tail_type[total_rig]);
			total_rig++;
		}
		tail_rig[rel] = total_rig;
		sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
	}
	fclose(f_type);
}

void prepare_binary() {
	struct stat statbuf1;
	if (stat((load_path + "entity2vec" + note + ".bin").c_str(), &statbuf1) != -1) {
		int fd = open((load_path + "entity2vec" + note + ".bin").c_str(), O_RDONLY);
		float* entity_vec_tmp = (float*)mmap(NULL, statbuf1.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
		memcpy(entity_vec, entity_vec_tmp, statbuf1.st_size);
		munmap(entity_vec_tmp, statbuf1.st_size);
		close(fd);
	}

	struct stat statbuf2;
	if (stat((load_path + "relation2vec" + note + ".bin").c_str(), &statbuf2) != -1) {
		int fd = open((load_path + "relation2vec" + note + ".bin").c_str(), O_RDONLY);
		float* relation_vec_tmp = (float*)mmap(NULL, statbuf2.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
		memcpy(relation_vec, relation_vec_tmp, statbuf2.st_size);
		munmap(relation_vec_tmp, statbuf2.st_size);
		close(fd);
	}
}

void prepare() {
	if (load_binary_flag) {
		prepare_binary();
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

float calc_sum(INT e1, INT e2, INT rel) {
	float res = 0;
	INT last1 = e1 * dimension;
	INT last2 = e2 * dimension;
	INT lastr = rel * dimension;
	for (INT i = 0; i < dimension; i++)
		res += fabs(entity_vec[last1 + i] + relation_vec[lastr + i] - entity_vec[last2 + i]);
	return res;
}

bool find(INT h, INT t, INT r) {
	INT lef = 0;
	INT rig = triple_total - 1;
	INT mid;
	while (lef + 1 < rig) {
		INT mid = (lef + rig) >> 1;
		if ((tripleList[mid].h < h) || (tripleList[mid].h == h && tripleList[mid].r < r) || (tripleList[mid].h == h && tripleList[mid].r == r && tripleList[mid].t < t)) lef = mid; else rig = mid;
	}
	if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
	if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
	return false;
}

float *l_filter_tot[6], *r_filter_tot[6], *l_tot[6], *r_tot[6];
float *l_filter_rank[6], *r_filter_rank[6], *l_rank[6], *r_rank[6];

void* testMode(void *con) {
	INT id;
	id = (unsigned long long)(con);
	INT lef = test_total / (threads) * id;
	INT rig = test_total / (threads) * (id + 1) - 1;
	if (id == threads - 1) rig = test_total - 1;
	for (INT i = lef; i <= rig; i++) {
		INT h = testList[i].h;
		INT t = testList[i].t;
		INT r = testList[i].r;
		INT label = testList[i].label;
		float minimal = calc_sum(h, t, r);
		INT l_filter_s = 0;
		INT l_s = 0;
		INT r_filter_s = 0;
		INT r_s = 0;
		INT l_filter_s_constrain = 0;
		INT l_s_constrain = 0;
		INT r_filter_s_constrain = 0;
		INT r_s_constrain = 0;
		INT type_head = head_lef[r], type_tail = tail_lef[r];
		for (INT j = 0; j < entity_total; j++) {
			if (j != h) {
				float value = calc_sum(j, t, r);
				if (value < minimal) {
					l_s += 1;
					if (not find(j, t, r))
						l_filter_s += 1;
				}
				while (type_head < head_rig[r] && head_type[type_head] < j) type_head++;
				if (type_head < head_rig[r] && head_type[type_head] == j) {
					if (value < minimal) {
						l_s_constrain += 1;
						if (not find(j, t, r))
							l_filter_s_constrain += 1;
					}
				}
			}
			if (j != t) {
				float value = calc_sum(h, j, r);
				if (value < minimal) {
					r_s += 1;
					if (not find(h, j, r))
						r_filter_s += 1;
				}
				while (type_tail < tail_rig[r] && tail_type[type_tail] < j) type_tail++;
				if (type_tail < tail_rig[r] && tail_type[type_tail] == j) {
					if (value < minimal) {
						r_s_constrain += 1;
						if (not find(h, j, r))
							r_filter_s_constrain += 1;
					}
				}
			}
		}
		if (l_filter_s < 10) l_filter_tot[0][id] += 1;
		if (l_s < 10) l_tot[0][id] += 1;
		if (r_filter_s < 10) r_filter_tot[0][id] += 1;
		if (r_s < 10) r_tot[0][id] += 1;

		l_filter_rank[0][id] += l_filter_s;
		r_filter_rank[0][id] += r_filter_s;
		l_rank[0][id] += l_s;
		r_rank[0][id] += r_s;

		if (l_filter_s < 10) l_filter_tot[label][id] += 1;
		if (l_s < 10) l_tot[label][id] += 1;
		if (r_filter_s < 10) r_filter_tot[label][id] += 1;
		if (r_s < 10) r_tot[label][id] += 1;

		l_filter_rank[label][id] += l_filter_s;
		r_filter_rank[label][id] += r_filter_s;
		l_rank[label][id] += l_s;
		r_rank[label][id] += r_s;

		if (l_filter_s_constrain < 10) l_filter_tot[5][id] += 1;
		if (l_s_constrain < 10) l_tot[5][id] += 1;
		if (r_filter_s_constrain < 10) r_filter_tot[5][id] += 1;
		if (r_s_constrain < 10) r_tot[5][id] += 1;

		l_filter_rank[5][id] += l_filter_s_constrain;
		r_filter_rank[5][id] += r_filter_s_constrain;
		l_rank[5][id] += l_s_constrain;
		r_rank[5][id] += r_s_constrain;
	}

	pthread_exit(NULL);
}

void* test(void *con) {
	for (int i = 0; i <= 5; i++) {
		l_filter_tot[i] = (float *)calloc(threads, sizeof(float));
		r_filter_tot[i] = (float *)calloc(threads, sizeof(float));
		l_tot[i] = (float *)calloc(threads, sizeof(float));
		r_tot[i] = (float *)calloc(threads, sizeof(float));

		l_filter_rank[i] = (float *)calloc(threads, sizeof(float));
		r_filter_rank[i] = (float *)calloc(threads, sizeof(float));
		l_rank[i] = (float *)calloc(threads, sizeof(float));
		r_rank[i] = (float *)calloc(threads, sizeof(float));
	}

	pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
	for (long a = 0; a < threads; a++)
		pthread_create(&pt[a], NULL, testMode, (void*)a);
	for (long a = 0; a < threads; a++)
		pthread_join(pt[a], NULL);
	free(pt);

	for (int i = 0; i <= 5; i++)
		for (INT a = 1; a < threads; a++) {
			l_filter_tot[i][a] += l_filter_tot[i][a - 1];
			r_filter_tot[i][a] += r_filter_tot[i][a - 1];
			l_tot[i][a] += l_tot[i][a - 1];
			r_tot[i][a] += r_tot[i][a - 1];

			l_filter_rank[i][a] += l_filter_rank[i][a - 1];
			r_filter_rank[i][a] += r_filter_rank[i][a - 1];
			l_rank[i][a] += l_rank[i][a - 1];
			r_rank[i][a] += r_rank[i][a - 1];
		}

	for (int i = 0; i <= 0; i++) {
		printf("left %f %f\n", l_rank[i][threads - 1] / test_total, l_tot[i][threads - 1] / test_total);
		printf("left(filter) %f %f\n", l_filter_rank[i][threads - 1] / test_total, l_filter_tot[i][threads - 1] / test_total);
		printf("right %f %f\n", r_rank[i][threads - 1] / test_total, r_tot[i][threads - 1] / test_total);
		printf("right(filter) %f %f\n", r_filter_rank[i][threads - 1] / test_total, r_filter_tot[i][threads - 1] / test_total);
	}

	for (int i = 5; i <= 5; i++) {
		printf("left %f %f\n", l_rank[i][threads - 1] / test_total, l_tot[i][threads - 1] / test_total);
		printf("left(filter) %f %f\n", l_filter_rank[i][threads - 1] / test_total, l_filter_tot[i][threads - 1] / test_total);
		printf("right %f %f\n", r_rank[i][threads - 1] / test_total, r_tot[i][threads - 1] / test_total);
		printf("right(filter) %f %f\n", r_filter_rank[i][threads - 1] / test_total, r_filter_tot[i][threads - 1] / test_total);
	}

	for (int i = 1; i <= 4; i++) {
		printf("left %f %f\n", l_rank[i][threads - 1] / nntotal[i], l_tot[i][threads - 1] / nntotal[i]);
		printf("left(filter) %f %f\n", l_filter_rank[i][threads - 1] / nntotal[i], l_filter_tot[i][threads - 1] / nntotal[i]);
		printf("right %f %f\n", r_rank[i][threads - 1] / nntotal[i], r_tot[i][threads - 1] / nntotal[i]);
		printf("right(filter) %f %f\n", r_filter_rank[i][threads - 1] / nntotal[i], r_filter_tot[i][threads - 1] / nntotal[i]);
	}
}

INT ArgPos(char *str, INT argc, char **argv) {
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
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) in_path = argv[i + 1];
	if ((i = ArgPos((char *)"-load", argc, argv)) > 0) load_path = argv[i + 1];
	if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-load-binary", argc, argv)) > 0) load_binary_flag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
}

int main(int argc, char **argv) {
	setparameters(argc, argv);
	init();
	prepare();
	test(NULL);
	return 0;
}
