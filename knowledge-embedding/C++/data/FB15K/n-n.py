# n-n.py
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com>

##################################################
# 从 train2id.txt, valid2id.txt、test2id.txt 读取三元组
##################################################

# lef 和 rig 类型为 {[]}, 外层是 <class 'dict'>, 内层是 <class 'list'>
# lef 外层的 key 为三元组 (训练集、验证集、测试集) (h, r)
# lef 外层的 value 为 (h, r) 对应的 t 的 list
# rig 外层的 key 为三元组 (训练集、验证集、测试集) (r, t)
# rig 外层的 value 为 (r, t) 对应的 h 的 list  
lef = {}
rig = {}

# rel_lef 和 rel_rig 类型为 {{}}, 外层是 <class 'dict'>, 内层是 <class 'dict'>
# rel_lef 外层的 key 为三元组 (训练集、验证集、测试集) r
# rel_lef 内层的 key 为 r 对应的 h, 内层的 value 为 1
# rel_rig 外层的 key 为三元组 (训练集、验证集、测试集) r
# rel_rig 内层的 key 为 r 对应的 t, 内层的 value 为 1
rel_lef = {}
rel_rig = {}

train_list = open("train2id.txt", "r")
valid_list = open("valid2id.txt", "r")
test_list = open("test2id.txt", "r")

tot = (int)(train_list.readline())
for i in range(tot):

	content = train_list.readline()
	h, t, r = content.strip().split()

	if not (h, r) in lef:
		lef[(h, r)] = []
	if not (r,t) in rig:
		rig[(r, t)] = []
	lef[(h, r)].append(t)
	rig[(r, t)].append(h)
	
	if not r in rel_lef:
		rel_lef[r] = {}
	if not r in rel_rig:
		rel_rig[r] = {}
	rel_lef[r][h] = 1
	rel_rig[r][t] = 1

tot = (int)(valid_list.readline())
for i in range(tot):

	content = valid_list.readline()
	h,t,r = content.strip().split()

	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)

	if not r in rel_lef:
		rel_lef[r] = {}
	if not r in rel_rig:
		rel_rig[r] = {}
	rel_lef[r][h] = 1
	rel_rig[r][t] = 1

tot = (int)(test_list.readline())
for i in range(tot):

	content = test_list.readline()
	h,t,r = content.strip().split()

	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)

	if not r in rel_lef:
		rel_lef[r] = {}
	if not r in rel_rig:
		rel_rig[r] = {}
	rel_lef[r][h] = 1
	rel_rig[r][t] = 1

test_list.close()
valid_list.close()
train_list.close()

##################################################
# 创建 type_constrain.txt
# type_constrain.txt: 类型约束文件, 第一行是关系的个数
# 下面的行是每个关系的类型限制 (训练集、验证集、测试集中每个关系存在的 head 和 tail 的类型)
# 每个关系有两行：
# 第一行：`id of relation` `Number of head types` `head1` `head2` ...
# 第二行: `id of relation` `number of tail types` `tail1` `tail2` ...
#
# For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733
# The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088
# 1200	4	3123	1034	58	5733
# 1200	4	12123	4388	11087	11088
##################################################

f = open("type_constrain.txt", "w")
f.write("%d\n"%(len(rel_lef)))
for i in rel_lef:
	f.write("%s\t%d"%(i, len(rel_lef[i])))
	for j in rel_lef[i]:
		f.write("\t%s"%(j))
	f.write("\n")
	f.write("%s\t%d"%(i, len(rel_rig[i])))
	for j in rel_rig[i]:
		f.write("\t%s"%(j))
	f.write("\n")
f.close()

##################################################

rel_lef = {}
totlef = {}
rel_rig = {}
totrig = {}

for i in lef:
	if not i[1] in rel_lef:
		rel_lef[i[1]] = 0
		totlef[i[1]] = 0
	rel_lef[i[1]] += len(lef[i])
	totlef[i[1]] += 1.0

for i in rig:
	if not i[0] in rel_rig:
		rel_rig[i[0]] = 0
		totrig[i[0]] = 0
	rel_rig[i[0]] += len(rig[i])
	totrig[i[0]] += 1.0

s11=0
s1n=0
sn1=0
snn=0
f = open("test2id.txt", "r")
tot = (int)(f.readline())
for i in range(tot):
	content = f.readline()
	h,t,r = content.strip().split()
	rign = rel_lef[r] / totlef[r]
	lefn = rel_rig[r] / totrig[r]
	if (rign <= 1.5 and lefn <= 1.5):
		s11+=1
	if (rign > 1.5 and lefn <= 1.5):
		s1n+=1
	if (rign <= 1.5 and lefn > 1.5):
		sn1+=1
	if (rign > 1.5 and lefn > 1.5):
		snn+=1
f.close()


f = open("test2id.txt", "r")
f11 = open("1-1.txt", "w")
f1n = open("1-n.txt", "w")
fn1 = open("n-1.txt", "w")
fnn = open("n-n.txt", "w")
fall = open("test2id_all.txt", "w")
tot = (int)(f.readline())
fall.write("%d\n"%(tot))
f11.write("%d\n"%(s11))
f1n.write("%d\n"%(s1n))
fn1.write("%d\n"%(sn1))
fnn.write("%d\n"%(snn))
for i in range(tot):
	content = f.readline()
	h,t,r = content.strip().split()
	rign = rel_lef[r] / totlef[r]
	lefn = rel_rig[r] / totrig[r]
	if (rign <= 1.5 and lefn <= 1.5):
		f11.write(content)
		fall.write("0"+"\t"+content)
	if (rign > 1.5 and lefn <= 1.5):
		f1n.write(content)
		fall.write("1"+"\t"+content)
	if (rign <= 1.5 and lefn > 1.5):
		fn1.write(content)
		fall.write("2"+"\t"+content)
	if (rign > 1.5 and lefn > 1.5):
		fnn.write(content)
		fall.write("3"+"\t"+content)
fall.close()
f.close()
f11.close()
f1n.close()
fn1.close()
fnn.close()