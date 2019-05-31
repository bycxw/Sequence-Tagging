# -*- coding:utf-8 -*-

import re
import numpy as np

from model.POS_CNN_CRF.utils import read_data
data_file = '../../data/trainset/train_pos_format'
sents_list, tags_list= read_data(data_file)


chars = {} # 统计词表
for s in sents_list:
    for c in s:
        if c in chars:
            chars[c] += 1
        else:
            chars[c] = 1

print('chars:',chars)
min_count = 2 # 过滤低频字
chars = {i:j for i,j in chars.items() if j >= min_count} # 过滤低频字
id2char = {i+1:j for i,j in enumerate(chars)} # id到字的映射
char2id = {j:i for i,j in id2char.items()} # 字到id的映射

# 共有43种tag
state_list = ['r', 'Tg', 'g', 'f', 'Rg', 'nt', 'p', 'v', 'n', 's', 'sub>', 'Dg', 'Bg', 'e', 'm', 'c', 'Vg', 'an', 'h', 'd', 'x', 'Mg', 'nr', 'w', 'l', 'ns', 'nx', 'y', 'Ng', 'j', 'q', 'u', 'i', 'a', 'o', 'z', 'Ag', 'ad', 't', 'k', 'sup>Tc', 'b', 'nrx']
id2tag = {}
for i in range(0, len(state_list)):
    id2tag[i] = state_list[i]
tag2id = {j:i for i,j in id2tag.items()}

# id2tag = ['r', 'Tg', 'g', 'f', 'Rg', 'nt', 'p', 'v', 'n', 's', 'sub>', 'Dg', 'Bg', 'e', 'm', 'c', 'Vg', 'an', 'h', 'd', 'x', 'Mg', 'nr', 'w', 'l', 'ns', 'nx', 'y', 'Ng', 'j', 'q', 'u', 'i', 'a', 'o', 'z', 'Ag', 'ad', 't', 'k', 'sup>Tc', 'b', 'nrx']
# id2tag = {0:'s', 1:'b', 2:'m', 3:'e'} # 标签（sbme）与id之间的映射

train_sents = sents_list[:-100] # 留下5000个句子做验证，剩下的都用来训练
valid_sents = sents_list[-100:]


from keras.utils import to_categorical

# from model.POS_CNN_CRF.utils import read_data
# data_file = '../../data/trainset/train_pos_format'
# sents_list, tags_list= read_data(data_file)


batch_size = 128
def train_generator(): # 定义数据生成器
    while True:
        X,Y = [],[]
        # for i,s in enumerate(train_sents): # 遍历每个句子
        for k in range (0,len(sents_list)):
            sx,sy = [],[]
            for w in sents_list[k]: # 遍历句子中的每个词
                sx.extend([char2id.get(c, 0) for c in w])  # 遍历词中的每个词
                # sx.extend([char2id.get(c, 0) for c in w]) # 遍历词中的每个字
            for w in tags_list[k]:  # 遍历句子中的每个词
                sy.extend([tag2id.get(c, 0) for c in w])

            X.append(sx)
            # print(sx)
            # print(sy)
            # print(X )
            Y.append(sy)
            if len(X) == batch_size or i == len(train_sents)-1: # 如果达到一个batch
                maxlen = max([len(x) for x in X]) # 找出最大字数
                X = [x+[0]*(maxlen-len(x)) for x in X] # 不足则补零
                Y = [y+[43]*(maxlen-len(y)) for y in Y] # 不足则补第五个标签
                yield np.array(X),to_categorical(Y, 44)
                X,Y = [],[]
    # print (X ,Y)



from model.POS_CNN_CRF.crf_model import CRF
from keras.layers import Dense, Embedding, Conv1D, Input
from keras.models import Model # 这里我们学习使用Model型的模型
import keras.backend as K # 引入Keras后端来自定义loss，注意Keras模型内的一切运算
                          # 必须要通过Keras后端完成，比如取对数要用K.log不能用np.log

embedding_size = 128
sequence = Input(shape=(None,), dtype='int32') # 建立输入层，输入长度设为None
embedding = Embedding(len(chars)+1,
                      embedding_size,
                     )(sequence) # 去掉了mask_zero=True
cnn = Conv1D(128, 3, activation='relu', padding='same')(embedding)
cnn = Conv1D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv1D(128, 3, activation='relu', padding='same')(cnn) # 层叠了3层CNN

crf = CRF(True) # 定义crf层，参数为True，自动mask掉最后一个标签
tag_score = Dense(44)(cnn) # 变成了5分类，第五个标签用来mask掉
tag_score = crf(tag_score) # 包装一下原来的tag_score

model = Model(inputs=sequence, outputs=tag_score)
model.summary()

model.compile(loss=crf.loss, # 用crf自带的loss
              optimizer='adam',
              metrics=[crf.accuracy] # 用crf自带的accuracy
             )


def max_in_dict(d): # 定义一个求字典中最大值的函数
    key,value = list(d.items())[0]
    for i,j in list(d.items())[1:]:
        if j > value:
            key,value = i,j
    return key,value


def viterbi(nodes, trans): # viterbi算法，跟前面的HMM一致
    paths = nodes[0] # 初始化起始路径
    for l in range(1, len(nodes)): # 遍历后面的节点
        paths_old,paths = paths,{}
        for n,ns in nodes[l].items(): # 当前时刻的所有节点
            max_path,max_score = '',-1e10
            for p,ps in paths_old.items(): # 截止至前一时刻的最优路径集合
                score = ns + ps + trans[p[-1]+n] # 计算新分数
                if score > max_score: # 如果新分数大于已有的最大分
                    max_path,max_score = p+n, score # 更新路径
            paths[max_path] = max_score # 储存到当前时刻所有节点的最优路径
    return max_in_dict(paths)

from keras.callbacks import Callback
from tqdm import tqdm

# 自定义Callback类
model.fit_generator(train_generator(),
                    steps_per_epoch=500,
                    epochs=10) # 训练并将evaluator加入到训练过程

model_path = 'cnn_crf_pos_model.h5'
model.save(model_path)
print('model saved t: ',model_path)


_ = model.get_weights()[-1][:43,:43] # 从训练模型中取出最新得到的转移矩阵
trans = {}
for i in state_list:
    for j in state_list:
        trans[i+j] = _[tag2id[i], tag2id[j]]


def pos_predict(s, trans): # 分词函数，也跟前面的HMM基本一致
    if not s: # 空字符直接返回
        return []
    # 字序列转化为id序列。注意，经过我们前面对语料的预处理，字符集是没有空格的，
    # 所以这里简单将空格的id跟句号的id等同起来
    sent_ids = np.array([[char2id.get(c, 0) if c != ' ' else char2id[u'。']
                          for c in s]])
    probas = model.predict(sent_ids)[0] # 模型预测
    nodes = [dict(zip(state_list, i)) for i in probas[:, :43]] # 只取前4个
    tags = viterbi(nodes, trans)[0]
    return tags

# result = pos_predict(['我','爱','你'], trans)
# print(result)
#
#
# test_file = '../../data/testset1/test_pos1_new.txt'
# fr = open(test_file,'r')
# data = fr.readlines()
# print (data[0])
# print (len(data))
# fr.close()
#
# data_seg = []
# for line in data:
#     str = ' '.join(cut(line,trans))
#     data_seg.append(str)
#
# result_file = '../../data/testset1/test_pos1_cnn_crf.txt'
# fw = open(result_file,'w')
# fw.writelines(data_seg)
# fw.close()
#
# print('test done')
# print('results saved to',result_file)
