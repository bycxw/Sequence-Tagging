# coding: utf-8

#****************************************************************
#
# utils.py
#
# 读取数据
#
#****************************************************************



def read_data(data_file):
    sents_list = []
    tags_list = []
    with open(data_file, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            try:
                if line == '\n':
                    if sentence:
                        sentence = tuple(zip(*sentence))
                        sents_list.append(sentence[0])
                        tags_list.append(sentence[1])
                        sentence = []
                else:
                    line = line.strip().split('\t')
                    sentence.append((line[0], line[1]))
            except Exception as e:
                print(e)

    return sents_list, tags_list


if __name__ == '__main__':

    data_file = '../../data/trainset/train_pos_format'
    sents_list, tags_list= read_data(data_file)

    print (sents_list[0],tags_list[0])
    print(type(sents_list))
    print (type(sents_list[0]))
    print(sents_list[1], tags_list[1])
    print (len(sents_list), len(tags_list))


