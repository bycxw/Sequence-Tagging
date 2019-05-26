# coding: utf-8
import sys

def read_data(data_file, build_vocab=True):
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

    if build_vocab:
        word2id = {}
        tag2id = {}
        for sent in sents_list:
            for word in sent:
                if word not in word2id:
                    word2id[word] = len(word2id)
        for tags in tags_list:
            for tag in tags:
                if tag not in tag2id:
                    tag2id[tag] = len(tag2id)
        word2id['<UNK>'] = len(word2id)
        word2id['<PAD>'] = len(word2id)
        tag2id['<UNK>'] = len(tag2id)
        tag2id['<PAD>'] = len(tag2id)
        return sents_list, tags_list, word2id, tag2id
    else:
        return sents_list, tags_list

def sort_by_length(sents_list, tags_list):
    data = list(zip(sents_list, tags_list))
    data = sorted(data, key=lambda x: len(x[0]), reverse=True)
    data = list(zip(*data))
    sents_list = data[0]
    tags_list = data[1]
    return sents_list, tags_list

def tensorize(batch, maps):
    UNK = maps.get('<UNK')
    PAD = maps.get('<PAD>')
    batch_size = len(batch)
    max_len = len(batch[0])
    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for i, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    lengths = [len(sent) for sent in batch]

    return batch_tensor, lengths


if __name__ == '__main__':
    data_file = sys.argv[1]
    sents_list, tags_list, word2id, tag2id = read_data(data_file)
    sents_list, tags_list = sort_by_length(sents_list, tags_list)
    print(sents_list[-2:])
    lengths = []
    for sent in tags_list:
        lengths.append(len(sent))
    print(lengths[-2:])
    print(len(word2id))
    print(tag2id)

