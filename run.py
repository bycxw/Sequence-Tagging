# coding: utf-8
from model.LSTM import BiLSTMTagger
from utils import read_data, sort_by_length

def LSTM_train_eval(train_file, dev_file, test_file):

    train_sents_list, train_tags_list, word2id, tag2id = read_data(train_file)
    dev_sents_list, dev_tags_list = read_data(dev_file, build_vocab=False)
    test_sents_list, test_tags_list = read_data(test_file, build_vocab=False)
    
    train_sents_list, train_tags_list = sort_by_length(train_sents_list, train_tags_list)
    dev_sents_list, dev_tags_list = sort_by_length(dev_sents_list, dev_tags_list)

    vocab_size = len(word2id)
    tagset_size = len(tag2id)
    model = BiLSTMTagger(vocab_size, tagset_size)

    model.train(train_sents_list, train_tags_list, dev_sents_list, dev_tags_list, word2id, tag2id)

if __name__ == '__main__':
    train_file = './data/trainset/train_ner_format'
    dev_file = './data/devset/val_ner_format'
    test_file = './data/testset1/test_ner_format'
    LSTM_train_eval(train_file, dev_file, test_file)


