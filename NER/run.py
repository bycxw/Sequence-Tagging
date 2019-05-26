# coding: utf-8


def LSTM_train_eval(train_file, dev_file, test_file):

    train_sents_list, train_tags_list, word2id, tag2id = read_data(train_file)
    dev_sents_list, dev_tags_list = read_data(dev_file, build_vocab=False)
    test_sents_list, test_tags_list = read_data(test_file, build_vocab=False)
    
    vocab_size = len(word2id)
    tag_size = len(tag2id)
    emb_dim = 200
    hidden_dim = 100

    epoches = 5
    B = 32

    model = LSTMTagger(vocab_size, emb_dim, hidden_dim, tag_size)

    for e in range(d1, epoches+1):
        losses = 0
        for ind in range(0, len(train_sents_list), B)
            batch_sents = train_sents_list[ind: ind+B]
            batch_tags = train_tags_list[ind: ind+B]
            model.train()
            batch_tensor, lengths = tensorize(batch_sents, word2id)
            tag_tensor, lengths = tensorize(batch_tags, tag2id)

            scores = model(batch_tensor, lengths)
    
    



    model = LSTM()

