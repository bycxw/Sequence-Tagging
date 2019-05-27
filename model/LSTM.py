# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, out_size, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=bidirectional)
        if bidirectional:
            self.lin = nn.Linear(2*hidden_dim, out_size)
        else:
            self.lin = nn.Linear(hidden_dim, out_size)

    def forward(self, sents_tensor, lengths):
        embeds = self.embeddings(sents_tensor)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.lin(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=-1)
        return tag_scores


class BiLSTMTagger(object):
    def __init__(self, vocab_size, tagset_size, bidirectional=True):
        """
        训练LSTM模型
        vocab_size: 词典大小
        tagset_size: 标签种类
        bidirectional: 是否双向LSTM
        """
        # self.emb_dim = config['LSTM']['emb_dim']
        # self.hidden_dim = config['LSTM']['hidden_dim']
        self.emb_dim = 200
        self.hidden_dim = 100
    
        self.model = BiLSTM(vocab_size, self.emb_dim, self.hidden_dim, tagset_size, bidirectional=bidirectional)

        # self.epoches = config['LSTM']['epoches']
        # self.lr = config['LSTM']['lr']
        # self.batch_size = config['LSTM']['batch_size']
        # self.print_step = config['LSTM']['print_step']
        self.epoches = 2
        self.lr = 0.001
        self.batch_size = 32
        self.print_step = 200

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.loss_func = nn.NLLLoss()
        self.step = 0
        self.best_val_loss = 1e10
        self.final_model = None

    def train(self, sents_list, tags_list, dev_sents_list, dev_tags_list, word2id, tag2id):

        total_step = len(sents_list) // self.batch_size + 1
        for e in range(1, self.epoches+1):
            for index in range(0, len(sents_list), self.batch_size):
                loss = 0
                self.step += 1
                batch_sents = sents_list[index: index+self.batch_size]
                batch_tags = tags_list[index: index+self.batch_size]
                loss = self.train_one_batch(batch_sents, batch_tags, word2id, tag2id)
            
                if self.step % self.print_step == 0:
                    print('epoch: {} step/total step: {}/{} Loss: {%.4f}'.format(e, self.step, total_step, loss))
            val_loss = self.validate(dev_sents_list, dev_tags_list, word2id, tag2id)
            print('val_loss: {.4%f}'.format(val_loss))
        self.final_model = self.model

    def train_one_batch(self, batch_sents, batch_tags, word2id, tag2id):

        self.model.train()
        batch_tensor, lengths = self.tensorize(batch_sents, word2id)
        tag_tensor, lengths = self.tensorize(batch_tags, tag2id)
        scores = self.model(batch_tensor, lengths)
        print(scores.size())
        print(tag_tensor.size())
        self.model.zero_grad()
        loss = self.loss_func(scores, tag_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def validate(self, dev_sents_list, dev_tags_list, word2id, tag2id):
        self.model.eval()
        val_losses = 0
        total_val_step = len(dev_sents_list) // self.batch_size + 1 
        with torch.no_grad():
            val_step = 0
            for index in range(0, len(dev_sents_list), self.batch_size):
                val_step += 1
                batch_sents = dev_sents_list[ind: index+self.batch_size]
                batch_tags = dev_tags_list[ind: index+self.batch_size]
                batch_tensor, lengths = self.tensorize(batch_sents, word2id)
                tag_tensor, lengths = self.tensorize(batch_tags, tag2id)

                scores = self.model(batch_tensor, lengths)

                loss = self.loss_func(scores, tag_tensor)
                val_losses += loss.item()

            val_loss = val_losses / total_val_step
        
        return val_loss

    def test(self, test_sents_list, test_tag_list, word2id, tag2id):

        pass

    def tensorize(self, batch, maps):
        UNK = maps.get('<UNK')
        PAD = maps.get('<PAD>')
        batch_size = len(batch)
        max_len = len(batch[0])
        batch_tensor = torch.ones(batch_size, max_len).long() * PAD
        for i, l in enumerate(batch):
            for j, e in enumerate(l):
                batch_tensor[i][j] = maps.get(e, UNK)
        lengths = [len(sent) for sent in batch]

        return batch_tensor, lengths
