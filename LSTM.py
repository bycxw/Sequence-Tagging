# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTMTagger(nn.model):
    def __init__(self, vocab_size, emb_dim, hidden_dim, tag_size, bidirectional=False):
        super(LSTMTagger, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=bidirectional)
        self.lin = nn.Linear(hidden_dim, tag_size)

    def forward(self, sents_tensor, lengths):
        embeds = self.embeddings(sents_tensor)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.lin(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
