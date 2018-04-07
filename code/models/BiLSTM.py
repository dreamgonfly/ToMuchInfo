import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BiLSTM(nn.Module):
    """
    TODO
    수렴하는거 보고나서 bidirectional로 전환.
    """

    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        ###########################################
        self.hidden_dim = 100
        self.label_size = 10
        self.layer_number = 2
        self.batch_size = 100
        ###########################################
        self.word_embeddings = nn.Embedding(config.vocabulary_size, config.embedding_size)
        self.lstm = nn.LSTM(config.embedding_size, self.hidden_dim,self.layer_number,bidirectional = True)
        self.B1 = nn.BatchNorm1d(self.hidden_dim)
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)
        self.B2 = nn.BatchNorm1d(self.label_size)
        self.label2score = nn.Linear(self.label_size,1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(self.layer_number, self.batch_size, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(self.layer_number, self.batch_size, self.hidden_dim)).cuda())

    def forward(self, reviews,features):
        embeds = self.word_embeddings(reviews)
        x = embeds.transpose(0,1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.B1(lstm_out[-1])
        y  = self.hidden2label(lstm_out[-1])
        y = self.B2(y)
        y = self.label2score(y)
#        log_probs = F.log_softmax(y,dim=1)
        return yimport torch.nn.functional as F
