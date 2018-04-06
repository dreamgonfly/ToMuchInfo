import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Variable

class SequenceWise(nn.Module):
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        vocabulary_size = config.vocabulary_size
        embedding_size = config.embedding_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        # hidden size 바꾸기 가능
        self.hidden_size = config.embedding_size
        # layer 수 변환 : 3
        self.lstm = nn.LSTM(config.embedding_size,self.hidden_size,3,bidirectional = True)
        self.h0 = Variable(torch.zeros(3 * 2,32,100)).cuda()
        self.c0 = Variable(torch.zeros(3 * 2,32,100)).cuda()

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, 1, bias=False)
        )

        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )


    def forward(self, x, features):
        """
        TODO:
            insert feature
        """
        x = self.embedding(x).transpose(0,1) # T x N x H
        x, _ = self.lstm(x,(self.h0,self.c0))
        x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        x = x.transpose(0, 1)
        x = F.log_softmax(x[:,1,:],dim=1)
        return x
