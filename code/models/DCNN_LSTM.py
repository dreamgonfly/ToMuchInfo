import torch
from torch import nn
from torch.autograd import Variable
import random

class SequenceWise(nn.Module):
    """
    총 두 번 사용된다. CNN의 output이 RNN으로 들어갈 때. RNN의 output이 FC로 들어갈 때.
    """
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

class DCNN_LSTM(nn.Module):
    def __init__(self, config, n_features):
        super(DCNN_LSTM, self).__init__()

        self.hidden_size = 128
        kernel_sizes = [3,4,5] # config.kernel_sizes
        self.kernel_sizes = kernel_sizes
        vocabulary_size = config.vocabulary_size
        embedding_size = config.embedding_size

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        convs = [nn.Conv1d(in_channels=embedding_size, out_channels=100, kernel_size=kernel_size) for kernel_size in
                 kernel_sizes]
        self.conv_modules = nn.ModuleList(convs)
        self.tanh = nn.Tanh()
        #self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.lstm = nn.LSTM(input_size=100*len(kernel_sizes),hidden_size =self.hidden_size,bidirectional=True)
        self.config = config
        self.B1 = SequenceWise(nn.BatchNorm1d(100*len(self.kernel_sizes)))
        self.batch_size = config.batch_size
        self.hidden = self.init_hidden()


        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(self.hidden_size,10, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

    def forward(self, reviews, features):

        x = self.embedding(reviews)
        x = x.transpose(1, 2)  # (batch_size, wordvec_size, sentence_length)

        feature_list = []
        for conv in self.conv_modules:
            feature_map = self.tanh(conv(x))
            feature_list.append(feature_map)

        max_len = max(feature_list[0].shape[2],feature_list[1].shape[2],feature_list[2].shape[2])
        new_feature_list = []
        for feat in feature_list:
            to_be_padded = max_len - feat.shape[2]
            if(round(random.random())): # overfitting 방지
                ze = nn.functional.pad(feat,(0,to_be_padded))
            else:
                ze = nn.functional.pad(feat,(to_be_padded,0))
            new_feature_list.append(ze)

        features = torch.cat(new_feature_list, dim=1)
        # T * N * H
        features = features.transpose(1,2).transpose(0,1).contiguous()


        #batch_norm
        features = self.B1(features)
        # ReLUx
        sigmoid = self.sigmoid(features)
        # dropout
        features = self.dropout(features)

        # now LSTM
        features ,self.hidden = self.lstm(features,self.hidden)
        x = features.view(features.size(0), features.size(1), 2, -1).sum(2).view(features.size(0), features.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        x = self.fc(x)
#        x, _ = x.transpose(0,1).max(dim=1)
        # 첫 번째 인자.
        x = x.transpose(0,1)[:,0]
        return x

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        if(self.config.use_gpu):
            return (Variable(torch.zeros(1 * 2, self.batch_size , 128)).cuda(),
                    Variable(torch.zeros(1 * 2, self.batch_size , 128)).cuda())
        else:
            return (Variable(torch.zeros(1 * 2, self.batch_size , 128)),
                    Variable(torch.zeros(1 * 2, self.batch_size , 128)))
