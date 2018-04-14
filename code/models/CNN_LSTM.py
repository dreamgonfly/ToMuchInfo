import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class SequenceWise(nn.Module):
    """
    Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
    Allows handling of variable sequence lengths and minibatch sizes.
    :param module: Module to apply input to.
    """
    """
    총 두 번 사용된다. CNN의 output이 RNN으로 들어갈 때. RNN의 output이 FC로 들어갈 때.
    그 필요성은 위의 딥 스피치 모델 그림에서 데이터가 transpose됨을 보면 알 수 있다.
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


class BatchRNN(nn.Module):
    def __init__(self, config, input_size, hidden_size=100, rnn_type=nn.LSTM, bidirectional=True, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.batch_size = 101
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(
            nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1  # why do I need this?
        self.config = config
        self.hidden = self.init_hidden()


    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, self.hidden = self.rnn(x, self.hidden)
        if self.bidirectional:
            # (TxNxH*2) -> (TxNxH) by sum
            x = x.view(x.size(0), x.size(1), 2, -
                       1).sum(2).view(x.size(0), x.size(1), -1)
            return x

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
####################################################3 cuda
#        return (autograd.Variable(torch.zeros(1 * 2, self.config.batch_size, self.hidden_size)).cuda(),
#                autograd.Variable(torch.zeros(1 * 2, self.config.batch_size, self.hidden_size)).cuda())
        if(self.config.use_gpu):
            return (autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_size)).cuda(),
                autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_size)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_size)),
                autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_size)))





class CNN_LSTM(nn.Module):
    def __init__(self, config, rnn_type=nn.LSTM, num_classes=1, rnn_hidden_size=100, audio_conf=None,
                 bidirectional=True):
        super(CNN_LSTM, self).__init__()

        self._hidden_size = rnn_hidden_size
#        self._hidden_layers = nb_layers
        self._rnn_type = rnn_type
        self.batch_size = 100
        self.config = config

        self.word_embeddings = nn.Embedding(
            config.vocabulary_size, config.embedding_size)
        self.word_embeddings.weight.requires_grad=False

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(16, 16, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.Hardtanh(0, 20, inplace=True)
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor(config.embedding_size) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 21) / 2)
        rnn_input_size *= 16 

        self.rnn = BatchRNN(config=config, input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                            bidirectional=bidirectional, batch_norm=True)

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

    def forward(self, reviews, features):
        x = self.word_embeddings(reviews)
        x = x.unsqueeze(1).transpose(2,3)

        x = self.conv(x)

        sizes = x.size()
        # Collapse feature dimension
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])

        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        x = self.rnn(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # x = F.log_softmax(x[:,-1])
        return x[:, -1]

    def init_hidden(self):
        self.rnn.batch_size = self.batch_size
        self.rnn.hidden = self.rnn.init_hidden()
        if(self.config.use_gpu):
            return (autograd.Variable(torch.zeros(1 * 2, self.config.batch_size, self._hidden_size)).cuda(),
		    autograd.Variable(torch.zeros(1 * 2, self.config.batch_size, self._hidden_size)).cuda())
        return (autograd.Variable(torch.zeros(1 * 2, self.batch_size, self._hidden_size)),
		    autograd.Variable(torch.zeros(1 * 2, self.batch_size, self._hidden_size)))


