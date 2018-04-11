import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import math

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


class BatchRNN(nn.Module):
    """
    batchnormalization + RNN 구현.
    굳이 이런식으로 붙이는 이유는 sequencewise 때문에. 
    input : 
        input_size - T * N * H 에서 H 의 값. 여기서 T 는 time, N 은 mini batch 갯수, H는 P * C(convolution layers)
        hidden_size - output의 값. 
                      여기서 output은 input과 같지 않냐! 고 생각하는데, 단순히 hi = W1 * xi + W2 * xi 라고 했을 때 matrix의 행의 크기. 
                      즉, T * N * H에서 H가 얼마나 압축될 것이냐의 의미.
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=True, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1 # why do I need this?

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        h0 = Variable(torch.zeros(1 * 2, 20, self.hidden_size)).cuda()#.cuda()# batch size = 20
        c0 = Variable(torch.zeros(1 * 2, 20, self.hidden_size)).cuda()#.cuda()
        x, _ = self.rnn(x,(h0,c0))
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
            return x

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1 * 2, self.batch_size , self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1 * 2, self.batch_size , self.hidden_dim)).cuda())


class DeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, num_classes=1 , rnn_hidden_size=100,
                 bidirectional=True):
        super(DeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        self._hidden_size = rnn_hidden_size
        self._rnn_type = rnn_type
        self._audio_conf = audio_conf or {}

        sample_rate = self._audio_conf.get("sample_rate", 16000)
        window_size = self._audio_conf.get("window_size", 0.02)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        self.rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=True)

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

    def forward(self, x):
        x = self.conv(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        x = self.rnn(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        x = F.log_softmax(x[:,-1,:])
        return x
    
    def init_hidden(self):
	self.rnn.self.rnn.init_hidden():
