import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

PAD_IDX = 0

class LSTM_Attention(nn.Module):
    def __init__(self, config, n_features):
        """
        V: input_size = vocab_size
        D: embedding_size
        H: hidden_size
        H_f: hidden_size (fully-connected)
        O: output_size (fully-connected)
        da: attenion_dimension (hyperparameter)
        r: keywords (different parts to be extracted from the sentence)
        """
        super(LSTM_Attention, self).__init__()

        V = config.vocabulary_size  # config.vocabulary_size
        D = config.embedding_size  # config.embedding_size
        H = 200  # hidden_size
        H_f = 1000
        O = 1
        da = 300
        self.r = 10
        num_layers = 3
        bidirec = True
        self.use_gpu = config.use_gpu

        if bidirec:
            num_directions = 2
        else:
            num_directions = 1

        self.embedding = nn.Embedding(V, D)
        self.lstm = nn.LSTM(D, H, num_layers, batch_first=True, bidirectional=bidirec)
        self.attn = nn.Linear(num_directions * H, da, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.attn2 = nn.Linear(da, self.r, bias=False)
        self.attn_dist = nn.Softmax(dim=2)

        self.fc = nn.Sequential(nn.Linear(self.r * H * num_directions, H_f), nn.ReLU(), nn.Linear(H_f, O), )

    def penalization_term(self, A):
        """
        A : B, r, T
        Frobenius Norm
        """
        eye = Variable(torch.eye(A.size(1)).expand(A.size(0), self.r, self.r))  # B, r, r
        if self.use_gpu:
            eye = eye.cuda()
        P = torch.bmm(A, A.transpose(1, 2)) - eye  # B, r, r
        loss_P = ((P ** 2).sum(1).sum(1) + 1e-10) ** 0.5
        loss_P = torch.sum(loss_P) / A.size(0)
        return loss_P

    def forward(self, reviews, features): # self, inputs, inputs_lengths
        """
        inputs: B, T, V
         - B: batch_size
         - T: max_len = seq_len
         - V: vocab_size
        inputs_lengths: length of each sentences
        """
        # inputs_lengths = (reviews != PAD_IDX).sum(1).data.cpu().numpy().tolist()
        embed = self.embedding(reviews)  # B, T, V  --> B, T, D

        # 패딩된 문장을 패킹(패딩은 연산 안들어가도록)
        # packed = pack_padded_sequence(embed, inputs_lengths, batch_first=True)
        # packed: B * T, D
        output, (hidden, cell) = self.lstm(embed)
        # output: B * T, 2H
        # hidden, cell: num_layers * num_directions, B, H

        # 패킹된 문장을 다시 unpack
        # output, output_lengths = pad_packed_sequence(output, batch_first=True)
        # output: B, T, 2H

        # Self Attention
        a1 = self.attn(output)  # Ws1(B, da, 2H) * output(B, T, 2H) -> B, T, da
        tanh_a1 = self.tanh(a1)  # B, T, da
        score = self.attn2(tanh_a1)  # Ws2(B, r, da) * tanh_a1(B, T, da) -> B, T, r
        self.A = self.attn_dist(score.transpose(1, 2))  # B, r, T
        self.M = self.A.bmm(output)  # B, r, T * B, T, 2H -> B, r, 2H

        # Penalization Term
        loss_P = self.penalization_term(self.A)

        output = self.fc(self.M.view(self.M.size(0), -1)).squeeze()  # B, r, 2H -> resize to B, r*2H -> B, H_f -> Relu -> B, 1

        return output, loss_P
