import torch
from torch import nn

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class RCNN(nn.Module):
    def __init__(self, config, n_features):
        super(RCNN, self).__init__()

        vocabulary_size = config.vocabulary_size
        embedding_dim = config.embedding_size
        kernel_size = 3
        self.kmax_pool_size = 2
        linear_hidden_size = 1000
        hidden_size = 256
        num_layers = 3

        self.encoder = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            bidirectional=True,
            )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size*2,
                      out_channels=512,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512,
                      out_channels=512,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(512,linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size,1)
        )

    def forward(self, reviews, features):
        reviews = self.encoder(reviews)

        reviews_out = self.lstm(reviews.permute(1,0,2))[0].permute(1,2,0)
        reviews.em = (reviews).permute(0,2,1)

        reviews_conv_out = kmax_pooling(self.conv(reviews_out),2,self.kmax_pool_size-1)
        reshaped = reviews_conv_out.view(reviews_conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits