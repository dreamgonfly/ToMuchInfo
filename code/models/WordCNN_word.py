import torch
from torch import nn


class WordCNN_word(nn.Module):
    def __init__(self, config, n_features):
        super(WordCNN_word, self).__init__()

        kernel_sizes = [1,2,3,4] # config.kernel_sizes

        vocabulary_size = config.vocabulary_size
        embedding_size = config.embedding_size

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        convs = [nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
                      out_channels=250,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=250,
                      out_channels=250,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(250),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(30 - kernel_size * 2 + 2))
        )
            for kernel_size in kernel_sizes]


        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * 250, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1)
        )


    def forward(self, reviews, features):

        title = self.embedding(reviews)
        title_out = [conv(title.transpose(1,2)) for conv in self.convs]
        conv_out = torch.cat(title_out, dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits