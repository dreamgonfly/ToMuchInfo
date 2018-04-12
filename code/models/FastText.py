import torch
from torch import nn

class FastText(nn.Module):
    def __init__(self, config, n_features):
        super(FastText, self).__init__()
        vocabulary_size = config.vocabulary_size
        embedding_size = config.embedding_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.pre = nn.Sequential(
            nn.Linear(embedding_size, embedding_size*2),
            nn.BatchNorm1d(embedding_size*2),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(embedding_size*2, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1)
        )

    def forward(self, reviews, features):

        embedding = self.embedding(reviews)
        # embedding = embedding.transpose(1,2)
        size = embedding.size()
        embedding_2 = self.pre(embedding.contiguous().view(-1, 300)).view(size[0], size[1], -1)

        embedding_ = torch.mean(embedding_2, dim=1)
        predictions = self.fc(embedding_)

        return predictions
