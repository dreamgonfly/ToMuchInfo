import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, config, n_features):
        super(TextCNN, self).__init__()

        V = config.vocabulary_size
        D = config.embedding_size
        C = 1
        Ci = 1
        Co = 4
        Ks = [1,2,3,4]

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci,Co,(K,D)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = (nn.ReLU(conv(x))).squeeze(3)
        x = nn.MaxPool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, reviews, features):
        x = self.embed(reviews)

        x = x.unsqueeze(1)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]

        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.fc(x)
        return logit
