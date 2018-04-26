import torch
from torch import nn


class WordCNN(nn.Module):
    def __init__(self, config, n_features):
        super(WordCNN, self).__init__()

        kernel_sizes = [3,4,5] # config.kernel_sizes

        vocabulary_size = config.vocabulary_size
        embedding_size = config.embedding_size

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        convs = [nn.Conv1d(in_channels=embedding_size, out_channels=100, kernel_size=kernel_size) for kernel_size in
                 kernel_sizes]
        self.conv_modules = nn.ModuleList(convs)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(in_features=300, out_features=10)

    def forward(self, reviews, features):

        x = self.embedding(reviews)
        x = x.transpose(1, 2)  # (batch_size, wordvec_size, sentence_length)

        feature_list = []
        for conv in self.conv_modules:
            feature_map = self.tanh(conv(x))
            max_pooled, argmax = feature_map.max(dim=2)
            feature_list.append(max_pooled)

        features = torch.cat(feature_list, dim=1)
        features_regularized = self.dropout(features)
        logits = self.linear(features_regularized)
        predictions = logits.squeeze()
        return predictions



