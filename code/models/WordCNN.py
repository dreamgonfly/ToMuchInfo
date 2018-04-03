import torch
from torch import nn


class WordCNN(nn.Module):
    def __init__(self, dictionary, config):
        super(WordCNN, self).__init__()

        kernel_sizes = [3,4,5] # config.kernel_sizes

        vocabulary_size = dictionary.vocabulary_size
        embedding_size = dictionary.embedding_size
        embedding_weight = dictionary.embedding
        if embedding_weight is not None:
            embedding_weight = torch.FloatTensor(embedding_weight)

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.embedding.weight = nn.Parameter(embedding_weight, requires_grad=False)

        convs = [nn.Conv1d(in_channels=embedding_size, out_channels=100, kernel_size=kernel_size) for kernel_size in
                 kernel_sizes]
        self.conv_modules = nn.ModuleList(convs)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(in_features=300, out_features=1)

    def forward(self, reviews, features):

        embedded = self.embedding(reviews)
        embedded = embedded.transpose(1, 2)  # (batch_size, wordvec_size, sentence_length)

        feature_list = []
        for conv in self.conv_modules:
            feature_map = self.tanh(conv(embedded))
            max_pooled, argmax = feature_map.max(dim=2)
            feature_list.append(max_pooled)

        features = torch.cat(feature_list, dim=1)
        features_regularized = self.dropout(features)
        logits = self.linear(features_regularized)
        predictions = logits.squeeze()
        return predictions