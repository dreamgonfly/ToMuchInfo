import torch
from torch import nn


class WordCNN_feat(nn.Module):
    def __init__(self, config, n_features):
        super(WordCNN_feat, self).__init__()

        kernel_sizes = [3,4,5] # config.kernel_sizes

        vocabulary_size = config.vocabulary_size
        embedding_size = config.embedding_size

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        convs = [nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=kernel_size) for kernel_size in
                 kernel_sizes]
        self.conv_modules = nn.ModuleList(convs)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(in_features=384, out_features=128)

        self.final_layers = nn.Sequential(
            nn.Linear(128 + n_features, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1),

        )

    def forward(self, reviews, features):

        x = self.embedding(reviews)
        x = x.transpose(1, 2)  # (batch_size, wordvec_size, sentence_length)

        feature_list = []
        for conv in self.conv_modules:
            feature_map = self.tanh(conv(x))
            max_pooled, argmax = feature_map.max(dim=2)
            feature_list.append(max_pooled)

        x = torch.cat(feature_list, dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        x_features = torch.cat([x, features], dim=1)
        final_output = self.final_layers(x_features)
        return final_output.squeeze()