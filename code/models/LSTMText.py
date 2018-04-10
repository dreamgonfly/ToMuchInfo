import torch
from torch import nn

# python2 main.py main --max_epoch=5 --plot_every=100 --env='lstm_word' --weight=1 --model='LSTMText'  \
# --batch-size=128  --lr=0.001 --lr2=0.0000 --lr_decay=0.5 --decay_every=10000  --type_='word'   --zhuge=True \
# --linear-hidden-size=2000 --hidden-size=320 --kmax-pooling=2  --augument=False

class LSTMText(nn.Module):
    def __init__(self, config, n_features):
        super(LSTMText, self).__init__()
        self.n_features = n_features
        kernel_sizes = 3 # config.kernel_sizes

        vocabulary_size = config.vocabulary_size
        embedding_size = config.embedding_size

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        self.lstm_layer = nn.LSTM(input_size=embedding_size,
                                  hidden_size=320,
                                  num_layers=3,
                                  bias=True,
                                  batch_first=False,
                                  bidirectional=False,
                                  dropout=0.2
                                  )
        self.fc = nn.Sequential(
            nn.Linear(320,128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128,1)
        )

    def forward(self, reviews, features):

        x = self.embedding(reviews)
        x = x.transpose(1, 2)  # (batch_size, wordvec_size, sentence_length)

        x = self.lstm_layer(x)
        x = self.fc(x)

        return x.squeeze()