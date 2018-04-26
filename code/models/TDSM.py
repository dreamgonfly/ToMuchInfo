from random import randint
import torch
from torch import nn
from torch.autograd import Variable


class TDSM(nn.Module):
    def __init__(self, config, n_features):
        super(TDSM, self).__init__()

        self.WORD_LENGTH = 20

        self.embedding = nn.Embedding(config.vocabulary_size, config.embedding_size, padding_idx=0)

        self.char_conv = nn.Sequential(nn.Conv1d(100, 10, kernel_size=4), nn.Dropout(), nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=4), nn.Dropout(), nn.ReLU(), nn.Conv1d(20, 30, kernel_size=4, stride=2),
            nn.Sigmoid())

        self.lstm = nn.LSTM(input_size=180, hidden_size=100, num_layers=1, batch_first=True, bidirectional=True)

        self.attention_layers = nn.Sequential(nn.Linear(200, 1), nn.Softmax(dim=1))

        self.output_layers = nn.Sequential(
            nn.Linear(380, 380), nn.ReLU(), nn.BatchNorm1d(num_features=380),
            nn.Linear(380, 190), nn.ReLU(), nn.BatchNorm1d(num_features=190),
            nn.Linear(190, 10)
        )

    def forward(self, inputs, features):
        # inputs : batch_size x (number of words x word length)
        batch_size, text_word_length = inputs.shape
        text_length = text_word_length // self.WORD_LENGTH
        inputs = inputs.view(batch_size, text_length, self.WORD_LENGTH)
        self.flatted = inputs.view(batch_size * text_length, self.WORD_LENGTH)
        self.embedded = self.embedding(self.flatted)

        self.topic_vectors = self.char_conv(self.embedded.transpose(1, 2)).view(batch_size, text_length, 180)
        #         print('topic_vectors', topic_vectors.shape)
        self.lstm_output, (h, c) = self.lstm(self.topic_vectors)
        #         print('h', h.shape)
        self.positional_features = h.view(batch_size, 200)
        #         print('lstm_output', lstm_output.shape)
        self.attention = self.attention_layers(self.lstm_output)
        #         print('topic_vectors * attention', topic_vectors.shape, attention.shape)
        self.sentence_embedded = torch.sum(self.topic_vectors * self.attention, dim=1)
        self.total_embedded = torch.cat([self.sentence_embedded, self.positional_features], dim=1)

        return self.output_layers(self.total_embedded)

if __name__ == '__main__':
    BATCH_SIZE = 2
    NUM_WORDS = 10
    inputs = Variable(torch.LongTensor(BATCH_SIZE, NUM_WORDS * 20).random_(128))
    model = TDSM()
    output = model(inputs)
    print(output)