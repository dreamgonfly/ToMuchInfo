import numpy as np
from collections import Counter
import io
import pickle

class RandomDictionary:
    """A dictionary that maps a word to an integer. No embedding word vectors."""

    def __init__(self, tokenizer, config):

        self.tokenizer = tokenizer
        self.vocabulary_size = config.vocabulary_size
        self.embedding_size = config.embedding_size
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

    def build_dictionary(self, data):

        self.vocab_words, self.word2idx, self.idx2word = self._build_vocabulary(data)
        self.embedding = None

    def indexer(self, word):
        try:
            return self.word2idx[word]
        except KeyError:
            return self.word2idx[self.UNK_TOKEN]

    def _build_vocabulary(self, data):

        counter = Counter([token for document, label in data for token in self.tokenizer.tokenize(document)])
        print("Total number of unique tokens:", len(counter))
        counter = {word:freq for word, freq in counter.most_common(self.vocabulary_size - 2)}  # for pad and unk

        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN]

        vocab_words += list(sorted(counter.keys()))

        word2idx = {word:idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words # instead of {idx:word for idx, word in enumerate(vocab_words)}

        return vocab_words, word2idx, idx2word

    def state_dict(self):
        state = {'idx2word': self.idx2word,
                 'word2idx': self.word2idx,
                 'vocab_words': self.vocab_words}
        return state

    def load_state_dict(self, state_dict):
        self.idx2word = state_dict['idx2word']
        self.word2idx = state_dict['word2idx']
        self.vocab_words = state_dict['vocab_words']


class FasttextDictionary:
    """A dictionary that maps a word to FastText embedding."""

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.vocabulary_size = config.vocabulary_size
        self.embedding_size = 300
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

    def build_dictionary(self, data):

        self.vocab_words, self.word2idx, self.idx2word = self._build_vocabulary(data)
        self.embedding = self.load_vectors()

    def indexer(self, word):
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']

    def _build_vocabulary(self, data):

        counter = Counter([word for document, label in data for word in self.tokenizer.tokenize(document)])
        print("Total number of unique tokens:", len(counter))
        counter = {word: freq for word, freq in counter.most_common(self.vocabulary_size - 2)}  # for pad and unk

        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN]
        vocab_words += list(sorted(counter.keys()))

        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words  # instead of {idx:word for idx, word in enumerate(vocab_words)}

        return vocab_words, word2idx, idx2word

    def load_vectors(self):
        fname = 'wordvectors/cc.ko.300.vec'
        print("Loading FastText...")
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        print("FastText loaded")
        n, d = map(int, fin.readline().split())
        # self.embedding_size = d

        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])

        word_vectors = []
        in_vocab_count, out_vocab_count = 0, 0
        for word in self.vocab_words:

            if word in data.keys():
                vector = data[word]
                in_vocab_count += 1
            else:
                vector = np.random.normal(scale=0.2, size=self.embedding_size)  # random vector
                out_vocab_count += 1

            word_vectors.append(vector)
        print("Words in embedding:", in_vocab_count, "out of embedding:", out_vocab_count)
        embedding = np.stack(word_vectors)
        return embedding

    def state_dict(self):
        state = {'idx2word': self.idx2word,
                 'word2idx': self.word2idx,
                 'vocab_words': self.vocab_words,
                 'embedding': self.embedding.tolist()}
        return state

    def load_state_dict(self, state_dict):
        self.idx2word = state_dict['idx2word']
        self.word2idx = state_dict['word2idx']
        self.vocab_words = state_dict['vocab_words']
        self.embedding = np.array(state_dict['embedding'])


class FastTextVectorizer:
    """A dictionary that maps a word to FastText embedding."""

    def __init__(self, tokenizer, config):
        self.tokenizer = Twitter()
        self.vocabulary_size = config.vocabulary_size
        self.embedding_size = 256
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.fasttext = None

    def build_dictionary(self, data):

        self.vocab_words, self.word2idx, self.idx2word = self._build_vocabulary(data)
        self.embedding = self.load_vectors()
        print(self.embedding.shape)

    def indexer(self, word):
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']

    def _build_vocabulary(self, data):
        reviews = [review for review, label in data]
        tokenized_reviews = [self.tokenizer.pos(review, norm=True) for review in reviews]

        tokens = [[token for token, pos in tokenized_list] for tokenized_list in tokenized_reviews]
        tags = [[pos for token, pos in tokenized_list] for tokenized_list in tokenized_reviews]

        self.fasttext = FastText(sentences=[' '.join(review) for review in tokens],
                                 size=self.embedding_size,
                                 max_vocab_size=self.vocabulary_size - 2)

        vocab_words = self.fasttext.wv.vocab
        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        word2idx['<UNK>'] = len(vocab_words)
        word2idx['<PAD>'] = len(vocab_words) + 1

        idx2word = {idx: word for idx, word in enumerate(vocab_words)}
        idx2word[len(vocab_words)] = '<UNK>'
        idx2word[len(vocab_words) + 1] = '<PAD>'

        return vocab_words, word2idx, idx2word

    def load_vectors(self):
        word_vectors = []
        for i in range(self.vocabulary_size):
            if i in self.idx2word:
                word = self.idx2word[i]
                if word in ['<UNK>', '<PAD>']:
                    vector = np.zeros(self.embedding_size)
                else:
                    vector = self.fasttext.wv[word]
                word_vectors.append(vector)
            else:
                word_vectors.append(np.zeros(self.embedding_size))
        embedding = np.stack(word_vectors)
        return embedding

    def state_dict(self):
        state = {'idx2word': self.idx2word,
                 'word2idx': self.word2idx,
                 'vocab_words': self.vocab_words,
                 'embedding': self.embedding.tolist()}
        return state

    def load_state_dict(self, state_dict):
        self.idx2word = state_dict['idx2word']
        self.word2idx = state_dict['word2idx']
        self.vocab_words = state_dict['vocab_words']
        self.embedding = np.array(state_dict['embedding'])


if __name__ == '__main__':

    class Config:
        max_vocab_size = 10
        min_count = 1
        vocabulary_size = 3000

    dictionary = FasttextDictionary(config=Config)
    dictionary.build_dictionary([])

