import numpy as np
from collections import Counter
import io
import pickle

class RandomWordDictionary:
    """A dictionary that maps a word to an integer. No embedding word vectors."""
    
    def __init__(self, tokenizer, config):

        self.tokenizer = tokenizer
        self.max_vocab_size = config.max_vocab_size
        self.min_count = config.min_count
        self.embedding_size = config.embedding_size        
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

    def build_dictionary(self, data):
        
        self.vocab_words, self.word2idx, self.idx2word = self._build_vocabulary(data)
        self.vocabulary_size = len(self.vocab_words)
        self.embedding = None
            
    def indexer(self, word):
        try:
            return self.word2idx[word]
        except:
            return self.word2idx[self.UNK_TOKEN]
    
    def _build_vocabulary(self, data):
        
        counter = Counter([token for document, label in data for token in self.tokenizer.tokenize(document)])
        if self.max_vocab_size:
            counter = {word:freq for word, freq in counter.most_common(self.max_vocab_size)}
        if self.min_count:
            counter = {word:freq for word, freq in counter.items() if freq >= self.min_count}
        
        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN]
        
        vocab_words += list(sorted(counter.keys()))
        
        word2idx = {word:idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words # instead of {idx:word for idx, word in enumerate(vocab_words)}
        
        return vocab_words, word2idx, idx2word


class FasttextDictionary:
    """A dictionary that maps a word to FastText embedding."""

    def __init__(self, config):
        self.max_vocab_size = config.max_vocab_size
        self.min_count = config.min_count
        self.embedding_size = 300
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

    def build_dictionary(self, data):

        self.vocab_words, self.word2idx, self.idx2word = self._build_vocabulary(data)
        self.vocabulary_size = len(self.vocab_words)
        self.embedding = self.load_vectors()

    def indexer(self, word):
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']

    def _build_vocabulary(self, data):

        counter = Counter([word for document, label in data for word in document])
        if self.max_vocab_size:
            counter = {word: freq for word, freq in counter.most_common(self.max_vocab_size)}
        if self.min_count:
            counter = {word: freq for word, freq in counter.items() if freq >= self.min_count}

        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN]
        vocab_words += list(sorted(counter.keys()))

        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words  # instead of {idx:word for idx, word in enumerate(vocab_words)}

        return vocab_words, word2idx, idx2word

    def load_vectors(self):
        fname = 'wordvectors/cc.ko.300.vec'
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        self.embedding_size = d

        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])

        word_vectors = []
        for word in self.vocab_words:

            if word in data.keys():
                vector = data[word]
            else:
                vector = np.random.normal(scale=0.2, size=self.embedding_size)  # random vector

            word_vectors.append(vector)

        embedding = np.stack(word_vectors)
        return embedding

    def save(self, params_filename, embedding_filename):
        params = {'vocab_words': self.vocab_words,
                  'word2idx': self.word2idx,
                  'idx2word': self.idx2word,
                  'vocabulary_size': self.vocabulary_size}

        with open(params_filename, 'wb') as params_file:
            pickle.dump(params, params_file)
        np.save(embedding_filename, self.embedding)

    @classmethod
    def load_from(cls, params_filename, embedding_filename):

        with open(params_filename, 'rb') as params_file:
            params = pickle.load(params_file)

        instance = cls()
        instance.vocab_words = params['vocab_words']
        instance.word2idx = params['word2idx']
        instance.idx2word = params['idx2word']
        instance.vocabulary_size = params['vocabulary_size']
        instance.embedding = np.load(embedding_filename)
        return instance

if __name__ == '__main__':

    class Config:
        max_vocab_size = 10
        min_count = 1

    dictionary = FasttextDictionary(config=Config)
    dictionary.build_dictionary([])
