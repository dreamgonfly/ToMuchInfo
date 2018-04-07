import numpy as np
from collections import Counter
from konlpy.tag import Twitter
from gensim.models import FastText

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


class FastTextVectorizer:
    """A dictionary that maps a word to FastText embedding."""

    def __init__(self, tokenizer, config):
        self.tokenizer = Twitter()
        self.vocabulary_size = config.vocabulary_size
        self.embedding_size = 300
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.fasttext = None

    def build_dictionary(self, data):

        self.vocab_words, self.word2idx, self.idx2word = self._build_vocabulary(data)
        self.embedding = self.load_vectors()

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
                                 max_vocab_size=self.vocabulary_size-2)

        vocab_words = self.fasttext.wv.vocab
        word2idx = {word:idx for idx, word in enumerate(vocab_words)}
        word2idx['<UNK>'] = self.vocabulary_size-1
        word2idx['<PAD>'] = self.vocabulary_size
        
        idx2word = {idx:word for idx, word in enumerate(vocab_words)}
        idx2word[self.vocabulary_size-1] = '<UNK>'
        idx2word[self.vocabulary_size] = '<PAD>'
        
        return vocab_words, word2idx, idx2word

    def load_vectors(self):
        word_vectors = []
        for i in self.idx2word:
            word = self.idx2word[i]
            if word in ['<UNK>', '<PAD>']:
                vector = np.zeros(self.embedding_size)
            else : vector = self.fasttext.wv[word]
            word_vectors.append(vector)
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
        tokenizer = 'TwitterTokenizer'

    dictionary = FastTextVectorizer(tokenizer=None, config=Config)
    dictionary.build_dictionary([])

