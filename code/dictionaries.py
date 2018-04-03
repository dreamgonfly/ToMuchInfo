import numpy as np
from collections import Counter


class RandomWordDictionary:
    """A dictionary that maps a word to an integer. No embedding word vectors."""
    
    def __init__(self, config):
        
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
        
        counter = Counter([word for document, label in data for word in document])
        if self.max_vocab_size:
            counter = {word:freq for word, freq in counter.most_common(self.max_vocab_size)}
        if self.min_count:
            counter = {word:freq for word, freq in counter.items() if freq >= self.min_count}
        
        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN]
        
        vocab_words += list(sorted(counter.keys()))
        
        word2idx = {word:idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words # instead of {idx:word for idx, word in enumerate(vocab_words)}
        
        return vocab_words, word2idx, idx2word