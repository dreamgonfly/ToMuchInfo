import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Random seed
np.random.seed(0)
random.seed(0)


def load_data(dataset_path, val_size=0.3):

    data_review = os.path.join(dataset_path, 'train', 'train_data')
    data_label = os.path.join(dataset_path, 'train', 'train_label')
    
    with open(data_review, 'rt', encoding='utf-8') as f:
        reviews = f.readlines()
    
    with open(data_label) as f:
        labels = [np.float32(x) for x in f.readlines()]
    
    data = [(review, label) for review, label in zip(reviews, labels)]
    train_data, val_data = train_test_split(data, test_size=val_size)
    
    return train_data, val_data


class Preprocessor:
    
    def __init__(self, tokenizer, feature_extractors, dictionary):
        
        self.tokenizer = tokenizer
        self.feature_extractors = feature_extractors
        self.dictionary = dictionary
        
    def preprocess(self, raw_text):
        
        tokenized_text = self.tokenizer.tokenize(raw_text)
        
        features_extracted = list()
        for feature_extractor in self.feature_extractors:
            
            feature_extracted = feature_extractor.extract_feature(raw_text, tokenized_text)
            features_extracted.append(feature_extracted)
        
        indexed_text = [self.dictionary.indexer(token) for token in tokenized_text]
        
        return indexed_text, features_extracted


def pad_text(text, pad, min_length=None, max_length=None):
    length = len(text)
    if min_length is not None and length < min_length:
        return text + [pad]*(min_length - length)
    if max_length is not None and length > max_length:
        return text[:max_length]
    return text


class MovieReviewDataset:
    
    def __init__(self, data, preprocessor, dictionary, sort=False, min_length=None, max_length=None):
        
        PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)
        
        self.data = [(preprocessor.preprocess(review), label) for review, label in data]
        
        if min_length or max_length:
            self.data = [((pad_text(review, PAD_IDX, min_length, max_length), feature), label) 
                    for (review, feature), label in self.data]
        if sort:
            self.data = sorted(self.data, key=lambda review_label: len(review_label[0][0]))
        
    def __getitem__(self, index):
        (review, feature), label = self.data[index]
        return (review, feature), label
        
    def __len__(self):
        return len(self.data)  
      

class MovieReviewDataLoader(object):
    """Data loader. Combines a dataset and a sampler, and provides an iterator over the dataset."""

    def __init__(self, dataset, dictionary, batch_size=1, shuffle=False, drop_last=False):
        """Initialize data loader.
        
        Args:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: False).
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: False) 
        """
        self.dataset = dataset
        self.dictionary = dictionary
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return len(self.dataset) // self.batch_size + 1

    def collate_fn(self, batch):
        """merges a list of samples to form a mini-batch."""

        text_lengths = [len(review) for (review, feature), label in batch]
        longest_length = max(text_lengths)

        reviews_padded = [pad_text(review, pad=self.PAD_IDX, min_length=longest_length) for (review, feature), label in
                          batch]
        features = [sum(feature, tuple()) for (review, feature), label in batch]
        labels = [label for (review, features), label in batch]

        reviews_array = np.array(reviews_padded)
        features_array = np.array(features)
        labels_array = np.array(labels)
        return reviews_array, features_array, labels_array


class DataLoaderIter:

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.shuffle = loader.shuffle
        self.drop_last = loader.drop_last

        self.collate_fn = loader.collate_fn
        
        data_size = len(self.dataset)
        if self.shuffle:
            indexes = list(range(data_size))
            random.shuffle(indexes)
            self.indexes = iter(indexes)
        else:
            self.indexes = iter(range(data_size))

    def __next__(self):
        batch_indices = []
        for _ in range(self.batch_size):
            try:
                next_index = next(self.indexes)
                batch_indices.append(next_index)
            except StopIteration:
                if self.drop_last:
                    raise StopIteration
                else:
                    break
        if len(batch_indices) == 0:
            raise StopIteration
        batch = self.collate_fn([self.dataset[i] for i in batch_indices])
        return batch




