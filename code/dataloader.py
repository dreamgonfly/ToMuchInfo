import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# Random seed
np.random.seed(0)
random.seed(0)

PAD_IDX = 0


def load_data(dataset_path, val_size=0.3):

    data_review = os.path.join(dataset_path, 'train', 'train_data')
    data_label = os.path.join(dataset_path, 'train', 'train_label')
    
    with open(data_review, 'rt', encoding='utf-8') as f:
        reviews = f.readlines()
    
    with open(data_label) as f:
        labels = [float(x) for x in f.readlines()]
    
    data = [(review, label) for review, label in zip(reviews, labels)]
    train_data, val_data = train_test_split(data, test_size=val_size)
    
    return train_data, val_data


class Preprocessor:
    
    def __init__(self, tokenizer, feature_extractors, dictionary):
        
        self.tokenizer = tokenizer
        self.feature_extractors = feature_extractors
        self.dictionary = dictionary
        
    def preprocess(self, raw_text):
        """Preprocess raw text

        Args:
            raw_text (string): For example, "무궁화 꽃이 피었습니다"

        Returns:
            A tuple that contains an indexed text and extracted features concatenated.
            For example,

            ([0, 2, 3], (1, 0.7, 0.1))
        """
        
        tokenized_text = self.tokenizer.tokenize(raw_text)
        
        features_extracted = tuple()
        for feature_name, feature_extractor in self.feature_extractors:
            
            feature_extracted = feature_extractor.extract_feature(raw_text, tokenized_text)
            features_extracted += feature_extracted
        
        indexed_text = [self.dictionary.indexer(token) for token in tokenized_text]
        
        return indexed_text, features_extracted
    
    def preprocess_all(self, raw_text):
        """raw_text (list of strings)"""
        
        data = [self.preprocess(review) for review in raw_text]
        text_lengths = [len(review) for review, feature in data]
        longest_length = max(text_lengths)
        reviews_padded = [pad_text(review, pad=PAD_IDX, min_length=longest_length) for review, feature in
                          data]
        features = [feature for review, feature in data]

        reviews_tensor = torch.LongTensor(reviews_padded)
        features_tensor = torch.FloatTensor(features)
        return reviews_tensor, features_tensor

    def state_dict(self):
        # self.tokenizer
        features_state_dict = {}
        for feature_name, feature_extractor in self.feature_extractors:
            features_state_dict[feature_name] = feature_extractor.state_dict()

        dictionary_dict = self.dictionary.state_dict()

        return {'features_state': features_state_dict,
                'dictionary_state': dictionary_dict}

    def load_state_dict(self, state_dict):

        for feature_name, feature_extractor in self.feature_extractors:
            feature_extractor.load_state_dict(state_dict['features_state'][feature_name])

        self.dictionary.load_state_dict(state_dict['dictionary_state'])


class MovieReviewDataset(Dataset):
    def __init__(self, data, preprocessor, sort=False, min_length=None, max_length=None):

        global PAD_IDX
        PAD_IDX = PAD_IDX # dictionary.indexer(dictionary.PAD_TOKEN)

        self.data = [(preprocessor.preprocess(review), label) for review, label in data]

        if min_length or max_length:
            self.data = [((pad_text(review, PAD_IDX, min_length, max_length), feature), label) for
                         (review, feature), label in self.data]
        if sort:
            self.data = sorted(self.data, key=lambda review_label: len(review_label[0][0]))

    def __getitem__(self, index):
        (review, feature), label = self.data[index]
        return (review, feature), label

    def __len__(self):
        return len(self.data)


def pad_text(text, pad=PAD_IDX, min_length=None, max_length=None):
    length = len(text)
    if min_length is not None and length < min_length:
        return text + [pad]*(min_length - length)
    if max_length is not None and length > max_length:
        return text[:max_length]
    return text


def collate_fn(batch):
    """merges a list of samples to form a mini-batch."""

    text_lengths = [len(review) for (review, feature), label in batch]
    longest_length = max(text_lengths)

    reviews_padded = [pad_text(review, pad=PAD_IDX, min_length=longest_length) for (review, feature), label in
                      batch]
    features = [feature for (review, feature), label in batch]
    labels = [label for (review, features), label in batch]

    reviews_tensor = torch.LongTensor(reviews_padded)
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.FloatTensor(labels)
    return reviews_tensor, features_tensor, labels_tensor