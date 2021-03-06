import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from ironyer import irony_deleter

# Random seed
np.random.seed(0)
random.seed(0)

PAD_IDX = 0


def load_data(dataset_path, val_size=0.3, small=False):

    data_review = os.path.join(dataset_path, 'train', 'train_data')
    data_label = os.path.join(dataset_path, 'train', 'train_label')
    
    with open(data_review, 'rt', encoding='utf-8') as f:
        reviews = f.readlines()
    
    with open(data_label) as f:
        labels = [int(x) for x in f.readlines()]

    if small:
        reviews = reviews[:1000]
        labels = labels[:1000]
    data = [(review, label) for review, label in zip(reviews, labels)]
    train_data, val_data = train_test_split(data, test_size=val_size)
    
    return train_data, val_data


class Preprocessor:
    
    def __init__(self, config, normalizer, tokenizer, feature_extractors, dictionary):

        self.min_length = config.min_length
        self.max_length = config.max_length
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.feature_extractors = feature_extractors
        self.n_features = sum([feature_extractor.n for feature_name, feature_extractor in feature_extractors])
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

        normalized_text = self.normalizer.normalize(raw_text)
        tokenized_text = self.tokenizer.tokenize(normalized_text)
        
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
        if self.max_length < longest_length:
            length = self.max_length
        elif self.min_length < longest_length <= self.max_length:
            length = longest_length
        elif longest_length <= self.min_length:
            length = self.min_length

        reviews_padded = [pad_text(review, pad=PAD_IDX, min_length=length, max_length=length) for review, feature in
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

        tokenizer_dict = self.tokenizer.state_dict()
        dictionary_dict = self.dictionary.state_dict()

        return {'features_state': features_state_dict,
                'dictionary_state': dictionary_dict,
                'tokenizer_state': tokenizer_dict,
                }

    def load_state_dict(self, state_dict):

        for feature_name, feature_extractor in self.feature_extractors:
            feature_extractor.load_state_dict(state_dict['features_state'][feature_name])
        self.tokenizer.load_state_dict(state_dict['tokenizer_state'])
        self.dictionary.load_state_dict(state_dict['dictionary_state'])


class MovieReviewDataset(Dataset):
    def __init__(self, data, preprocessor, sort=False, min_length=None, max_length=None):

        PAD_IDX = preprocessor.dictionary.indexer(preprocessor.dictionary.PAD_TOKEN)

        data_deleted = [(review, label) for review, label in data if irony_deleter(review, label)]
        self.data = [(preprocessor.preprocess(review), label - 1) for review, label in data_deleted]

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
    # print(labels)
    labels_tensor = torch.LongTensor(labels) # classification
    # print(labels_tensor)
    # labels_tensor = torch.FloatTensor(labels) # for regression
    return reviews_tensor, features_tensor, labels_tensor
