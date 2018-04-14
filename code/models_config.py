MODELS_CONFIG = {
    'WordCNN_basic': {
        'model': 'WordCNN',
        'normalizer': 'BasicNormalizer',
        'tokenizer': 'TwitterTokenizer',
        'dictionary': 'RandomDictionary',
        'learning_rate': 0.005,
        'sort_dictionary': True,
        'lr_schedule': True,
        'embedding_size': 300,
    },
    'VDCNN_last-best': {
        'model': 'VDCNN',
        'normalizer': 'BasicNormalizer',
        'tokenizer': 'JamoTokenizer',
        'dictionary': 'RandomDictionary',
        'vocabulary_size': 100000,
        'shuffle_dataset': True,
        'lr_schedule': True,
        'learning_rate': 0.001,
        'min_length': 64,
        'max_length': 100,
        'embedding_size': 16
    }
    'DCNN_LSTM_oh_my_god': {
        'model': 'DCNN_LSTM',
        'normalizer': 'BasicNormalizer',
	'tokenizer': 'SoyNLPTokenizer',
        'dictionary': 'RandomDictionary',
        'vocabulary_size': 100000,
        'shuffle_dataset': True,
        'lr_schedule': True,
        'learning_rate': 0.001,
        'min_length': 16,
        'max_length': 100,
        'embedding_size': 300 

    }
}
