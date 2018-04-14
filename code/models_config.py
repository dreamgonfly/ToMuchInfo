MODELS_CONFIG = {
    'WordCNN_basic': {
        'model': 'WordCNN',

    },
    'VDCNN_last-best': {
        'model': 'VDCNN',
        'normalizer': 'BasicNormalizer',
        'tokenizer': 'JamoTokenizer',
        'dictionary': 'RandomDictionary',
        'epochs': 50,
        'sort_dictionary': True,
        'lr_schedule': True,
        'learning_rate': 0.005,
        'min_length': 64,
        'max_length': 100,
        'embedding_size': 16
    }
}