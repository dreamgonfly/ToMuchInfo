MODELS_CONFIG = {
    'VDCNN_1': {
        'model': 'VDCNN',
        'normalizer': 'BasicNormalizer',
        'tokenizer': 'JamoTokenizer',
        'dictionary': 'RandomDictionary',
        'vocabulary_size': 50000,
        'shuffle_dataset': True,
        'lr_schedule': True,
        'learning_rate': 0.001,
        'min_length': 64,
        'max_length': 100,
        'embedding_size': 16,
        'best_epoch': 1,
        # 'best_epoch' : 15
    },
    'WordCNN_1': {
        'model': 'WordCNN',
        # 'normalizer': 'AdvancedNormalizer',
        # 'tokenizer': 'JamoTokenizer',
        # 'dictionary': 'RandomDictionary',
        # 'vocabulary_size': 50000,
        # 'shuffle_dataset': True,
        # 'lr_schedule': True,
        # 'learning_rate': 0.001,
        # 'min_length': 64,
        # 'max_length': 150,
        # 'embedding_size': 16,
        'best_epoch': 1,
        # 'best_epoch': 12
    },
}
