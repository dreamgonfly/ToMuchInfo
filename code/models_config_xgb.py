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
        'best_epoch' : 1,
        # 'best_epoch' : 15
    },
    'VDCNN_2': {
        'model': 'VDCNN',
        'normalizer': 'AdvancedNormalizer',
        'tokenizer': 'JamoTokenizer',
        'dictionary': 'RandomDictionary',
        'vocabulary_size': 50000,
        'shuffle_dataset': True,
        'lr_schedule': True,
        'learning_rate': 0.001,
        'min_length': 64,
        'max_length': 150,
        'embedding_size': 16,
        'best_epoch' : 1,
        # 'best_epoch' : 12
    },
    'DCNN_LSTM_oh_my_god': {
        'model': 'DCNN_LSTM',
        'normalizer': 'BasicNormalizer',
	    'tokenizer': 'SoyNLPTokenizer',
        'dictionary': 'RandomDictionary',
        'vocabulary_size': 50000,
        'shuffle_dataset': False,
        'lr_schedule': True,
        'learning_rate': 0.003,
        'min_length': 10,
        'max_length': 300,
        'embedding_size': 300,
        'best_epoch' : 1,
        # 'best_epoch' : 7
    },

}
