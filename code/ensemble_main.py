import argparse
import os
import numpy as np
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

import models
import normalizers
import tokenizers
import feature_extractors
import dictionaries
from dataloader import load_data, collate_fn, Preprocessor, MovieReviewDataset
from trainers import Trainer, EnsembleTrainer
import utils
from models_config import MODELS_CONFIG

import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
from LSUV import LSUVinit

INFER_THRESHOLD = 4.5

# Random seed
np.random.seed(0)
torch.manual_seed(0)

args = argparse.ArgumentParser()
# DONOTCHANGE: They are reserved for nsml
args.add_argument('--mode', type=str, default='train')
args.add_argument('--pause', type=int, default=0)
args.add_argument('--iteration', type=str, default='0')

# User options
# config serves as a default set
args.add_argument('--model', type=str, default='WordCNN')
args.add_argument('--normalizer', type=str, default='DummyNormalizer')
args.add_argument('--tokenizer', type=str, default='JamoTokenizer')
args.add_argument('--features', type=str, default='LengthFeatureExtractor')  # LengthFeatureExtractor_MovieActorFeaturesExtractor ...
args.add_argument('--dictionary', type=str, default='RandomDictionary')
args.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available() or GPU_NUM)
args.add_argument('--output', type=int, default=1)
args.add_argument('--epochs', type=int, default=10)
args.add_argument('--batch_size', type=int, default=64)
args.add_argument('--vocabulary_size', type=int, default=50000)
args.add_argument('--embedding_size', type=int, default=100)
args.add_argument('--min_length', type=int, default=5)
args.add_argument('--max_length', type=int, default=300)
args.add_argument('--sort_dataset', action='store_true')
args.add_argument('--shuffle_dataset', action='store_true')
args.add_argument('--learning_rate', type=float, default=0.01)
args.add_argument('--lr_schedule', action='store_true')
args.add_argument('--print_every', type=int, default=1)
args.add_argument('--save_every', type=int, default=1)
args.add_argument('--down_sampling', type=bool, default=False)
args.add_argument('--min_lr', type=float, default=0)
args.add_argument('--loss_weights', default=False, type=bool, help='loss_weights')
args.add_argument('--small', default=False)
default_config = args.parse_args()

logger = utils.get_logger('Ensemble')

ensemble_models = defaultdict(dict)

class BaseConfig:
    pass

# Set defaults
for config_name in MODELS_CONFIG:
    for default_argument in [c for c in dir(default_config) if not c.startswith('_')]:
        if default_argument not in MODELS_CONFIG[config_name]:
            MODELS_CONFIG[config_name][default_argument] = getattr(default_config, default_argument)
    config = BaseConfig()
    for argument in MODELS_CONFIG[config_name]:
        setattr(config, argument, MODELS_CONFIG[config_name][argument])
    ensemble_models[config_name]['config'] = config
    logger.info('Config {config_name}: {config}'.format(config_name=config_name, config=MODELS_CONFIG[config_name]))

for config_name in ensemble_models:
    config = ensemble_models[config_name]['config']
    Normalizer = getattr(normalizers, config.normalizer)
    normalizer = Normalizer(config)

    Tokenizer = getattr(tokenizers, config.tokenizer)
    tokenizer = Tokenizer(config)

    Dictionary = getattr(dictionaries, config.dictionary)
    dictionary = Dictionary(tokenizer, config)

    feature_extractor_list = []
    for feature_name in config.features.split('_'):
        FeatureExtractor = getattr(feature_extractors, feature_name)
        feature_extractor = FeatureExtractor(config)
        feature_extractor_list.append((feature_name, feature_extractor))

    preprocessor = Preprocessor(config, normalizer, tokenizer, feature_extractor_list, dictionary)

    ensemble_models[config_name]['preprocessor'] = preprocessor

for config_name in ensemble_models:
    config = ensemble_models[config_name]['config']
    preprocessor = ensemble_models[config_name]['preprocessor']
    Model = getattr(models, config.model)
    model = Model(config, n_features=preprocessor.n_features)
    if default_config.use_gpu:
        model = model.cuda()
    ensemble_models[config_name]['model'] = model
    logger.info("Number of features of {config_name} : {n_features}".format(config_name=config_name,
                                                                      n_features=preprocessor.n_features))
if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
    DATASET_PATH = 'data/small/' # 'data/movie_review_phase1/'

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': {config_name:ensemble_models[config_name]['model'].state_dict() for config_name in ensemble_models},
            'preprocessor': {config_name:ensemble_models[config_name]['preprocessor'].state_dict() for config_name in ensemble_models},
            'best_losses': {config_name: ensemble_models[config_name]['best_loss'] for config_name in ensemble_models}
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        for config_name in ensemble_models:
            model = ensemble_models[config_name]['model']
            preprocessor = ensemble_models[config_name]['preprocessor']

            model.load_state_dict(checkpoint['model'][config_name])
            preprocessor.load_state_dict(checkpoint['preprocessor'][config_name])
            ensemble_models[config_name]['best_loss'] = checkpoint['best_losses'][config_name]
        print('Checkpoint loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다

        predictions = []
        for config_name in ensemble_models:

            # INFER_THRESHOLD보다 높은 loss를 가진 모델은 제외
            if ensemble_models[config_name]['best_loss'] > INFER_THRESHOLD:
                continue
            preprocessor = ensemble_models[config_name]['preprocessor']
            model = ensemble_models[config_name]['model']
            reviews, features = preprocessor.preprocess_all(raw_data)
            reviews, features = Variable(reviews), Variable(features)

            if ensemble_models[config_name]['config'].use_gpu:
                reviews, features = reviews.cuda(), features.cuda()

            model.eval()
            if hasattr(model, 'init_hidden'):
                model.batch_size = len(reviews)
                model.hidden = model.init_hidden()
            # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
            output_prediction = model(reviews, features)
            score_tensor = Variable(torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
            if default_config.use_gpu:
                score_tensor = score_tensor.cuda()
            prediction = (softmax(output_prediction, dim=1) * score_tensor).sum(dim=1)

            ensemble_models[config_name]['prediction'] = prediction
            predictions.append(output_prediction)

        ensemble_predictions = sum(predictions) / len(predictions)
        prediction_clipped = torch.clamp(ensemble_predictions, min=1, max=10)

        point = prediction_clipped.data.tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

# DONOTCHANGE: Reserved for nsml use
bind_model('model', 'config')

# DONOTCHANGE: They are reserved for nsml
if config.pause:
    nsml.paused(scope=locals())

# 학습 모드일 때 사용합니다. (기본값)
if config.mode == 'train':
    # 데이터를 로드합니다.
    logger.info("Loading data...")
    train_data, val_data = load_data(DATASET_PATH, val_size=0.03, small=default_config.small)
    # print('using only 1000 samples for test')
    # train_data, val_data = train_data[:1000], val_data[:1000] # For test

    logger.info("Building preprocessor...")
    for config_name in ensemble_models:
        preprocessor = ensemble_models[config_name]['preprocessor']
        config = ensemble_models[config_name]['config']
        model = ensemble_models[config_name]['model']

        for feature_name, feature_extractor in preprocessor.feature_extractors:
            feature_extractor.fit(train_data)

        preprocessor.tokenizer.fit(train_data)
        preprocessor.dictionary.build_dictionary(train_data)

        logger.info("Making dataset & dataloader for {} ...".format(config_name))
        train_dataset = MovieReviewDataset(train_data, preprocessor, sort=config.sort_dataset, min_length=config.min_length, max_length=config.max_length)
        val_dataset = MovieReviewDataset(val_data, preprocessor, sort=config.sort_dataset, min_length=config.min_length, max_length=config.max_length)

        if default_config.down_sampling:
            train_labels = [label for train, label in train_data]
            class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in
                                           range(1, 11)])  # dataset has 10 class-1 samples, 1 class-2 samples, etc.
            weights = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1 / 6]) # 1 / torch.FloatTensor(class_sample_count) #
            weights = weights.double().cuda()
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, config.batch_size)
            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                                                           num_workers=2, sampler=sampler, )
        else:
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                          shuffle=config.shuffle_dataset, collate_fn=collate_fn, num_workers=2)

        val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=True,
                                    collate_fn=collate_fn, num_workers=2)

        ensemble_models[config_name]['train_dataloader'] = train_dataloader
        ensemble_models[config_name]['val_dataloader'] = val_dataloader

        ## initialize model param w/ LSUV
        for inputs, features, targets in train_dataloader:
            if default_config.use_gpu:
                inputs = Variable(inputs).cuda()
            else:
                inputs = Variable(inputs)
            LSUVinit(model, inputs, needed_std=1.0, std_tol=0.1, max_attempts=100, do_orthonorm=False, cuda=default_config.use_gpu)
            break

        if preprocessor.dictionary.embedding is not None:
            embedding_weights = torch.FloatTensor(preprocessor.dictionary.embedding)
            if config.use_gpu:
                embedding_weights = embedding_weights.cuda()
            model.embedding.weight = nn.Parameter(embedding_weights, requires_grad=True)

        if config.loss_weights:
            weights = [9, 67, 66, 55, 35, 26, 19, 13, 11, 2]
            weights = torch.FloatTensor(weights)
            if config.use_gpu:
                weights = weights.cuda()
        else:
            weights = None
        criterion = nn.CrossEntropyLoss(size_average=False, weight=weights)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params=trainable_params, lr=config.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)  # .ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.00005)

        ensemble_models[config_name]['criterion'] = criterion
        ensemble_models[config_name]['optimizer'] = optimizer
        ensemble_models[config_name]['lr_scheduler'] = lr_scheduler

    trainer = EnsembleTrainer(ensemble_models, use_gpu=default_config.use_gpu, logger=logger)
    trainer.run(epochs=default_config.epochs)

# 로컬 테스트 모드일때 사용합니다
# 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
# [(0.0, 9.045), (0.0, 5.91), ... ]
elif config.mode == 'test_local':
    with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
        reviews = f.readlines()
    res = nsml.infer(reviews)
    print(res)
