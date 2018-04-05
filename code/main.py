import argparse
import os
from os.path import dirname, abspath, join, exists
import numpy as np

import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader

import tokenizers
import feature_extractors
import dictionaries
from dataloader import load_data, collate_fn, Preprocessor, MovieReviewDataset
from trainers import Trainer
import utils

import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

from models.WordCNN import WordCNN
from models.VDCNN import VDCNN

# Random seed
np.random.seed(0)
torch.manual_seed(0)

args = argparse.ArgumentParser()
# DONOTCHANGE: They are reserved for nsml
args.add_argument('--mode', type=str, default='train')
args.add_argument('--pause', type=int, default=0)
args.add_argument('--iteration', type=str, default='0')

# User options
args.add_argument('--model', type=str, default='WordCNN', choices=['WordCNN', 'VDCNN'])
args.add_argument('--tokenizer', type=str, default='JamoTokenizer', choices=['JamoTokenizer', 'DummyTokenizer'])
args.add_argument('--features', type=str, default='LengthFeatureExtractor')
args.add_argument('--dictionary', type=str, default='RandomDictionary', choices=['RandomDictionary', 'FasttextDictionary'])
args.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available() or GPU_NUM)
args.add_argument('--output', type=int, default=1)
args.add_argument('--epochs', type=int, default=10)
args.add_argument('--batch_size', type=int, default=64)
args.add_argument('--max_vocab_size', type=int, default=50000)
args.add_argument('--min_count', type=int, default=None)
args.add_argument('--min_length', type=int, default=5)
args.add_argument('--max_length', type=int, default=300)
args.add_argument('--sort_dataset', action='store_true')
args.add_argument('--shuffle_dataset', action='store_true')
args.add_argument('--embedding_size', type=int, default=100)
args.add_argument('--learning_rate', type=float, default=0.01)
args.add_argument('--lr_schedule', action='store_true')
args.add_argument('--print_every', type=int, default=1)
args.add_argument('--save_every', type=int, default=1)
config = args.parse_args()

logger = utils.get_logger('MovieReview')
logger.info('Arguments: {}'.format(config))

if config.model == 'WordCNN': Model = WordCNN
elif config.model == 'VDCNN': Model = VDCNN

Tokenizer = getattr(tokenizers, config.tokenizer)
tokenizer = Tokenizer(config)

Dictionary = getattr(dictionaries, config.dictionary)
dictionary = Dictionary(tokenizer, config)

feature_extractor_list = []
for feature_name in config.features.split():
    FeatureExtractor = getattr(feature_extractors, feature_name)
    feature_extractor = FeatureExtractor(config)
    feature_extractor_list.append((feature_name, feature_extractor))
    
# base_dir = dirname(abspath(__file__))
preprocess_params_dir = 'preprocess_params' # join(base_dir, 'preprocess_params')

dictionary_params_filename = join(preprocess_params_dir, config.dictionary)
dictionary.load(dictionary_params_filename + '.pkl', dictionary_params_filename + '-embedding.pkl')

for feature_name, feature_extractor in feature_extractor_list:
    # Load parameters of feature extractors
    feature_params_filename = join(preprocess_params_dir, feature_name + '.pkl')
    feature_extractor.load(feature_params_filename)
preprocessor = Preprocessor(tokenizer, feature_extractor_list, dictionary)

model = Model(dictionary, config)
model = model.cuda()

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        reviews, features = preprocessor.preprocess_all(raw_data)
        reviews, features = Variable(reviews.cuda()), Variable(features.cuda())
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(reviews, features)
        point = output_prediction.data.tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

# DONOTCHANGE: Reserved for nsml use
bind_model(model, config)
    
if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
    DATASET_PATH = 'data/movie_review_phase1/'

# DONOTCHANGE: They are reserved for nsml
if config.pause:
    nsml.paused(scope=locals())

# 학습 모드일 때 사용합니다. (기본값)
if config.mode == 'train':
    # 데이터를 로드합니다.
    logger.info("Loading data...")
    train_data, val_data = load_data(DATASET_PATH, val_size=0.1)

    logger.info("Building preprocessor...")
    if not exists(preprocess_params_dir):
        os.mkdir(preprocess_params_dir)
    
    for feature_name, feature_extractor in feature_extractor_list:
        feature_extractor.fit(train_data)
        # Save parameters of feature extractors
        feature_params_filename = join(preprocess_params_dir, feature_name + '.pkl')
        feature_extractor.save(feature_params_filename)
        
    dictionary.build_dictionary(train_data)
    dictionary_params_filename = join(preprocess_params_dir, config.dictionary)
    dictionary.save(dictionary_params_filename + '.pkl', dictionary_params_filename + '-embedding.pkl')
    preprocessor = Preprocessor(tokenizer, feature_extractor_list, dictionary)

    logger.info("Making dataset & dataloader...")
    train_dataset = MovieReviewDataset(train_data, preprocessor, sort=config.sort_dataset, min_length=config.min_length, max_length=config.max_length)
    val_dataset = MovieReviewDataset(val_data, preprocessor, sort=config.sort_dataset, min_length=config.min_length, max_length=config.max_length)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=config.shuffle_dataset, collate_fn=collate_fn,
                              num_workers=2)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=2)

    model = Model(dictionary, config)
    if config.use_gpu:
        model = model.cuda()

    criterion = nn.MSELoss(size_average=False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params=trainable_params, lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)  # .ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.00005)

    trainer = Trainer(model, train_dataloader, val_dataloader, criterion=criterion, optimizer=optimizer,
                      lr_schedule=config.lr_schedule, lr_scheduler=lr_scheduler, use_gpu=config.use_gpu, logger=logger)
    trainer.run(epochs=config.epochs)

# 로컬 테스트 모드일때 사용합니다
# 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
# [(0.0, 9.045), (0.0, 5.91), ... ]
elif config.mode == 'test_local':
    with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
        reviews = f.readlines()
    res = nsml.infer(reviews[:10])
    print(res)