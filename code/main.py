import argparse
import os
import numpy as np
import tensorflow as tf

from tokenizers import DummyTokenizer
from feature_extractors import LengthFeatureExtractor
from dictionaries import RandomWordDictionary
from dataloader import load_data, Preprocessor, MovieReviewDataset, MovieReviewDataLoader
from models.BaselineRegression import BaselineRegression

# import nsml
# from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
IS_ON_NSML = None
HAS_DATASET = None

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        pass
        # TODO: TF로 바꾸기
#         checkpoint = {
#             'model': model.state_dict()
#         }
#         torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        # TODO: TF로 바꾸기
#         checkpoint = torch.load(filename)
#         model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocessor.preprocess(raw_data)
#         model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data)

        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

args = argparse.ArgumentParser()
# DONOTCHANGE: They are reserved for nsml
args.add_argument('--mode', type=str, default='train')
args.add_argument('--pause', type=int, default=0)
args.add_argument('--iteration', type=str, default='0')

# User options
args.add_argument('--output', type=int, default=1)
args.add_argument('--epochs', type=int, default=10)
args.add_argument('--batch_size', type=int, default=64)
args.add_argument('--max_vocab_size', type=int, default=10000)
args.add_argument('--min_count', type=int, default=3)
args.add_argument('--sentence_length', type=int, default=20)
args.add_argument('--embedding_size', type=int, default=100)
args.add_argument('--learning_rate', type=float, default=0.01)
args.add_argument('--print_every', type=int, default=1)
args.add_argument('--save_every', type=int, default=1)
config = args.parse_args()

if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
    DATASET_PATH = 'data/movie_review_phase1/'

# DONOTCHANGE: They are reserved for nsml
if config.pause:
    nsml.paused(scope=locals())

# 학습 모드일 때 사용합니다. (기본값)
if config.mode == 'train':
    # 데이터를 로드합니다.
    train_data, val_data = load_data(DATASET_PATH, val_size=0.3)
    
    tokenizer = DummyTokenizer(config)
    feature_extractor1 = LengthFeatureExtractor(config)
    feature_extractors = [feature_extractor1]
    dictionary = RandomWordDictionary(config)
    dictionary.build_dictionary(train_data)

    preprocessor = Preprocessor(tokenizer, feature_extractors, dictionary)
    
    train_dataset = MovieReviewDataset(train_data, preprocessor, dictionary, sort=False, min_length=config.sentence_length, max_length=config.sentence_length)
    val_dataset = MovieReviewDataset(val_data, preprocessor, dictionary, sort=False, min_length=config.sentence_length, max_length=config.sentence_length)
    
    train_dataloader = MovieReviewDataLoader(dataset=train_dataset, dictionary=dictionary, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_dataloader = MovieReviewDataLoader(dataset=val_dataset, dictionary=dictionary, batch_size=64, shuffle=True, drop_last=False)

    model = BaselineRegression(dictionary, config)
    model.build_model()
    placeholders = [model.reviews, model.features, model.labels]

    # DONOTCHANGE: Reserved for nsml use
    # bind_model(model, config)

    loss = tf.losses.mean_squared_error(labels=model.labels, predictions=model.predictions)

    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    variables_to_train = tf.trainable_variables()
    trainer = optimizer.minimize(loss, var_list=variables_to_train)

    session = tf.Session()
    tf.global_variables_initializer().run(session=session)
    saver = tf.train.Saver(max_to_keep=1000)

    epoch_losses = []
    epoch_val_losses = []
    checkpoint_path = os.path.join('checkpoints', 'model_ckpt')

    for epoch in range(config.epochs):

        batch_losses = []
        batch_metrics = []
        for train_batch in train_dataloader:

            _, loss_value = session.run([trainer, loss], feed_dict=dict(zip(placeholders, train_batch)))

            batch_losses.append(loss_value)

            if epoch == 0:
                break  # for test

        epoch_loss = np.array(batch_losses).mean()
        epoch_losses.append(epoch_loss)

        if epoch % config.print_every == 0:

            # Calculate val loss
            val_loss_values = []
            for val_batch in val_dataloader:

                val_loss_value = session.run([loss], feed_dict=dict(
                    zip(placeholders, val_batch)))

                val_loss_values.append(val_loss_value)

            epoch_val_loss = np.array(val_loss_values).mean()
            epoch_val_losses.append(epoch_val_loss)

            # Leave log
            base_message = ("Epoch: {epoch:<3d} Loss: {loss:<.6} "
                            "Val Loss: {val_loss:<.6} "
                            "Learning rate: {learning_rate:<.4}")
            message = base_message.format(epoch=epoch, loss=epoch_loss,
                val_loss=epoch_val_loss, learning_rate=config.learning_rate)

            print(message, flush=True)
            # nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
            #             train__loss=epoch_loss, step=epoch)

        if epoch % config.save_every == 0:
            # Save
            saver.save(session, checkpoint_path, global_step=epoch)
            # nsml.save(epoch)

    # Final save
    saver.save(session, checkpoint_path, global_step=epoch)
    # nsml.save(epoch)

# 로컬 테스트 모드일때 사용합니다
# 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
# [(0.0, 9.045), (0.0, 5.91), ... ]
elif config.mode == 'test_local':
    with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
        reviews = f.readlines()
    res = nsml.infer(reviews)
    print(res)