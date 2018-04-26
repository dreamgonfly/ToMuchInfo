from os.path import dirname, abspath, join, exists
import os
from datetime import datetime
from tqdm import tqdm
import torch
from torch.autograd import Variable
from collections import defaultdict
from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import train_test_split

import nsml

STOP_TRAIN_AFTER = 4

class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer, lr_schedule, lr_scheduler, min_lr,
                 use_gpu=False, print_every=1, save_every=1, logger=None):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.lr_scheduler = lr_scheduler
        self.min_lr = min_lr

        self.print_every = print_every
        self.save_every = save_every

        self.epoch = 0
        self.epoch_losses = []
#         self.epoch_metrics = []
        self.val_epoch_losses = []
#         self.val_epoch_metrics = []
        self.use_gpu = use_gpu
        self.logger = logger

        self.base_message = ("Epoch: {epoch:<3d} "
                             "Progress: {progress:<.1%} ({elapsed}) "
                             "Train Loss: {train_loss:<.6} "
#                              "Train Acc: {train_metric:<.1%} "
                             "Val Loss: {val_loss:<.6} "
#                              "Val Acc: {val_metric:<.1%} "
                             "Learning rate: {learning_rate:<.4} ")

        self.start_time = datetime.now()

    def train(self):
        self.model.train()

        self.batch_losses = []
#         self.batch_metrics = []
        for inputs, features, targets in tqdm(self.train_dataloader):

            if self.use_gpu:
                self.inputs, self.features, self.targets = Variable(inputs.cuda()), Variable(features.cuda()), Variable(targets.cuda())
            else:
                self.inputs, self.features, self.targets = Variable(inputs), Variable(features), Variable(targets)

            self.optimizer.zero_grad()
            if hasattr(self.model, 'init_hidden'):
                self.model.batch_size = len(inputs)
                self.model.hidden = self.model.init_hidden()
            self.outputs = self.model(self.inputs, self.features)
            if type(self.outputs) == tuple:
                batch_loss = self.criterion(self.outputs[0], self.targets) + self.outputs[1]
            else:
                batch_loss = self.criterion(self.outputs, self.targets)
#             batch_metric = self.accuracy(self.outputs, self.targets)

            batch_loss.backward()
            self.optimizer.step()

            self.batch_losses.append(batch_loss.data)
#             self.batch_metrics.append(batch_metric.data)
            if self.epoch == 0:  # for testing
                break

        # validation
        self.model.eval()
        self.val_batch_losses = []
#         self.val_batch_metrics = []
        for val_inputs, val_features, val_targets in self.val_dataloader:
            if self.use_gpu:
                self.val_inputs, self.val_features, self.val_targets = Variable(val_inputs.cuda()), Variable(val_features.cuda()), Variable(val_targets.cuda())
            else:
                self.val_inputs, self.val_features, self.val_targets = Variable(val_inputs), Variable(val_features), Variable(val_targets)

            if hasattr(self.model, 'init_hidden'):
                self.model.batch_size = len(val_inputs)
                self.model.hidden = self.model.init_hidden()
            self.val_outputs = self.model(self.val_inputs, self.val_features)
            self.val_outputs = torch.clamp(self.val_outputs, min=1, max=10)
            if type(self.val_outputs) == tuple:
                val_batch_loss = self.criterion(self.val_outputs[0], self.val_targets) + self.val_outputs[1]
            else:
                val_batch_loss = self.criterion(self.val_outputs, self.val_targets)
#             val_batch_metric = self.accuracy(self.val_outputs, self.val_targets)
            self.val_batch_losses.append(val_batch_loss.data)
#             self.val_batch_metrics.append(val_batch_metric.data)

        train_data_size = len(self.train_dataloader.dataset)
        epoch_loss = torch.cat(self.batch_losses).sum() / train_data_size
#         epoch_metric = torch.cat(self.batch_metrics).sum() / train_data_size

        val_data_size = len(self.val_dataloader.dataset)
        val_epoch_loss = torch.cat(self.val_batch_losses).sum() / val_data_size
#         val_epoch_metric = torch.cat(self.val_batch_metrics).sum() / val_data_size

        return epoch_loss, val_epoch_loss

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            epoch_loss, val_epoch_loss = self.train()
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.lr_schedule and current_lr > self.min_lr:
                self.lr_scheduler.step()

            self.epoch_losses.append(epoch_loss)
#             self.epoch_metrics.append(epoch_metric)
            self.val_epoch_losses.append(val_epoch_loss)
#             self.val_epoch_metrics.append(val_epoch_metric)

            if epoch % self.print_every == 0:
                message = self.base_message.format(epoch=epoch, progress=epoch / epochs, train_loss=epoch_loss,
                                                   val_loss=val_epoch_loss,
                                                   learning_rate=current_lr,
                                                   elapsed=self.elapsed_time())
                self.logger.info(message)
                nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=epochs,
                            train__loss=epoch_loss, val__loss=val_epoch_loss, step=epoch)

            if epoch % self.save_every == 0:
                self.logger.info("Saving the model...")
                self.save_model()

                # DONOTCHANGE (You can decide how often you want to save the model)
                nsml.save(epoch)

#     def accuracy(self, outputs, labels):

#         maximum, argmax = outputs.max(dim=1)
#         corrects = argmax == labels  # ByteTensor
#         n_corrects = corrects.float().sum()  # FloatTensor
#         return n_corrects

    def elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed)

    def save_model(self):
        base_dir = dirname(abspath(__file__))
        checkpoint_dir = join(base_dir, 'checkpoints')
        if not exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        model_name = self.model.__class__.__name__
        base_filename = '{model_name}-{start_time}-{epoch}.pth'
        checkpoint_filename = base_filename.format(model_name=model_name, start_time=self.start_time, epoch=self.epoch)
        checkpoint_filepath = join(checkpoint_dir, checkpoint_filename)
        torch.save(self.model.state_dict(), checkpoint_filepath)
        self.last_checkpoint_filepath = checkpoint_filepath
        if max(self.val_epoch_losses) == self.val_epoch_losses[-1]:  # if last run is the best
            self.best_checkpoint_filepath = checkpoint_filepath


class EnsembleTrainer():
    def __init__(self, ensemble_models, use_gpu=False, print_every=1, save_every=1, logger=None):

        self.ensemble_models = ensemble_models

        self.print_every = print_every
        self.save_every = save_every

        self.epoch = 0
        self.epoch_losses = defaultdict(list)
#         self.epoch_metrics = []
        self.val_epoch_losses = defaultdict(list)
#         self.val_epoch_metrics = []
        self.use_gpu = use_gpu
        self.logger = logger

        self.base_message = ("Epoch: {epoch:<3d} "
                             "Progress: {progress:<.1%} ({elapsed}) "
                             "Train Loss: {train_loss:<.6} "
#                              "Train Acc: {train_metric:<.1%} "
                             "Val Loss: {val_loss:<.6} "
#                              "Val Acc: {val_metric:<.1%} "
                             "Learning rate: {learning_rate:<.4} ")

        self.start_time = datetime.now()

    def train(self, model, criterion, optimizer, train_dataloader, val_dataloader, ):
        model.train()

        batch_losses = []
#         self.batch_metrics = []
        for inputs, features, targets in tqdm(train_dataloader):

            if self.use_gpu:
                inputs, features, targets = Variable(inputs.cuda()), Variable(features.cuda()), Variable(targets.cuda())
            else:
                inputs, features, targets = Variable(inputs), Variable(features), Variable(targets)

            optimizer.zero_grad()
            if hasattr(model, 'init_hidden'):
                model.batch_size = len(inputs)
                model.hidden = model.init_hidden()
            outputs = model(inputs, features)
            if type(outputs) == tuple:
                batch_loss = criterion(outputs[0], targets) + outputs[1]
            else:
                batch_loss = criterion(outputs, targets)
#             batch_metric = self.accuracy(self.outputs, self.targets)

            batch_loss.backward()
            optimizer.step()

            batch_losses.append(batch_loss.data)
#             self.batch_metrics.append(batch_metric.data)
            if self.epoch == 0:  # for testing
                break

        # validation
        model.eval()
        val_batch_losses = []
#         self.val_batch_metrics = []
        for val_inputs, val_features, val_targets in val_dataloader:
            if self.use_gpu:
                val_inputs, val_features, val_targets = Variable(val_inputs.cuda()), Variable(val_features.cuda()), Variable(val_targets.cuda())
            else:
                val_inputs, val_features, val_targets = Variable(val_inputs), Variable(val_features), Variable(val_targets)

            if hasattr(model, 'init_hidden'):
                model.batch_size = len(val_inputs)
                model.hidden = model.init_hidden()
            val_outputs = model(val_inputs, val_features)
            val_outputs = torch.clamp(val_outputs, min=1, max=10)
            if type(val_outputs) == tuple:
                val_batch_loss = criterion(val_outputs[0], val_targets) + val_outputs[1]
            else:
                val_batch_loss = criterion(val_outputs, val_targets)
#             val_batch_metric = self.accuracy(self.val_outputs, self.val_targets)
            val_batch_losses.append(val_batch_loss.data)
#             self.val_batch_metrics.append(val_batch_metric.data)

        train_data_size = len(train_dataloader.dataset)
        epoch_loss = torch.cat(batch_losses).sum() / train_data_size
#         epoch_metric = torch.cat(self.batch_metrics).sum() / train_data_size

        val_data_size = len(val_dataloader.dataset)
        val_epoch_loss = torch.cat(val_batch_losses).sum() / val_data_size
#         val_epoch_metric = torch.cat(self.val_batch_metrics).sum() / val_data_size

        return epoch_loss, val_epoch_loss

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            for config_name in self.ensemble_models:
                if len(self.val_epoch_losses[config_name]) >= STOP_TRAIN_AFTER and \
                                self.val_epoch_losses[config_name][-STOP_TRAIN_AFTER] < self.val_epoch_losses[config_name][-3] and \
                                self.val_epoch_losses[config_name][-3] < self.val_epoch_losses[config_name][-2] and \
                                self.val_epoch_losses[config_name][-3] < self.val_epoch_losses[config_name][-1]:
                    self.logger.info("Skip {}".format(config_name))
                    continue
                self.logger.info("Training {}".format(config_name))
                model = self.ensemble_models[config_name]['model']
                criterion = self.ensemble_models[config_name]['criterion']
                optimizer = self.ensemble_models[config_name]['optimizer']
                lr_schedule = self.ensemble_models[config_name]['config'].lr_schedule
                lr_scheduler = self.ensemble_models[config_name]['lr_scheduler']
                min_lr = self.ensemble_models[config_name]['config'].min_lr
                train_dataloader = self.ensemble_models[config_name]['train_dataloader']
                val_dataloader = self.ensemble_models[config_name]['val_dataloader']

                epoch_loss, val_epoch_loss = self.train(model, criterion, optimizer, train_dataloader, val_dataloader, )
                current_lr = optimizer.param_groups[0]['lr']
                if lr_schedule and current_lr > min_lr:
                    lr_scheduler.step()

                self.epoch_losses[config_name].append(epoch_loss)
    #             self.epoch_metrics.append(epoch_metric)
                self.val_epoch_losses[config_name].append(val_epoch_loss)
    #             self.val_epoch_metrics.append(val_epoch_metric)

                config_result_message = (
                    "Config: {config} "
                    "Epoch: {epoch:<3d} "
                    "Progress: {progress:<.1%} ({elapsed}) "
                    "Train Loss: {train_loss:<.6} "
                    #                              "Train Acc: {train_metric:<.1%} "
                    "Val Loss: {val_loss:<.6} "
                    #                              "Val Acc: {val_metric:<.1%} "
                    "Learning rate: {learning_rate:<.4} ")

                if epoch % self.print_every == 0:
                    message = config_result_message.format(config=config_name, epoch=epoch, progress=epoch / epochs, train_loss=epoch_loss,
                                                       val_loss=val_epoch_loss, learning_rate=current_lr,
                                                       elapsed=self.elapsed_time())
                    self.logger.info(message)

                if 'best_loss' not in self.ensemble_models[config_name] \
                        or val_epoch_loss < self.ensemble_models[config_name]['best_loss']:
                    # self.logger.info("Saving the model for {}".format(config_name))
                    self.ensemble_models[config_name]['best_loss'] = val_epoch_loss
                    # self.ensemble_models[config_name]['best_model'] = model.state_dict()

                self.logger.info('best_loss {}'.format(self.ensemble_models[config_name]['best_loss']))

            # if epoch % self.print_every == 0:
            #     current_lr = self.optimizer.param_groups[0]['lr']
            #     message = self.base_message.format(epoch=epoch, progress=epoch / epochs, train_loss=epoch_loss,
            #                                        val_loss=val_epoch_loss,
            #                                        learning_rate=current_lr,
            #                                        elapsed=self.elapsed_time())
            #     self.logger.info(message)
            #     nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=epochs,
            #                 train__loss=epoch_loss, val__loss=val_epoch_loss, step=epoch)

            if epoch % self.save_every == 0:
                # self.logger.info("Saving the model...")
                # self.save_model()

                # DONOTCHANGE (You can decide how often you want to save the model)
                nsml.save(epoch)

#     def accuracy(self, outputs, labels):

#         maximum, argmax = outputs.max(dim=1)
#         corrects = argmax == labels  # ByteTensor
#         n_corrects = corrects.float().sum()  # FloatTensor
#         return n_corrects

    def elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed)

    def save_model(self):
        base_dir = dirname(abspath(__file__))
        checkpoint_dir = join(base_dir, 'checkpoints')
        if not exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        model_name = self.model.__class__.__name__
        base_filename = '{model_name}-{start_time}-{epoch}.pth'
        checkpoint_filename = base_filename.format(model_name=model_name, start_time=self.start_time, epoch=self.epoch)
        checkpoint_filepath = join(checkpoint_dir, checkpoint_filename)
        torch.save(self.model.state_dict(), checkpoint_filepath)
        self.last_checkpoint_filepath = checkpoint_filepath
        if max(self.val_epoch_losses) == self.val_epoch_losses[-1]:  # if last run is the best
            self.best_checkpoint_filepath = checkpoint_filepath


class EnsembleTrainer_xgb():
    def __init__(self, ensemble_models, use_gpu=False, print_every=1, save_every=1, logger=None, xgb=None):

        self.ensemble_models = ensemble_models

        self.print_every = print_every
        self.save_every = save_every

        self.epoch = 0
        self.epoch_losses = defaultdict(list)
#         self.epoch_metrics = []
        self.val_epoch_losses = defaultdict(list)
#         self.val_epoch_metrics = []
        self.use_gpu = use_gpu
        self.logger = logger

        self.base_message = ("Epoch: {epoch:<3d} "
                             "Progress: {progress:<.1%} ({elapsed}) "
                             "Train Loss: {train_loss:<.6} "
#                              "Train Acc: {train_metric:<.1%} "
                             "Val Loss: {val_loss:<.6} "
#                              "Val Acc: {val_metric:<.1%} "
                             "Learning rate: {learning_rate:<.4} ")

        self.start_time = datetime.now()
        self.train_predictions = defaultdict(list)
        self.train_labels = defaultdict(list)
        self.xgb = xgb

    def train(self, model, criterion, optimizer, train_dataloader, val_dataloader, config_name, is_best_epoch=False):
        model.train()

        batch_losses = []
#         self.batch_metrics = []

        for inputs, features, targets in tqdm(train_dataloader):

            if self.use_gpu:
                inputs, features, targets = Variable(inputs.cuda()), Variable(features.cuda()), Variable(targets.cuda())
            else:
                inputs, features, targets = Variable(inputs), Variable(features), Variable(targets)

            optimizer.zero_grad()
            if hasattr(model, 'init_hidden'):
                model.batch_size = len(inputs)
                model.hidden = model.init_hidden()
            outputs = model(inputs, features)
            if type(outputs) == tuple:
                batch_loss = criterion(outputs[0], targets) + outputs[1]
            else:
                batch_loss = criterion(outputs, targets)
#             batch_metric = self.accuracy(self.outputs, self.targets)

            batch_loss.backward()
            optimizer.step()

            batch_losses.append(batch_loss.data)
#             self.batch_metrics.append(batch_metric.data)
#             print("[ENSEMBLE XGB] outputs : Type : {}, Length : {}".format(type(outputs), len(outputs)))
#             print("[ENSEMBLE XGB] targets : Type : {}, Length : {}".format(type(targets), len(targets)))

            if is_best_epoch:
                self.train_predictions[config_name] += outputs.data.tolist()
                self.train_labels[config_name] += outputs.data.tolist()


        # validation
        model.eval()
        val_batch_losses = []
#         self.val_batch_metrics = []
        for val_inputs, val_features, val_targets in val_dataloader:
            if self.use_gpu:
                val_inputs, val_features, val_targets = Variable(val_inputs.cuda()), Variable(val_features.cuda()), Variable(val_targets.cuda())
            else:
                val_inputs, val_features, val_targets = Variable(val_inputs), Variable(val_features), Variable(val_targets)

            if hasattr(model, 'init_hidden'):
                model.batch_size = len(val_inputs)
                model.hidden = model.init_hidden()
            val_outputs = model(val_inputs, val_features)
            val_outputs = torch.clamp(val_outputs, min=1, max=10)
            if type(val_outputs) == tuple:
                val_batch_loss = criterion(val_outputs[0], val_targets) + val_outputs[1]
            else:
                val_batch_loss = criterion(val_outputs, val_targets)
#             val_batch_metric = self.accuracy(self.val_outputs, self.val_targets)
            val_batch_losses.append(val_batch_loss.data)
#             self.val_batch_metrics.append(val_batch_metric.data)

        train_data_size = len(train_dataloader.dataset)

        epoch_loss = torch.cat(batch_losses).sum() / train_data_size
#         epoch_metric = torch.cat(self.batch_metrics).sum() / train_data_size

        val_data_size = len(val_dataloader.dataset)
        val_epoch_loss = torch.cat(val_batch_losses).sum() / val_data_size
#         val_epoch_metric = torch.cat(self.val_batch_metrics).sum() / val_data_size

        return epoch_loss, val_epoch_loss

    def run(self, epochs=10):

        for epoch in range(2):
            self.epoch = epoch

            for config_name in self.ensemble_models:
                self.logger.info("[ENSEMBLE XGB] Training {}".format(config_name))
                model = self.ensemble_models[config_name]['model']
                criterion = self.ensemble_models[config_name]['criterion']
                optimizer = self.ensemble_models[config_name]['optimizer']
                lr_schedule = self.ensemble_models[config_name]['config'].lr_schedule
                lr_scheduler = self.ensemble_models[config_name]['lr_scheduler']
                min_lr = self.ensemble_models[config_name]['config'].min_lr
                train_dataloader = self.ensemble_models[config_name]['train_dataloader']
                val_dataloader = self.ensemble_models[config_name]['val_dataloader']

                best_epoch = self.ensemble_models[config_name]['config'].best_epoch

                for i in range(best_epoch):
                    if i == best_epoch-1:
                        epoch_loss, val_epoch_loss = self.train(model, criterion, optimizer, train_dataloader,
                                                                val_dataloader, config_name, is_best_epoch=True)
                    else:
                        epoch_loss, val_epoch_loss = self.train(model, criterion, optimizer, train_dataloader, val_dataloader, config_name, is_best_epoch=False)

                    current_lr = optimizer.param_groups[0]['lr']

                    if lr_schedule and current_lr > min_lr:
                        lr_scheduler.step()

                    config_result_message = (
                        "Config: {config} "
                        "Epoch: {epoch:<3d} "
                        "Progress: {progress:<.1%} ({elapsed}) "
                        "Train Loss: {train_loss:<.6} "
                        "Val Loss: {val_loss:<.6} "
                        "Learning rate: {learning_rate:<.4} ")

                    if i % self.print_every == 0:
                        message = config_result_message.format(config=config_name, epoch=i, progress=epoch / epochs, train_loss=epoch_loss,
                                                           val_loss=val_epoch_loss, learning_rate=current_lr,
                                                           elapsed=self.elapsed_time())
                        self.logger.info(message)

                print("[ENSEMBLE XGB] train_predictions : {}".format(len(self.train_predictions[config_name])))
                print("[ENSEMBLE XGB] train_labels : {}".format(len(self.train_labels[config_name])))

                predictions = None
                for model_name in sorted(self.train_predictions):
                    model_prediction = self.train_predictions[model_name]
                    if not predictions:
                        predictions = np.array(model_prediction)
                        labels = self.train_labels[model_name]
                    else:
                        predictions = np.concatenate((predictions, np.array(model_prediction)))
                X_train, X_val, y_train, y_val = train_test_split(predictions, labels)
                print("Ensemble Training Start!")
                self.xgb.fit(X_train, y_train)
                print("Ensemble loss : {}".format(self.xgb.score(X_val, y_val)))

            # if epoch % self.print_every == 0:
            #     current_lr = self.optimizer.param_groups[0]['lr']
            #     message = self.base_message.format(epoch=epoch, progress=epoch / epochs, train_loss=epoch_loss,
            #                                        val_loss=val_epoch_loss,
            #                                        learning_rate=current_lr,
            #                                        elapsed=self.elapsed_time())
            #     self.logger.info(message)
            #     nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=epochs,
            #                 train__loss=epoch_loss, val__loss=val_epoch_loss, step=epoch)

            if epoch % self.save_every == 0:
                # self.logger.info("Saving the model...")
                # self.save_model()

                # DONOTCHANGE (You can decide how often you want to save the model)
                nsml.save(epoch)

    def elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed)

    def save_model(self):
        base_dir = dirname(abspath(__file__))
        checkpoint_dir = join(base_dir, 'checkpoints')
        if not exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        model_name = self.model.__class__.__name__
        base_filename = '{model_name}-{start_time}-{epoch}.pth'
        checkpoint_filename = base_filename.format(model_name=model_name, start_time=self.start_time, epoch=self.epoch)
        checkpoint_filepath = join(checkpoint_dir, checkpoint_filename)
        torch.save(self.model.state_dict(), checkpoint_filepath)
        self.last_checkpoint_filepath = checkpoint_filepath
        if max(self.val_epoch_losses) == self.val_epoch_losses[-1]:  # if last run is the best
            self.best_checkpoint_filepath = checkpoint_filepath