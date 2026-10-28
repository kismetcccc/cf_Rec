# coding: utf-8

import os
import itertools
import json
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


def _checkpoint_safe_values(value):
    """Keep tensor weights, but serialize metadata without NumPy pickle types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {key: _checkpoint_safe_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_safe_values(item) for item in value]
    return value


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.saved_model_file = None
        self.best_model_state = None
        self.best_epoch_idx = None

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    'Non-finite loss at epoch {}, batch {}'.format(epoch_idx, batch_idx)
                )
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
        return total_loss

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        best_model_state = None
        best_epoch_idx = None
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss = self._train_epoch(train_data, epoch_idx)
            self.lr_scheduler.step()

            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            should_evaluate = (epoch_idx + 1) % self.eval_step == 0
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # Select checkpoints from validation only. Test is evaluated once
            # after the best validation state has been restored.
            stop_flag = False
            valid_result = None
            if should_evaluate:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                if not np.isfinite(valid_score):
                    raise FloatingPointError('Non-finite validation score at epoch {}'.format(epoch_idx))
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    best_epoch_idx = epoch_idx
                    best_model_state = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }
                    self.best_model_state = best_model_state
                    self.best_epoch_idx = best_epoch_idx
                    if saved:
                        self._save_checkpoint(best_model_state, best_epoch_idx)

            # Fixed VALID interventions are descriptive curves only. Neither
            # early stopping nor the retained best state uses these results.
            # Its cadence is independent of the complete-VALID selection cadence.
            interval = self.config['missing_eval_interval']
            if (self.config['missing_diagnostics'] is True and interval
                    and (epoch_idx + 1) % interval == 0
                    and self.config['model'] == 'LGMRecOpt'):
                diagnosis_start = time()
                if valid_result is None:
                    _, valid_result = self._valid_epoch(valid_data)
                # A's curve uses zero filling; C reuses A only at final diagnosis.
                mode = ('neighbor' if self.model.neighbor_imputation
                        and self.model.modality_mask_rate > 0 else 'zero')
                curve = self.evaluator.diagnose_missing(
                    self.model, valid_data, valid_result, self.config['missing_eval_seed'],
                    imputation_modes=(mode,), include_groups=False)
                self.logger.info('Missing VALID curve: %s', json.dumps({
                    'epoch': epoch_idx, 'state_origin': 'current_epoch',
                    'used_for_selection': False, 'mask_seed': curve['mask_seed'],
                    'results': [{key: row[key] for key in ('modality', 'ratio', 'imputation', 'metrics')}
                                for row in curve['results']],
                    'seconds': time() - diagnosis_start,
                }, allow_nan=False))

            if stop_flag:
                break

        if best_model_state is not None:
            self.best_model_state = best_model_state
            self.best_epoch_idx = best_epoch_idx
            self.model.load_state_dict(best_model_state)
            if verbose:
                self.logger.info(
                    '+++++Finished training, best eval result in epoch %d'
                    % best_epoch_idx
                )
            if test_data is not None:
                _, self.best_test_upon_valid = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(
                        'selected checkpoint test result: \n'
                        + dict2str(self.best_test_upon_valid)
                    )
            if saved and test_data is not None:
                # Add the final TEST result to the already durable VALID state.
                self._save_checkpoint(best_model_state, best_epoch_idx)
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    def _save_checkpoint(self, model_state, epoch_idx):
        checkpoint_dir = os.path.abspath(
            self.config['checkpoint_dir'] or 'saved'
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        ablation = self.config['ablation'] or 'full'
        filename = '{}-{}-{}-seed{}-{}.pth'.format(
            self.config['model'],
            self.config['dataset'],
            ablation,
            self.config['seed'],
            get_local_time(),
        )
        from uuid import uuid4
        filename = filename[:-4] + '-' + uuid4().hex[:8] + '.pth'
        # One unique file per Trainer/run, updated only after a VALID improvement.
        # Older runs remain untouched and interruption keeps the last complete save.
        checkpoint_file = self.saved_model_file or os.path.join(checkpoint_dir, filename)
        effective_config = dict(getattr(self.config, 'final_config_dict', self.config))
        checkpoint = _checkpoint_safe_values({
                'model_state_dict': model_state,
                'epoch': epoch_idx,
                'best_valid_score': self.best_valid_score,
                'best_valid_result': self.best_valid_result,
                'test_result': self.best_test_upon_valid if effective_config.get('report_test', True) else None,
                'config': effective_config,
                'model': self.config['model'],
                'dataset': self.config['dataset'],
                'seed': self.config['seed'],
                'protocol': self.config['protocol'],
                'evaluation_version': self.config['evaluation_version'],
                'search_version': self.config['search_version'],
                'implementation_version': self.config[
                    'implementation_version'
                ],
                'ablation': self.config['ablation'],
            })
        temporary_file = checkpoint_file + '.' + uuid4().hex + '.tmp'
        try:
            with open(temporary_file, 'xb') as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_file, checkpoint_file)
        finally:
            if os.path.exists(temporary_file):
                os.remove(temporary_file)
        self.saved_model_file = checkpoint_file
        self.logger.info('Saved checkpoint: ' + self.saved_model_file)


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        self.model.eval()
        batches = self.evaluator.rank(self.model, eval_data)
        return self.evaluator.evaluate(batches, eval_data, is_test=is_test, idx=idx)
