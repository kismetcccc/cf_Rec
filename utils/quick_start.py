# coding: utf-8

from logging import getLogger
from itertools import product
import pandas as pd
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.experiment_state import fingerprint_inputs
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
import torch


def _grid_values(value):
    """Normalize one grid-search value while accepting convenient scalars."""
    if value is None:
        return [None]
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _best_result_index(hyper_results, metric):
    """Select a hyperparameter run using validation data only."""
    if not hyper_results:
        raise ValueError('hyper_results must not be empty')
    return max(
        range(len(hyper_results)),
        key=lambda index: hyper_results[index][1][metric],
    )


def _prepare_train_only_protocol(train_data, train_dataset, negative_count=1):
    """Uniform full-catalog negatives, excluding ONLY observed train positives."""
    item_count = train_dataset.get_item_num()
    history = {
        user: set(items.values)
        for user, items in train_dataset.df.groupby(train_dataset.uid_field)[train_dataset.iid_field]
    }
    if any(len(items) >= item_count for items in history.values()):
        raise ValueError('Cannot sample negatives for users that observed every item in training')
    train_data.history_items_per_u = history
    train_data.all_items = list(range(item_count))
    train_data.all_items_set = set(train_data.all_items)
    train_data.all_item_len = item_count
    negative_count = int(negative_count)
    if negative_count < 1:
        raise ValueError('negative_count must be positive')
    train_data.negative_candidate_num = negative_count


def quick_start(model, dataset, config_dict, save_model=False, initialize_logger=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    report_test = config['report_test'] is True
    if config['protocol'] == 'train_only':
        config['input_hashes'], config['data_signature'] = fingerprint_inputs(config)
    if initialize_logger:
        init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    test_history_dataset = train_dataset
    protocol = config['protocol']
    if protocol == 'train_only':
        # All current variants use the baseline TRAIN-only single-negative sampler.
        _prepare_train_only_protocol(train_data, train_dataset, 1)
        test_history_dataset = train_dataset.copy(pd.concat(
            [train_dataset.df, valid_dataset.df], ignore_index=True, sort=False))
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=test_history_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    idx = 0
    best_result_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(_grid_values(config[i]))
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    grid_search = total_loops > 1
    selected_state = None
    selected_epoch = None
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data,
            valid_data=valid_data,
            test_data=test_data if report_test and not grid_search else None,
            saved=save_model and not grid_search,
        )
        if trainer.best_model_state is None or trainer.best_epoch_idx is None:
            raise RuntimeError('Training did not produce a valid checkpoint; this run is not complete')
        #########
        hyper_ret.append(
            (
                hyper_tuple,
                best_valid_result,
                best_test_upon_valid if report_test and not grid_search else None,
            )
        )

        # Hyperparameters must be selected by validation data.  The previous
        # implementation selected by the test metric and leaked test labels.
        best_result_idx = _best_result_index(hyper_ret, val_metric)
        if grid_search and best_result_idx == len(hyper_ret) - 1:
            selected_state = trainer.best_model_state
            selected_epoch = trainer.best_epoch_idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        if not grid_search and report_test:
            logger.info(
                'test result: {}'.format(dict2str(best_test_upon_valid))
            )
        logger.info(
            '████Current BEST████:\nParameters: {}={},\nValid: {}{}'.format(
                config['hyper_parameters'],
                hyper_ret[best_result_idx][0],
                dict2str(hyper_ret[best_result_idx][1]),
                '\n\n\n'
                if grid_search or not report_test
                else ',\nTest: {}\n\n\n'.format(
                    dict2str(hyper_ret[best_result_idx][2])
                ),
            )
        )
        if grid_search:
            del trainer
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if grid_search:
        if selected_state is None or selected_epoch is None:
            raise RuntimeError('Grid search did not produce a valid checkpoint')
        selected_tuple = hyper_ret[best_result_idx][0]
        for name, value in zip(config['hyper_parameters'], selected_tuple):
            config[name] = value
        selected_model = get_model(config['model'])(config, train_data).to(
            config['device']
        )
        selected_model.load_state_dict(selected_state)
        selected_trainer = get_trainer()(config, selected_model)
        selected_trainer.best_valid_score = hyper_ret[best_result_idx][1][
            val_metric
        ]
        selected_trainer.best_valid_result = hyper_ret[best_result_idx][1]
        selected_test = None
        if report_test:
            _, selected_test = selected_trainer._valid_epoch(test_data)
        selected_trainer.best_test_upon_valid = selected_test
        if save_model:
            selected_trainer._save_checkpoint(selected_state, selected_epoch)
        hyper_ret[best_result_idx] = (
            selected_tuple,
            hyper_ret[best_result_idx][1],
            selected_test,
        )
        logger.info(
            '+++++Finished training, best eval result in epoch {}'.format(
                selected_epoch
            )
        )
        if report_test:
            logger.info('selected checkpoint test result: \n' + dict2str(selected_test))
        logger.info(
            'best valid result: {}'.format(
                dict2str(hyper_ret[best_result_idx][1])
            )
        )
        if report_test:
            logger.info('test result: {}'.format(dict2str(selected_test)))

    # log info
    logger.info('Selected effective configuration: %s', config)
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), '-' if v is None else dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_result_idx][0],
                                                                   dict2str(hyper_ret[best_result_idx][1]),
                                                                   '-' if hyper_ret[best_result_idx][2] is None else dict2str(hyper_ret[best_result_idx][2])))
    if not report_test:
        logger.info('validation-only run complete; test was not evaluated')
    final_trainer = selected_trainer if grid_search else trainer
    final_model = selected_model if grid_search else model
    behavior_view = getattr(final_model, 'behavior_view', None)
    sparse_hypergraph = getattr(final_model, 'sparse_hypergraph', None)
    return {'valid_result': hyper_ret[best_result_idx][1],
            'effective_config': {k: str(v) if isinstance(v, torch.device) else v
                                 for k, v in config.final_config_dict.items()},
            'behavior_graph': behavior_view.metadata if behavior_view is not None else None,
            'sparse_hypergraph': sparse_hypergraph.metadata if sparse_hypergraph is not None else None,
            'best_epoch': selected_epoch if grid_search else trainer.best_epoch_idx,
            'checkpoint': getattr(final_trainer, 'saved_model_file', None),
            'data_signature': config['data_signature'],
            'implementation_version': config['implementation_version'],
            'test_evaluated': report_test}
    

