import os
import sys
import copy
import csv
import yaml
import shutil
import logging
import traceback
# import random
# import joblib
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import optuna

from src.pairdata import get_dataloader
from src.metric import get_metrics
from src.log import AverageMeter, update_av_meters, update_writers
from src.utils import set_seed, get_timestamp, debug_setting
from src.network.model import get_model
from config.config import get_config
from eval_dataset import main as eval_dataset

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def inference(model, minibatch):
    model.to(device)
    sup_in = minibatch[0].to(device, dtype=torch.float32)
    inf_in = minibatch[1].to(device, dtype=torch.float32)

    img_size = cfg.data.img_size
    sup_in = sup_in.view(-1, 1, img_size, img_size)
    inf_in = inf_in.view(-1, 1, img_size, img_size)

    with autocast(enabled=cfg.learning.use_amp):
        sup_outs = model(sup_in)
        inf_outs = model(inf_in)

    return sup_outs, inf_outs


def train(model, train_loader, optimizer, av_meters):
    for value in av_meters.values():
        value.reset()

    # autocast mixed precision
    scaler = GradScaler(enabled=cfg.learning.use_amp)

    # Save previous model/optimizer for avoiding NaN loss/parameter
    # https://qiita.com/syoamakase/items/a9b3146e09f9fcafbb66
    prev_model = model.state_dict()
    prev_optimizer = optimizer.state_dict()

    model.train()
    for i, minibatch in enumerate(tqdm(train_loader)):
        sup_outs, inf_outs = inference(model, minibatch)
        label_sim = minibatch[2].to(device)

        meters, sizes = get_metrics(cfg, sup_outs, inf_outs, label_sim)

        if torch.isnan(meters['total_loss']):
            model.load_state_dict(prev_model)
            optimizer = init_optimizer(model)
            optimizer.load_state_dict(prev_optimizer)
        else:
            prev_model = copy.deepcopy(model.state_dict())
            prev_optimizer = copy.deepcopy(optimizer.state_dict())
            loss = meters['total_loss'] / \
                cfg.learning.optimizer.accumulate_epoch
            scaler.scale(loss).backward()

        # Accumulated gradients
        if (i + 1) % cfg.learning.optimizer.accumulate_epoch == 0:
            # gradient clipping
            # https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.learning.optimizer.clip_gradient)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        update_av_meters(av_meters, meters, sizes)


def test(model, test_loader, av_meters):
    for value in av_meters.values():
        value.reset()

    model.eval()
    with torch.no_grad():
        for minibatch in tqdm(test_loader):
            sup_outs, inf_outs = inference(model, minibatch)
            label_sim = minibatch[2]
            meters, sizes = get_metrics(cfg, sup_outs, inf_outs, label_sim)
            update_av_meters(av_meters, meters, sizes)


def main(trial=None) -> float:

    set_seed(cfg.seed)
    debug_setting(cfg.debug)

    timestamp = get_timestamp()
    target_dir = f'./learning_logs/{cfg.data.target}/{cfg.model.architecture}/{timestamp}'
    os.makedirs(target_dir, exist_ok=True)
    with open(f'{target_dir}/config.yaml', 'w') as config_file:
        yaml.dump(cfg.to_dict(), config_file)
    csv_file = open(f'{target_dir}/cv_result.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    acc_on_cv = np.empty(0)

    # NOTE: Cross validation is disabled when hyperparameter tuning
    if (cfg.learning.cross_validation is False) or (trial is not None):
        split_ids = [0]
    else:
        split_ids = [0, 1, 2]

    for split_id in split_ids:

        log_dir = f'{target_dir}/split_id={split_id}'
        tb_writer = SummaryWriter(log_dir)

        av_meters = {
            'dif_loss': AverageMeter(),
            'sim_loss': AverageMeter(),
            'total_loss': AverageMeter(),
            'dif_acc': AverageMeter(),
            'sim_acc': AverageMeter(),
            'total_acc': AverageMeter()
        }

        # dataset
        train_loader = get_dataloader(cfg, 'train', split_id)
        test_loader = get_dataloader(cfg, 'test', split_id)

        model = get_model(cfg, 'learning')
        initial_lr = cfg.learning.optimizer.initial_lr
        optimizer = init_optimizer(model, initial_lr)
        scheduler = StepLR(
            optimizer,
            step_size=cfg.learning.optimizer.decrease_epoch,
            gamma=cfg.learning.optimizer.gamma
        )

        state = {
            'best_loss': float('inf'),
            'best_model': model.cpu().state_dict(),
            'best_epoch': 0,
            'best_accuracy': 0.0,
            'latest_model': model.cpu().state_dict(),
            'latest_epoch': 0,
            'latest_optimizer': optimizer.state_dict()
        }

        for epoch in range(1, cfg.learning.epochs + 1):
            print(f'Epoch {epoch}')

            train(model, train_loader, optimizer, av_meters)
            update_writers(tb_writer, av_meters, 'train', epoch)
            train_acc = av_meters["total_acc"].avg
            print(f' train accuracy : {train_acc}')

            test(model, test_loader, av_meters)
            update_writers(tb_writer, av_meters, 'test', epoch)
            test_acc = av_meters["total_acc"].avg
            print(f' test  accuracy : {test_acc}')

            # save best model (without optimizer)
            if state['best_accuracy'] < test_acc:
                state['best_epoch'] = epoch
                state['best_loss'] = av_meters["total_loss"].avg
                state['best_model'] = copy.deepcopy(model).cpu().state_dict()
                state['best_accuracy'] = test_acc

            state['latest_model'] = copy.deepcopy(model).cpu().state_dict()
            state['latest_epoch'] = epoch
            state['latest_optimizer'] = copy.deepcopy(optimizer).state_dict()
            scheduler.step()
            torch.save(state, f'{log_dir}/state_dict.pt')

            # pruning
            if trial is not None:
                trial.report(state['best_accuracy'], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        acc_on_cv = np.append(acc_on_cv, state['best_accuracy'])
        print(acc_on_cv)
        csv_writer.writerow([state['best_accuracy']])
        # metric_dict = {
        #     'hparam/best_acc': float(state['best_accuracy']),
        #     'hparam/best_epoch': state['best_epoch']
        # }
        # tb_writer.add_hparams(cfg.to_dict(), metric_dict)
        tb_writer.close()

        if cfg.learning.eval_dataset:
            eval_dataset(log_dir)

    csv_file.close()

    return np.average(acc_on_cv)  # average on k-folds cross validation


def init_optimizer(model, initial_lr=None):
    optimizer_algorithm = cfg.learning.optimizer.algorithm
    if optimizer_algorithm == 'Adam':
        # eps needs to be set for stability
        # https://discuss.pytorch.org/t/adam-optimizer-fp16-autocast/101814
        if initial_lr is None:
            optimizer = optim.Adam(model.parameters())
        else:
            optimizer = optim.Adam(model.parameters(), lr=initial_lr, eps=1e-4)
    elif optimizer_algorithm == 'SGD':
        if initial_lr is None:
            optimizer = optim.SGD(model.parameters())
        else:
            optimizer = optim.SGD(model.parameters(
            ), lr=initial_lr, momentum=cfg.learning.optimizer.sgd_momentum)
    return optimizer


def objective(trial):
    # model
    cfg.model.base = trial.suggest_categorical(
        'base', ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    if cfg.model.architecture != 'PDR':
        cfg.model.disable_bad = trial.suggest_categorical(
            'disable_bad', [False, True])
    # data
    cfg.data.feature = trial.suggest_categorical(
        'feature', ['mel_spectrogram', 'mfcc'])
    cfg.data.time_len = trial.suggest_float('time_len', 0.5, 3.0)
    # learning
    # cfg.learning.batch_size = trial.suggest_int('batch_size', 1, 64)
    # sampling
    cfg.learning.sampling.method = trial.suggest_categorical(
        'sampling', ['sparse', 'dense'])
    cfg.learning.sampling.n_frame = trial.suggest_int('n_frame', 1, 16)
    # loss
    cfg.learning.loss.method = trial.suggest_categorical(
        'loss', ['marginal_loss', 'softplus'])
    cfg.learning.loss.dif_weight = trial.suggest_float('dif_weight', 0.0, 1.0)
    # optimizer
    cfg.learning.optimizer.algorithm = trial.suggest_categorical('optimizer', [
                                                                 'SGD', 'Adam'])
    cfg.learning.optimizer.initial_lr = trial.suggest_loguniform(
        'initial_lr', 1e-3, 1e-1)
    # cfg.learning.optimizer.decrease_epoch = trial.suggest_int('decrease_epoch', 10, 100)
    cfg.learning.optimizer.gamma = trial.suggest_loguniform('gamma', 1e-2, 1.0)
    cfg.learning.optimizer.accumulate_epoch = trial.suggest_int(
        'accumulate_epoch', 1, 16)
    # cfg.learning.optimizer.clip_gradient = trial.suggest_float('clip_gradient', 0.5, 3.0)
    # augmentation
    # TODO
    # cfg.learning.augmentation.add_noise = trial.suggest_categorical('add_noise',[False, True])
    cfg.learning.augmentation.time_masking = trial.suggest_categorical(
        'time_masking', [False, True])

    try:
        return main(trial)
    except Exception:
        raise optuna.TrialPruned()


def hyperparameter_tuning():
    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout))

    study_name = f"{cfg.model.architecture}_tuning"
    storage_name = f"{study_name}.db"

    # NOTE: default sampler is TPE
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///./optuna/{storage_name}",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(min_resource=10)
    )

    study.optimize(objective, n_trials=50, gc_after_trial=True)
    print(f"Best trial config: {study.best_params}")
    print(f"Best trial value: {study.best_value}")


if __name__ == "__main__":
    try:
        global cfg
        cfg = get_config()
        print(cfg)
        if cfg.tuning:
            hyperparameter_tuning()
        else:
            main()
    except Exception:
        print(traceback.format_exc())
        try:
            shutil.rmtree(log_dir)
        except Exception:
            pass
