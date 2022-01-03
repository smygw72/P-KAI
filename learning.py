import os
import sys
import copy
import csv
import shutil
import logging
import traceback
import random
import joblib
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import optuna
import hydra

from src.data import get_dataloader
from src.metric import get_metrics
from src.log import AverageMeter, update_av_meters, update_writers
from src.utils import set_seed, get_timestamp
from src.network.model import MyModel
from config.config import CONFIG
from eval_dataset import main as eval_dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_amp = CONFIG.learning.use_amp

def inference(model, minibatch):
    model.to(device)
    sup_in = minibatch[0].to(device, dtype=torch.float32)
    inf_in = minibatch[1].to(device, dtype=torch.float32)

    img_size = CONFIG.data.img_size
    sup_in = sup_in.view(-1, 1, img_size, img_size)
    inf_in = inf_in.view(-1, 1, img_size, img_size)

    with autocast(enabled=use_amp):
        sup_outs = model(sup_in)
        inf_outs = model(inf_in)

    return sup_outs, inf_outs


def train(model, train_loader, optimizer, av_meters):
    for value in av_meters.values():
        value.reset()

    # autocast mixed precision
    scaler = GradScaler(enabled=use_amp)

    # Save previous model/optimizer for avoiding NaN loss/parameter
    # https://qiita.com/syoamakase/items/a9b3146e09f9fcafbb66
    prev_model = model.state_dict()
    prev_optimizer = optimizer.state_dict()

    model.train()
    for i, minibatch in enumerate(tqdm(train_loader)):
        sup_outs, inf_outs = inference(model, minibatch)
        label_sim = minibatch[2].to(device)

        meters, sizes = get_metrics(sup_outs, inf_outs, label_sim)

        if torch.isnan(meters['total_loss']):
            model.load_state_dict(prev_model)
            optimizer = init_optimizer(model)
            optimizer.load_state_dict(prev_optimizer)
        else:
            prev_model = copy.deepcopy(model.state_dict())
            prev_optimizer = copy.deepcopy(optimizer.state_dict())
            loss = meters['total_loss'] / CONFIG.learning.accumulate_epoch
            scaler.scale(loss).backward()

        # Accumulated gradients
        if (i + 1) % CONFIG.learning.accumulate_epoch == 0:
            # gradient clipping
            # https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.learning.clip_gradient)
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
            meters, sizes = get_metrics(sup_outs, inf_outs, label_sim)
            update_av_meters(av_meters, meters, sizes)


def main(trial=None) -> float:
    set_seed(CONFIG.seed)
    torch.autograd.set_detect_anomaly(True)

    timestamp = get_timestamp()
    target_dir = f'./learning_logs/{CONFIG.data.target}/{CONFIG.model.architecture}/{timestamp}'
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(f'./config/{CONFIG.model.architecture}.yaml', target_dir)
    csv_file = open(f'{target_dir}/cv_result.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    acc_on_cv = np.empty(0)

    # NOTE: Cross validation is disabled when hyperparameter tuning
    if (CONFIG.learning.cross_validation is False) or (trial is not None):
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
        train_loader = get_dataloader('train', split_id)
        test_loader = get_dataloader('test', split_id)

        model = MyModel()
        initial_lr = CONFIG.learning.optimizer.initial_lr
        optimizer = init_optimizer(model, initial_lr)
        scheduler = StepLR(
            optimizer,
            step_size=CONFIG.learning.optimizer.decrease_epoch,
            gamma=CONFIG.learning.optimizer.gamma
        )

        state = {
            'best_loss': float('inf'),
            'best_model': model.cpu().state_dict(),
            'best_epoch': 0,
            'best_accuracy': 0,
            'latest_model': model.cpu().state_dict(),
            'latest_epoch': 0,
            'latest_optimizer': optimizer.state_dict()
        }
        count = 0

        for epoch in range(1, CONFIG.learning.epochs + 1):
            print(f'Epoch {epoch}')

            train(model, train_loader, optimizer, av_meters)
            update_writers(tb_writer, av_meters, 'train', epoch)
            train_loss = av_meters["total_loss"].avg
            print(f' train loss : {train_loss}')

            test(model, test_loader, av_meters)
            update_writers(tb_writer, av_meters, 'test', epoch)
            test_loss = av_meters["total_loss"].avg
            print(f' test  loss : {test_loss}')

            # save best model (without optimizer)
            if state['best_loss'] > test_loss:
                count += 1
                if count >= CONFIG.learning.save_ths:
                    state['best_epoch'] = epoch
                    state['best_loss'] = test_loss
                    state['best_model'] = model.cpu().state_dict()
                    state['best_accuracy'] = av_meters['total_acc'].avg
            else:
                count = 0
            state['latest_model'] = model.cpu().state_dict()
            state['latest_epoch'] = epoch
            state['latest_optimizer'] = optimizer.state_dict()
            scheduler.step()
            torch.save(state, f'{log_dir}/state_dict.pt')

            # pruning
            if trial is not None:
                trial.report(state['best_epoch'], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        best_acc = state['best_accuracy']
        acc_on_cv.append(best_acc)
        csv_writer.writerow([best_acc])
        tb_writer.add_hparams(CONFIG, {'best_acc': state['best_acc']})
        tb_writer.close()

    csv_file.close()
    if CONFIG.learning.eval_dataset:
        eval_dataset(log_dir)

    return np.average(acc_on_cv)  # average on k-folds cross validation


def init_optimizer(model, initial_lr=None):
    optimizer_algorithm = CONFIG.learning.optimizer.algorithm
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
            optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    return optimizer


def objective(trial):
    # model
    CONFIG.model.base = trial.suggest_categorical('base_arch', ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    if CONFIG.model.architecture != 'PDR':
        CONFIG.model.disable_bad = trial.suggest_categorical('disable_bad', [False, True])
    if CONFIG.model.architecture in ['APR_TCN', 'TCN_APR']:
        CONFIG.model.tcn.levels = trial.suggest_int('tcn_levels', 1, 6)
        CONFIG.model.tcn.kernel_size = trial.suggest_int('tcn_ksize', 1, 6)
        CONFIG.model.tcn.n_unit = trial.suggest_int('tcn_n_unit', 16, 2048)
        CONFIG.model.tcn.dropout = trial.suggest_float('tcn_dropout', 0.0, 1.0)
    # data
    # CONFIG.data.mfcc_window = trial.suggest_float('batch_size', 0.1, 3.0) # TODO
    # learning
    # CONFIG.learning.batch_size = trial.suggest_int('batch_size', 1, 64)
    # sampling
    CONFIG.learning.sampling.method = trial.suggest_categorical('sampling', ['sparse', 'dense'])
    # CONFIG.learning.sampling.n_frame = trial.suggest_int('n_frame', 1, 32)
    # loss
    CONFIG.learning.loss.method = trial.suggest_categorical('loss', ['marginal_loss', 'softplus'])
    CONFIG.learning.loss.enable_sim_loss = trial.suggest_categorical('enable_simloss', [False, True])
    # optimizer
    CONFIG.learning.optimizer.algorithm = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    CONFIG.learning.optimizer.initial_lr = trial.suggest_loguniform('initial_lr', 1e-5, 1e-1)
    # CONFIG.learning.optimizer.decrease_epoch = trial.suggest_int('decrease_epoch', 10, 100)
    # CONFIG.learning.optimizer.gamma = trial.suggest_float('gamma', 0.1, 1.0)
    CONFIG.learning.accumulate_epoch = trial.suggest_int('clip_gradient', 1, 16)
    # CONFIG.learning.clip_gradient = trial.suggest_float('clip_gradient', 0.5, 3.0)
    # augmentation
    # CONFIG.learning.augmentation.add_noise = trial.suggest_categorical('add_noise',[False, True]) # TODO
    CONFIG.learning.augmentation.time_masking = trial.suggest_categorical('time_masking',[False, True])

    return main(trial)

def hyperparameter_tuning():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # NOTE: default sampler is TPE
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner
    )

    study.optimize(objective, n_trials=20, gc_after_trial=True)
    print(f"Best trial config: {study.best_params}")
    print(f"Best trial value: {study.best_value}")
    timestamp = get_timstamp()
    joblib.dump(study, f"./optuna/{timestamp}/study.pkl")

    # save
    tb_writer = SummaryWriter(f'./optuna/{timestamp}/')
    df = study.trials_dataframe()
    df_records = df.to_dict(orient='records')
    for i in range(len(df_records)):
        df_records[i]['datetime_start'] = str(df_records[i]['datetime_start'])
        df_records[i]['datetime_complete'] = str(df_records[i]['datetime_complete'])
        value = df_records[i].pop('value')
        value_dict = {'value': value}
        tb_writer.add_hparams(df_records[i], value_dict)
    tb_writer.close()

if __name__ == "__main__":
    try:
        # uncomment desirable one
        hyperparameter_tuning()
        # main(split_id=0)
    except Exception as e:
        print(traceback.format_exc())
        shutil.rmtree(log_dir)
