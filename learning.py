import os
import copy
import shutil
import traceback
import random
import numpy as np
from pytz import timezone
from datetime import datetime
from tqdm import tqdm
import hydra
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from src.data import get_dataloader
from src.metric import get_metrics
from src.log import AverageMeter, update_av_meters, update_writers
from src.utils import set_seed
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
    for minibatch in tqdm(train_loader):
        sup_outs, inf_outs = inference(model, minibatch)
        label_sim = minibatch[2].to(device)

        meters, sizes = get_metrics(sup_outs, inf_outs, label_sim)
        if torch.isnan(meters['total_loss']):
            model.load_state_dict(prev_model)
            optimizer = init_optimizer(model)
            optimizer.load_state_dict(prev_optimizer)
        prev_model = copy.deepcopy(model.state_dict())
        prev_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad()
        scaler.scale(meters['total_loss']).backward()

        # gradient clipping
        # https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.learning.clip_gradient)

        scaler.step(optimizer)
        scaler.update()

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


def main():
    set_seed(CONFIG.seed)
    torch.autograd.set_detect_anomaly(True)

    global log_dir
    utc_now = datetime.now(timezone('UTC'))
    jst_now = utc_now.astimezone(timezone('Asia/Tokyo'))
    timestamp = datetime.strftime(jst_now, '%m-%d-%H-%M-%S')
    log_dir = f'./learning_logs/{CONFIG.data.target}/{CONFIG.model.architecture}/{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(f'./config/{CONFIG.model.architecture}.yaml', log_dir)

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
    train_loader = get_dataloader('train')
    test_loader = get_dataloader('test')

    model = MyModel()
    initial_lr = CONFIG.learning.optimizer.initial_lr
    optimizer = init_optimizer(model, initial_lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    state = {
        'best_loss': float('inf'),
        'best_model': model.cpu().state_dict(),
        'best_epoch': 0,
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
        else:
            count = 0
        state['latest_model'] = model.cpu().state_dict()
        state['latest_epoch'] = epoch
        state['latest_optimizer'] = optimizer.state_dict()
        scheduler.step()
        torch.save(state, f'{log_dir}/state_dict.pt')

    tb_writer.close()
    if CONFIG.learning.eval_dataset:
        eval_dataset(log_dir)


def init_optimizer(model, initial_lr=None):
    optimizer_algorithm = CONFIG.learning.optimizer.algorithm
    if optimizer_algorithm == 'Adam':
        # eps needs to be set for stability
        # https://discuss.pytorch.org/t/adam-optimizer-fp16-autocast/101814
        if initial_lr is not None:
            optimizer = optim.Adam(model.parameters(), lr=initial_lr, eps=1e-4)
        else:
            optimizer = optim.Adam(model.parameters())
    elif optimizer_algorithm == 'SGD':
        if initial_lr is not None:
            optimizer = optim.SGD(model.parameters(), lr=initial_lr)
        else:
            optimizer = optim.SGD(model.parameters())

    return optimizer


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
        shutil.rmtree(log_dir)
