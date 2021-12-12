import os
import random
import numpy as np
from pytz import timezone
from datetime import datetime
from tqdm import tqdm
import hydra
import mlflow
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloader
from metric import cal_metrics
from log import AverageMeter, MlflowWriter, update_av_meters, update_writers

from network.interface import get_model
from config import CONFIG

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def inference(model, minibatch):
    sup_input = minibatch[0].to(device, dtype=torch.float32)
    inf_input = minibatch[1].to(device, dtype=torch.float32)

    input_size = CONFIG.common.input_size
    sup_input = sup_input.view(-1, 1, input_size, input_size)
    inf_input = inf_input.view(-1, 1, input_size, input_size)

    sup_output = model(sup_input)
    inf_output = model(inf_input)

    sup_output = sup_output.view(-1, CONFIG.learning.n_sample)
    inf_output = inf_output.view(-1, CONFIG.learning.n_sample)
    return sup_output, inf_output


def train(model, train_loader, optimizer, av_meters):
    for value in av_meters.values():
        value.reset()

    use_amp = CONFIG.learning.use_amp
    scaler = GradScaler(enabled=use_amp)

    model.train()
    for i, minibatch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            sup_output, inf_output = inference(model, minibatch)
            label_sim = minibatch[2].to(device)

            meters, sizes = cal_metrics(sup_output, inf_output, label_sim)
            scaler.scale(meters['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()

        update_av_meters(av_meters, meters, sizes)


def test(model, test_loader, av_meters):
    for value in av_meters.values():
        value.reset()

    model.eval()
    with torch.no_grad():
        for i, minibatch in enumerate(tqdm(test_loader)):
            sup_output, inf_output = inference(model, minibatch)

            label_sim = minibatch[2]
            meters, sizes = cal_metrics(sup_output, inf_output, label_sim)
            update_av_meters(av_meters, meters, sizes)


def main():
    set_seed(CONFIG.seed)

    # timestamp
    utc_now = datetime.now(timezone('UTC'))
    jst_now = utc_now.astimezone(timezone('Asia/Tokyo'))
    timestamp = datetime.strftime(jst_now, '%m-%d-%H-%M-%S')

    mlflow.set_tracking_uri(os.getcwd() + "/mlruns")
    ml_writer = MlflowWriter(imestamp)
    ml_writer.log_params_from_omegaconf_dict(CONFIG)

    tb_writer = SummaryWriter(f'{CONFIG.learning.log.dir}/{timestamp}')

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

    model = get_model(CONFIG.common.arch, pretrained=True).to(device)
    initial_lr = CONFIG.learning.optimizer.initial_lr
    if CONFIG.learning.optimizer.algorithm == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    elif CONFIG.learning.optimizer.algorithm == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float('inf')
    best_model = model
    count = 0

    for epoch in range(1, CONFIG.learning.epochs + 1):
        print(f'Epoch {epoch}')

        train(model, train_loader, optimizer, av_meters)
        update_writers(tb_writer, ml_writer, av_meters, 'train', epoch)
        train_loss = av_meters["total_loss"].avg
        print(f' train loss : {train_loss}')

        test(model, test_loader, av_meters)
        update_writers(tb_writer, ml_writer, av_meters, 'test', epoch)
        test_loss = av_meters["total_loss"].avg
        print(f' test  loss : {test_loss}')

        # save best model (without optimizer)
        if best_loss > test_loss:
            count += 1
            if count >= CONFIG.learning.save_ths:
                best_loss = test_loss
                best_model = model
        else:
            count = 0
        scheduler.step()

    tb_writer.close()
    ml_writer.log_torch_model(best_model)
    ml_writer.set_terminated()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
