import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from data import get_dataloader
from network import get_model
from metric import get_dif_loss, get_sim_loss, get_dif_acc, get_sim_acc
from learning_config import CONFIG


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class AverageMeter(object):
    """Compute and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, train_loader, optimizer, av_meters, change_lr=None):
    for value in av_meters.values():
        value.reset()

    model.train()
    for i, minibatch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        sup_input = minibatch[0].to(device, dtype=torch.float32)
        inf_input = minibatch[1].to(device, dtype=torch.float32)
        label_sim = minibatch[2]

        sup_input = sup_input.view(
            -1, 1, CONFIG.input_size, CONFIG.input_size
        )
        inf_input = inf_input.view(
            -1, 1, CONFIG.input_size, CONFIG.input_size
        )

        sup_output = model(sup_input).to('cpu')
        inf_output = model(inf_input).to('cpu')

        sup_output = sup_output.view(-1, CONFIG.n_sample)
        inf_output = inf_output.view(-1, CONFIG.n_sample)

        dif_size = len(label_sim[~label_sim])
        sim_size = len(label_sim[label_sim])
        total_size = dif_size + sim_size

        dif_loss = get_dif_loss(
            sup_output[~label_sim], inf_output[~label_sim], dif_size
        )
        sim_loss = get_sim_loss(
            sup_output[label_sim], inf_output[label_sim], sim_size
        )
        total_loss = dif_loss + sim_loss  # weighted averrage

        sup_score = torch.mean(sup_output, dim=1)
        inf_score = torch.mean(inf_output, dim=1)

        dif_acc = get_dif_acc(
            sup_score[~label_sim], inf_score[~label_sim], dif_size
        )
        sim_acc = get_sim_acc(
            sup_score[label_sim], inf_score[label_sim], sim_size
        )
        total_acc = (
            dif_size * dif_acc.item() + sim_size * sim_acc.item()
        ) / total_size

        update_meters(
            av_meters,
            dif_loss, dif_acc, dif_size,
            sim_loss, sim_acc, sim_size,
            total_loss, total_acc, total_size
        )

        total_loss.backward()
        optimizer.step()

    total_loss_avg = av_meters['total_loss'].avg
    total_acc_avg = av_meters['total_acc'].avg

    return total_acc_avg, total_loss_avg


def test(model, test_loader, av_meters):
    for value in av_meters.values():
        value.reset()

    model.eval()
    with torch.no_grad():
        for i, minibatch in enumerate(test_loader):
            sup_input = minibatch[0].to(device, dtype=torch.float32)
            inf_input = minibatch[1].to(device, dtype=torch.float32)
            label_sim = minibatch[2]

            sup_input = sup_input.view(
                -1, 1, CONFIG.input_size, CONFIG.input_size
            )
            inf_input = inf_input.view(
                -1, 1, CONFIG.input_size, CONFIG.input_size
            )

            sup_output = model(sup_input).to('cpu').detach()
            inf_output = model(inf_input).to('cpu').detach()

            sup_output = sup_output.view(-1, CONFIG.n_sample)
            inf_output = inf_output.view(-1, CONFIG.n_sample)

            dif_size = len(label_sim[~label_sim])
            sim_size = len(label_sim[label_sim])
            total_size = dif_size + sim_size

            dif_loss = get_dif_loss(
                sup_output[~label_sim], inf_output[~label_sim], dif_size
            )
            sim_loss = get_sim_loss(
                sup_output[label_sim], inf_output[label_sim], sim_size
            )
            total_loss = dif_loss + sim_loss

            sup_score = torch.mean(sup_output, dim=1)
            inf_score = torch.mean(inf_output, dim=1)

            dif_acc = get_dif_acc(
                sup_score[~label_sim], inf_score[~label_sim], dif_size
            )
            sim_acc = get_sim_acc(
                sup_score[label_sim], inf_score[label_sim], sim_size
            )
            total_acc = (
                dif_size * dif_acc.item() + sim_size * sim_acc.item()
            ) / total_size

            update_meters(
                av_meters,
                dif_loss, dif_acc, dif_size,
                sim_loss, sim_acc, sim_size,
                total_loss, total_acc, total_size
            )

        total_loss_avg = av_meters['total_loss'].avg
        total_acc_avg = av_meters['total_acc'].avg

        return total_acc_avg, total_loss_avg


def update_meters(av_meters,
                  dif_loss, dif_acc, dif_size,
                  sim_loss, sim_acc, sim_size,
                  total_loss, total_acc, total_size):

    if dif_size != 0:
        av_meters['dif_loss'].update(dif_loss.item(), dif_size)
        av_meters['dif_acc'].update(dif_acc.item(), dif_size)

    if sim_size != 0:
        av_meters['sim_loss'].update(sim_loss.item(), sim_size)
        av_meters['sim_acc'].update(sim_acc.item(), sim_size)

    av_meters['total_loss'].update(total_loss, total_size)
    av_meters['total_acc'].update(total_acc, total_size)


def lr_decay(optimizer, epoch):
    if epoch % 10 == 0:
        new_lr = CONFIG.lr / (10 ** (epoch // 10))
        optimizer.param_groups[0]['lr'] = new_lr
    return optimizer


def main():
    set_seed(CONFIG.seed)

    # log
    writer = SummaryWriter(f'{CONFIG.path.log_dir}')
    av_meters = {
        'dif_loss': AverageMeter(),
        'sim_loss': AverageMeter(),
        'total_loss': AverageMeter(),
        'dif_acc': AverageMeter(),
        'sim_acc': AverageMeter(),
        'total_acc': AverageMeter()
    }

    train_loader = get_dataloader('train')
    test_loader = get_dataloader('test')

    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.lr)

    best_loss = float('inf')
    count = 0

    for epoch in range(1, CONFIG.epochs + 1):
        print(f'Epoch {epoch}')
        optimizer = lr_decay(optimizer, epoch)

        train_acc, train_loss = train(
            model, train_loader, optimizer, av_meters, lr_decay
        )
        # print(f'train acc: {train_acc}')
        test_acc, test_loss = test(
            model, test_loader, av_meters
        )
        print(f'test acc : {test_acc}')

        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)

        # save latest model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f'{CONFIG.path.model_dir}/{CONFIG.version}_latest.tar'
        )

        # save best model (without optimizer)
        if best_loss > test_loss:
            count += 1
            if count >= CONFIG.save_ths:
                best_loss = test_loss
                torch.save(
                    model.state_dict(),
                    f'{CONFIG.path.model_dir}/{CONFIG.version}.pth'
                )
        else:
            count = 0

    writer.close()


if __name__ == "__main__":
    main()
