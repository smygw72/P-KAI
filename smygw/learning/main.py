import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from data import get_dataloader
from network import get_model
from metric import get_accuracy, get_loss
from config import CONFIG


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, train_loader, optimizer, change_lr=None):
    model.train()

    sup_outputs = torch.Tensor()
    inf_outputs = torch.Tensor()

    if change_lr:
        optimizer = change_lr(optimizer, epoch)

    for i, (sup_input, inf_input) in enumerate(train_loader):
        optimizer.zero_grad()

        sup_input = sup_input.to(device, dtype=torch.float32)
        inf_input = inf_input.to(device, dtype=torch.float32)

        sup_output = model(sup_input)
        inf_output = model(inf_input)

        sup_outputs = torch.cat([sup_outputs, sup_output.detach(), dim=0])
        inf_outputs = torch.cat([inf_outputs, inf_output.detach(), dim=0])

        loss = get_loss(sup_output, inf_output)
        loss.backward()
        optimizer.step()

    total_acc = get_accuracy(sup_outputs, inf_outputs)
    total_loss = get_loss(sup_outputs, inf_outputs)

    return total_acc, total_loss


def test(model, test_loader):
    model.eval()

    sup_outputs = torch.Tensor()
    inf_outputs = torch.Tensor()

    with torch.no_grad():
        for (sup_input, inf_input) in test_loader:
            sup_input = sup_input.to(device, dtype=torch.float32)
            inf_input = inf_input.to(device, dtype=torch.float32)

            sup_output = model(sup_input).detach()
            inf_output = model(inf_input).detach()

            sup_outputs = torch.cat([sup_outputs, sup_output, dim=0])
            inf_outputs = torch.cat([inf_outputs, inf_output, dim=0])

        total_acc = get_accuracy(sup_outputs, inf_outputs)
        total_loss = get_loss(sup_outputs, inf_outputs)

        return total_acc, total_loss


def lr_decay(optimizer, epoch):
    if epoch % 10 == 0:
        new_lr = learning_rate / (10**(epoch//10))
        optimizer = setlr(optimizer, new_lr)
        print(f'Changed learning rate to {new_lr}')
    return optimizer


def main():
    set_seed(CONFIG.other.seed)

    writer = SummaryWriter(f'./smygw/learning/logs/')

    train_loader = get_dataloader('train')
    test_loader = get_dataloader('test')

    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning.lr)

    best_loss = float('inf')
    count = 0

    for epoch in tqdm(range(1, CONFIG.learning.epochs+1)):
        train_acc, train_loss = train(
            model, train_loader, optimizer, lr_decay
        )
        test_acc, test_loss = test(
            model, test_loader
        )

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
            "./smygw/model/latest_model.tar",
        )

        # save best model (without optimizer)
        if best_loss > test_loss:
            count += 1
            if count >= CONFIG.other.save_ths:
                best_loss = test_loss
                torch.save(model.state_dict(), './smygw/model/best_model.pth')
        else:
            count = 0

    writer.close()


if __name__ == "__main__":
    main()
