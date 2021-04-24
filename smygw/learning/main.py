import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from data import get_dataloader
from network import get_model
from metric import get_accuracy, get_loss
from learning_config import CONFIG


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

    for i, minibatch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        sup_input = minibatch[0].to(device, dtype=torch.float32)
        inf_input = minibatch[1].to(device, dtype=torch.float32)
        similar = minibatch[2]

        sup_input = sup_input.view(-1, 1, CONFIG.input_size, CONFIG.input_size)
        inf_input = inf_input.view(-1, 1, CONFIG.input_size, CONFIG.input_size)

        sup_output = model(sup_input).to('cpu')
        inf_output = model(inf_input).to('cpu')

        loss = get_loss(sup_output, inf_output)
        loss.backward()
        optimizer.step()

        sup_outputs = torch.cat([sup_outputs, sup_output.detach()], dim=0)
        inf_outputs = torch.cat([inf_outputs, inf_output.detach()], dim=0)

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

            sup_outputs = torch.cat([sup_outputs, sup_output], dim=0)
            inf_outputs = torch.cat([inf_outputs, inf_output], dim=0)

        total_acc = get_accuracy(sup_outputs, inf_outputs)
        total_loss = get_loss(sup_outputs, inf_outputs)

        return total_acc, total_loss


def lr_decay(optimizer, epoch):
    if epoch % 10 == 0:
        new_lr = CONFIG.lr / (10 ** (epoch // 10))
        optimizer.param_groups[0]['lr'] = new_lr
    return optimizer


def main():
    set_seed(CONFIG.seed)

    writer = SummaryWriter(f'{CONFIG.path.log_dir}')

    train_loader = get_dataloader('train')
    test_loader = get_dataloader('test')

    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.lr)

    best_loss = float('inf')
    count = 0

    for epoch in range(1, CONFIG.epochs + 1):
        print(f'Epoch {epoch}')
        # optimizer = lr_decay(optimizer, epoch)

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
            f'{CONFIG.path.model_dor}/{CONFIG.version}_latest.tar'
        )

        # save best model (without optimizer)
        if best_loss > test_loss:
            count += 1
            if count >= CONFIG.save_ths:
                best_loss = test_loss
                torch.save(
                    model.state_dict(),
                    f'{CONFIG.path.model_dor}/{CONFIG.version}.pth'
                )
        else:
            count = 0

    writer.close()


if __name__ == "__main__":
    main()
