import torch

from learning_config import CONFIG


def get_loss(sup_output, inf_output):
    diff = CONFIG.margin - sup_output + inf_output
    diff *= torch.gt(diff, torch.zeros_like(diff))
    loss = torch.sum(diff)

    return loss


def get_accuracy(sup_outputs, inf_outputs):
    batch_size = len(sup_outputs)
    acc = torch.sum(torch.gt(sup_outputs, inf_outputs)).item() / batch_size
    return acc
