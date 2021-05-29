import torch

import _paths
from config import CONFIG


def get_dif_loss(dif1, dif2, dif_size):
    if dif_size == 0:
        return torch.sum(torch.zeros(1))
    tmp = CONFIG.learning.margin - dif1 + dif2
    tmp *= torch.gt(tmp, torch.zeros_like(tmp))
    dif_loss = torch.sum(tmp) / CONFIG.learning.n_sample / dif_size

    return dif_loss


def get_sim_loss(sim1, sim2, sim_size):
    if sim_size == 0:
        return torch.sum(torch.zeros(1))

    tmp = torch.abs(sim1 - sim2) - CONFIG.learning.margin
    tmp *= torch.gt(tmp, torch.zeros_like(tmp))
    sim_loss = torch.sum(tmp) / CONFIG.learning.n_sample / sim_size
    return sim_loss


def get_dif_acc(dif1, dif2, dif_size):
    if dif_size == 0:
        return torch.sum(torch.zeros(1))

    dif_acc = torch.sum(torch.gt(dif1, dif2)) / dif_size

    return dif_acc


def get_sim_acc(sim1, sim2, sim_size):
    if sim_size == 0:
        return torch.sum(torch.zeros(1))

    # TODO: similarity version
    sim_acc = torch.sum(torch.zeros(1)) / sim_size
    return sim_acc
