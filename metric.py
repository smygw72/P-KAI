import torch

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


def cal_metrics(sup_output, inf_output, label_sim):
    dif_size = len(label_sim[~label_sim])
    sim_size = len(label_sim[label_sim])
    total_size = dif_size + sim_size

    dif_loss = get_dif_loss(
        sup_output[~label_sim], inf_output[~label_sim], dif_size)
    sim_loss = get_sim_loss(
        sup_output[label_sim], inf_output[label_sim], sim_size)
    total_loss = dif_loss + sim_loss

    sup_score = torch.mean(sup_output, dim=1)
    inf_score = torch.mean(inf_output, dim=1)

    dif_acc = get_dif_acc(sup_score[~label_sim],
                          inf_score[~label_sim],
                          dif_size)
    sim_acc = get_sim_acc(sup_score[label_sim],
                          inf_score[label_sim],
                          sim_size)
    total_acc = (dif_size * dif_acc + sim_size * sim_acc) / total_size

    meters = {
        'dif_loss': dif_loss,
        'sim_loss': sim_loss,
        'total_loss': total_loss,
        'dif_acc': dif_acc,
        'sim_acc': sim_acc,
        'total_acc': total_acc
    }
    sizes = {
        'dif_size': dif_size,
        'sim_size': sim_size,
        'total_size': total_size
    }

    return meters, sizes
