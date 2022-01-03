import torch
import torch.nn as nn

from config.config import CONFIG

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
softplus = nn.Softplus(threshold=20)

def _split_dif_sim(outs, label_sim):
    outs_dif = outs.copy()
    outs_sim = outs.copy()
    for i in range(len(outs)):
        if outs[i] is not None:
            outs_dif[i] = outs[i][~label_sim]
            outs_sim[i] = outs[i][label_sim]
    return outs_dif, outs_sim


def mean_scores(outs):
    for i in [0, 1, 3, 4]:  # {rx/ax}_{good/bad}
        if outs[i] is not None:
          outs[i] = torch.mean(outs[i], dim=1)
    return outs


def _PDR_loss(sup_outs_dif, sup_outs_sim, inf_outs_dif, inf_outs_sim):
    dif_loss = _get_dif_loss(sup_outs_dif[0], inf_outs_dif[0])
    sim_loss = _get_sim_loss(sup_outs_sim[0], inf_outs_sim[0])
    return dif_loss, sim_loss


def _APR_loss(sup_outs_dif, sup_outs_sim, inf_outs_dif, inf_outs_sim):

    dif_loss_rx_good = _get_dif_loss(sup_outs_dif[0], inf_outs_dif[0])
    dif_loss_ax_good = _get_dif_loss(sup_outs_dif[1], inf_outs_dif[1])
    sim_loss_rx_good = _get_sim_loss(sup_outs_sim[0], inf_outs_sim[0])
    sim_loss_ax_good = _get_sim_loss(sup_outs_sim[1], inf_outs_sim[1])

    dif_loss = dif_loss_rx_good + dif_loss_ax_good
    sim_loss = sim_loss_rx_good + sim_loss_ax_good

    if CONFIG.model.disable_bad is False:
        dif_loss_rx_bad = _get_dif_loss(sup_outs_dif[3], inf_outs_dif[3])
        dif_loss_ax_bad = _get_dif_loss(sup_outs_dif[4], inf_outs_dif[4])
        sim_loss_rx_bad = _get_sim_loss(sup_outs_sim[3], inf_outs_sim[3])
        sim_loss_ax_bad = _get_sim_loss(sup_outs_sim[4], inf_outs_sim[4])
        dif_loss = dif_loss + dif_loss_rx_bad + dif_loss_ax_bad
        sim_loss = sim_loss + sim_loss_rx_bad + sim_loss_ax_bad

    return dif_loss, sim_loss


def _get_acc(sup_outs_dif, sup_outs_sim, inf_outs_dif, inf_outs_sim):
    arch = CONFIG.model.architecture

    if (arch == 'PDR') or (CONFIG.model.disable_bad is True):
        sup_score = sup_outs_dif[0]
        inf_score = inf_outs_dif[0]
    else:
        sup_score = sup_outs_dif[0] - sup_outs_dif[3]
        inf_score = inf_outs_dif[0] - inf_outs_dif[3]

    dif_acc = _get_dif_acc(sup_score, inf_score)
    sim_acc = _get_sim_acc(sup_score, inf_score)

    return dif_acc, sim_acc


def _get_dif_loss(dif1, dif2):
    if len(dif1) == 0:
        return torch.zeros(1).to(device)
    if CONFIG.learning.loss.method == 'marginal_loss':
        loss = CONFIG.learning.loss.margin - dif1 + dif2
        is_positive = torch.gt(loss, 0.0).float()
        loss = loss * is_positive
    elif CONFIG.learning.loss.method == 'softplus':
        loss = softplus(-dif1 + dif2)
    loss_avg = torch.mean(loss)
    return loss_avg


def _get_sim_loss(sim1, sim2):
    if len(sim1) == 0:
        return torch.zeros(1).to(device)
    loss = torch.abs(sim1 - sim2) - CONFIG.learning.loss.margin
    is_positive = torch.gt(loss, 0.0).float()
    loss = loss * is_positive.float()
    loss_avg = torch.mean(loss)
    return loss_avg


def _get_dif_acc(dif1, dif2):
    if len(dif1) == 0:
        return torch.zeros(1).to(device)
    dif_acc = torch.mean(torch.gt(dif1, dif2).float())
    return dif_acc


def _get_sim_acc(sim1, sim2):
    if len(sim1) == 0:
        return torch.zeros(1).to(device)
    # TODO: similarity version
    sim_acc = torch.zeros(1).to(device)
    return sim_acc


def get_metrics(sup_outs, inf_outs, label_sim):

    arch = CONFIG.model.architecture

    # split into dif/sim tensors
    sup_outs_dif, sup_outs_sim = _split_dif_sim(sup_outs, label_sim)
    inf_outs_dif, inf_outs_sim = _split_dif_sim(inf_outs, label_sim)
    dif_size = len(label_sim[~label_sim])
    sim_size = len(label_sim[label_sim])
    total_size = dif_size + sim_size

    # loss
    loss_func = _PDR_loss if arch == 'PDR' else _APR_loss
    dif_loss, sim_loss = loss_func(sup_outs_dif, sup_outs_sim, inf_outs_dif, inf_outs_sim)
    if CONFIG.learning.loss.enable_sim_loss:
        total_loss = dif_loss + sim_loss
    else:
        total_loss = dif_loss

    # average frame-by-frame scores for each video
    sup_outs_dif = mean_scores(sup_outs_dif)
    inf_outs_dif = mean_scores(inf_outs_dif)
    sup_outs_sim = mean_scores(sup_outs_sim)
    inf_outs_sim = mean_scores(inf_outs_sim)

    # accuracy
    dif_acc, sim_acc = _get_acc(sup_outs_dif, sup_outs_sim, inf_outs_dif, inf_outs_sim)
    # TODO: sim_accを実装
    total_acc = dif_acc
    # total_acc = (dif_size * dif_acc + sim_size * sim_acc) / total_size

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
