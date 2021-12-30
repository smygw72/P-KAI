import torch
import torch.nn as nn

from config.config import CONFIG


softplus = nn.Softplus()


def _split_dif_sim(outs, label_sim):
    outs_dif = outs.copy()
    outs_sim = outs.copy()
    for i in range(len(outs)):
        if outs[i] is not None:
            outs_dif[i] = outs[i][~label_sim]
            outs_sim[i] = outs[i][label_sim]
    return outs_dif, outs_sim


def _mean_scores(outs):
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
    dif_loss_rx_bad = _get_dif_loss(sup_outs_dif[3], inf_outs_dif[3])
    dif_loss_ax_bad = _get_dif_loss(sup_outs_dif[4], inf_outs_dif[4])

    if CONFIG.model.disable_bad is False:
        dif_loss = dif_loss_rx_good + dif_loss_ax_good + dif_loss_rx_bad + dif_loss_ax_bad
    elif CONFIG.model.disable_bad is True:
        dif_loss = dif_loss_rx_good + dif_loss_ax_good

    sim_loss_rx_good = _get_sim_loss(sup_outs_sim[0], inf_outs_sim[0])
    sim_loss_ax_good = _get_sim_loss(sup_outs_sim[1], inf_outs_sim[1])
    sim_loss_rx_bad = _get_sim_loss(sup_outs_sim[3], inf_outs_sim[3])
    sim_loss_ax_bad = _get_sim_loss(sup_outs_sim[4], inf_outs_sim[4])

    if CONFIG.model.disable_bad is False:
        sim_loss = sim_loss_rx_good + sim_loss_ax_good + sim_loss_rx_bad + sim_loss_ax_bad
    elif CONFIG.model.disable_bad is True:
        sim_loss = sim_loss_rx_good + sim_loss_ax_good

    return dif_loss, sim_loss


def _PDR_acc(sup_outs_dif, sup_outs_sim, inf_outs_dif, inf_outs_sim):
    dif_acc = _get_dif_acc(sup_outs_dif[0], inf_outs_dif[0])
    sim_acc = _get_sim_acc(sup_outs_sim[0], inf_outs_sim[0])
    return dif_acc, sim_acc


def _APR_acc(sup_outs_dif, sup_outs_sim, inf_outs_dif, inf_outs_sim):
    if CONFIG.model.disable_bad is False: # Equation (4) in APR paper.
        dif_acc = _get_dif_acc(sup_outs_dif[0], inf_outs_dif[3])
        sim_acc = _get_sim_acc(sup_outs_sim[0], inf_outs_sim[3])
    else:
        dif_acc = _get_dif_acc(sup_outs_dif[0], inf_outs_dif[0])
        sim_acc = _get_sim_acc(sup_outs_sim[0], inf_outs_sim[0])
    return dif_acc, sim_acc


def _get_dif_loss(dif1, dif2):
    if len(dif1) == 0:
        return torch.zeros(1)
    if CONFIG.learning.loss.method == 'marginal_loss':
        loss = CONFIG.learning.loss.margin - dif1 + dif2
        loss *= torch.gt(loss, torch.zeros_like(loss))
    elif CONFIG.learning.loss.method == 'softplus':
        loss = softplus(-dif1 + dif2)
    loss_avg = torch.mean(loss)
    return loss_avg


def _get_sim_loss(sim1, sim2):
    if len(sim1) == 0:
        return torch.zeros(1)
    loss = torch.abs(sim1 - sim2) - CONFIG.learning.loss.margin
    loss *= torch.gt(loss, torch.zeros_like(loss))
    loss_avg = torch.mean(loss)
    return loss_avg


def _get_dif_acc(dif1, dif2):
    if len(dif1) == 0:
        return torch.sum(torch.zeros(1))
    dif_acc = torch.mean(torch.gt(dif1, dif2).float())
    return dif_acc


def _get_sim_acc(sim1, sim2):
    if (len(sim1) == 0) or (CONFIG.learning.loss.enable_sim_loss is True):
        return torch.sum(torch.zeros(1))
    # TODO: similarity version
    sim_acc = torch.mean(torch.zeros(1))
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
    total_loss = dif_loss + sim_loss

    # average frame-by-frame scores for each video
    sup_outs_dif = _mean_scores(sup_outs_dif)
    inf_outs_dif = _mean_scores(inf_outs_dif)
    sup_outs_sim = _mean_scores(sup_outs_sim)
    inf_outs_sim = _mean_scores(inf_outs_sim)

    # accuracy
    acc_func = _PDR_acc if arch == 'PDR' else _APR_acc
    dif_acc, sim_acc = acc_func(sup_outs_dif, sup_outs_sim, inf_outs_dif, inf_outs_sim)
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
