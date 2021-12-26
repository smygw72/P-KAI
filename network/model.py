import torch.nn as nn

from config import CONFIG
from network.block import BasicBlock, Bottleneck
from network.resnet import resnet34, resnet50
from network.attention_branch import AttentionBranch
from network.ranking_branch import RankingBranch
from network.tcn import TCN


def get_model():
    if CONFIG.model.architecture == 'PDR':
        return PDR()
    elif CONFIG.model.architecture == 'APR':
        return APR()
    elif CONFIG.model.architecture == 'APR_TCN':
        return APR_TCN()
    elif CONFIG.model.architecture == 'TCN_APR':
        return TCN_APR()

def get_resnet():
    pretrained = CONFIG.model.pretrained
    if CONFIG.model.base == 'resnet18':
        block = BasicBlock
        layers = [2, 2, 2, 2]
        model = resnet18(block, layers, pretrained)
    elif CONFIG.model.base == 'resnet34':
        block = BasicBlock
        layers = [3, 4, 6, 3]
        model = resnet34(block, layers, pretrained)
    elif CONFIG.model.base == 'resnet50':
        block = Bottleneck
        layers = [3, 4, 6, 3]
        model = resnet50(block, layers, pretrained)
    elif CONFIG.model.base == 'resnet101':
        block = Bottleneck
        layers = [3, 4, 23, 3]
        model = resnet101(block, layers, pretrained)
    elif CONFIG.model.base == 'resnet152':
        block = Bottleneck
        layers = [3, 8, 36, 3]
        model = resnet152(block, layers, pretrained)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    return model, block, layers


# Pairwise Deep Ranking
class PDR(nn.Module):
    def __init__(self):
        super(PDR, self).__init__()
        self.base_model, block, layers = get_resnet()
        self.ranking_branch = RankingBranch(block, layers)

    def forward(self, x):
        x = self.base_model(x)
        x = self.ranking_branch(x)
        return x, None, None, None, None, None

# Attention Pairwiser Ranking
# refer to https://github.com/mosa-mprg/attention_pairwise_ranking/blob/master/resnet.py
class APR(nn.Module):
    def __init__(self):
        super(APR, self).__init__()
        self.base_model, block, layers = get_resnet()
        self.attention_branch = AttentionBranch(block, layers)
        self.ranking_branch = RankingBranch(block, layers)

    def forward(self, x):
        x = self.base_model(x)
        ax_good, att_good, ax_bad, att_bad = self.attention_branch(x)
        x_good = x * att_good
        x_bad = x * att_bad
        rx_good = self.ranking_branch(x_good)
        rx_bad = self.ranking_branch(x_bad)
        return rx_good, ax_good, att_good, rx_bad, ax_bad, att_bad

# Our model #1
class APR_TCN(nn.Module):
    def __init__(self):
        super(APR_TCN, self).__init__()
        self.base_model, block, layers = get_resnet()
        self.attention_branch = AttentionBranch(block, layers)
        self.ranking_branch = RankingBranch(block, layers)
        self.tcn = TCN()

    def forward(self, x):
        x = self.base_model(x)
        ax_good, att_good, ax_bad, att_bad = self.attention_branch(x)
        x_good = x * att_good
        x_bad = x * att_bad
        x_good = self.tcn(x_good)
        x_bad = self.tcn(x_bad)
        rx_good = self.ranking_branch(x_good)
        rx_bad = self.ranking_branch(x_bad)
        return rx_good, ax_good, att_good, rx_bad, ax_bad, att_bad

# Our model #2
class TCN_APR(nn.Module):
    def __init__(self):
        super(TCN_APR, self).__init__()
        self.base_model, block, layers = get_resnet()
        self.attention_branch = AttentionBranch(block, layers)
        self.ranking_branch = RankingBranch(block, layers)

    def forward(self, x):
        x = self.base_model(x)
        x = self.tcn(x)
        ax_good, att_good, ax_bad, att_bad = self.attention_branch(x)
        x_good = x * att_good
        x_bad = x * att_bad
        rx_good = self.ranking_branch(x_good)
        rx_bad = self.ranking_branch(x_bad)
        return rx_good, ax_good, att_good, rx_bad, ax_bad, att_bad
