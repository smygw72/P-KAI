import torch
import torch.nn as nn

from src.network.block import BasicBlock, Bottleneck
from src.network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from src.network.attention_branch import AttentionBranch
from src.network.ranking_branch import RankingBranch
from src.network.tcn import TemporalConvNet


class MyModel(nn.Module):
    def __init__(self, cfg, learning_or_inference):
        super(MyModel, self).__init__()

        self.arch = cfg.model.architecture
        if learning_or_inference == 'learning':
            self.n_frame = cfg.learning.sampling.n_frame
        elif learning_or_inference == 'inference':
            self.n_frame = cfg.inference.n_frame
        base = cfg.model.base
        pretrained = cfg.model.pretrained
        tcn_levels = cfg.model.tcn.levels
        n_unit = cfg.model.tcn.n_unit
        hidden_channels = [n_unit] * tcn_levels
        kernel_size = cfg.model.tcn.kernel_size
        dropout = cfg.model.tcn.dropout
        disable_attention = cfg.model.disable_attention
        self.disable_bad = cfg.model.disable_bad

        self.network = _get_network(
            self.arch, base, pretrained,
            disable_attention, self.disable_bad,
            hidden_channels, kernel_size, dropout, self.n_frame
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = int(x.size(0) / self.n_frame)

        rx_good, ax_good, att_good, rx_bad, ax_bad, att_bad = self.network(x)

        # (B*L, C=1) -> (B, L)
        rx_good = rx_good.view(batch_size, self.n_frame)

        if self.arch == 'PDR':
            return [rx_good, None, None, None, None, None]

        ax_good = ax_good.view(batch_size, self.n_frame)
        # (B*L, C=1, H=14, W=14) -> (B, L, C=1, H=14, W=14)
        att_good = att_good.view(batch_size, self.n_frame, 1, 14, 14)

        if self.disable_bad is True:
            return [rx_good, ax_good, att_good, None, None, None]
        else:
            rx_bad = rx_bad.view(batch_size, self.n_frame)
            ax_bad = ax_bad.view(batch_size, self.n_frame)
            att_bad = att_bad.view(batch_size, self.n_frame, 1, 14, 14)
            return [rx_good, ax_good, att_good, rx_bad, ax_bad, att_bad]


# Pairwise Deep Ranking
class PDR(nn.Module):
    def __init__(self, base, pretrained):
        super(PDR, self).__init__()

        self.base_model, block, layers, base_out_channel, __ = _get_resnet(base, pretrained)
        self.ranking_branch = RankingBranch(block, layers, base_out_channel)
        self.new_fc = nn.Linear(512 * block.expansion, 1)
        # self.Tanh = nn.Tanh()

    def forward(self, x):
        # (B*L, C=1, H=224, W=224) -> (B*L, C, H=14, W=14)
        x = self.base_model(x)
        # (B*L, C, H=14, W=14) -> (B*L, C, H=1, W=1)
        x = self.ranking_branch(x)
        # (B*L, C, H=1, W=1) -> (B*L, C)
        x = x.squeeze()
        # (B*L, C) -> (B*L, 1)
        x = self.new_fc(x)
        # x = self.Tanh(x)
        return [x, None, None, None, None, None]

# Attention Pairwiser Ranking
# refer to https://github.com/mosa-mprg/attention_pairwise_ranking/blob/master/resnet.py
class APR(nn.Module):
    def __init__(self, base, pretrained, disable_attention, disable_bad):
        super(APR, self).__init__()

        self.disable_attention = disable_attention
        self.disable_bad = disable_bad

        self.base_model, block, layers, base_out_channel, __ = _get_resnet(base, pretrained)
        self.attention_branch = AttentionBranch(block, layers)
        self.ranking_branch_good = RankingBranch(block, layers, base_out_channel)
        self.new_fc_good = nn.Linear(512 * block.expansion, 1)
        if self.disable_bad is False:
            self.ranking_branch_bad = RankingBranch(block, layers, base_out_channel)
            self.new_fc_bad = nn.Linear(512 * block.expansion, 1)
        # self.Tanh = nn.Tanh()

    def forward(self, x):
        # (B*L, C=1, H=224, W=224) -> (B*L, C, H=14, W=14)
        x = self.base_model(x)

        # (B*L, C, H=14, W=14) ->
        # ax_good, ax_bad: (B*L, C=1, H=1, W=1)
        # att_good att_bad: (B*L, C=1, H=14, W=14)
        ax_good, att_good, ax_bad, att_bad = self.attention_branch(x)

        if self.disable_attention is False:
            x_good = x * att_good
            x_bad = x * att_bad
        else:
            x_good = x
            x_bad = x

        # good network
        # (B*L, C, H=14, W=14) -> (B*L, C, H=1, W=1)
        rx_good = self.ranking_branch_good(x_good)
        # (B*L, C, H=1, W=1) -> (B*L, C)
        rx_good = rx_good.squeeze()
        # (B*L, C) -> (B*L, C=1)
        rx_good = self.new_fc_good(rx_good)
        # rx_good = self.Tanh(rx_good)

        # bad network
        if self.disable_bad is True:
            return [rx_good, ax_good, att_good, None, None, None]
        else:
            rx_bad = self.ranking_branch_bad(x_bad)
            rx_bad = rx_bad.squeeze()
            rx_bad = self.new_fc_bad(rx_bad)
            # rx_bad = self.Tanh(rx_bad)
            return [rx_good, ax_good, att_good, rx_bad, ax_bad, att_bad]


# Our model #1
class APR_TCN(nn.Module):
    def __init__(self, base, pretrained, disable_attention, disable_bad, hidden_channels, kernel_size, dropout, n_frame):
        super(APR_TCN, self).__init__()

        self.n_frame = n_frame
        self.disable_attention = disable_attention
        self.disable_bad = disable_bad

        self.base_model, block, layers, base_out_channel, rb_out_channel = _get_resnet(base, pretrained)
        self.attention_branch = AttentionBranch(block, layers)
        self.ranking_branch_good = RankingBranch(block, layers, base_out_channel)
        self.tcn = TemporalConvNet(rb_out_channel, hidden_channels, kernel_size, dropout)
        self.new_fc_good = nn.Linear(hidden_channels[-1], 1)
        self.new_fc_bad = nn.Linear(hidden_channels[-1], 1)
        if self.disable_bad is False:
            self.ranking_branch_bad = RankingBranch(block, layers, base_out_channel)
            self.new_fc_bad = nn.Linear(hidden_channels[-1], 1)
        # self.Tanh = nn.Tanh()

    def forward(self, x):

        batch_size = int(x.size(0) / self.n_frame)

        # (B*L, C_in=1, H=224, W=224) -> (B*L, C_out, H=14, W=14)
        x = self.base_model(x)
        # (B*L, C, H=14, W=14) ->
        # ax_good, ax_bad: (B*L, C=1, H=1, W=1)
        # att_good att_bad: (B*L, C=1, H=14, W=14)
        ax_good, att_good, ax_bad, att_bad = self.attention_branch(x)
        att_height, att_width = att_good.size(2), att_good.size(3)

        if self.disable_attention is False:
            x_good = x * att_good
            x_bad = x * att_bad
        else:
            x_good = x
            x_bad = x

        # (B*L, C, H=14, W=14) -> (B*L, C, H=1, W=1)
        rx_good = self.ranking_branch_good(x_good)
        # (B*L, C, H=1, W=1) -> (B, L, C)
        rx_good = rx_good.view(batch_size, self.n_frame, -1)
        # (B, L, C) -> (B, C, L)
        rx_good = rx_good.transpose(2, 1)
        rx_good = self.tcn(rx_good)
        # (B, C, L) -> (B, L, C)
        rx_good = rx_good.transpose(2, 1)
        # (B, L, C) -> (B*L, C)
        rx_good = rx_good.reshape(batch_size * self.n_frame, -1)
        # (B*L, C) -> (B*L, C=1)
        rx_good = self.new_fc_good(rx_good)

        # bad network
        if self.disable_bad is True:
            return [rx_good, ax_good, att_good, None, None, None]
        else:
            rx_bad = self.ranking_branch_bad(x_bad)
            rx_bad = rx_bad.view(batch_size, self.n_frame, -1)
            rx_bad = rx_bad.transpose(2, 1)
            rx_bad = self.tcn(rx_bad)
            rx_bad = rx_bad.transpose(2, 1)
            rx_bad = rx_bad.reshape(batch_size * self.n_frame, -1)
            rx_bad = self.new_fc_bad(rx_bad)
            return [rx_good, ax_good, att_good, rx_bad, ax_bad, att_bad]

# Our model #2
class TCN_APR(nn.Module):
    def __init__(self, base, pretrained, disable_attention, disable_bad, hidden_channels, kernel_size, dropout, n_frame):
        super(TCN_APR, self).__init__()

        self.n_frame = n_frame
        self.disable_attention = disable_attention
        self.disable_bad = disable_bad

        self.base_model, block, layers, base_out_channel, rb_out_channel = _get_resnet(base, pretrained)
        hidden_channels[-1] = base_out_channel
        self.tcn = TemporalConvNet(base_out_channel, hidden_channels, kernel_size, dropout)
        self.attention_branch = AttentionBranch(block, layers)
        self.ranking_branch_good = RankingBranch(block, layers, base_out_channel)
        self.new_fc_good = nn.Linear(rb_out_channel, 1)
        if self.disable_bad is False:
            self.ranking_branch_bad = RankingBranch(block, layers, base_out_channel)
            self.new_fc_bad = nn.Linear(rb_out_channel, 1)
        # self.Tanh = nn.Tanh()


    def forward(self, x):

        batch_size = int(x.size(0) / self.n_frame)

        # (B*L, C_in=1, H=224, W=224) -> (B*L, C_out, H=14, W=14)
        x = self.base_model(x)

        # (B*L, C, H=14, W=14) -> (B, L, C, H*W)
        height, width = x.size(2), x.size(3)
        channel = x.size(1)
        x = x.view(batch_size, self.n_frame, channel, -1)
        # (B, L, C, H*W) -> (B, H*W, C, L)
        x = x.transpose(3, 1)
        # (B, H*W, C, L) -> (B*H*W, C, L)
        x = x.reshape(batch_size*height*width, channel, self.n_frame)
        # (B*H*W, C_in, L) -> (B, H*W, C_out, L)
        x = self.tcn(x)
        # (B*H*W, C, L) -> (B, H*W, C, L)
        x = x.view(batch_size, height*width, -1, self.n_frame)
        # (B, H*W, C, L) -> (B, L, C, H*W)
        x = x.transpose(3, 1)
        # (B, L, C, H*W) -> (B*L, C, H, W)
        x = x.reshape(batch_size*self.n_frame, -1, height, width)
        # (B*L, C, H=14, W=14) ->
        # ax_good, ax_bad: (B*L, C=1, H=1, W=1)
        # att_good att_bad: (B*L, C=1, H=14, W=14)
        ax_good, att_good, ax_bad, att_bad = self.attention_branch(x)

        if self.disable_attention is False:
            x_good = x * att_good
            x_bad = x * att_bad
        else:
            x_good = x
            x_bad = x

        # (B*L, C_in, H=14, W=14) -> (B*L, C_out, H=1, W=1)
        rx_good = self.ranking_branch_good(x_good)
        # (B*L, C, H=1, W=1) -> (B*L, C)
        rx_good = rx_good.squeeze()
        # (B*L, C_in) -> (B*L, C_out=1)
        rx_good = self.new_fc_good(rx_good)
        # rx_good = self.Tanh(rx_good)

        if self.disable_bad is True:
            return [rx_good, ax_good, att_good, None, None, None]
        else:
            rx_bad = self.ranking_branch_bad(x_bad)
            rx_bad = rx_bad.squeeze()
            rx_bad = self.new_fc_bad(rx_bad)
            # rx_bad = self.Tanh(rx_bad)
            return [rx_good, ax_good, att_good, rx_bad, ax_bad, att_bad]


def _get_network(arch, base, pretrained, disable_attention, disable_bad, hidden_channels, kernel_size, dropout, n_frame):
    if arch == 'PDR':
        model = PDR(base, pretrained)
    elif arch == 'APR':
        model = APR(base, pretrained, disable_attention, disable_bad)
    elif arch == 'APR_TCN':
        model = APR_TCN(base, pretrained, disable_attention, disable_bad, hidden_channels, kernel_size, dropout, n_frame)
    elif arch == 'TCN_APR':
        model = TCN_APR(base, pretrained, disable_attention, disable_bad, hidden_channels, kernel_size, dropout, n_frame)
    return model


def _get_resnet(base, pretrained):

    if base == 'resnet18':
        block = BasicBlock
        layers = [2, 2, 2, 2]
        model = resnet18(block, layers, pretrained)
        base_out_channel = 256
        rb_out_channel = 512
    elif base == 'resnet34':
        block = BasicBlock
        layers = [3, 4, 6, 3]
        model = resnet34(block, layers, pretrained)
        base_out_channel = 256
        rb_out_channel = 512
    elif base == 'resnet50':
        block = Bottleneck
        layers = [3, 4, 6, 3]
        model = resnet50(block, layers, pretrained)
        base_out_channel = 1024
        rb_out_channel = 2048
    elif base == 'resnet101':
        block = Bottleneck
        layers = [3, 4, 23, 3]
        model = resnet101(block, layers, pretrained)
        base_out_channel = 1024
        rb_out_channel = 2048
    elif base == 'resnet152':
        block = Bottleneck
        layers = [3, 8, 36, 3]
        model = resnet152(block, layers, pretrained)
        base_out_channel = 1024
        rb_out_channel = 2048
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    return model, block, layers, base_out_channel, rb_out_channel

