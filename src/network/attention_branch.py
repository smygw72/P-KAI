import torch
import torch.nn as nn


def min_max(inputs):
    min_val = torch.min(inputs)
    max_val = torch.max(inputs)
    output = (inputs - min_val) / (max_val  - min_val)
    return output

class SingleBranch(nn.Module):
    def __init__(self, block, layers):
        super(SingleBranch, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.att_conv4 = nn.Conv2d(block.channel_num*2, block.channel_num*2, kernel_size=3, padding=1, bias=False)
        self.att_conv6 = nn.Conv2d(block.channel_num*2, block.channel_num, kernel_size=3, padding=1, bias=False)
        self.bn_att4 = nn.BatchNorm2d(block.channel_num*2)
        self.bn_att6 = nn.BatchNorm2d(block.channel_num)
        self.att_wgp = nn.Conv2d(block.channel_num, 1, kernel_size=14, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        ax = self.relu(self.bn_att4(self.att_conv4(x)))
        ax = self.relu(self.bn_att6(self.att_conv6(ax)))
        att_map = torch.sum(ax, dim=1, keepdim=True)
        att = min_max(att_map)
        ax = self.att_wgp(ax)
        return ax, att


class AttentionBranch(nn.Module):
    def __init__(self, block, layers):
        super(AttentionBranch, self).__init__()
        self.good_branch = SingleBranch(block, layers)
        self.bad_branch = SingleBranch(block, layers)

    def forward(self, x):
        ax_good, att_good = self.good_branch(x)
        ax_bad, att_bad = self.bad_branch(x)
        return ax_good, att_good, ax_bad, att_bad