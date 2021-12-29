import torch
import torch.nn as nn


def min_max(input):
    min = torch.min(input)
    max = torch.max(input)
    output = (input - min) / (max - min)
    return output

class SingleBranch(nn.Module):
    def __init__(self, block, layers):
        super(SingleBranch, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.att_conv4  = nn.Conv2d(block.channel_num*2, block.channel_num*2, kernel_size=3, padding=1, bias=False)
        self.att_conv6  = nn.Conv2d(block.channel_num*2, block.channel_num, kernel_size=3, padding=1, bias=False)
        self.bn_att4 = nn.BatchNorm2d(block.channel_num*2)
        self.bn_att6 = nn.BatchNorm2d(block.channel_num)
        self.att_wgp = nn.Conv2d(block.channel_num, 1, (14,14), padding=0, bias=False)

    def forward(self, x):
        x = self.relu(self.bn_att4(self.att_conv4(x)))
        x = self.relu(self.bn_att6(self.att_conv6(x)))
        att_map = torch.sum(x, dim=1,keepdim = True)
        att = min_max(att_map)
        x = self.att_wgp(x)
        return x, att


class AttentionBranch(nn.Module):
    def __init__(self, block, layers):
        super(AttentionBranch, self).__init__()
        self.good_branch = SingleBranch(block, layers)
        self.bad_branch = SingleBranch(block, layers)

    def forward(self, x):
        x_good, att_good = self.good_branch(x)
        x_bad, att_bad = self.bad_branch(x)
        return x_good, att_good, x_bad, att_bad