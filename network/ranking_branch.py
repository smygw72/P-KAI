import torch
import torch.nn as nn


class RankingBranch(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(RankingBranch, self).__init__()
        self.inplanes = 1024
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, down_size=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.new_fc = nn.Linear(512 * block.expansion, num_classes)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.new_fc(x)
        x = self.Tanh(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)
