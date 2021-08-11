import torch.nn as nn
from torchvision.models import resnet34, resnet50


def get_resnet(n_layer, pretrained):
    if n_layer == '34':
        model = resnet34(pretrained)
        model.fc = nn.Linear(512, 1)
    elif n_layer == '50':
        model = resnet50(pretrained)
        model.fc = nn.Linear(2048, 1)

    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )
    return model
