import torch.nn as nn
from torchvision.models import resnet34


def get_model():
    model = resnet34(pretrained=True)
    model.fc = nn.Linear(512, 1)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )
    return model