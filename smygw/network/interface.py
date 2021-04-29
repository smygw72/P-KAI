from .resnet import get_resnet


def get_model(arch):
    if arch == 'resnet':
        return get_resnet()
