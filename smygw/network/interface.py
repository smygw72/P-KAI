from .resnet import get_resnet


def get_model(arch):
    if arch == 'resnet34':
        return get_resnet('34')
    elif arch == 'resnet50':
        return get_resnet('50')
